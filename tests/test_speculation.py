# coding=utf-8

from copy import deepcopy

from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model
from smolagents.speculation import (
    BranchConfidence,
    ConstantConfidencePredictor,
    SalvageSpeculatingToolCallingAgent,
)
from smolagents.tools import Tool


def make_tool_message(call_id: str, name: str, arguments, content: str = "") -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=content or f"Calling {name}",
        tool_calls=[
            ChatMessageToolCall(
                id=call_id,
                type="function",
                function=ChatMessageToolCallFunction(name=name, arguments=arguments),
            )
        ],
    )


class StepwiseToolCallModel(Model):
    def __init__(self, responses: list[ChatMessage]):
        super().__init__()
        self.responses = responses
        self.index = 0

    def generate(self, messages, tools_to_call_from=None, stop_sequences=None, **kwargs):
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return deepcopy(response)


class CountingSearchTool(Tool):
    name = "search_tool"
    description = "Returns a deterministic result for a search query."
    inputs = {"query": {"type": "string", "description": "Query to search for."}}
    output_type = "string"

    def __init__(self):
        self.calls: list[str] = []
        super().__init__()

    def forward(self, query: str) -> str:
        self.calls.append(query)
        return f"search-result:{query}"


class DangerousDeleteTool(Tool):
    name = "delete_record"
    description = "Deletes a record and should never be speculatively executed."
    inputs = {"record_id": {"type": "string", "description": "Record to delete."}}
    output_type = "string"

    def __init__(self):
        self.calls: list[str] = []
        super().__init__()

    def forward(self, record_id: str) -> str:
        self.calls.append(record_id)
        return f"deleted:{record_id}"


class QueryAwarePredictor:
    def predict(self, tool_call, tags, task, recent_records):
        if tool_call.name == "delete_record":
            return BranchConfidence(correctness=0.05, usefulness=0.05, risk=0.99)
        query = tool_call.arguments.get("query") if isinstance(tool_call.arguments, dict) else ""
        if query == "france capital city":
            return BranchConfidence(correctness=0.20, usefulness=0.95, risk=0.05)
        return BranchConfidence(correctness=0.10, usefulness=0.10, risk=0.05)


def test_salvage_speculation_accepts_exact_match_and_reuses_speculative_output():
    search_tool = CountingSearchTool()
    actor_model = StepwiseToolCallModel(
        [
            make_tool_message("actor_0", "search_tool", {"query": "alpha"}),
            make_tool_message("actor_1", "final_answer", {"answer": "done"}),
        ]
    )
    speculator_model = StepwiseToolCallModel(
        [
            make_tool_message("spec_0", "search_tool", {"query": "alpha"}),
            make_tool_message("spec_1", "final_answer", {"answer": "unused"}),
        ]
    )

    agent = SalvageSpeculatingToolCallingAgent(
        tools=[search_tool],
        model=actor_model,
        speculator_model=speculator_model,
        confidence_predictor=ConstantConfidencePredictor(correctness=0.9, usefulness=0.2, risk=0.05),
        max_steps=2,
        verbosity_level=0,
    )

    result = agent.run("Find alpha")

    assert result == "done"
    assert search_tool.calls == ["alpha"]
    assert agent.get_speculation_summary()["accepted"] == 1
    assert "speculation_records" in agent.memory.get_full_steps()[1]
    assert any(
        record["source"] == "speculator" and record["decision"] == "accepted"
        for record in agent.memory.get_full_steps()[1]["speculation_records"]
    )


def test_salvage_speculation_salvages_branch_and_reuses_cache_later():
    search_tool = CountingSearchTool()
    actor_model = StepwiseToolCallModel(
        [
            make_tool_message("actor_0", "search_tool", {"query": "capital of france"}),
            make_tool_message("actor_1", "search_tool", {"query": "france capital city"}),
            make_tool_message("actor_2", "final_answer", {"answer": "Paris"}),
        ]
    )
    speculator_model = StepwiseToolCallModel(
        [
            make_tool_message("spec_0", "search_tool", {"query": "france capital city"}),
            make_tool_message("spec_1", "final_answer", {"answer": "unused"}),
            make_tool_message("spec_2", "final_answer", {"answer": "unused"}),
        ]
    )

    agent = SalvageSpeculatingToolCallingAgent(
        tools=[search_tool],
        model=actor_model,
        speculator_model=speculator_model,
        confidence_predictor=QueryAwarePredictor(),
        max_steps=3,
        verbosity_level=0,
    )

    result = agent.run("What is the capital of France?")

    assert result == "Paris"
    assert len(search_tool.calls) == 2
    assert "capital of france" in search_tool.calls
    assert "france capital city" in search_tool.calls
    assert agent.get_speculation_summary()["salvaged"] >= 1
    assert agent.get_speculation_summary()["cache_hits"] >= 1
    assert any(
        record["source"] == "actor" and record["cache_hit"]
        for record in agent.memory.get_full_steps()[2]["speculation_records"]
    )


def test_salvage_speculation_suppresses_risky_branches():
    search_tool = CountingSearchTool()
    dangerous_tool = DangerousDeleteTool()
    actor_model = StepwiseToolCallModel(
        [
            make_tool_message("actor_0", "search_tool", {"query": "alpha"}),
            make_tool_message("actor_1", "final_answer", {"answer": "done"}),
        ]
    )
    speculator_model = StepwiseToolCallModel(
        [
            make_tool_message("spec_0", "delete_record", {"record_id": "42"}),
            make_tool_message("spec_1", "final_answer", {"answer": "unused"}),
        ]
    )

    agent = SalvageSpeculatingToolCallingAgent(
        tools=[search_tool, dangerous_tool],
        model=actor_model,
        speculator_model=speculator_model,
        confidence_predictor=QueryAwarePredictor(),
        max_steps=2,
        verbosity_level=0,
    )

    result = agent.run("Find alpha")

    assert result == "done"
    assert search_tool.calls == ["alpha"]
    assert dangerous_tool.calls == []
    assert agent.get_speculation_summary()["suppressed"] >= 1
    assert any(
        record["source"] == "speculator"
        and record["decision"] == "suppressed"
        and record["action_type"] == "delete_record"
        for record in agent.memory.get_full_steps()[1]["speculation_records"]
    )
