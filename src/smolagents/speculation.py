#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .agent_types import AgentAudio, AgentImage
from .agents import ActionOutput, ToolCallingAgent, ToolCall, ToolOutput
from .memory import ActionStep
from .models import ChatMessage, Model, parse_json_if_needed
from .monitoring import LogLevel
from .tools import Tool
from .utils import AgentGenerationError, make_json_serializable


__all__ = [
    "BranchConfidence",
    "BranchDecision",
    "CalibratedConfidencePredictor",
    "ConstantConfidencePredictor",
    "HeuristicConfidencePredictor",
    "ReuseCachedObservationTool",
    "SalvageSpeculatingToolCallingAgent",
    "SemanticTagBundle",
    "SemanticTagExtractor",
    "SpeculationCache",
    "SpeculationRecord",
]


RISKY_KEYWORDS = {
    "book",
    "buy",
    "create",
    "delete",
    "deploy",
    "drop",
    "email",
    "insert",
    "merge",
    "post",
    "purchase",
    "remove",
    "send",
    "submit",
    "transfer",
    "update",
    "upload",
    "write",
}
READ_ONLY_HINTS = {
    "archive",
    "browse",
    "crawl",
    "extract",
    "find",
    "inspect",
    "lookup",
    "page",
    "read",
    "scrape",
    "search",
    "visit",
}
LOCATOR_PREFIX = "locator:"
TOOL_PREFIX = "tool:"
ARG_PREFIX = "arg:"
TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


class BranchDecision(str, Enum):
    ACCEPT = "accepted"
    SALVAGE = "salvaged"
    DISCARD = "discarded"
    SUPPRESS = "suppressed"


@dataclass
class BranchConfidence:
    correctness: float
    usefulness: float
    risk: float

    def dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "usefulness": self.usefulness,
            "risk": self.risk,
        }


@dataclass
class SemanticTagBundle:
    semantic_tags: list[str]
    required_tags: list[str]
    goal_tags: list[str]


@dataclass
class SpeculationRecord:
    action_type: str
    tool_arguments: Any
    textual_observation: str | None
    semantic_tags: list[str]
    source: str
    decision: str
    required_tags: list[str]
    goal_tags: list[str]
    correctness: float | None = None
    usefulness: float | None = None
    risk: float | None = None
    exact_match: bool = False
    cache_hit: bool = False
    step_number: int | None = None

    def dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "tool_arguments": make_json_serializable(self.tool_arguments),
            "textual_observation": self.textual_observation,
            "semantic_tags": self.semantic_tags,
            "source": self.source,
            "decision": self.decision,
            "required_tags": self.required_tags,
            "goal_tags": self.goal_tags,
            "correctness": self.correctness,
            "usefulness": self.usefulness,
            "risk": self.risk,
            "exact_match": self.exact_match,
            "cache_hit": self.cache_hit,
            "step_number": self.step_number,
        }


@dataclass
class CacheEntry:
    action_type: str
    output: Any
    observation_text: str
    semantic_tags: tuple[str, ...]
    required_tags: tuple[str, ...]
    goal_tags: tuple[str, ...]
    source: str
    created_at: float = field(default_factory=time.time)


@dataclass
class _PreparedSpeculativeBranch:
    tool_call: ToolCall
    tags: SemanticTagBundle
    confidence: BranchConfidence
    output: Any = None
    observation_text: str | None = None
    decision: BranchDecision | None = None
    exact_match: bool = False
    finalized: bool = False


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _canonicalize_arguments(arguments: Any) -> str:
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except TypeError:
        return str(arguments)


def _flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        flattened = []
        for inner_value in value.values():
            flattened.extend(_flatten_values(inner_value))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for inner_value in value:
            flattened.extend(_flatten_values(inner_value))
        return flattened
    return [str(value)]


def _looks_like_locator(value: str) -> bool:
    return value.startswith(("http://", "https://", "/", "./", "../")) or (
        len(value) > 3 and "." in value and "/" in value
    )


class SemanticTagExtractor:
    """Extracts lightweight semantic tags used for speculation salvage and cache lookup."""

    def __init__(self, max_task_tokens: int = 16, max_argument_tokens: int = 16, max_observation_tokens: int = 24):
        self.max_task_tokens = max_task_tokens
        self.max_argument_tokens = max_argument_tokens
        self.max_observation_tokens = max_observation_tokens

    def _tokenize(self, text: str, limit: int) -> list[str]:
        return list(dict.fromkeys(TOKEN_RE.findall(text.lower())))[:limit]

    def extract(
        self,
        tool_name: str,
        arguments: Any,
        observation_text: str | None = None,
        task: str | None = None,
    ) -> SemanticTagBundle:
        required_tags = {f"{TOOL_PREFIX}{tool_name}"}
        goal_tags: list[str] = []

        if isinstance(arguments, dict):
            for key, value in arguments.items():
                required_tags.add(f"{ARG_PREFIX}{key.lower()}")
                for flattened_value in _flatten_values(value):
                    if _looks_like_locator(flattened_value):
                        required_tags.add(f"{LOCATOR_PREFIX}{flattened_value.lower()}")
                    goal_tags.extend(self._tokenize(flattened_value, self.max_argument_tokens))
        else:
            for flattened_value in _flatten_values(arguments):
                if _looks_like_locator(flattened_value):
                    required_tags.add(f"{LOCATOR_PREFIX}{flattened_value.lower()}")
                goal_tags.extend(self._tokenize(flattened_value, self.max_argument_tokens))

        if task:
            goal_tags.extend(self._tokenize(task, self.max_task_tokens))
        if observation_text:
            goal_tags.extend(self._tokenize(observation_text, self.max_observation_tokens))

        deduped_goal_tags = list(dict.fromkeys(goal_tags))
        semantic_tags = list(dict.fromkeys(sorted(required_tags) + deduped_goal_tags))
        return SemanticTagBundle(
            semantic_tags=semantic_tags,
            required_tags=sorted(required_tags),
            goal_tags=deduped_goal_tags,
        )


class SpeculationCache:
    """Stores salvaged or accepted tool outputs for later reuse."""

    def __init__(self, max_entries: int = 256, min_goal_overlap: int = 1):
        self.max_entries = max_entries
        self.min_goal_overlap = min_goal_overlap
        self.entries: list[CacheEntry] = []

    def clear(self) -> None:
        self.entries.clear()

    def store(self, entry: CacheEntry) -> None:
        fingerprint = (
            entry.action_type,
            entry.required_tags,
            entry.goal_tags,
        )
        self.entries = [
            cached_entry
            for cached_entry in self.entries
            if (
                cached_entry.action_type,
                cached_entry.required_tags,
                cached_entry.goal_tags,
            )
            != fingerprint
        ]
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def lookup(self, required_tags: list[str], goal_tags: list[str], action_type: str | None = None) -> CacheEntry | None:
        required = set(required_tags)
        goal = set(goal_tags)
        best_match: tuple[float, CacheEntry] | None = None

        for entry in reversed(self.entries):
            if action_type is not None and entry.action_type != action_type:
                continue
            if not required.issubset(set(entry.required_tags)):
                continue

            goal_overlap = len(goal & set(entry.goal_tags))
            if goal and goal_overlap < self.min_goal_overlap:
                continue

            score = float(goal_overlap)
            if entry.action_type == action_type:
                score += 0.5
            score += entry.created_at * 1e-9

            if best_match is None or score > best_match[0]:
                best_match = (score, entry)

        return best_match[1] if best_match else None

    def __len__(self) -> int:
        return len(self.entries)


class ConfidencePredictor(Protocol):
    def predict(
        self,
        tool_call: ToolCall,
        tags: SemanticTagBundle,
        task: str | None,
        recent_records: list[SpeculationRecord],
    ) -> BranchConfidence: ...


class HeuristicConfidencePredictor:
    """Default lightweight predictor used when no offline calibration is provided."""

    def extract_features(
        self,
        tool_call: ToolCall,
        tags: SemanticTagBundle,
        task: str | None,
        recent_records: list[SpeculationRecord],
    ) -> dict[str, float]:
        task_tokens = set(TOKEN_RE.findall((task or "").lower()))
        goal_tokens = set(tags.goal_tags)
        recent_goal_tokens = {
            token
            for record in recent_records[-8:]
            for token in record.goal_tags
            if record.source in {"speculator", "actor"}
        }
        recent_tools = {record.action_type for record in recent_records[-8:]}
        tool_name_lower = tool_call.name.lower()
        flattened_arguments = " ".join(_flatten_values(tool_call.arguments)).lower()

        read_only_tool = float(self._is_read_only_tool(tool_name_lower, flattened_arguments))
        task_overlap = len(goal_tokens & task_tokens) / max(len(goal_tokens), 1)
        recent_overlap = len(goal_tokens & recent_goal_tokens) / max(len(goal_tokens), 1)

        return {
            "read_only_tool": read_only_tool,
            "task_overlap": task_overlap,
            "recent_overlap": recent_overlap,
            "tool_seen_recently": float(tool_call.name in recent_tools),
            "has_locator": float(any(tag.startswith(LOCATOR_PREFIX) for tag in tags.required_tags)),
            "is_final_answer": float(tool_call.name == "final_answer"),
            "risky_keyword": float(any(keyword in f"{tool_name_lower} {flattened_arguments}" for keyword in RISKY_KEYWORDS)),
        }

    def predict(
        self,
        tool_call: ToolCall,
        tags: SemanticTagBundle,
        task: str | None,
        recent_records: list[SpeculationRecord],
    ) -> BranchConfidence:
        features = self.extract_features(tool_call, tags, task, recent_records)

        correctness = _clamp_probability(
            0.15
            + 0.35 * features["task_overlap"]
            + 0.20 * features["recent_overlap"]
            + 0.15 * features["tool_seen_recently"]
            + 0.10 * features["read_only_tool"]
            + 0.05 * features["has_locator"]
            - 0.30 * features["risky_keyword"]
        )
        usefulness = _clamp_probability(
            0.20
            + 0.30 * features["task_overlap"]
            + 0.25 * features["recent_overlap"]
            + 0.20 * features["read_only_tool"]
            + 0.10 * features["has_locator"]
        )
        risk = _clamp_probability(
            0.05
            + 0.50 * (1.0 - features["read_only_tool"])
            + 0.25 * features["is_final_answer"]
            + 0.35 * features["risky_keyword"]
        )

        return BranchConfidence(correctness=correctness, usefulness=usefulness, risk=risk)

    def _is_read_only_tool(self, tool_name: str, argument_text: str) -> bool:
        if any(keyword in f"{tool_name} {argument_text}" for keyword in RISKY_KEYWORDS):
            return False
        return any(hint in tool_name for hint in READ_ONLY_HINTS) or any(
            hint in argument_text for hint in READ_ONLY_HINTS
        )


class CalibratedConfidencePredictor(HeuristicConfidencePredictor):
    """Logistic predictor calibrated from offline labels stored in JSON."""

    def __init__(self, calibration: dict[str, Any]):
        self.calibration = calibration

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibratedConfidencePredictor":
        with open(path, encoding="utf-8") as fp:
            calibration = json.load(fp)
        return cls(calibration=calibration)

    def _score(self, head_name: str, features: dict[str, float]) -> float:
        head = self.calibration[head_name]
        bias = head.get("bias", 0.0)
        weights = head.get("weights", {})
        linear_value = bias + sum(weights.get(feature_name, 0.0) * feature_value for feature_name, feature_value in features.items())
        return _sigmoid(linear_value)

    def predict(
        self,
        tool_call: ToolCall,
        tags: SemanticTagBundle,
        task: str | None,
        recent_records: list[SpeculationRecord],
    ) -> BranchConfidence:
        features = self.extract_features(tool_call, tags, task, recent_records)
        return BranchConfidence(
            correctness=self._score("correctness", features),
            usefulness=self._score("usefulness", features),
            risk=self._score("risk", features),
        )


class ConstantConfidencePredictor:
    """Small helper predictor useful for tests or ablations."""

    def __init__(self, correctness: float, usefulness: float, risk: float):
        self.confidence = BranchConfidence(correctness=correctness, usefulness=usefulness, risk=risk)

    def predict(
        self,
        tool_call: ToolCall,
        tags: SemanticTagBundle,
        task: str | None,
        recent_records: list[SpeculationRecord],
    ) -> BranchConfidence:
        return self.confidence


class ReuseCachedObservationTool(Tool):
    name = "reuse_cached_observation"
    description = (
        "Looks up previously cached observations. Use `required_tags` for hard filters such as tool names, URLs, or files, "
        "and `goal_tags` for the information you are trying to recover."
    )
    inputs = {
        "required_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Mandatory tags that the cached observation must contain.",
        },
        "goal_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Semantic tags that describe the information you hope to recover.",
        },
        "tool_name": {
            "type": "string",
            "description": "Optional tool name to narrow the lookup.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, cache: SpeculationCache):
        self.cache = cache
        super().__init__()

    def forward(self, required_tags: list[str], goal_tags: list[str], tool_name: str | None = None) -> str:
        entry = self.cache.lookup(required_tags=required_tags, goal_tags=goal_tags, action_type=tool_name)
        if entry is None:
            return "CACHE_MISS"
        tags = ", ".join(entry.semantic_tags)
        return f"CACHE_HIT\naction_type={entry.action_type}\nsemantic_tags={tags}\nobservation={entry.observation_text}"


class SalvageSpeculatingToolCallingAgent(ToolCallingAgent):
    """
    Tool-calling agent that runs a fast speculator in parallel with the actor and salvages useful misses into a cache.
    """

    cache_lookup_tool_name = ReuseCachedObservationTool.name

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        speculator_model: Model,
        confidence_predictor: ConfidencePredictor | None = None,
        speculation_cache: SpeculationCache | None = None,
        tag_extractor: SemanticTagExtractor | None = None,
        tau_c: float = 0.85,
        tau_u: float = 0.55,
        tau_r: float = 0.35,
        cache_actor_outputs: bool = True,
        enable_cache_lookup_tool: bool = True,
        speculation_allowlist: list[str] | None = None,
        speculation_blocklist: list[str] | None = None,
        semantic_match_threshold: float = 0.6,
        max_speculative_tool_threads: int | None = None,
        instructions: str | None = None,
        **kwargs,
    ):
        self.speculator_model = speculator_model
        self.confidence_predictor = confidence_predictor or HeuristicConfidencePredictor()
        self.speculation_cache = speculation_cache or SpeculationCache()
        self.tag_extractor = tag_extractor or SemanticTagExtractor()
        self.tau_c = tau_c
        self.tau_u = tau_u
        self.tau_r = tau_r
        self.cache_actor_outputs = cache_actor_outputs
        self.semantic_match_threshold = semantic_match_threshold
        self.max_speculative_tool_threads = max_speculative_tool_threads
        self.speculation_allowlist = set(speculation_allowlist or [])
        self.speculation_blocklist = set(speculation_blocklist or [])
        self._pending_speculative_branches: list[_PreparedSpeculativeBranch] = []
        self._speculation_records: list[SpeculationRecord] = []

        tool_names = {tool.name for tool in tools}
        tools = list(tools)
        if enable_cache_lookup_tool and self.cache_lookup_tool_name not in tool_names:
            tools.append(ReuseCachedObservationTool(self.speculation_cache))

        speculation_instructions = (
            "If a read-only tool may have already been called, you can use `reuse_cached_observation` first. "
            "Use `required_tags` for exact anchors like tool names, URLs, or file paths, and `goal_tags` for the semantic intent."
        )
        merged_instructions = speculation_instructions if not instructions else f"{instructions}\n\n{speculation_instructions}"

        super().__init__(tools=tools, model=model, instructions=merged_instructions, **kwargs)

    @property
    def speculation_tools_and_managed_agents(self) -> list[Any]:
        return [
            tool
            for tool in self.tools_and_managed_agents
            if getattr(tool, "name", None) not in {self.cache_lookup_tool_name}
        ]

    def get_speculation_records(self) -> list[dict[str, Any]]:
        return [record.dict() for record in self._speculation_records]

    def get_speculation_summary(self) -> dict[str, Any]:
        summary = {
            BranchDecision.ACCEPT.value: 0,
            BranchDecision.SALVAGE.value: 0,
            BranchDecision.DISCARD.value: 0,
            BranchDecision.SUPPRESS.value: 0,
            "actor_accepted": 0,
            "cache_hits": 0,
            "cached_entries": len(self.speculation_cache),
        }
        for record in self._speculation_records:
            if record.source == "speculator" and record.decision in summary:
                summary[record.decision] += 1
            if record.source == "actor" and record.decision == BranchDecision.ACCEPT.value:
                summary["actor_accepted"] += 1
            if record.cache_hit:
                summary["cache_hits"] += 1
        return summary

    def _step_stream(self, memory_step: ActionStep):
        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()
        memory_step.model_input_messages = input_messages

        speculation_future = None
        speculation_executor = None
        if self.speculator_model is not None:
            speculation_executor = ThreadPoolExecutor(max_workers=1)
            speculation_future = speculation_executor.submit(self._prepare_speculative_branches, deepcopy(input_messages))

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                yield from super()._step_stream(memory_step)
                return

            chat_message: ChatMessage = self.model.generate(
                input_messages,
                stop_sequences=["Observation:", "Calling tools:"],
                tools_to_call_from=self.tools_and_managed_agents,
            )
            self.logger.log_markdown(
                content=str(chat_message.content or chat_message.raw or ""),
                title="Output message of the LLM:",
                level=LogLevel.DEBUG,
            )
            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e
        finally:
            if speculation_executor is not None:
                speculation_executor.shutdown(wait=False)

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentGenerationError(f"Error while parsing tool call from model output: {e}", self.logger) from e
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)

        speculative_branches: list[_PreparedSpeculativeBranch] = []
        if speculation_future is not None:
            try:
                speculative_branches = speculation_future.result()
            except Exception as e:
                self.logger.log(f"Speculation failed: {e}", level=LogLevel.DEBUG)
        self._pending_speculative_branches = speculative_branches

        final_answer, got_final_answer = None, False
        try:
            for output in self.process_tool_calls(chat_message, memory_step):
                yield output
                if isinstance(output, ToolOutput) and output.is_final_answer:
                    if len(chat_message.tool_calls) > 1:
                        raise AgentGenerationError(
                            "If you want to return an answer, please do not perform any other tool calls than the final answer tool call!",
                            self.logger,
                        )
                    if got_final_answer:
                        raise AgentGenerationError(
                            "You returned multiple final answers. Please return only one single final answer!",
                            self.logger,
                        )
                    final_answer = output.output
                    got_final_answer = True
                    if isinstance(final_answer, str) and final_answer in self.state.keys():
                        final_answer = self.state[final_answer]
        finally:
            self._pending_speculative_branches = []

        yield ActionOutput(
            output=final_answer,
            is_final_answer=got_final_answer,
        )

    def process_tool_calls(self, chat_message: ChatMessage, memory_step: ActionStep):
        parallel_calls: dict[str, ToolCall] = {}
        step_records: list[SpeculationRecord] = []
        speculative_branches = list(self._pending_speculative_branches or [])

        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name,
                arguments=chat_tool_call.function.arguments,
                id=chat_tool_call.id,
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        outputs: dict[str, ToolOutput] = {}
        for tool_call in parallel_calls.values():
            actor_tags = self.tag_extractor.extract(tool_call.name, tool_call.arguments, task=self.task)
            accepted_branch = self._select_accepted_branch(tool_call, actor_tags, speculative_branches)

            if accepted_branch is not None:
                accepted_branch.decision = BranchDecision.ACCEPT
                accepted_branch.finalized = True
                accepted_branch.exact_match = self._is_exact_match(accepted_branch.tool_call, tool_call)
                self._maybe_store_cache_entry(
                    tool_call=tool_call,
                    tags=accepted_branch.tags,
                    output=accepted_branch.output,
                    observation_text=accepted_branch.observation_text or "",
                    source="speculator",
                )
                step_records.append(
                    SpeculationRecord(
                        action_type=tool_call.name,
                        tool_arguments=accepted_branch.tool_call.arguments,
                        textual_observation=accepted_branch.observation_text,
                        semantic_tags=accepted_branch.tags.semantic_tags,
                        source="speculator",
                        decision=BranchDecision.ACCEPT.value,
                        required_tags=accepted_branch.tags.required_tags,
                        goal_tags=accepted_branch.tags.goal_tags,
                        correctness=accepted_branch.confidence.correctness,
                        usefulness=accepted_branch.confidence.usefulness,
                        risk=accepted_branch.confidence.risk,
                        exact_match=accepted_branch.exact_match,
                        step_number=memory_step.step_number,
                    )
                )
                tool_output = self._build_tool_output(
                    tool_call=tool_call,
                    result=accepted_branch.output,
                    observation_override=accepted_branch.observation_text,
                    log_prefix="Accepted speculative output",
                )
            else:
                cache_entry = self._lookup_cache_entry(tool_call, actor_tags)
                if cache_entry is not None:
                    tool_output = self._build_tool_output(
                        tool_call=tool_call,
                        result=cache_entry.output,
                        observation_override=f"[cache reuse] {cache_entry.observation_text}",
                        log_prefix="Reused cached output",
                    )
                    step_records.append(
                        SpeculationRecord(
                            action_type=tool_call.name,
                            tool_arguments=tool_call.arguments,
                            textual_observation=tool_output.observation,
                            semantic_tags=list(cache_entry.semantic_tags),
                            source="actor",
                            decision=BranchDecision.ACCEPT.value,
                            required_tags=list(cache_entry.required_tags),
                            goal_tags=list(cache_entry.goal_tags),
                            cache_hit=True,
                            step_number=memory_step.step_number,
                        )
                    )
                else:
                    self.logger.log(
                        f"Calling tool: '{tool_call.name}' with arguments: {tool_call.arguments}",
                        level=LogLevel.INFO,
                    )
                    tool_result = self.execute_tool_call(tool_call.name, tool_call.arguments or {})
                    tool_output = self._build_tool_output(tool_call=tool_call, result=tool_result)
                    self._maybe_store_cache_entry(
                        tool_call=tool_call,
                        tags=actor_tags,
                        output=tool_result,
                        observation_text=tool_output.observation,
                        source="actor",
                    )
                    step_records.append(
                        SpeculationRecord(
                            action_type=tool_call.name,
                            tool_arguments=tool_call.arguments,
                            textual_observation=tool_output.observation,
                            semantic_tags=actor_tags.semantic_tags,
                            source="actor",
                            decision=BranchDecision.ACCEPT.value,
                            required_tags=actor_tags.required_tags,
                            goal_tags=actor_tags.goal_tags,
                            step_number=memory_step.step_number,
                        )
                    )

            outputs[tool_output.id] = tool_output
            yield tool_output

        for branch in speculative_branches:
            if branch.finalized:
                continue

            if branch.decision == BranchDecision.SUPPRESS:
                branch.finalized = True
                step_records.append(self._branch_record(branch, memory_step.step_number))
                continue

            if branch.output is not None and branch.confidence.usefulness >= self.tau_u:
                branch.decision = BranchDecision.SALVAGE
                self._maybe_store_cache_entry(
                    tool_call=branch.tool_call,
                    tags=branch.tags,
                    output=branch.output,
                    observation_text=branch.observation_text or "",
                    source="speculator",
                )
            else:
                branch.decision = BranchDecision.DISCARD

            branch.finalized = True
            step_records.append(self._branch_record(branch, memory_step.step_number))

        memory_step.tool_calls = [parallel_calls[key] for key in sorted(parallel_calls.keys())]
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[key] for key in sorted(outputs.keys())]:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )
        memory_step.speculation_records = [record.dict() for record in step_records]
        self._speculation_records.extend(step_records)

    def _prepare_speculative_branches(self, input_messages: list[ChatMessage | dict]) -> list[_PreparedSpeculativeBranch]:
        chat_message = self.speculator_model.generate(
            input_messages,
            stop_sequences=["Observation:", "Calling tools:"],
            tools_to_call_from=self.speculation_tools_and_managed_agents,
        )
        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.speculator_model.parse_tool_calls(chat_message)
            except Exception:
                return []
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)

        branches: list[_PreparedSpeculativeBranch] = []
        tool_calls = chat_message.tool_calls or []
        if len(tool_calls) == 0:
            return branches

        max_workers = self.max_speculative_tool_threads or max(1, len(tool_calls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for chat_tool_call in tool_calls:
                tool_call = ToolCall(
                    name=chat_tool_call.function.name,
                    arguments=chat_tool_call.function.arguments,
                    id=chat_tool_call.id,
                )
                tags = self.tag_extractor.extract(tool_call.name, tool_call.arguments, task=self.task)
                confidence = self.confidence_predictor.predict(tool_call, tags, self.task, self._speculation_records)
                branch = _PreparedSpeculativeBranch(tool_call=tool_call, tags=tags, confidence=confidence)
                branches.append(branch)

                if self._should_suppress_branch(branch):
                    branch.decision = BranchDecision.SUPPRESS
                    branch.finalized = False
                    continue

                future_map[executor.submit(self.execute_tool_call, tool_call.name, tool_call.arguments or {})] = branch

            for future, branch in future_map.items():
                try:
                    result = future.result()
                    branch.output = result
                    branch.observation_text = self._result_to_observation(result, commit_state=False)
                except Exception as e:
                    branch.observation_text = f"Speculation error: {e}"
                    branch.decision = BranchDecision.DISCARD
                    branch.finalized = False

        return branches

    def _should_suppress_branch(self, branch: _PreparedSpeculativeBranch) -> bool:
        if branch.tool_call.name == "final_answer":
            return True
        if branch.tool_call.name in self.managed_agents:
            return True
        if branch.tool_call.name == self.cache_lookup_tool_name:
            return True
        if self.speculation_allowlist and branch.tool_call.name not in self.speculation_allowlist:
            return True
        if branch.tool_call.name in self.speculation_blocklist:
            return True
        return branch.confidence.risk >= self.tau_r

    def _branch_record(self, branch: _PreparedSpeculativeBranch, step_number: int) -> SpeculationRecord:
        return SpeculationRecord(
            action_type=branch.tool_call.name,
            tool_arguments=branch.tool_call.arguments,
            textual_observation=branch.observation_text,
            semantic_tags=branch.tags.semantic_tags,
            source="speculator",
            decision=(branch.decision or BranchDecision.DISCARD).value,
            required_tags=branch.tags.required_tags,
            goal_tags=branch.tags.goal_tags,
            correctness=branch.confidence.correctness,
            usefulness=branch.confidence.usefulness,
            risk=branch.confidence.risk,
            exact_match=branch.exact_match,
            step_number=step_number,
        )

    def _select_accepted_branch(
        self,
        actor_tool_call: ToolCall,
        actor_tags: SemanticTagBundle,
        speculative_branches: list[_PreparedSpeculativeBranch],
    ) -> _PreparedSpeculativeBranch | None:
        exact_match = next(
            (
                branch
                for branch in speculative_branches
                if not branch.finalized
                and branch.output is not None
                and self._is_exact_match(branch.tool_call, actor_tool_call)
            ),
            None,
        )
        if exact_match is not None:
            return exact_match

        for branch in speculative_branches:
            if branch.finalized or branch.output is None:
                continue
            if branch.tool_call.name != actor_tool_call.name:
                continue
            if branch.confidence.correctness < self.tau_c:
                continue
            if self._goal_overlap(branch.tags.goal_tags, actor_tags.goal_tags) < self.semantic_match_threshold:
                continue
            if not set(actor_tags.required_tags).issubset(set(branch.tags.required_tags + branch.tags.goal_tags)):
                continue
            return branch
        return None

    def _lookup_cache_entry(self, tool_call: ToolCall, tags: SemanticTagBundle) -> CacheEntry | None:
        if not self._can_cache_or_reuse(tool_call.name):
            return None
        return self.speculation_cache.lookup(
            required_tags=tags.required_tags,
            goal_tags=tags.goal_tags,
            action_type=tool_call.name,
        )

    def _maybe_store_cache_entry(
        self,
        tool_call: ToolCall,
        tags: SemanticTagBundle,
        output: Any,
        observation_text: str,
        source: str,
    ) -> None:
        if tool_call.name == self.cache_lookup_tool_name:
            return
        if source == "actor" and not self.cache_actor_outputs:
            return
        if not self._can_cache_or_reuse(tool_call.name):
            return
        self.speculation_cache.store(
            CacheEntry(
                action_type=tool_call.name,
                output=output,
                observation_text=observation_text,
                semantic_tags=tuple(tags.semantic_tags),
                required_tags=tuple(tags.required_tags),
                goal_tags=tuple(tags.goal_tags),
                source=source,
            )
        )

    def _can_cache_or_reuse(self, tool_name: str) -> bool:
        if tool_name == "final_answer":
            return False
        if tool_name in self.managed_agents:
            return False
        combined_name = tool_name.lower()
        if any(keyword in combined_name for keyword in RISKY_KEYWORDS):
            return False
        if self.speculation_allowlist and tool_name not in self.speculation_allowlist:
            return False
        if tool_name in self.speculation_blocklist:
            return False
        return True

    def _is_exact_match(self, left: ToolCall, right: ToolCall) -> bool:
        return left.name == right.name and _canonicalize_arguments(left.arguments) == _canonicalize_arguments(right.arguments)

    def _goal_overlap(self, left_tags: list[str], right_tags: list[str]) -> float:
        left, right = set(left_tags), set(right_tags)
        if not left or not right:
            return 0.0
        return len(left & right) / max(len(left | right), 1)

    def _store_state_from_result(self, result: Any) -> str | None:
        if isinstance(result, AgentImage):
            self.state["image.png"] = result
            return "image.png"
        if isinstance(result, AgentAudio):
            self.state["audio.mp3"] = result
            return "audio.mp3"
        return None

    def _result_to_observation(self, result: Any, commit_state: bool) -> str:
        stored_name = self._store_state_from_result(result) if commit_state else None
        if stored_name is not None:
            return f"Stored '{stored_name}' in memory."
        if isinstance(result, (AgentImage, AgentAudio)):
            return "Produced multimodal output."
        return str(result).strip()

    def _build_tool_output(
        self,
        tool_call: ToolCall,
        result: Any,
        observation_override: str | None = None,
        log_prefix: str | None = None,
    ) -> ToolOutput:
        if observation_override is None:
            observation = self._result_to_observation(result, commit_state=True)
        else:
            stored_name = self._store_state_from_result(result)
            observation = observation_override
            if stored_name is not None:
                observation = f"Stored '{stored_name}' in memory."
        if log_prefix:
            self.logger.log(f"{log_prefix} for '{tool_call.name}'.", level=LogLevel.INFO)
        else:
            self.logger.log(f"Observations: {observation.replace('[', '|')}", level=LogLevel.INFO)
        if log_prefix:
            self.logger.log(f"Observations: {observation.replace('[', '|')}", level=LogLevel.INFO)
        return ToolOutput(
            id=tool_call.id,
            output=result,
            is_final_answer=tool_call.name == "final_answer",
            observation=observation,
            tool_call=tool_call,
        )
