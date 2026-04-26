# EXAMPLE COMMAND: from folder examples/open_deep_research, run: python run_gaia.py --concurrency 32 --run-name generate-traces-03-apr-noplanning --model-id gpt-4o
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from scripts.gaia_scorer import question_scorer
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    CalibratedConfidencePredictor,
    CodeAgent,
    GoogleSearchTool,
    HeuristicConfidencePredictor,
    InferenceClientModel,
    LiteLLMModel,
    Model,
    SalvageSpeculatingToolCallingAgent,
    ToolCallingAgent,
    TransformersModel,
)


load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

append_answer_lock = threading.Lock()
model_cache_local = threading.local()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--model-type",
        type=str,
        default="litellm",
        choices=["litellm", "transformers", "inference-client"],
    )
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument(
        "--speculator-model-type",
        type=str,
        default=None,
        choices=["litellm", "transformers", "inference-client"],
    )
    parser.add_argument("--speculator-model-id", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="Run only the specified GAIA task_id. Repeat the flag to run multiple specific tasks.",
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--set-to-run", type=str, default="validation")
    parser.add_argument("--use-open-models", type=bool, default=False)
    parser.add_argument("--use-raw-dataset", action="store_true")
    parser.add_argument("--enable-salvage-speculation", action="store_true")
    parser.add_argument("--speculation-calibration-path", type=str, default=None)
    parser.add_argument("--speculation-tau-c", type=float, default=0.85)
    parser.add_argument("--speculation-tau-u", type=float, default=0.55)
    parser.add_argument("--speculation-tau-r", type=float, default=0.35)
    parser.add_argument("--transformers-device-map", type=str, default="auto")
    parser.add_argument("--transformers-torch-dtype", type=str, default=None)
    parser.add_argument("--transformers-trust-remote-code", action="store_true")
    parser.add_argument("--inference-provider", type=str, default=None)
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated any VPN like Tailscale, else some URLs will be blocked!")

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

TRANSIENT_PROVIDER_ERROR_PATTERNS = (
    "ratelimiterror",
    "rate limit",
    "quota",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "service unavailable",
    "server disconnected",
    "connection reset",
    "connection aborted",
    "internal server error",
    "payload too large",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
)


user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def get_search_provider() -> str:
    if os.getenv("SERPER_API_KEY"):
        return "serper"
    if os.getenv("SERPAPI_API_KEY"):
        return "serpapi"
    raise ValueError("Missing search API key. Set either SERPER_API_KEY or SERPAPI_API_KEY.")


def get_generation_budget(args, speculative: bool) -> int:
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    return 2048 if speculative else 4096


def is_transient_provider_error(error: Exception | str) -> bool:
    error_text = str(error).lower()
    return any(pattern in error_text for pattern in TRANSIENT_PROVIDER_ERROR_PATTERNS)


def fallback_prediction_from_error(error: Exception) -> str | None:
    if is_transient_provider_error(error):
        return "Unable to determine"
    return None


def summarize_agent_error(error_text: str | None) -> str | None:
    if not error_text:
        return None

    rate_limit_match = re.search(r"(RateLimitError|AuthenticationError|BadRequestError|APIConnectionError)", error_text)
    if rate_limit_match:
        return rate_limit_match.group(1)

    status_match = re.search(r"\b(429|500|502|503|504)\b", error_text)
    if status_match:
        return status_match.group(1)

    first_line = error_text.strip().splitlines()[0].strip()
    return first_line[:160] if first_line else None


def get_model_cache() -> dict[tuple[Any, ...], Model]:
    if not hasattr(model_cache_local, "models"):
        model_cache_local.models = {}
    return model_cache_local.models


def get_model_cache_key(args, model_id: str, speculative: bool, selected_model_type: str) -> tuple[Any, ...]:
    return (
        selected_model_type,
        model_id,
        speculative,
        get_generation_budget(args, speculative),
        args.transformers_device_map,
        args.transformers_torch_dtype,
        args.transformers_trust_remote_code,
        args.inference_provider,
    )


def build_model(args, model_id: str, speculative: bool = False, model_type: str | None = None) -> Model:
    selected_model_type = model_type or args.model_type
    cache_key = get_model_cache_key(args, model_id, speculative, selected_model_type)
    model_cache = get_model_cache()
    if cache_key in model_cache:
        return model_cache[cache_key]

    if selected_model_type == "litellm":
        model_params: dict[str, Any] = {
            "model_id": model_id,
            "custom_role_conversions": custom_role_conversions,
        }
        if model_id == "o1":
            model_params["reasoning_effort"] = "low" if speculative else "high"
            model_params["max_completion_tokens"] = 4096 if speculative else 8192
        else:
            model_params["max_tokens"] = get_generation_budget(args, speculative)
        model = LiteLLMModel(**model_params)
        model_cache[cache_key] = model
        return model

    if selected_model_type == "transformers":
        model_params = {
            "model_id": model_id,
            "max_new_tokens": get_generation_budget(args, speculative),
            "device_map": args.transformers_device_map,
            "trust_remote_code": args.transformers_trust_remote_code,
        }
        if args.transformers_torch_dtype:
            model_params["torch_dtype"] = args.transformers_torch_dtype
        model = TransformersModel(**model_params)
        model_cache[cache_key] = model
        return model

    if selected_model_type == "inference-client":
        model = InferenceClientModel(
            model_id=model_id,
            provider=args.inference_provider,
            token=os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY"),
        )
        model_cache[cache_key] = model
        return model

    raise ValueError(f"Unsupported model type: {selected_model_type}")


def serialize_messages(messages: list[Any]) -> list[Any]:
    serialized_messages = []
    for message in messages:
        if hasattr(message, "model_dump_json"):
            serialized_messages.append(json.loads(message.model_dump_json()))
        elif hasattr(message, "dict"):
            serialized_messages.append(message.dict())
        else:
            serialized_messages.append(message)
    return serialized_messages


def build_confidence_predictor(args) -> HeuristicConfidencePredictor | CalibratedConfidencePredictor:
    if args.speculation_calibration_path:
        return CalibratedConfidencePredictor.from_json(args.speculation_calibration_path)
    return HeuristicConfidencePredictor()


def create_agent_team(model: Model, token_counts: dict[str, int], args):
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    search_provider = get_search_provider()

    WEB_TOOLS = [
        GoogleSearchTool(provider=search_provider),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    def increment_web_agent_token_counts(final_answer, memory_step, agent):
        token_counts_web = agent.monitor.get_total_token_counts()
        token_counts["input_tokens"] += token_counts_web.input_tokens
        token_counts["output_tokens"] += token_counts_web.output_tokens
        return True

    browser_agent_kwargs = dict(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    This agent cannot inspect local attached images, audio files, or other local files directly; use it only for web lookup.
    """,
        provide_run_summary=True,
        final_answer_checks=[increment_web_agent_token_counts],
    )

    if args.enable_salvage_speculation:
        speculator_model_type = args.speculator_model_type or args.model_type
        speculator_model_id = args.speculator_model_id or args.model_id
        if (
            speculator_model_type == "transformers"
            and speculator_model_id == args.model_id
            and args.speculator_model_id is None
        ):
            raise ValueError(
                "Salvage speculation would load a second copy of the same local Transformers model. "
                "Disable `--enable-salvage-speculation` or provide a smaller `--speculator-model-id`/"
                "`--speculator-model-type`."
            )
        text_webbrowser_agent = SalvageSpeculatingToolCallingAgent(
            speculator_model=build_model(
                args,
                speculator_model_id,
                speculative=True,
                model_type=speculator_model_type,
            ),
            confidence_predictor=build_confidence_predictor(args),
            tau_c=args.speculation_tau_c,
            tau_u=args.speculation_tau_u,
            tau_r=args.speculation_tau_r,
            speculation_allowlist=[tool.name for tool in WEB_TOOLS],
            **browser_agent_kwargs,
        )
    else:
        text_webbrowser_agent = ToolCallingAgent(**browser_agent_kwargs)

    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.
    Never attempt to interpret attached local images, screenshots, sheet music, or other local visual files: the manager must use the `visualizer` tool for those."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str) -> datasets.Dataset:
    if not os.path.exists("data/gaia"):
        if use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            # WARNING: this dataset is gated: make sure you visit the repo to require access.
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )

    split_root = Path("data/gaia") / "2023" / set_to_run
    metadata_file = split_root / "metadata.jsonl"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Could not find GAIA metadata file at {metadata_file.resolve()}")

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = str(split_root / row["file_name"])
        return row

    eval_ds = datasets.load_dataset(
        "json",
        data_files={set_to_run: str(metadata_file)},
        split=set_to_run,
    )

    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
    eval_ds = eval_ds.map(preprocess_file_paths)
    return eval_ds


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fp.flush()

        # Keep a human-readable companion file without breaking the line-delimited
        # JSON format that downstream tooling expects from the primary results file.
        pretty_path = jsonl_path.with_suffix(".pretty.json")
        with open(jsonl_path, encoding="utf-8") as source_fp:
            entries = [json.loads(line) for line in source_fp if line.strip()]
        with open(pretty_path, "w", encoding="utf-8") as pretty_fp:
            json.dump(entries, pretty_fp, indent=2, ensure_ascii=False)

    assert jsonl_path.exists(), "File not found!"
    print("Answer exported to file:", jsonl_path.resolve())
    print("Readable results exported to file:", pretty_path.resolve())


def answer_single_question(
    example: dict, args, answers_file: str, visual_inspection_tool: TextInspectorTool
) -> None:
    model = build_model(args, args.model_id, speculative=False)
    # model = InferenceClientModel(model_id="Qwen/Qwen3-32B", provider="novita", max_tokens=4096)
    document_inspection_tool = TextInspectorTool(model, 100000)

    total_token_counts = {"input_tokens": 0, "output_tokens": 0}
    exception = None

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
            prompt_use_files += get_single_file_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        augmented_question += prompt_use_files
        file_extension = str(example["file_name"]).lower().rsplit(".", 1)[-1]
        if file_extension in {"png", "jpg", "jpeg"}:
            augmented_question += (
                "\n\nImportant: the attached file is a local image. "
                "Do not ask `search_agent` to inspect, transcribe, or interpret this local image. "
                "Use the `visualizer` tool directly for all image understanding."
            )

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = None
    intermediate_steps = []
    parsing_error = False
    iteration_limit_exceeded = False
    raised_exception = False
    search_agent = None
    speculation_summary = None
    speculation_records = None

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        agent = create_agent_team(model, total_token_counts, args)
        try:
            # Run agent 🚀
            agent.run(augmented_question)

            agent_memory = agent.write_memory_to_messages()
            final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

            output = str(final_result)
            for memory_step in agent.memory.steps:
                memory_step.model_input_messages = None
            intermediate_steps = serialize_messages(agent_memory)

            # Check for parsing errors which indicate the LLM failed to follow the required format
            parsing_error = any("AgentParsingError" in json.dumps(step) for step in intermediate_steps)

            # check if iteration limit exceeded
            iteration_limit_exceeded = "Agent stopped due to iteration limit or time limit." in output
            raised_exception = False
            exception = None
            search_agent = agent.managed_agents.get("search_agent")
            if hasattr(search_agent, "get_speculation_summary"):
                speculation_summary = search_agent.get_speculation_summary()
                speculation_records = search_agent.get_speculation_records()
            token_counts_manager = agent.monitor.get_total_token_counts()
            total_token_counts["input_tokens"] += token_counts_manager.input_tokens
            total_token_counts["output_tokens"] += token_counts_manager.output_tokens
            break

        except Exception as e:
            token_counts_manager = agent.monitor.get_total_token_counts()
            total_token_counts["input_tokens"] += token_counts_manager.input_tokens
            total_token_counts["output_tokens"] += token_counts_manager.output_tokens
            exception = e
            raised_exception = True
            search_agent = agent.managed_agents.get("search_agent")
            if hasattr(search_agent, "get_speculation_summary"):
                speculation_summary = search_agent.get_speculation_summary()
                speculation_records = search_agent.get_speculation_records()

            if attempt < max_attempts and is_transient_provider_error(e):
                print(f"[{example['task_id']}] transient error on attempt {attempt}: {e}. Retrying...")
                time.sleep(float(attempt))
                continue

            print("Error on ", augmented_question, e)
            output = fallback_prediction_from_error(e)
            intermediate_steps = []
            parsing_error = False
            iteration_limit_exceeded = False
            break
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_token_counts["total_tokens"] = total_token_counts["input_tokens"] + total_token_counts["output_tokens"]
    scored_correctly = None
    if str(example["true_answer"]) != "?" and output is not None:
        scored_correctly = question_scorer(str(output), str(example["true_answer"]))

    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "task": example["task"],
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": total_token_counts,
        "speculation_enabled": args.enable_salvage_speculation,
        "speculator_model_id": (args.speculator_model_id or args.model_id) if args.enable_salvage_speculation else None,
        "speculation_summary": speculation_summary,
        "speculation_records": speculation_records,
        "scored_correctly": scored_correctly,
    }
    append_answer(annotated_example, answers_file)
    if scored_correctly is None:
        print(f"[{example['task_id']}] prediction={output!r}")
    else:
        print(
            f"[{example['task_id']}] correct={scored_correctly} "
            f"prediction={output!r} true_answer={example['true_answer']!r}"
        )


def get_examples_to_answer(answers_file: str, eval_ds: datasets.Dataset) -> list[dict]:
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [line for line in eval_ds.to_list() if line["question"] not in done_questions and line["file_name"]]


def filter_examples_by_task_ids(examples: list[dict], task_ids: list[str] | None) -> list[dict]:
    if not task_ids:
        return examples

    normalized_task_ids = {str(task_id).strip() for task_id in task_ids if str(task_id).strip()}
    return [example for example in examples if str(example["task_id"]) in normalized_task_ids]


def print_run_summary(answers_file: str, tasks_to_run: list[dict]) -> None:
    if not tasks_to_run:
        print("Run summary: no tasks were queued.")
        return

    results = pd.read_json(answers_file, lines=True)
    task_ids = {task["task_id"] for task in tasks_to_run}
    run_results = results[results["task_id"].isin(task_ids)].drop_duplicates(subset=["task_id"], keep="last")

    scored_results = run_results[run_results["scored_correctly"].notna()]
    correct_count = int(scored_results["scored_correctly"].astype(bool).sum())
    scored_count = int(len(scored_results))
    total_count = int(len(tasks_to_run))
    unscored_count = total_count - scored_count
    wrong_results = scored_results[~scored_results["scored_correctly"].astype(bool)]
    wrong_task_ids = wrong_results["task_id"].astype(str).tolist()
    unscored_results = run_results[run_results["scored_correctly"].isna()]
    unscored_task_ids = unscored_results["task_id"].astype(str).tolist()
    error_counter = Counter(
        summarized_error
        for summarized_error in run_results["agent_error"].dropna().map(summarize_agent_error)
        if summarized_error
    )

    if scored_count > 0:
        accuracy = 100 * correct_count / scored_count
        print(
            f"Run summary: correct={correct_count}/{scored_count} scored "
            f"({accuracy:.1f}%), total_queued={total_count}, unscored={unscored_count}."
        )
    else:
        print(f"Run summary: correct=0/0 scored, total_queued={total_count}, unscored={unscored_count}.")

    if wrong_task_ids:
        print(f"Wrong task_ids: {', '.join(wrong_task_ids)}")
    if unscored_task_ids:
        print(f"Unscored task_ids: {', '.join(unscored_task_ids)}")
    if error_counter:
        print("Agent error counts:")
        for error_label, count in error_counter.most_common():
            print(f"  - {error_label}: {count}")


def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    eval_ds = load_gaia_dataset(args.use_raw_dataset, args.set_to_run)
    print("Loaded evaluation dataset:")
    print(pd.DataFrame(eval_ds)["task"].value_counts())

    answers_file = f"output/{args.set_to_run}/{args.run_name}.jsonl"
    if args.task_id:
        tasks_to_run = filter_examples_by_task_ids(eval_ds.to_list(), args.task_id)
        if not tasks_to_run:
            raise ValueError(f"No tasks matched --task-id values: {args.task_id}")
    else:
        tasks_to_run = get_examples_to_answer(answers_file, eval_ds)
    if args.max_examples is not None:
        tasks_to_run = tasks_to_run[: args.max_examples]
    print(f"Queued {len(tasks_to_run)} task(s) for this run.")

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args, answers_file, visualizer)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    # for example in tasks_to_run:
    #     answer_single_question(example, args.model_id, answers_file, visualizer)
    print_run_summary(answers_file, tasks_to_run)
    print("All tasks processed.")


if __name__ == "__main__":
    main()
