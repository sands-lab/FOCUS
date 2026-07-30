#!/usr/bin/env python3
"""FOCUS README throughput profiler for the SGLang implementation.

This mirrors the LMDeploy FOCUS benchmark front end while running SGLang's
in-process Engine from this repository.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs


DatasetFormat = str
DEFAULT_HF_SHUFFLE_BUFFER_SIZE = 100_000
GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DEFAULT_CONFIG = "main"
GSM8K_SUPPORTED_SPLIT = "test"
GSM8K_FALLBACK_SPLIT = "validation"


def _is_gsm8k_dataset(dataset_id: str) -> bool:
    return dataset_id.strip().lower() == GSM8K_DATASET_ID


def _get_hf_split_candidates(dataset_id: str, split: str) -> Tuple[str, ...]:
    if not _is_gsm8k_dataset(dataset_id):
        return (split,)
    if split == GSM8K_SUPPORTED_SPLIT:
        return (GSM8K_SUPPORTED_SPLIT, GSM8K_FALLBACK_SPLIT)
    if split == GSM8K_FALLBACK_SPLIT:
        return (GSM8K_FALLBACK_SPLIT, GSM8K_SUPPORTED_SPLIT)
    return (split,)


def _resolve_math_prompt_and_solution(
    example: Dict[str, Any],
    prompt_keys: Tuple[str, ...],
    solution_keys: Tuple[str, ...],
) -> Optional[Tuple[str, str]]:
    prompt = next((example.get(key) for key in prompt_keys if example.get(key) is not None), None)
    solution = next((example.get(key) for key in solution_keys if example.get(key) is not None), None)
    if prompt is None or solution is None:
        return None
    prompt_text = str(prompt).strip()
    solution_text = str(solution).strip()
    if not prompt_text or not solution_text:
        return None
    return prompt_text, solution_text


def _extract_math_messages(
    example: Dict[str, Any],
    *,
    prompt_keys: Tuple[str, ...] = ("problem", "question"),
    solution_keys: Tuple[str, ...] = ("solution", "answer"),
) -> Optional[List[Dict[str, str]]]:
    prompt_and_solution = _resolve_math_prompt_and_solution(example, prompt_keys, solution_keys)
    if prompt_and_solution is None:
        return None
    problem_text, solution_text = prompt_and_solution
    return [
        {"role": "user", "content": problem_text},
        {"role": "assistant", "content": solution_text},
    ]


def _looks_like_math(example: Dict[str, Any]) -> bool:
    return (
        isinstance(example, dict)
        and _resolve_math_prompt_and_solution(
            example,
            prompt_keys=("problem", "question"),
            solution_keys=("solution", "answer"),
        )
        is not None
    )


def _normalize_role(role: Any) -> str:
    role = str(role).strip().lower()
    if role in ("human", "user"):
        return "user"
    if role in ("gpt", "assistant", "bot", "model"):
        return "assistant"
    if role == "system":
        return "system"
    return role


def _extract_messages(
    example: Dict[str, Any],
    dataset_format: DatasetFormat = "auto",
) -> Optional[List[Dict[str, str]]]:
    if dataset_format == "math":
        return _extract_math_messages(example)
    if dataset_format == "gsm8k":
        return _extract_math_messages(example, prompt_keys=("question",), solution_keys=("answer",))
    if dataset_format == "auto" and _looks_like_math(example):
        messages = _extract_math_messages(example)
        if messages is not None:
            return messages

    if dataset_format == "sharegpt":
        key_order = ("conversations", "conversation", "messages")
    elif dataset_format == "wildchat":
        key_order = ("conversation", "messages", "conversations")
    else:
        key_order = ("conversation", "conversations", "messages")

    turns = None
    for key in key_order:
        if key in example:
            turns = example[key]
            break
    if not isinstance(turns, list):
        return None

    messages: List[Dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = None
        content = None
        if "role" in turn and "content" in turn:
            role = turn.get("role")
            content = turn.get("content")
        elif "from" in turn and "value" in turn:
            role = turn.get("from")
            content = turn.get("value")
        elif "speaker" in turn and "text" in turn:
            role = turn.get("speaker")
            content = turn.get("text")

        if role is None or content is None:
            continue
        content = str(content)
        if not content.strip():
            continue
        messages.append({"role": _normalize_role(role), "content": content})

    return messages or None


def _pick_prompt_completion(messages: List[Dict[str, str]]) -> Optional[Tuple[List[Dict[str, str]], str]]:
    for idx, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        prompt_messages = [
            m for m in messages[:idx] if m.get("role") == "system" and m.get("content")
        ] + [msg]
        for j in range(idx + 1, len(messages)):
            if messages[j].get("role") == "assistant" and messages[j].get("content"):
                return prompt_messages, messages[j]["content"]
    return None


def _iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_pairs_from_rows(
    rows: Iterable[Dict[str, Any]],
    dataset_format: DatasetFormat = "auto",
) -> Iterable[Tuple[List[Dict[str, str]], str]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        messages = _extract_messages(row, dataset_format=dataset_format)
        if not messages or len(messages) < 2:
            continue
        picked = _pick_prompt_completion(messages)
        if picked is not None:
            yield picked


def _iter_pairs_from_file(
    dataset_path: str,
    dataset_format: DatasetFormat = "auto",
    *,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Iterable[Tuple[List[Dict[str, str]], str]]:
    rng = random.Random(seed)
    if dataset_path.endswith(".jsonl"):
        rows = list(_iter_jsonl(dataset_path))
        if shuffle:
            rng.shuffle(rows)
        return _iter_pairs_from_rows(rows, dataset_format=dataset_format)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset JSON root type: {type(data)}")
    if shuffle:
        rng.shuffle(data)
    return _iter_pairs_from_rows(data, dataset_format=dataset_format)


def _iter_pairs_from_hf(
    dataset_id: str,
    split: str,
    streaming: bool,
    config: Optional[str] = None,
    seed: Optional[int] = None,
    dataset_format: DatasetFormat = "auto",
) -> Iterable[Tuple[List[Dict[str, str]], str]]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise ImportError("HuggingFace dataset loading requires the `datasets` package.") from e

    load_kwargs = dict(split=split, streaming=streaming)
    if config is not None:
        load_kwargs["name"] = config

    ds = None
    last_error = None
    for candidate_split in _get_hf_split_candidates(dataset_id, split):
        load_kwargs["split"] = candidate_split
        try:
            ds = load_dataset(dataset_id, **load_kwargs)
            break
        except Exception as e:
            last_error = e
    if ds is None:
        assert last_error is not None
        raise last_error

    if seed is not None:
        try:
            if streaming:
                ds = ds.shuffle(seed=seed, buffer_size=DEFAULT_HF_SHUFFLE_BUFFER_SIZE)
            else:
                ds = ds.shuffle(seed=seed)
        except Exception:
            pass

    def _iter_rows() -> Iterator[Tuple[List[Dict[str, str]], str]]:
        it = iter(ds)
        first = next(it, None)
        if first is None:
            return
        for row in itertools.chain([first], it):
            if not isinstance(row, dict):
                continue
            messages = _extract_messages(row, dataset_format=dataset_format)
            if not messages or len(messages) < 2:
                continue
            picked = _pick_prompt_completion(messages)
            if picked is not None:
                yield picked

    return _iter_rows()


def _download_hf_data_file(
    dataset_id: str,
    *,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    if filename is None:
        files = list_repo_files(repo_id=dataset_id, repo_type="dataset", revision=revision)
        candidates = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
        preferred = "ShareGPT_V3_unfiltered_cleaned_split.json"
        if preferred in candidates:
            filename = preferred
        elif len(candidates) == 1:
            filename = candidates[0]
        else:
            preview = ", ".join(candidates[:20])
            raise ValueError(
                f"Cannot infer a JSON/JSONL file in `{dataset_id}`; use --hf-data-file. "
                f"Found candidates: {preview}"
            )

    return str(
        hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
        )
    )


def _format_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if tokenizer.bos_token:
            prompt = prompt.replace(tokenizer.bos_token, "")
        return prompt
    except Exception:
        return messages[-1]["content"]


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    dataset_format: DatasetFormat = "auto",
    hf_split: str = "train",
    hf_streaming: bool = False,
    hf_config: Optional[str] = None,
    hf_data_file: Optional[str] = None,
    hf_revision: Optional[str] = None,
    max_scan_examples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Tuple[str, int]]:
    if os.path.isfile(dataset_path):
        pairs_iter = _iter_pairs_from_file(dataset_path, dataset_format, seed=seed, shuffle=True)
    else:
        try:
            pairs_iter = _iter_pairs_from_hf(
                dataset_path,
                split=hf_split,
                streaming=hf_streaming,
                config=hf_config,
                seed=seed,
                dataset_format=dataset_format,
            )
        except Exception as e:
            try:
                from datasets.exceptions import DataFilesNotFoundError
            except Exception:
                DataFilesNotFoundError = ()  # type: ignore[assignment]
            if (
                hf_data_file is None
                and DataFilesNotFoundError
                and not isinstance(e, DataFilesNotFoundError)
            ):
                raise
            local_path = _download_hf_data_file(
                dataset_path,
                filename=hf_data_file,
                revision=hf_revision,
            )
            pairs_iter = _iter_pairs_from_file(local_path, dataset_format, seed=seed, shuffle=True)

    filtered: List[Tuple[str, int]] = []
    scanned = 0
    for messages, completion in pairs_iter:
        if len(filtered) == num_requests:
            break
        scanned += 1
        if max_scan_examples is not None and scanned > max_scan_examples:
            break

        prompt = _format_prompt(tokenizer, messages)
        prompt_len = len(tokenizer.encode(prompt))
        output_len = len(tokenizer.encode(completion))
        if prompt_len < 4 or output_len < 4:
            continue
        filtered.append((prompt, prompt_len))

    return filtered


def _write_algorithm_config(
    output_dir: Path,
    *,
    block_size: int,
    threshold: float,
    enable_focus: bool,
    enable_delayed_cache: bool,
    focus_alpha: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "block_size": int(block_size),
        "threshold": float(threshold),
        "enable_focus": bool(enable_focus),
        "enable_delayed_cache": bool(enable_delayed_cache or enable_focus),
        "focus_alpha": float(focus_alpha),
    }
    stem = (
        f"low_confidence_b{block_size}_thr{threshold:g}"
        f"_focus{int(enable_focus)}_delay{int(enable_delayed_cache or enable_focus)}"
        f"_alpha{focus_alpha:g}"
    ).replace(".", "p")
    path = output_dir / f"{stem}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=True)
    return path


def _safe_basename(value: str) -> str:
    return Path(value.rstrip("/")).name.replace(" ", "_")


def _top_k_for_sampling(top_k: int, temperature: float) -> int:
    if top_k == 0:
        return -1
    if temperature <= 0:
        return 1
    return top_k


def _normalize_outputs(outputs: Any) -> List[Dict[str, Any]]:
    if isinstance(outputs, dict):
        return [outputs]
    return list(outputs)


def _extract_completion_tokens(output: Dict[str, Any]) -> int:
    meta = output.get("meta_info") or {}
    value = meta.get("completion_tokens")
    if value is not None:
        return int(value)
    output_ids = output.get("output_ids")
    if output_ids is not None:
        return len(output_ids)
    text = output.get("text")
    return len(text) if isinstance(text, str) else 0


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LMDeploy-README-style FOCUS throughput on SGLang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Dataset path or HuggingFace dataset ID.")
    parser.add_argument("model_path_arg", help="Model path or HuggingFace model ID.")
    parser.add_argument(
        "--dataset-format",
        default="auto",
        choices=["auto", "sharegpt", "wildchat", "math", "gsm8k"],
    )
    parser.add_argument("--hf-split", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-streaming", action="store_true", default=False)
    parser.add_argument("--hf-data-file", default=None)
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--max-scan-examples", type=int, default=None)
    parser.add_argument("-c", "--concurrency", type=int, default=256)
    parser.add_argument("-n", "--num-prompts", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument(
        "--skip-detokenize",
        action="store_true",
        help="Recorded for parity with LMDeploy; SGLang still returns standard metadata.",
    )
    parser.add_argument("--repeat-block-detect", action="store_true")
    parser.add_argument("--repeat-block-window", type=int, default=None)
    parser.add_argument("--repeat-block-threshold", type=int, default=3)
    parser.add_argument("--dllm-block-length", type=int, default=32)
    parser.add_argument("--dllm-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--eager-mode", action="store_true")
    parser.add_argument("--csv", default="results/focus_sglang/profile_throughput_sglang.csv")
    parser.add_argument("--json-result", default=None)
    parser.add_argument("--output-dir", default="results/focus_sglang")
    parser.add_argument("--no-force-trust-remote-code", action="store_true")
    parser.add_argument("--no-force-attention-backend", action="store_true")

    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    if args.hf_split is None:
        args.hf_split = GSM8K_SUPPORTED_SPLIT if _is_gsm8k_dataset(args.dataset) else "train"
    if _is_gsm8k_dataset(args.dataset):
        if args.hf_split != GSM8K_SUPPORTED_SPLIT:
            parser.error(f"`{GSM8K_DATASET_ID}` is supported with the `{GSM8K_SUPPORTED_SPLIT}` split only.")
        if args.hf_config is None:
            args.hf_config = GSM8K_DEFAULT_CONFIG
    if args.repeat_block_threshold < 2:
        parser.error("--repeat-block-threshold must be >= 2.")
    if args.repeat_block_window is not None and args.repeat_block_window <= 0:
        parser.error("--repeat-block-window must be positive.")
    if args.repeat_block_detect and args.repeat_block_window is None:
        args.repeat_block_window = args.dllm_block_length

    args.model_path = args.model_path or args.model_path_arg
    args.tokenizer_path = args.tokenizer_path or args.model_path
    args.dllm_algorithm = args.dllm_algorithm or "LowConfidence"

    if args.max_running_requests is None:
        args.max_running_requests = args.concurrency
    if args.eager_mode and hasattr(args, "disable_cuda_graph"):
        args.disable_cuda_graph = True
    if not args.no_force_trust_remote_code and hasattr(args, "trust_remote_code"):
        args.trust_remote_code = True
    if not args.no_force_attention_backend and getattr(args, "attention_backend", None) is None:
        args.attention_backend = "flashinfer"

    if args.dllm_algorithm_config is None:
        cfg_path = _write_algorithm_config(
            Path(args.output_dir) / "configs",
            block_size=args.dllm_block_length,
            threshold=args.dllm_confidence_threshold,
            enable_focus=args.dllm_enable_focus,
            enable_delayed_cache=args.dllm_enable_delayed_cache,
            focus_alpha=args.dllm_focus_alpha,
        )
        args.dllm_algorithm_config = str(cfg_path)

    return args


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=getattr(args, "trust_remote_code", True),
    )
    requests = sample_requests(
        dataset_path=args.dataset,
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        dataset_format=args.dataset_format,
        hf_split=args.hf_split,
        hf_streaming=args.hf_streaming,
        hf_config=args.hf_config,
        hf_data_file=args.hf_data_file,
        hf_revision=args.hf_revision,
        max_scan_examples=args.max_scan_examples,
        seed=args.seed,
    )
    if not requests:
        raise ValueError(f"No valid prompts were sampled from `{args.dataset}`.")
    if len(requests) < args.num_prompts:
        print(
            f"[INFO] Requested {args.num_prompts} prompts but sampled only {len(requests)} from `{args.dataset}`."
        )

    prompts = [prompt for prompt, _ in requests]
    prompt_lens = [prompt_len for _, prompt_len in requests]
    sampling_params: List[Dict[str, Any]] = []
    custom_params = None
    if args.repeat_block_detect:
        custom_params = {
            "repeat_block_window": int(args.repeat_block_window),
            "repeat_block_threshold": int(args.repeat_block_threshold),
        }
    for _ in prompts:
        params: Dict[str, Any] = {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": _top_k_for_sampling(args.top_k, args.temperature),
            "max_new_tokens": int(args.max_new_tokens),
            "ignore_eos": False,
        }
        if custom_params is not None:
            params["custom_params"] = dict(custom_params)
        sampling_params.append(params)

    input_ids = None
    if args.skip_tokenize:
        input_ids = [tokenizer.encode(prompt) for prompt in prompts]
        prompts_for_engine = None
    else:
        prompts_for_engine = prompts

    # ServerArgs is immutable after the current upstream resolution pipeline
    # runs, so set the declaration on the CLI namespace before constructing it.
    args.max_running_requests = min(args.concurrency, len(requests))
    server_args = ServerArgs.from_cli_args(args)

    print("SGLang FOCUS throughput configuration")
    print(f"  dataset                  : {args.dataset}")
    print(f"  model                    : {args.model_path}")
    print(f"  prompts requested/sampled: {args.num_prompts}/{len(requests)}")
    print(f"  concurrency              : {args.concurrency}")
    print(f"  max new tokens           : {args.max_new_tokens}")
    print(f"  dllm algorithm/config    : {args.dllm_algorithm} / {args.dllm_algorithm_config}")
    print(f"  delayed cache / focus    : {args.dllm_enable_delayed_cache} / {args.dllm_enable_focus}")
    print(f"  repeat block             : {custom_params}")
    print(f"  skip tokenize/detokenize : {args.skip_tokenize}/{args.skip_detokenize}")
    print(f"  input tokens             : {sum(prompt_lens)}")

    engine = None
    start = time.perf_counter()
    try:
        engine = Engine(**dataclasses.asdict(server_args))
        gen_start = time.perf_counter()
        outputs = engine.generate(
            prompt=prompts_for_engine,
            input_ids=input_ids,
            sampling_params=sampling_params,
        )
        latency = time.perf_counter() - gen_start
        outputs_list = _normalize_outputs(outputs)
        server_info = _json_safe(engine.get_server_info())
    finally:
        if engine is not None:
            engine.shutdown()
    total_wall = time.perf_counter() - start

    total_input_tokens = int(sum(prompt_lens))
    total_output_tokens = int(sum(_extract_completion_tokens(o) for o in outputs_list))
    successful_requests = len(outputs_list)
    result = {
        "backend": "sglang",
        "dataset": args.dataset,
        "dataset_name": _safe_basename(args.dataset),
        "model": args.model_path,
        "model_name": _safe_basename(args.model_path),
        "successful_requests": successful_requests,
        "requested_prompts": args.num_prompts,
        "sampled_prompts": len(requests),
        "concurrency": args.concurrency,
        "total_latency": latency,
        "total_wall_time": total_wall,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": successful_requests / latency if latency > 0 else 0.0,
        "input_throughput": total_input_tokens / latency if latency > 0 else 0.0,
        "output_throughput": total_output_tokens / latency if latency > 0 else 0.0,
        "total_throughput": (total_input_tokens + total_output_tokens) / latency if latency > 0 else 0.0,
        "max_new_tokens": args.max_new_tokens,
        "dllm_block_length": args.dllm_block_length,
        "dllm_confidence_threshold": args.dllm_confidence_threshold,
        "dllm_enable_delayed_cache": args.dllm_enable_delayed_cache,
        "dllm_enable_focus": args.dllm_enable_focus,
        "dllm_focus_alpha": args.dllm_focus_alpha,
        "dllm_algorithm_config": args.dllm_algorithm_config,
        "repeat_block_detect": args.repeat_block_detect,
        "repeat_block_window": args.repeat_block_window if args.repeat_block_detect else None,
        "repeat_block_threshold": args.repeat_block_threshold if args.repeat_block_detect else None,
        "server_info": server_info,
    }

    print("\n================ SGLang FOCUS Throughput Result ================")
    print(f"{'Successful requests:':<38} {successful_requests}")
    print(f"{'Benchmark duration (s):':<38} {latency:.2f}")
    print(f"{'Total input tokens:':<38} {total_input_tokens}")
    print(f"{'Total generated tokens:':<38} {total_output_tokens}")
    print(f"{'Request throughput (req/s):':<38} {result['request_throughput']:.2f}")
    print(f"{'Input token throughput (tok/s):':<38} {result['input_throughput']:.2f}")
    print(f"{'Output token throughput (tok/s):':<38} {result['output_throughput']:.2f}")
    print(f"{'Total token throughput (tok/s):':<38} {result['total_throughput']:.2f}")
    print("================================================================")

    json_result = args.json_result
    if json_result is None:
        kind = (
            "focus"
            if args.dllm_enable_focus
            else ("delayed_cache" if args.dllm_enable_delayed_cache else "base")
        )
        json_result = str(
            output_dir
            / (
                f"{kind}_{_safe_basename(args.dataset)}_{_safe_basename(args.model_path)}"
                f"_batch_{args.concurrency}.json"
            )
        )
    with open(json_result, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"JSON result: {json_result}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        fields = [
            "backend",
            "dataset",
            "model",
            "concurrency",
            "sampled_prompts",
            "max_new_tokens",
            "dllm_block_length",
            "dllm_confidence_threshold",
            "dllm_enable_delayed_cache",
            "dllm_enable_focus",
            "dllm_focus_alpha",
            "repeat_block_detect",
            "total_latency",
            "total_input_tokens",
            "total_output_tokens",
            "request_throughput",
            "input_throughput",
            "output_throughput",
            "total_throughput",
            "json_result",
        ]
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            row = {field: result.get(field) for field in fields}
            row["json_result"] = json_result
            writer.writerow(row)
        print(f"CSV appended: {csv_path}")


if __name__ == "__main__":
    main()
