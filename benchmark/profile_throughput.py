# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import asyncio
import itertools
import json
import os
import random
from queue import Queue
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
try:
    from transformers import PreTrainedTokenizerBase
except Exception:  # pragma: no cover
    PreTrainedTokenizerBase = Any  # type: ignore

from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.profiler import Profiler, Session
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import get_logger

get_logger('lmdeploy').setLevel('ERROR')
os.environ['TM_LOG_LEVEL'] = 'ERROR'

DatasetFormat = str
DEFAULT_HF_SHUFFLE_BUFFER_SIZE = 100_000
GSM8K_DATASET_ID = 'openai/gsm8k'
GSM8K_DEFAULT_CONFIG = 'main'
GSM8K_SUPPORTED_SPLIT = 'test'
GSM8K_FALLBACK_SPLIT = 'validation'


def _is_gsm8k_dataset(dataset_id: str) -> bool:
    return dataset_id.strip().lower() == GSM8K_DATASET_ID


def _get_hf_split_candidates(dataset_id: str, split: str) -> Tuple[str, ...]:
    if not _is_gsm8k_dataset(dataset_id):
        return (split, )
    if split == GSM8K_SUPPORTED_SPLIT:
        return (GSM8K_SUPPORTED_SPLIT, GSM8K_FALLBACK_SPLIT)
    if split == GSM8K_FALLBACK_SPLIT:
        return (GSM8K_FALLBACK_SPLIT, GSM8K_SUPPORTED_SPLIT)
    return (split, )


def _resolve_math_prompt_and_solution(example: Dict[str, Any],
                                      prompt_keys: Tuple[str, ...],
                                      solution_keys: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
    prompt = next((example.get(key) for key in prompt_keys if example.get(key) is not None), None)
    solution = next((example.get(key) for key in solution_keys if example.get(key) is not None), None)
    if prompt is None or solution is None:
        return None
    prompt_text = str(prompt).strip()
    solution_text = str(solution).strip()
    if not prompt_text or not solution_text:
        return None
    return prompt_text, solution_text


def _looks_like_math(example: Dict[str, Any]) -> bool:
    if not isinstance(example, dict):
        return False
    return _resolve_math_prompt_and_solution(example,
                                             prompt_keys=('problem', 'question'),
                                             solution_keys=('solution', 'answer')) is not None


def _extract_math_messages(example: Dict[str, Any],
                           *,
                           prompt_keys: Tuple[str, ...] = ('problem', 'question'),
                           solution_keys: Tuple[str, ...] = ('solution', 'answer')) -> Optional[List[Dict[str, str]]]:
    prompt_and_solution = _resolve_math_prompt_and_solution(example, prompt_keys, solution_keys)
    if prompt_and_solution is None:
        return None
    problem_text, solution_text = prompt_and_solution
    return [
        {'role': 'user', 'content': problem_text},
        {'role': 'assistant', 'content': solution_text},
    ]


def _normalize_role(role: Any) -> str:
    role = str(role).strip().lower()
    if role in ('human', 'user'):
        return 'user'
    if role in ('gpt', 'assistant', 'bot', 'model'):
        return 'assistant'
    if role in ('system',):
        return 'system'
    return role


def _extract_messages(example: Dict[str, Any], dataset_format: DatasetFormat = 'auto') -> Optional[List[Dict[str, str]]]:
    """Extract a list of {role, content} messages from a dataset row."""
    if dataset_format == 'math':
        return _extract_math_messages(example)
    if dataset_format == 'gsm8k':
        return _extract_math_messages(example, prompt_keys=('question',), solution_keys=('answer',))
    if dataset_format == 'auto' and _looks_like_math(example):
        messages = _extract_math_messages(example)
        if messages is not None:
            return messages
    turns = None
    if dataset_format == 'sharegpt':
        key_order = ('conversations', 'conversation', 'messages')
    elif dataset_format == 'wildchat':
        key_order = ('conversation', 'messages', 'conversations')
    else:
        key_order = ('conversation', 'conversations', 'messages')

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
        if 'role' in turn and 'content' in turn:
            role = turn.get('role')
            content = turn.get('content')
        elif 'from' in turn and 'value' in turn:
            role = turn.get('from')
            content = turn.get('value')
        elif 'speaker' in turn and 'text' in turn:
            role = turn.get('speaker')
            content = turn.get('text')

        if role is None or content is None:
            continue
        content = str(content)
        if not content.strip():
            continue
        messages.append({'role': _normalize_role(role), 'content': content})

    return messages or None


def _pick_prompt_completion(messages: List[Dict[str, str]]) -> Optional[Tuple[List[Dict[str, str]], str]]:
    """Pick a (prompt_messages, completion_text) pair from a conversation."""
    for idx, msg in enumerate(messages):
        if msg.get('role') != 'user':
            continue
        prompt_messages: List[Dict[str, str]] = [
            m for m in messages[:idx] if m.get('role') == 'system' and m.get('content')
        ] + [msg]
        for j in range(idx + 1, len(messages)):
            if messages[j].get('role') == 'assistant' and messages[j].get('content'):
                return prompt_messages, messages[j]['content']
    return None


def _iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_pairs_from_rows(rows: Iterable[Dict[str, Any]],
                          dataset_format: DatasetFormat = 'auto') -> Iterable[Tuple[List[Dict[str, str]], str]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        messages = _extract_messages(row, dataset_format=dataset_format)
        if not messages or len(messages) < 2:
            continue
        picked = _pick_prompt_completion(messages)
        if picked is None:
            continue
        yield picked


def _iter_pairs_from_file(dataset_path: str,
                          dataset_format: DatasetFormat = 'auto',
                          *,
                          streaming: bool = False,
                          seed: Optional[int] = None,
                          shuffle: bool = True) -> Iterable[Tuple[List[Dict[str, str]], str]]:
    rng = random.Random(seed)
    if dataset_path.endswith('.jsonl'):
        rows: Iterable[Dict[str, Any]] = _iter_jsonl(dataset_path)
        if shuffle:
            if streaming:
                buffer_size = DEFAULT_HF_SHUFFLE_BUFFER_SIZE

                def _buffered() -> Iterator[Dict[str, Any]]:
                    it = iter(rows)
                    buf: List[Dict[str, Any]] = []
                    for _ in range(buffer_size):
                        item = next(it, None)
                        if item is None:
                            break
                        buf.append(item)
                    while buf:
                        idx = rng.randrange(len(buf))
                        yield buf[idx]
                        nxt = next(it, None)
                        if nxt is None:
                            buf.pop(idx)
                        else:
                            buf[idx] = nxt

                rows = _buffered()
            else:
                materialized = list(rows)
                rng.shuffle(materialized)
                rows = materialized
        return _iter_pairs_from_rows(rows, dataset_format=dataset_format)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'Unsupported dataset JSON root type: {type(data)}')
    if shuffle:
        rng.shuffle(data)
    return _iter_pairs_from_rows(data, dataset_format=dataset_format)


def _iter_pairs_from_hf(dataset_id: str,
                        split: str,
                        streaming: bool,
                        config: Optional[str] = None,
                        seed: Optional[int] = None,
                        dataset_format: DatasetFormat = 'auto') -> Iterable[Tuple[List[Dict[str, str]], str]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError('HuggingFace dataset loading requires the `datasets` package. '
                          'Install it (e.g. `pip install datasets`).') from e

    load_kwargs = dict(split=split, streaming=streaming)
    if config is not None:
        load_kwargs['name'] = config
    ds = None
    last_error = None
    for candidate_split in _get_hf_split_candidates(dataset_id, split):
        load_kwargs['split'] = candidate_split
        try:
            ds = load_dataset(dataset_id, **load_kwargs)
            split = candidate_split
            break
        except Exception as e:
            last_error = e
    if ds is None:
        assert last_error is not None
        raise last_error

    if not streaming and seed is not None:
        try:
            ds = ds.shuffle(seed=seed)
        except Exception:
            # Best-effort: if shuffling is unsupported, fall back to dataset order.
            pass
    elif streaming and seed is not None:
        # Streaming datasets cannot be fully shuffled without scanning everything; use buffered shuffle.
        try:
            ds = ds.shuffle(seed=seed, buffer_size=DEFAULT_HF_SHUFFLE_BUFFER_SIZE)
        except TypeError:
            try:
                ds = ds.shuffle(buffer_size=DEFAULT_HF_SHUFFLE_BUFFER_SIZE, seed=seed)
            except Exception:
                pass
        except Exception:
            # Best-effort: if shuffling is unsupported, fall back to dataset order.
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
            if picked is None:
                continue
            yield picked

    return _iter_rows()


def _download_hf_data_file(dataset_id: str,
                           *,
                           filename: Optional[str] = None,
                           revision: Optional[str] = None) -> str:
    """Download a JSON/JSONL data file from a HuggingFace dataset repo."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
    except Exception as e:
        raise ImportError('Downloading a HuggingFace dataset file requires `huggingface_hub`. '
                          'Install it (e.g. `pip install huggingface_hub`).') from e

    if filename is None:
        files = list_repo_files(repo_id=dataset_id, repo_type='dataset', revision=revision)
        candidates = [f for f in files if f.endswith('.json') or f.endswith('.jsonl')]
        preferred = 'ShareGPT_V3_unfiltered_cleaned_split.json'
        if preferred in candidates:
            filename = preferred
        elif len(candidates) == 1:
            filename = candidates[0]
        else:
            preview = ', '.join(candidates[:20])
            raise ValueError('Cannot infer which dataset file to download from '
                             f'`{dataset_id}`; use `--hf-data-file`. '
                             f'Found candidates: {preview}')

    local_path = hf_hub_download(repo_id=dataset_id,
                                 repo_type='dataset',
                                 filename=filename,
                                 revision=revision)
    return str(local_path)


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    chat_template=None,  # Add chat_template parameter
    dataset_format: DatasetFormat = 'auto',
    hf_split: str = 'train',
    hf_streaming: bool = False,
    hf_config: Optional[str] = None,
    hf_data_file: Optional[str] = None,
    hf_revision: Optional[str] = None,
    max_scan_examples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Sample requests from dataset for DLLM benchmarking.
    
    Returns:
        List of tuples containing (prompt, prompt_len)
    """
    pairs_iter: Iterable[Tuple[List[Dict[str, str]], str]]
    if os.path.isfile(dataset_path):
        if dataset_path.endswith('.jsonl'):
            pairs_iter = _iter_pairs_from_rows(_iter_jsonl(dataset_path), dataset_format=dataset_format)
        else:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list) or not data:
                return []
            # Best-effort shuffle for JSON arrays without materializing the extracted pairs list.
            random.shuffle(data)
            pairs_iter = _iter_pairs_from_rows(data, dataset_format=dataset_format)
    else:
        # Treat as a HuggingFace dataset ID (e.g. allenai/WildChat).
        try:
            pairs_iter = _iter_pairs_from_hf(dataset_path,
                                             split=hf_split,
                                             streaming=hf_streaming,
                                             config=hf_config,
                                             seed=seed,
                                             dataset_format=dataset_format)
        except ImportError:
            # Allow file-based fallback even without `datasets` (common for ShareGPT dataset repos).
            local_path = _download_hf_data_file(dataset_path, filename=hf_data_file, revision=hf_revision)
            pairs_iter = _iter_pairs_from_file(local_path,
                                               dataset_format=dataset_format,
                                               streaming=hf_streaming,
                                               seed=seed,
                                               shuffle=True)
            # Continue; no `datasets`-based iteration available.
        except Exception as e:
            try:
                from datasets.exceptions import DataFilesNotFoundError  # type: ignore
            except Exception:
                raise
            if not isinstance(e, DataFilesNotFoundError):
                raise
            # Fallback for dataset repos that don't contain supported data files for `datasets.load_dataset`,
            # such as ShareGPT repos that host the JSON directly.
            local_path = _download_hf_data_file(dataset_path, filename=hf_data_file, revision=hf_revision)
            pairs_iter = _iter_pairs_from_file(local_path,
                                               dataset_format=dataset_format,
                                               streaming=hf_streaming,
                                               seed=seed,
                                               shuffle=True)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int]] = []
    scanned = 0

    for messages, completion in pairs_iter:
        if len(filtered_dataset) == num_requests:
            break

        scanned += 1
        if max_scan_examples is not None and scanned > max_scan_examples:
            break

        # Apply chat template to format the prompt.
        if chat_template is not None:
            prompt = chat_template.messages2prompt(messages, sequence_start=True, enable_thinking=False)
        else:
            # Fallback: use the selected user message.
            prompt = messages[-1]['content']

        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)
        completion_token_ids = tokenizer.encode(completion)
        output_len = len(completion_token_ids)
        
        # Apply filters for prompt length only (no output length filtering)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
            
        filtered_dataset.append((prompt, prompt_len))

    print(f'#Input tokens: {np.sum([x[1] for x in filtered_dataset])}')
    print(f'#Prompts: {len(filtered_dataset)}')
    return filtered_dataset


def _subset_elapsed_time(sessions, start_ts, fallback):
    """Estimate elapsed time for a subset of profiler sessions."""
    if not sessions or start_ts is None:
        return fallback
    finish_ts = max((sess.ts[-1] for sess in sessions if sess.ts), default=None)
    if finish_ts is None:
        return fallback
    elapsed = finish_ts - start_ts
    return elapsed if elapsed > 0 else fallback


def _summarize_profiler_slice(profiler: Profiler, sessions, title: str, hyperparams):
    """Print profiler summary for the provided `sessions`."""
    if not sessions:
        return
    partial_profiler = Profiler(profiler.stream_output, profiler.percentages)
    partial_profiler.sessions = list(sessions)
    partial_profiler.t_start = getattr(profiler, 't_start', None)
    fallback_elapsed = getattr(profiler, 'elapsed_time', 0) or 1e-9
    partial_profiler.elapsed_time = _subset_elapsed_time(partial_profiler.sessions,
                                                         partial_profiler.t_start,
                                                         fallback_elapsed)
    partial_profiler.compute_metrics()
    partial_profiler.summarize(title=title, hyperparams=hyperparams)


class _UInt64RollingHasher:
    """Simple rolling hash helper that works modulo 2**64."""

    __slots__ = ('base', 'mask', 'powers', 'prefix')

    def __init__(self, base: int):
        self.base = base
        self.mask = (1 << 64) - 1
        self.prefix = [0]
        self.powers = [1]

    def append(self, value: int):
        mask = self.mask
        next_prefix = (self.prefix[-1] * self.base + value + 1) & mask
        self.prefix.append(next_prefix)
        self.powers.append((self.powers[-1] * self.base) & mask)

    def range_hash(self, start: int, end: int) -> int:
        mask = self.mask
        prefix = self.prefix
        power = self.powers[end - start]
        return (prefix[end] - (prefix[start] * power & mask)) & mask


class RepetitionDetector:
    """Detect repeated blocks of generated tokens using rolling hashes."""

    MIN_CHECK_BLOCK = 8
    HASH_BASES = (911382323, 972663749)

    def __init__(self, block_size: int, repeats: int):
        self.block_size = block_size
        self.repeats = repeats

    def create_state(self):
        """Return a new incremental detector state for a single request."""
        return _RepetitionState(self)

    def should_stop(self, generated: List[int]) -> bool:
        """Compatibility helper that runs the detector on a full sequence."""
        state = self.create_state()
        return state.observe(generated)


class _RepetitionState:
    """Track rolling hashes for a single decoding stream."""

    __slots__ = ('detector', 'hashers', 'token_count')

    def __init__(self, detector: RepetitionDetector):
        self.detector = detector
        self.token_count = 0
        self.hashers = [_UInt64RollingHasher(base) for base in detector.HASH_BASES]

    def observe(self, tokens: Iterable[int]) -> bool:
        for token in tokens:
            if self.observe_token(token):
                return True
        return False

    def observe_token(self, token: int) -> bool:
        self.token_count += 1
        for hasher in self.hashers:
            hasher.append(token)
        return self._detect_tail()

    def _hash_range(self, start: int, end: int):
        return tuple(hasher.range_hash(start, end) for hasher in self.hashers)

    def _detect_tail(self) -> bool:
        detector = self.detector
        total = self.token_count
        if total < detector.repeats:
            return False
        max_block = min(detector.block_size, total // detector.repeats)
        if max_block <= 0:
            return False
        min_block = detector.MIN_CHECK_BLOCK if max_block >= detector.MIN_CHECK_BLOCK else 1
        if max_block < min_block:
            return False
        for block_size in range(max_block, min_block - 1, -1):
            tail_hash = self._hash_range(total - block_size, total)
            repeated = True
            for idx in range(2, detector.repeats + 1):
                block_start = total - block_size * idx
                if block_start < 0:
                    repeated = False
                    break
                if self._hash_range(block_start, block_start + block_size) != tail_hash:
                    repeated = False
                    break
            if repeated:
                return True
        return False

class Engine:

    def __init__(self,
                 model_path: str,
                 engine_config: Union[PytorchEngineConfig, TurbomindEngineConfig],
                 repetition_detector: Optional[RepetitionDetector] = None):
        self.tokenizer = Tokenizer(model_path)
        
        # Automatically detect and use the model's corresponding chat template
        from lmdeploy.model import ChatTemplateConfig, MODELS
        chat_template_name = 'base'
        for name, model in MODELS.module_dict.items():
            if model.match(model_path):
                chat_template_name = name
                break
        self.chat_template_config = ChatTemplateConfig(model_name=chat_template_name, model_path=model_path)
        self.chat_template = self.chat_template_config.chat_template
        
        if isinstance(engine_config, TurbomindEngineConfig):
            from lmdeploy.turbomind import TurboMind
            tm_model = TurboMind.from_pretrained(model_path,
                                           engine_config=engine_config,
                                           chat_template_name=chat_template_name)
            self.backend = 'turbomind'
        elif isinstance(engine_config, PytorchEngineConfig):
            from lmdeploy.pytorch.engine import Engine as PytorchEngine
            tm_model = PytorchEngine.from_pretrained(model_path, engine_config=engine_config)
            self.backend = 'pytorch'

        self.tm_model = tm_model
        self.pbar = None
        self.dllm_processed_tokens = 0
        self.dllm_decode_steps = 0
        self.repetition_detector = repetition_detector
        self.repetition_stops = 0

    async def _inference(self,
                         req_queue: Queue,
                         session_id: int,
                         temperature: float,
                         top_p: float,
                         top_k: int,
                         stream_output: bool,
                         skip_tokenize: bool,
                         skip_detokenize: bool,
                         concurrency: int,
                         max_new_tokens: int,
                         repetition_detector: Optional[RepetitionDetector] = None):
        model_inst = self.tm_model.create_instance()
        sess: Session = None
        for prompt, input_len, cancel_after, sess in iter(req_queue.get_nowait, None):

            sess.tick(0)

            if skip_tokenize:
                input_ids = prompt
            else:
                print("Prompt:")
                print(prompt)
                input_ids = self.tokenizer(prompt).input_ids

            state = DetokenizeState(len(input_ids))

            n_token = 0
            token_ids = input_ids.copy()
            stop_due_to_repetition = False
            repetition_state = repetition_detector.create_state() if repetition_detector else None
            last_dllm_stats = None

            generator = model_inst.async_stream_infer(session_id,
                                                      input_ids=input_ids,
                                                      gen_config=GenerationConfig(max_new_tokens=max_new_tokens,
                                                                                  temperature=temperature,
                                                                                  top_p=top_p,
                                                                                  top_k=top_k,
                                                                                  stop_token_ids=[self.tokenizer.eos_token_id]),
                                                      sequence_start=True,
                                                      sequence_end=True,
                                                      stream_output=stream_output)
            try:
                async for outputs in generator:
                    chunk_tokens = outputs.token_ids
                    processed_count = 0
                    repetition_hit = False
                    if repetition_state:
                        for token in chunk_tokens:
                            token_ids.append(token)
                            processed_count += 1
                            if repetition_state.observe_token(token):
                                repetition_hit = True
                                break
                    else:
                        token_ids.extend(chunk_tokens)
                        processed_count = len(chunk_tokens)
                    if processed_count and not skip_detokenize:
                        text, state = self.tokenizer.detokenize_incrementally(token_ids, state)
                        print(text, end='')
                    n_token += processed_count
                    sess.tick(n_token)
                    stats = getattr(outputs.req_metrics, 'dllm_stats', None) if outputs.req_metrics else None
                    if stats:
                        last_dllm_stats = stats
                    if repetition_hit:
                        stop_due_to_repetition = True
                        if not skip_detokenize:
                            print()
                            print('[INFO] Early stop: repetitive output detected.')
                        await model_inst.async_cancel(session_id)
                        break
                if last_dllm_stats:
                    self.dllm_processed_tokens += last_dllm_stats.get('processed_tokens', 0)
                    self.dllm_decode_steps += last_dllm_stats.get('decode_steps', 0)
                sess.finish(Session.SUCCESS)
                if not skip_detokenize:
                    print()
                if stop_due_to_repetition:
                    self.repetition_stops += 1
            finally:
                await generator.aclose()

            # for pytorch engine to restart a session
            if self.backend == 'pytorch':
                await model_inst.async_end(session_id)

            self.pbar.update(1)
            session_id += concurrency

    def process_request(self, requests, profiler: Profiler, concurrency, temperature, top_p, top_k, stream_output,
                        skip_tokenize, skip_detokenize, max_new_tokens):
        req_queue = Queue()
        self.dllm_processed_tokens = 0
        self.dllm_decode_steps = 0

        # feed request to q
        for prompt, input_len in requests:
            # Set a large cancel_after value since we're generating until EOS
            cancel_after = max_new_tokens
            sess = profiler.new_session(input_len, 0)  # Start with 0 output tokens
            req = [prompt, input_len, cancel_after, sess]
            if skip_tokenize:
                req[0] = self.tokenizer.encode(prompt)
            req_queue.put(req)
        for i in range(concurrency):
            req_queue.put(None)

        # start threads
        tasks = []
        for i in range(concurrency):
            task = self._inference(req_queue, i, temperature, top_p, top_k, stream_output, skip_tokenize,
                                   skip_detokenize, concurrency, max_new_tokens, self.repetition_detector)
            tasks.append(task)

        async def _gather_tasks(tasks):
            profiler.start()
            ret = await asyncio.gather(*tasks)
            profiler.finish()
            return ret

        self.pbar = tqdm(total=len(requests))

        asyncio.run(_gather_tasks(tasks))

        self.pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark DLLM request throughput of lmdeploy '
                                     'in localhost',
                                     formatter_class=DefaultsAndTypesHelpFormatter)
    parser.add_argument('dataset',
                        type=str,
                        help='Dataset path (.json/.jsonl) or HuggingFace dataset ID '
                        '(e.g. allenai/WildChat, nlile/hendrycks-MATH-benchmark, openai/gsm8k).')
    parser.add_argument('model_path',
                        type=str,
                        help='the path of model in localhost or '
                        'the repo_id of model in huggingface.co')
    parser.add_argument('--dataset-format',
                        type=str,
                        default='auto',
                        choices=['auto', 'sharegpt', 'wildchat', 'math', 'gsm8k'],
                        help='Dataset format: ShareGPT JSON, WildChat (HF or JSON/JSONL), '
                        'Hendrycks MATH, GSM8K, or auto-detect.')
    parser.add_argument('--hf-split',
                        type=str,
                        default=None,
                        help='HuggingFace dataset split when `dataset` is a dataset ID '
                        '(defaults to `train`, except `openai/gsm8k` uses `test`).')
    parser.add_argument('--hf-config',
                        type=str,
                        default=None,
                        help='Optional HuggingFace dataset config/subset name '
                        '(e.g. `main` or `socratic` for `openai/gsm8k`).')
    hf_streaming_group = parser.add_mutually_exclusive_group()
    hf_streaming_group.add_argument('--hf-streaming',
                                    dest='hf_streaming',
                                    action='store_true',
                                    default=False,
                                    help='Enable HuggingFace streaming mode (iterates without loading the full split).')
    hf_streaming_group.add_argument('--no-hf-streaming',
                                    dest='hf_streaming',
                                    action='store_false',
                                    help='Disable HuggingFace streaming mode (default; loads the full split into memory).')
    parser.add_argument('--hf-data-file',
                        type=str,
                        default=None,
                        help='Optional HF dataset filename to download when the repo hosts JSON/JSONL directly '
                        '(e.g. ShareGPT).')
    parser.add_argument('--hf-revision',
                        type=str,
                        default=None,
                        help='Optional HuggingFace dataset revision (branch/tag/commit).')
    parser.add_argument('--max-scan-examples',
                        type=int,
                        default=None,
                        help='Optional cap on how many dataset rows to scan while sampling prompts.')
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        help='Number of working threads to process the sampled prompts',
                        default=256)
    parser.add_argument('-n',
                        '--num-prompts',
                        type=int,
                        help='Target number of prompts to process; automatically capped by the number of '
                        'available dataset samples.',
                        default=5000)
    parser.add_argument('--no-stream-output', action='store_true', help='Use stream output')
    parser.add_argument('--skip-tokenize', action='store_true', help='Pre-tokenize input prompts before starting')
    parser.add_argument('--skip-detokenize', action='store_true', help='Skip detokenizing output tokens')
    parser.add_argument('--use-uvloop', action='store_true')
    parser.add_argument('--csv', type=str, help='Where to save the result.', default='./profile_throughput_dllm.csv')
    parser.add_argument('--seed', type=int, default=0, help='Seed used in sampling prompts from dataset')
    parser.add_argument('--distributed-executor-backend',
                        type=str,
                        default=None,
                        choices=['uni', 'mp', 'ray'],
                        help='backend of executor backend')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=16384,
        help='Maximum number of new tokens to generate for each request.',
    )
    parser.add_argument(
        '--repeat-block-detect',
        action='store_true',
        help='Stop requests early when repetitive output blocks are detected (defaults to dllm-block-length unless --repeat-block-window is set).',
    )
    parser.add_argument('--repeat-block-window',
                        type=int,
                        default=None,
                        help='Optional block size for repetition detection; defaults to dllm-block-length when unset.')
    parser.add_argument('--repeat-block-threshold',
                        type=int,
                        default=3,
                        help='Number of consecutive repeated blocks required to trigger early stop.')
    # other args
    ArgumentHelper.top_p(parser)
    ArgumentHelper.temperature(parser)
    ArgumentHelper.top_k(parser)
    ArgumentHelper.backend(parser)

    # pytorch engine args
    pt_group = parser.add_argument_group('PyTorch engine arguments')
    ArgumentHelper.eager_mode(pt_group)
    ArgumentHelper.dllm_block_length(pt_group)
    ArgumentHelper.dllm_unmasking_strategy(pt_group)
    ArgumentHelper.dllm_denoising_steps(pt_group)
    ArgumentHelper.dllm_confidence_threshold(pt_group)
    ArgumentHelper.dllm_enable_delayed_cache(pt_group)
    ArgumentHelper.dllm_enable_focus(pt_group)
    ArgumentHelper.dllm_focus_alpha(pt_group)
    ArgumentHelper.dllm_track(pt_group)

    tp_act = ArgumentHelper.tp(pt_group)
    cache_count_act = ArgumentHelper.cache_max_entry_count(pt_group)
    cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
    prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)
    quant_policy_act = ArgumentHelper.quant_policy(pt_group, default=0)
    dtype_act = ArgumentHelper.dtype(pt_group)

    # turbomind engine args
    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(cache_count_act)
    tb_group._group_actions.append(cache_block_seq_len_act)
    tb_group._group_actions.append(prefix_caching_act)
    tb_group._group_actions.append(quant_policy_act)
    tb_group._group_actions.append(dtype_act)

    ArgumentHelper.dp(tb_group)
    ArgumentHelper.cp(tb_group)
    ArgumentHelper.model_format(tb_group, default='hf')
    ArgumentHelper.num_tokens_per_iter(tb_group)
    ArgumentHelper.max_prefill_iters(tb_group)
    ArgumentHelper.communicator(tb_group)

    args = parser.parse_args()

    if args.hf_split is None:
        args.hf_split = GSM8K_SUPPORTED_SPLIT if _is_gsm8k_dataset(args.dataset) else 'train'

    if _is_gsm8k_dataset(args.dataset):
        if args.hf_split != GSM8K_SUPPORTED_SPLIT:
            parser.error(f'`{GSM8K_DATASET_ID}` is supported with the `{GSM8K_SUPPORTED_SPLIT}` split only.')
        if args.hf_config is None:
            args.hf_config = GSM8K_DEFAULT_CONFIG

    if args.repeat_block_threshold < 2:
        parser.error('--repeat-block-threshold must be >= 2.')

    if args.repeat_block_window is not None and args.repeat_block_window <= 0:
        parser.error('--repeat-block-window must be a positive integer when provided.')

    if args.repeat_block_detect:
        block_window = args.repeat_block_window
        if block_window is None:
            block_window = args.dllm_block_length
        if block_window is None or block_window <= 0:
            parser.error('--repeat-block-detect requires a positive block size via --repeat-block-window or --dllm-block-length.')
        args.repeat_block_window = block_window
    else:
        args.repeat_block_window = None

    return args


def main():
    args = parse_args()
    assert args.backend == 'pytorch', 'only support pytorch backend now'
    random.seed(args.seed)
    if args.backend == 'turbomind':
        engine_config = TurbomindEngineConfig(
            max_batch_size=args.concurrency // args.dp,
            tp=args.tp,
            dp=args.dp,
            cp=args.cp,
            cache_max_entry_count=args.cache_max_entry_count,
            cache_block_seq_len=args.cache_block_seq_len,
            model_format=args.model_format,
            quant_policy=args.quant_policy,
            num_tokens_per_iter=args.num_tokens_per_iter,
            max_prefill_iters=args.max_prefill_iters,
            enable_prefix_caching=args.enable_prefix_caching,
            dtype=args.dtype,
            communicator=args.communicator,
        )
    elif args.backend == 'pytorch':
        engine_config = PytorchEngineConfig(
            cache_max_entry_count=args.cache_max_entry_count,
            block_size=args.cache_block_seq_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            eager_mode=args.eager_mode,
            enable_prefix_caching=args.enable_prefix_caching,
            quant_policy=args.quant_policy,
            dtype=args.dtype,
            distributed_executor_backend=args.distributed_executor_backend,
            dllm_block_length=args.dllm_block_length,
            dllm_unmasking_strategy=args.dllm_unmasking_strategy,
            dllm_denoising_steps=args.dllm_denoising_steps,
            dllm_confidence_threshold=args.dllm_confidence_threshold,
            dllm_enable_delayed_cache=args.dllm_enable_delayed_cache,
            dllm_enable_focus=args.dllm_enable_focus,
            dllm_focus_alpha=args.dllm_focus_alpha,
            dllm_track=args.dllm_track,
            max_prefill_token_num=args.concurrency * args.dllm_block_length if args.dllm_block_length is not None else 4096,
        )

    if args.use_uvloop:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    repetition_detector = None
    if args.repeat_block_detect:
        repetition_detector = RepetitionDetector(args.repeat_block_window, args.repeat_block_threshold)

    engine = Engine(args.model_path, engine_config, repetition_detector=repetition_detector)

    requests = sample_requests(
        dataset_path=args.dataset,
        num_requests=args.num_prompts,
        tokenizer=engine.tokenizer.model.model,
        chat_template=engine.chat_template,  # Pass the automatically detected chat template
        dataset_format=args.dataset_format,
        hf_split=args.hf_split,
        hf_streaming=args.hf_streaming,
        hf_config=args.hf_config,
        hf_data_file=args.hf_data_file,
        hf_revision=args.hf_revision,
        max_scan_examples=args.max_scan_examples,
        seed=args.seed,
    )
    actual_num_prompts = len(requests)
    if actual_num_prompts == 0:
        raise ValueError(f'No valid prompts were sampled from dataset `{args.dataset}`.')
    if actual_num_prompts < args.num_prompts:
        print(f'[INFO] Requested {args.num_prompts} prompts but only sampled {actual_num_prompts} from '
              f'`{args.dataset}`; continuing with the available prompts.')

    stream_output = not args.no_stream_output

    profiler = Profiler(stream_output, [50, 75, 95, 99])
    effective_concurrency = min(args.concurrency, actual_num_prompts)

    engine.process_request(requests,
                           profiler,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           top_k=args.top_k,
                           concurrency=effective_concurrency,
                           stream_output=not args.no_stream_output,
                           skip_tokenize=args.skip_tokenize,
                           skip_detokenize=args.skip_detokenize,
                           max_new_tokens=args.max_new_tokens)

    # Get and display the used chat template name
    chat_template_name = engine.chat_template_config.model_name
    hyperparams = [('Concurrency', args.concurrency),
                   ('Prompts requested', args.num_prompts),
                   ('Prompts sampled', actual_num_prompts),
                   ('Max new tokens', args.max_new_tokens),
                   ('Stream output', str(stream_output).lower()),
                   ('Skip tokenize', str(args.skip_tokenize).lower()),
                   ('Skip detokenize', str(args.skip_detokenize).lower()),
                   ('Chat template', chat_template_name),
                   ('Repeat block detect', 'true' if args.repeat_block_detect else 'false')]
    if args.repeat_block_detect:
        hyperparams.extend([
            ('Repeat block window', args.repeat_block_window),
            ('Repeat block threshold', args.repeat_block_threshold),
        ])
    profiler.compute_metrics()
    profiler.summarize(title='Profile LLM Throughput', hyperparams=hyperparams)
    if getattr(engine, 'repetition_stops', 0):
        print(f'Requests stopped early due to repetition: {engine.repetition_stops}')
    dllm_processed = getattr(engine, 'dllm_processed_tokens', 0)
    dllm_steps = getattr(engine, 'dllm_decode_steps', 0)
    total_generated = getattr(profiler, 'total_output', 0)
    if dllm_processed > 0 or dllm_steps > 0:
        processed_ratio = (dllm_processed / total_generated)
        gen_per_step = (total_generated / dllm_steps)
        print('\nDLLM Metrics')
        print(f'  Processed tokens            : {dllm_processed}')
        print(f'  Generated tokens            : {total_generated}')
        print(f'  Decode steps                : {dllm_steps}')
        print(f'  processed/generated         : {processed_ratio:.3f}')
        print(f'  generated tokens per step   : {gen_per_step:.3f}')
    if args.csv:
        repeat_window_str = args.repeat_block_window if args.repeat_block_detect else '-'
        repeat_threshold = args.repeat_block_threshold if args.repeat_block_detect else '-'
        profiler.save_csv(args.csv, (
            ('backend', args.backend),
            ('bs', args.concurrency),
            ('max_new_tokens', args.max_new_tokens),
            ('num_prompts', actual_num_prompts),
            ('chat_template', chat_template_name),
            ('repeat_block_detect', str(args.repeat_block_detect).lower()),
            ('repeat_block_window', repeat_window_str),
            ('repeat_block_threshold', repeat_threshold),
        ))


if __name__ == '__main__':
    main()
