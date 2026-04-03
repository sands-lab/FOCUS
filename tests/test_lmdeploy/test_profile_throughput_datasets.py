import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _add_argument(parser, *names, **kwargs):
    if not names:
        return None
    return parser.add_argument(*names, **kwargs)


def _install_lmdeploy_stubs():
    if 'lmdeploy.cli.utils' in sys.modules:
        return

    lmdeploy_pkg = types.ModuleType('lmdeploy')
    cli_pkg = types.ModuleType('lmdeploy.cli')
    cli_utils = types.ModuleType('lmdeploy.cli.utils')
    messages_mod = types.ModuleType('lmdeploy.messages')
    profiler_mod = types.ModuleType('lmdeploy.profiler')
    tokenizer_mod = types.ModuleType('lmdeploy.tokenizer')
    utils_mod = types.ModuleType('lmdeploy.utils')

    class ArgumentHelper:

        @staticmethod
        def top_p(parser):
            return _add_argument(parser, '--top-p', type=float, default=1.0)

        @staticmethod
        def temperature(parser):
            return _add_argument(parser, '--temperature', type=float, default=1.0)

        @staticmethod
        def top_k(parser):
            return _add_argument(parser, '--top-k', type=int, default=1)

        @staticmethod
        def backend(parser):
            return _add_argument(parser, '--backend', type=str, default='pytorch')

        @staticmethod
        def eager_mode(parser):
            return _add_argument(parser, '--eager-mode', action='store_true')

        @staticmethod
        def dllm_block_length(parser):
            return _add_argument(parser, '--dllm-block-length', type=int, default=None)

        @staticmethod
        def dllm_unmasking_strategy(parser):
            return _add_argument(parser, '--dllm-unmasking-strategy', type=str, default=None)

        @staticmethod
        def dllm_denoising_steps(parser):
            return _add_argument(parser, '--dllm-denoising-steps', type=int, default=None)

        @staticmethod
        def dllm_confidence_threshold(parser):
            return _add_argument(parser, '--dllm-confidence-threshold', type=float, default=None)

        @staticmethod
        def dllm_enable_delayed_cache(parser):
            return _add_argument(parser, '--dllm-enable-delayed-cache', action='store_true')

        @staticmethod
        def dllm_enable_focus(parser):
            return _add_argument(parser, '--dllm-enable-focus', action='store_true')

        @staticmethod
        def dllm_focus_alpha(parser):
            return _add_argument(parser, '--dllm-focus-alpha', type=float, default=None)

        @staticmethod
        def dllm_track(parser):
            return _add_argument(parser, '--dllm-track', action='store_true')

        @staticmethod
        def tp(parser):
            return _add_argument(parser, '--tp', type=int, default=1)

        @staticmethod
        def cache_max_entry_count(parser):
            return _add_argument(parser, '--cache-max-entry-count', type=float, default=0.8)

        @staticmethod
        def cache_block_seq_len(parser):
            return _add_argument(parser, '--cache-block-seq-len', type=int, default=64)

        @staticmethod
        def enable_prefix_caching(parser):
            return _add_argument(parser, '--enable-prefix-caching', action='store_true')

        @staticmethod
        def quant_policy(parser, default=0):
            return _add_argument(parser, '--quant-policy', type=int, default=default)

        @staticmethod
        def dtype(parser):
            return _add_argument(parser, '--dtype', type=str, default='auto')

        @staticmethod
        def dp(parser):
            return _add_argument(parser, '--dp', type=int, default=1)

        @staticmethod
        def cp(parser):
            return _add_argument(parser, '--cp', type=int, default=1)

        @staticmethod
        def model_format(parser, default='hf'):
            return _add_argument(parser, '--model-format', type=str, default=default)

        @staticmethod
        def num_tokens_per_iter(parser):
            return _add_argument(parser, '--num-tokens-per-iter', type=int, default=0)

        @staticmethod
        def max_prefill_iters(parser):
            return _add_argument(parser, '--max-prefill-iters', type=int, default=0)

        @staticmethod
        def communicator(parser):
            return _add_argument(parser, '--communicator', type=str, default='default')

    class _DummyConfig:

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _DummyProfiler:
        pass

    class _DummySession:
        pass

    class _DummyTokenizer:
        pass

    class _DummyDetokenizeState:
        pass

    class _DummyLogger:

        def setLevel(self, *_args, **_kwargs):
            return None

    cli_utils.ArgumentHelper = ArgumentHelper
    cli_utils.DefaultsAndTypesHelpFormatter = argparse.HelpFormatter
    messages_mod.GenerationConfig = _DummyConfig
    messages_mod.PytorchEngineConfig = _DummyConfig
    messages_mod.TurbomindEngineConfig = _DummyConfig
    profiler_mod.Profiler = _DummyProfiler
    profiler_mod.Session = _DummySession
    tokenizer_mod.DetokenizeState = _DummyDetokenizeState
    tokenizer_mod.Tokenizer = _DummyTokenizer
    utils_mod.get_logger = lambda *_args, **_kwargs: _DummyLogger()

    lmdeploy_pkg.cli = cli_pkg
    cli_pkg.utils = cli_utils

    sys.modules['lmdeploy'] = lmdeploy_pkg
    sys.modules['lmdeploy.cli'] = cli_pkg
    sys.modules['lmdeploy.cli.utils'] = cli_utils
    sys.modules['lmdeploy.messages'] = messages_mod
    sys.modules['lmdeploy.profiler'] = profiler_mod
    sys.modules['lmdeploy.tokenizer'] = tokenizer_mod
    sys.modules['lmdeploy.utils'] = utils_mod


_install_lmdeploy_stubs()


MODULE_PATH = Path(__file__).resolve().parents[2] / 'benchmark' / 'profile_throughput.py'
SPEC = importlib.util.spec_from_file_location('benchmark_profile_throughput', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
profile_throughput = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_throughput)


def test_extract_gsm8k_messages():
    row = {
        'question': 'If Alice has 2 apples and buys 2 more, how many apples does she have?',
        'answer': 'Alice has 2 + 2 = 4 apples. #### 4',
    }

    expected = [
        {'role': 'user', 'content': row['question']},
        {'role': 'assistant', 'content': row['answer']},
    ]

    assert profile_throughput._extract_messages(row, dataset_format='gsm8k') == expected
    assert profile_throughput._extract_messages(row, dataset_format='auto') == expected


def test_gsm8k_split_candidates():
    assert profile_throughput._get_hf_split_candidates('openai/gsm8k', 'test') == ('test', 'validation')
    assert profile_throughput._get_hf_split_candidates('openai/gsm8k', 'validation') == ('validation', 'test')
    assert profile_throughput._get_hf_split_candidates('allenai/WildChat', 'train') == ('train',)


def test_parse_args_defaults_gsm8k_to_test_split_and_main_config(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['profile_throughput.py', 'openai/gsm8k', '/tmp/model'])
    args = profile_throughput.parse_args()

    assert args.dataset_format == 'auto'
    assert args.hf_split == 'test'
    assert args.hf_config == 'main'


def test_parse_args_rejects_non_test_split_for_gsm8k(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['profile_throughput.py', 'openai/gsm8k', '/tmp/model', '--hf-split', 'train'])

    with pytest.raises(SystemExit):
        profile_throughput.parse_args()
