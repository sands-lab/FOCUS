import unittest
from array import array
from unittest import mock

import torch

# ruff: noqa: E402

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.config import DllmConfig, infer_sdar_block_size_from_model_path
from sglang.srt.dllm.algorithm.low_confidence import (
    LowConfidence,
    _compute_start_from_masks,
    _focus_processing_positions_from_state,
)
from sglang.srt.dllm.mixin.req import DllmReqPhase, ReqDllmMixin
from sglang.srt.dllm.focus import (
    _focus_importance_loop,
    delayed_cache_enabled,
    focus_build_processing_batch,
    focus_build_suffix_batch,
    focus_enabled,
    focus_init_block_progress,
    focus_importance,
    focus_mark_cached_from_input_ids,
    focus_select_retain_metadata_from_importance,
    focus_select_retain_positions,
    focus_update_block_progress,
)
from sglang.srt.layers.attention.flashinfer_backend import FlashInferIndicesUpdaterPrefill
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.common import Range

class _DummyDllmReq(ReqDllmMixin):
    def set_extend_range(self, start: int, end: int) -> None:
        self.extend_range = Range(start, end)


class _FakeTreeCache:
    def supports_mamba(self):
        return False

    def evictable_size(self):
        return 1_000_000

    def swa_evictable_size(self):
        return 0


class _FakeAllocator:
    def available_size(self):
        return 1_000_000


class _FakeGroupAllocator:
    def __init__(self):
        self.group_begin_count = 0
        self.group_end_count = 0

    def free_group_begin(self):
        self.group_begin_count += 1

    def free_group_end(self):
        self.group_end_count += 1


class _FakeMetricsReporter:
    def __init__(self):
        self.num_generated_tokens = 0
        self.report_prefill_stats_calls = 0

    def report_prefill_stats(self, **kwargs):
        self.report_prefill_stats_calls += 1


class _FakeOutputStreamer:
    def __init__(self):
        self.stream_calls = 0

    def stream_output(self, reqs, return_logprob):
        self.stream_calls += 1


class _FakeScheduleBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.return_logprob = False
        self.prefill_stats = None
        self.dp_cooperation_info = None

    def batch_size(self):
        return len(self.reqs)


class _FakeModelOutput:
    def __init__(self, logits_output):
        self.logits_output = logits_output
        self.can_run_graph = False


class _RecordingModelRunner:
    def __init__(self, logits_output):
        self.logits_outputs = (
            list(logits_output)
            if isinstance(logits_output, (list, tuple))
            else [logits_output]
        )
        self.forward_batches = []

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.forward_batches.append(forward_batch)
        index = min(len(self.forward_batches) - 1, len(self.logits_outputs) - 1)
        return _FakeModelOutput(self.logits_outputs[index])


class TestFocusHelpers(CustomTestCase):
    def test_sdar_block_size_infers_from_model_path(self):
        self.assertEqual(
            infer_sdar_block_size_from_model_path("JetLM/SDAR-8B-Chat-b32"), 32
        )
        self.assertEqual(
            infer_sdar_block_size_from_model_path(
                "/cache/models--JetLM--SDAR-8B-Chat-b64/snapshots/abc"
            ),
            64,
        )
        self.assertIsNone(
            infer_sdar_block_size_from_model_path("inclusionAI/LLaDA2.0-mini")
        )

    def test_focus_processing_positions_tensor_warmup(self):
        uncached = torch.tensor(
            [
                [False, False, False, False],
                [True, False, True, False],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        )
        needs_warmup = torch.tensor([True, False, False], dtype=torch.bool)

        positions = _focus_processing_positions_from_state(uncached, needs_warmup)

        self.assertEqual(
            [pos.tolist() for pos in positions],
            [[0, 1, 2, 3], [0, 2], [0, 1, 2, 3]],
        )

    def _low_confidence(self, threshold=0.8):
        return LowConfidence(
            DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": threshold},
                block_size=4,
                mask_id=99,
                max_running_requests=2,
                enable_focus=True,
                focus_alpha=1.0,
            )
        )

    def _forward_batch(self, input_ids, batch_size=2):
        return ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=batch_size,
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req_pool_indices=torch.arange(batch_size, dtype=torch.int32),
            seq_lens=torch.full((batch_size,), 4, dtype=torch.int32),
            out_cache_loc=torch.arange(batch_size * 4, dtype=torch.int64),
            seq_lens_sum=batch_size * 4,
        )

    def test_original_low_confidence_keeps_sync_and_fdfo_contracts(self):
        logits = torch.zeros((4, 8), dtype=torch.float32)
        logits[2, 5] = 10
        logits[3, 6] = 10

        for fdfo in (False, True):
            dllm_config = DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": 0.8},
                block_size=4,
                mask_id=99,
                max_running_requests=1,
                first_done_first_out_mode=fdfo,
            )
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DLLM_EXTEND,
                batch_size=1,
                input_ids=torch.tensor([1, 2, 99, 99], dtype=torch.long),
                req_pool_indices=torch.tensor([0], dtype=torch.int32),
                seq_lens=torch.tensor([4], dtype=torch.int32),
                out_cache_loc=torch.arange(4, dtype=torch.int64),
                seq_lens_sum=4,
                dllm_config=dllm_config,
            )
            runner = _RecordingModelRunner(
                [
                    LogitsProcessorOutput(next_token_logits=None, full_logits=logits),
                    LogitsProcessorOutput(next_token_logits=None, full_logits=logits),
                ]
            )

            run_output = LowConfidence(dllm_config).run(runner, forward_batch)

            self.assertEqual(len(run_output), 5)
            self.assertEqual(len(runner.forward_batches), 1 if fdfo else 2)
            if fdfo:
                self.assertEqual(run_output[2], [0])
                self.assertEqual(run_output[1][0], [1, 2, 5, 6])
            else:
                self.assertIsNone(run_output[2])
                self.assertEqual(run_output[1][0].tolist(), [5, 6])

    def test_original_scheduler_commits_sync_and_carries_fdfo_state(self):
        from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin

        class FakeScheduler(SchedulerDllmMixin):
            def __init__(self, dllm_config):
                self.dllm_config = dllm_config
                self.token_to_kv_pool_allocator = _FakeGroupAllocator()
                self.metrics_reporter = _FakeMetricsReporter()
                self.output_streamer = _FakeOutputStreamer()
                self.tree_cache = object()

        def make_req(dllm_config):
            req = Req(
                rid="rid",
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3, 4]),
                sampling_params=SamplingParams(max_new_tokens=8),
                dllm_config=dllm_config,
                vocab_size=128,
            )
            req.sampling_params.stop_strs = []
            req.sampling_params.stop_regex_strs = []
            req.full_untruncated_fill_ids = array("q", [1, 2, 3, 4] + [99] * 4)
            req.set_extend_range(4, 8)
            return req

        sync_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
        )
        sync_req = make_req(sync_config)
        sync_scheduler = FakeScheduler(sync_config)
        sync_scheduler.process_batch_result_dllm(
            _FakeScheduleBatch([sync_req]),
            GenerationBatchResult(
                next_token_ids=[torch.tensor([5, 6])],
                can_run_cuda_graph=False,
            ),
        )
        self.assertEqual(sync_req.output_ids.tolist(), [5, 6])
        self.assertEqual(sync_req.full_untruncated_fill_ids.tolist()[-2:], [5, 6])
        self.assertEqual(sync_scheduler.metrics_reporter.num_generated_tokens, 2)

        fdfo_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            first_done_first_out_mode=True,
        )
        fdfo_req = make_req(fdfo_config)
        fdfo_scheduler = FakeScheduler(fdfo_config)
        carried_state = {"step": 1}
        fdfo_scheduler.process_batch_result_dllm(
            _FakeScheduleBatch([fdfo_req]),
            GenerationBatchResult(
                next_token_ids=[[5, 6, 7, 8]],
                accept_length_per_req_cpu=[0],
                dllm_algo_state=[carried_state],
                can_run_cuda_graph=False,
            ),
        )
        self.assertEqual(fdfo_req.dllm_incomplete_ids.tolist(), [5, 6, 7, 8])
        self.assertIs(fdfo_req.dllm_algo_state, carried_state)

    def test_dllm_next_round_uses_live_prefix_as_block_offset(self):
        req = _DummyDllmReq()
        req.origin_input_ids = array("q", range(96))
        req.output_ids = array("q", range(32))
        req.full_untruncated_fill_ids = array("q")
        req.fill_len = 128
        req.prefix_indices = torch.arange(128, dtype=torch.int64)
        req.init_diffusion_llm(
            DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": 0.8},
                block_size=32,
                mask_id=99,
                max_running_requests=2,
                enable_delayed_cache=True,
            )
        )
        req.dllm_block_offset = 32
        req.fill_len = 128
        req.prefix_indices = torch.arange(128, dtype=torch.int64)

        req._init_fill_ids_for_dllm()

        self.assertEqual(req.dllm_block_offset, 128)
        self.assertEqual(req.fill_len, 160)
        self.assertEqual(req.full_untruncated_fill_ids[-32:].tolist(), [99] * 32)

    def test_dllm_promote_committed_prefix_keeps_generated_kv_live(self):
        req = _DummyDllmReq()
        req.origin_input_ids = array("q", range(48))
        req.output_ids = array("q", range(48))
        req.full_untruncated_fill_ids = array("q", range(96))
        req.fill_len = 96
        req.prefix_indices = torch.arange(32, dtype=torch.int64)
        req.req_pool_idx = 0
        req.init_diffusion_llm(
            DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": 0.8},
                block_size=32,
                mask_id=99,
                max_running_requests=2,
                enable_delayed_cache=True,
            )
        )
        req.dllm_block_offset = 32
        req.prefix_indices = torch.arange(32, dtype=torch.int64)

        class Pool:
            req_to_token = torch.arange(128, dtype=torch.int32).view(1, 128)

        safe_len = req.promote_dllm_committed_prefix(Pool())

        self.assertEqual(safe_len, 96)
        self.assertEqual(req.dllm_block_offset, 96)
        self.assertEqual(req.prefix_indices.tolist(), list(range(96)))

    def test_dllm_promoted_prefix_skips_resolved_nomask_block(self):
        req = _DummyDllmReq()
        req.origin_input_ids = array("q", range(48))
        req.output_ids = array("q", range(48))
        req.full_untruncated_fill_ids = array("q", range(96))
        req.fill_len = 96
        req.req_pool_idx = 0
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.init_diffusion_llm(
            DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": 0.8},
                block_size=32,
                mask_id=99,
                max_running_requests=2,
                enable_delayed_cache=True,
            )
        )

        class Pool:
            req_to_token = torch.arange(128, dtype=torch.int32).view(1, 128)

        req.promote_dllm_committed_prefix(Pool())
        req._init_fill_ids_for_dllm()
        req.determine_dllm_phase()

        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)
        self.assertEqual(req.dllm_block_offset, 96)
        self.assertEqual(req.fill_len, 128)
        self.assertEqual(req.full_untruncated_fill_ids[96:].tolist(), [99] * 32)

    def test_dllm_next_round_skips_radix_prefix_cache(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=2,
            enable_delayed_cache=True,
        )
        req = Req(
            rid="rid",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3, 4]),
            sampling_params=SamplingParams(max_new_tokens=8),
            dllm_config=dllm_config,
            vocab_size=128,
        )
        req.output_ids = array("q", [10, 11, 12, 13])
        req.req_pool_idx = 0
        req.prefix_indices = torch.arange(8, dtype=torch.int64)
        req.cache_protected_len = 8

        class TreeCache:
            root_node = object()

            def swa_reprefill_tail_tokens(self):
                return 0

            def match_prefix(self, params):
                raise AssertionError("DLLM must not use radix prefix matching")

        tree_cache = TreeCache()
        req.init_next_round_input(tree_cache)

        self.assertEqual(req.prefix_indices.tolist(), list(range(8)))
        self.assertIs(req.last_node, tree_cache.root_node)
        self.assertEqual(req.num_matched_prefix_tokens, 0)
        self.assertEqual(req.cache_protected_len, 0)
        self.assertIsNone(req.extend_range)

    def test_dllm_unfinished_cache_insert_is_noop(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=2,
            enable_delayed_cache=True,
        )
        req = Req(
            rid="rid",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3, 4]),
            sampling_params=SamplingParams(max_new_tokens=8),
            dllm_config=dllm_config,
            vocab_size=128,
        )
        tree_cache = mock.Mock()

        maybe_cache_unfinished_req(req, tree_cache)

        tree_cache.cache_unfinished_req.assert_not_called()

    def test_dllm_resolved_partial_block_stays_decode_phase(self):
        req = _DummyDllmReq()
        req.origin_input_ids = array("q", range(4))
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", [0, 1, 2, 3, 10, 11, 12, 13])
        req.fill_len = 8
        req.prefix_indices = torch.arange(4, dtype=torch.int64)
        req.init_diffusion_llm(
            DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": 0.8},
                block_size=4,
                mask_id=99,
                max_running_requests=2,
                enable_delayed_cache=True,
            )
        )
        req.dllm_partial_block = array("q", [10, 11, 12, 13])

        req.determine_dllm_phase()

        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)

    def test_dllm_empty_result_finishes_at_max_new_tokens(self):
        from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin

        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=2,
            enable_delayed_cache=True,
        )
        req = Req(
            rid="rid",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3, 4]),
            sampling_params=SamplingParams(max_new_tokens=4),
            dllm_config=dllm_config,
            vocab_size=128,
        )
        req.req_pool_idx = 0
        req.output_ids = array("q", [10, 11, 12, 13])
        req.fill_len = 8
        req.kv_committed_len = 8
        req.kv_allocated_len = 8
        req.inflight_middle_chunks = 1
        req.dllm_partial_block = array("q", [20, 21, 22, 23])

        class FakeScheduler(SchedulerDllmMixin):
            def __init__(self):
                self.token_to_kv_pool_allocator = _FakeGroupAllocator()
                self.metrics_reporter = _FakeMetricsReporter()
                self.output_streamer = _FakeOutputStreamer()
                self.tree_cache = object()
                self.dllm_config = dllm_config
                self.stashed = []

            def stash_chunked_request(self, req):
                self.stashed.append(req)

        scheduler = FakeScheduler()
        batch = _FakeScheduleBatch([req])
        result = GenerationBatchResult(
            next_token_ids=[torch.empty((0,), dtype=torch.long)],
            can_run_cuda_graph=False,
        )

        with mock.patch(
            "sglang.srt.dllm.mixin.scheduler.release_kv_cache"
        ) as release_kv_cache:
            scheduler.process_batch_result_dllm(batch, result)

        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.to_json(), {"type": "length", "length": 4})
        self.assertFalse(req.has_dllm_partial_block())
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertEqual(scheduler.stashed, [])
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        self.assertEqual(scheduler.token_to_kv_pool_allocator.group_begin_count, 1)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.group_end_count, 1)

    def test_dllm_staging_prefill_uses_computed_budget(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=32,
            mask_id=99,
            max_running_requests=4,
            enable_delayed_cache=True,
        )
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.page_size = 32
        adder.tree_cache = _FakeTreeCache()
        adder.token_to_kv_pool_allocator = _FakeAllocator()
        adder.rem_input_tokens = 1024
        adder.rem_chunk_tokens = None
        adder.rem_total_token_offset = 0
        adder.cur_rem_token_offset = 0
        adder.rem_swa_token_offset = 0
        adder.is_all_swa = False
        adder.is_hybrid_swa = False
        adder.is_hybrid_ssm_cache = False
        adder.can_run_list = []
        adder.log_hit_tokens = 0
        adder.log_input_tokens = 0
        adder.reprocessed_log_hit_tokens = 0
        adder.reprocessed_log_input_tokens = 0
        adder.dllm_config = dllm_config
        adder._init_dllm_meta(dllm_config)
        req = _DummyDllmReq()
        req.origin_input_ids = array("q", range(160))
        req.output_ids = array("q")
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.dllm_config = dllm_config
        req.dllm_phase = DllmReqPhase.STAGING_PREFILL
        req.fill_len = 160
        req.retracted_stain = False
        req.has_dllm_partial_block = lambda: False

        result = adder.add_dllm_staging_req(req)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(req.extend_range.length, 128)
        self.assertEqual(req.fill_len, 128)

    def test_dllm_staging_decode_does_not_reserve_future_ar_tokens(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=32,
            mask_id=99,
            max_running_requests=4,
            enable_delayed_cache=True,
        )
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.page_size = 32
        adder.tree_cache = _FakeTreeCache()
        adder.token_to_kv_pool_allocator = _FakeAllocator()
        adder.rem_input_tokens = 1024
        adder.rem_chunk_tokens = None
        adder.rem_total_token_offset = 0
        adder.cur_rem_token_offset = 0
        adder.rem_swa_token_offset = 0
        adder.is_all_swa = False
        adder.is_hybrid_swa = False
        adder.is_hybrid_ssm_cache = False
        adder.can_run_list = []
        adder.log_hit_tokens = 0
        adder.log_input_tokens = 0
        adder.reprocessed_log_hit_tokens = 0
        adder.reprocessed_log_input_tokens = 0
        adder.dllm_config = dllm_config
        adder._init_dllm_meta(dllm_config)
        req = _DummyDllmReq()
        req.output_ids = array("q")
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.dllm_config = dllm_config
        req.dllm_phase = DllmReqPhase.STAGING_DECODE
        req.full_untruncated_fill_ids = array("q", [99] * 32)
        req.fill_len = 32
        req.retracted_stain = False
        req.sampling_params = SamplingParams(max_new_tokens=2048)

        result = adder.add_dllm_staging_req(req)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(req.extend_range.length, 32)
        self.assertEqual(adder.rem_total_token_offset, 64)
        self.assertEqual(adder.cur_rem_token_offset, 64)
        self.assertEqual(adder.rem_dllm_tokens, 96)

    def test_low_confidence_apply_logits_full_batch(self):
        algo = self._low_confidence()
        forward_batch = self._forward_batch([99, 99, 5, 6, 7, 99, 99, 8])
        logits = torch.zeros((8, 10), dtype=torch.float32)
        logits[0, 1] = 10.0
        logits[1, 2] = 1.0
        logits[5, 3] = 1.0
        logits[6, 4] = 1.0

        algo._apply_logits(
            forward_batch=forward_batch,
            logits_output=LogitsProcessorOutput(
                next_token_logits=None, full_logits=logits
            ),
            kept_positions=None,
            focus_token_sum=None,
            focus_steps=None,
        )

        self.assertEqual(forward_batch.input_ids.tolist(), [1, 99, 5, 6, 7, 3, 99, 8])

    def test_low_confidence_apply_logits_focus_kept_positions(self):
        algo = self._low_confidence()
        forward_batch = self._forward_batch([99, 99, 5, 6, 7, 99, 99, 8])
        logits = torch.zeros((4, 10), dtype=torch.float32)
        logits[0, 1] = 10.0
        logits[1, 5] = 10.0
        logits[2, 3] = 1.0
        logits[3, 9] = 10.0
        focus_token_sum = torch.zeros(2, dtype=torch.float32)
        focus_steps = torch.zeros(2, dtype=torch.int32)

        algo._apply_logits(
            forward_batch=forward_batch,
            logits_output=LogitsProcessorOutput(
                next_token_logits=None, full_logits=logits
            ),
            kept_positions=[[0, 2], [1, 3]],
            focus_token_sum=focus_token_sum,
            focus_steps=focus_steps,
        )

        self.assertEqual(forward_batch.input_ids.tolist(), [1, 99, 5, 6, 7, 3, 99, 8])
        self.assertEqual(focus_token_sum.tolist(), [1.0, 1.0])
        self.assertEqual(focus_steps.tolist(), [1, 1])

    def test_low_confidence_apply_logits_focus_kept_position_tensors(self):
        algo = self._low_confidence()
        forward_batch = self._forward_batch([99, 99, 5, 6, 7, 99, 99, 8])
        logits = torch.zeros((4, 10), dtype=torch.float32)
        logits[0, 1] = 10.0
        logits[1, 5] = 10.0
        logits[2, 3] = 1.0
        logits[3, 9] = 10.0
        focus_token_sum = torch.zeros(2, dtype=torch.float32)
        focus_steps = torch.zeros(2, dtype=torch.int32)

        algo._apply_logits(
            forward_batch=forward_batch,
            logits_output=LogitsProcessorOutput(
                next_token_logits=None, full_logits=logits
            ),
            kept_positions={
                "positions": torch.tensor([0, 2, 1, 3], dtype=torch.long),
                "lengths": torch.tensor([2, 2], dtype=torch.int32),
                "rightmost_positions": torch.tensor([2, 3], dtype=torch.int32),
            },
            focus_token_sum=focus_token_sum,
            focus_steps=focus_steps,
        )

        self.assertEqual(forward_batch.input_ids.tolist(), [1, 99, 5, 6, 7, 3, 99, 8])
        self.assertEqual(focus_token_sum.tolist(), [1.0, 1.0])
        self.assertEqual(focus_steps.tolist(), [1, 1])

    def test_low_confidence_nomask_uses_full_stable_cache_forward(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([10, 11, 12, 13], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
            seq_lens_sum=8,
            positions=torch.arange(4, dtype=torch.int64),
            extend_seq_lens=torch.tensor([4], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([4], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[4],
            extend_prefix_lens_cpu=[4],
            extend_num_tokens=4,
            dllm_config=dllm_config,
            dllm_partial_start_offsets=torch.tensor([1], dtype=torch.int32),
            dllm_partial_uncached_positions=torch.tensor(
                [[False, True, False, True]], dtype=torch.bool
            ),
            dllm_partial_needs_warmup=torch.tensor([False], dtype=torch.bool),
            dllm_focus_token_sum=torch.tensor([2.0], dtype=torch.float32),
            dllm_focus_steps=torch.tensor([1], dtype=torch.int32),
            dllm_focus_block_progress=torch.tensor([3], dtype=torch.int32),
        )
        runner = _RecordingModelRunner(
            LogitsProcessorOutput(
                next_token_logits=None,
                full_logits=torch.zeros((4, 16), dtype=torch.float32),
            )
        )

        _, next_token_ids, _, partial_state = LowConfidence(dllm_config).run(
            runner, forward_batch
        )

        self.assertIsNone(partial_state)
        self.assertEqual(len(runner.forward_batches), 1)
        stable_batch = runner.forward_batches[0]
        self.assertEqual(stable_batch.input_ids.tolist(), [10, 11, 12, 13])
        self.assertFalse(stable_batch.dllm_focus_active)
        self.assertFalse(stable_batch.dllm_delayed_active)
        self.assertTrue(stable_batch.dllm_focus_disabled)
        self.assertFalse(focus_enabled(stable_batch))
        self.assertTrue(stable_batch.dllm_nomask_forward)
        self.assertEqual(next_token_ids[0].tolist(), [11, 12, 13])

    def test_low_confidence_delays_commit_until_nomask_stable_cache_pass(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([10, 11, 99, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
            seq_lens_sum=8,
            positions=torch.arange(4, dtype=torch.int64),
            extend_seq_lens=torch.tensor([4], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([4], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[4],
            extend_prefix_lens_cpu=[4],
            extend_num_tokens=4,
            dllm_config=dllm_config,
            dllm_partial_start_offsets=torch.tensor([2], dtype=torch.int32),
            dllm_partial_uncached_positions=torch.tensor(
                [[False, False, True, True]], dtype=torch.bool
            ),
            dllm_partial_needs_warmup=torch.tensor([False], dtype=torch.bool),
            dllm_focus_token_sum=torch.tensor([2.0], dtype=torch.float32),
            dllm_focus_steps=torch.tensor([1], dtype=torch.int32),
            dllm_focus_block_progress=torch.tensor([1], dtype=torch.int32),
        )
        logits = torch.zeros((2, 16), dtype=torch.float32)
        logits[0, 5] = 10.0
        logits[1, 6] = 10.0
        runner = _RecordingModelRunner(
            LogitsProcessorOutput(
                next_token_logits=None,
                full_logits=logits,
                customized_info={
                    "focus_kept_positions": {
                        "positions": torch.tensor([2, 3], dtype=torch.long),
                        "lengths": torch.tensor([2], dtype=torch.int32),
                        "rightmost_positions": torch.tensor([3], dtype=torch.int32),
                    }
                },
            )
        )

        _, next_token_ids, _, partial_state = LowConfidence(dllm_config).run(
            runner, forward_batch
        )

        self.assertEqual(len(runner.forward_batches), 1)
        self.assertEqual(runner.forward_batches[0].input_ids.tolist(), [99, 99])
        self.assertEqual(next_token_ids[0].numel(), 0)
        self.assertIsNotNone(partial_state)
        self.assertEqual(partial_state["blocks"][0].tolist(), [10, 11, 5, 6])
        self.assertEqual(partial_state["start_offsets"].tolist(), [2])
        self.assertEqual(
            partial_state["uncached_positions"].tolist(),
            [[False, False, True, True]],
        )

    def test_low_confidence_alpha999_uses_delayed_only_processing_path(self):
        def run_with_alpha(alpha):
            dllm_config = DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": 0.8},
                block_size=4,
                mask_id=99,
                max_running_requests=1,
                enable_delayed_cache=True,
                enable_focus=True,
                focus_alpha=alpha,
            )
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DLLM_EXTEND,
                batch_size=1,
                input_ids=torch.tensor([10, 11, 99, 99], dtype=torch.long),
                req_pool_indices=torch.tensor([0], dtype=torch.int32),
                seq_lens=torch.tensor([8], dtype=torch.int32),
                out_cache_loc=torch.arange(4, dtype=torch.int64),
                seq_lens_sum=8,
                positions=torch.arange(4, dtype=torch.int64),
                extend_seq_lens=torch.tensor([4], dtype=torch.int32),
                extend_prefix_lens=torch.tensor([4], dtype=torch.int32),
                extend_start_loc=torch.tensor([0], dtype=torch.int32),
                extend_seq_lens_cpu=[4],
                extend_prefix_lens_cpu=[4],
                extend_num_tokens=4,
                dllm_config=dllm_config,
                dllm_partial_start_offsets=torch.tensor([2], dtype=torch.int32),
            )
            logits = torch.zeros((4, 16), dtype=torch.float32)
            logits[2, 5] = 10.0
            logits[3, 6] = 10.0
            runner = _RecordingModelRunner(
                LogitsProcessorOutput(
                    next_token_logits=None,
                    full_logits=logits,
                    customized_info={
                        "focus_kept_positions": {
                            "positions": torch.tensor([2], dtype=torch.long),
                            "lengths": torch.tensor([1], dtype=torch.int32),
                            "rightmost_positions": torch.tensor([2], dtype=torch.int32),
                        }
                    },
                )
            )

            LowConfidence(dllm_config).run(runner, forward_batch)
            return runner.forward_batches[0], forward_batch

        alpha999_batch, alpha999_forward_batch = run_with_alpha(999.0)
        self.assertFalse(alpha999_batch.dllm_focus_active)
        self.assertTrue(alpha999_batch.dllm_delayed_active)
        self.assertEqual(alpha999_batch.forward_mode, ForwardMode.DLLM_EXTEND)
        self.assertFalse(focus_enabled(alpha999_batch))
        self.assertEqual(
            alpha999_forward_batch.dllm_focus_block_progress.tolist(), [3]
        )

        alpha1_batch, alpha1_forward_batch = run_with_alpha(1.0)
        self.assertTrue(alpha1_batch.dllm_focus_active)
        self.assertFalse(alpha1_batch.dllm_delayed_active)
        self.assertEqual(alpha1_batch.forward_mode, ForwardMode.EXTEND)
        self.assertTrue(focus_enabled(alpha1_batch))
        self.assertEqual(alpha1_forward_batch.dllm_focus_block_progress.tolist(), [2])

    def test_low_confidence_focus_reprocesses_mask_from_stale_uncached_state(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([10, 99, 99, 13], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
            seq_lens_sum=8,
            positions=torch.arange(4, dtype=torch.int64),
            extend_seq_lens=torch.tensor([4], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([4], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[4],
            extend_prefix_lens_cpu=[4],
            extend_num_tokens=4,
            dllm_config=dllm_config,
            dllm_partial_start_offsets=torch.tensor([1], dtype=torch.int32),
            dllm_partial_uncached_positions=torch.tensor(
                [[True, False, False, True]], dtype=torch.bool
            ),
            dllm_partial_needs_warmup=torch.tensor([False], dtype=torch.bool),
            dllm_focus_token_sum=torch.tensor([1.0], dtype=torch.float32),
            dllm_focus_steps=torch.tensor([1], dtype=torch.int32),
            dllm_focus_block_progress=torch.tensor([3], dtype=torch.int32),
        )
        logits = torch.zeros((1, 16), dtype=torch.float32)
        logits[0, 5] = 10.0
        runner = _RecordingModelRunner(
            LogitsProcessorOutput(
                next_token_logits=None,
                full_logits=logits,
                customized_info={
                    "focus_kept_positions": {
                        "positions": torch.tensor([1], dtype=torch.long),
                        "lengths": torch.tensor([1], dtype=torch.int32),
                        "rightmost_positions": torch.tensor([1], dtype=torch.int32),
                    }
                },
            )
        )

        _, next_token_ids, _, partial_state = LowConfidence(dllm_config).run(
            runner, forward_batch
        )

        self.assertEqual(len(runner.forward_batches), 1)
        self.assertEqual(runner.forward_batches[0].input_ids.tolist(), [10, 99, 99, 13])
        self.assertEqual(forward_batch.input_ids.tolist(), [10, 5, 99, 13])
        self.assertEqual(next_token_ids[0].numel(), 0)
        self.assertIsNotNone(partial_state)
        self.assertEqual(partial_state["blocks"][0].tolist(), [10, 5, 99, 13])

    def test_low_confidence_delays_newly_stable_rows_in_mixed_partial_batch(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=2,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=2,
            input_ids=torch.tensor([10, 11, 12, 13, 20, 21, 99, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([8, 8], dtype=torch.int32),
            out_cache_loc=torch.arange(8, dtype=torch.int64),
            seq_lens_sum=16,
            positions=torch.arange(8, dtype=torch.int64),
            extend_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([4, 4], dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            extend_seq_lens_cpu=[4, 4],
            extend_prefix_lens_cpu=[4, 4],
            extend_num_tokens=8,
            dllm_config=dllm_config,
            dllm_partial_start_offsets=torch.tensor([1, 2], dtype=torch.int32),
            dllm_partial_uncached_positions=torch.tensor(
                [
                    [False, True, False, True],
                    [False, False, True, True],
                ],
                dtype=torch.bool,
            ),
            dllm_partial_needs_warmup=torch.tensor([False, False], dtype=torch.bool),
            dllm_focus_token_sum=torch.tensor([2.0, 2.0], dtype=torch.float32),
            dllm_focus_steps=torch.tensor([1, 1], dtype=torch.int32),
            dllm_focus_block_progress=torch.tensor([3, 1], dtype=torch.int32),
        )
        logits = torch.zeros((4, 16), dtype=torch.float32)
        logits[2, 5] = 10.0
        runner = _RecordingModelRunner(
            LogitsProcessorOutput(
                next_token_logits=None,
                full_logits=logits,
            )
        )

        _, next_token_ids, _, partial_state = LowConfidence(dllm_config).run(
            runner, forward_batch
        )

        self.assertEqual(len(runner.forward_batches), 1)
        self.assertEqual(runner.forward_batches[0].input_ids.tolist(), [11, 13, 99, 99])
        self.assertFalse(runner.forward_batches[0].dllm_focus_active)
        self.assertTrue(runner.forward_batches[0].dllm_delayed_active)
        self.assertEqual(next_token_ids[0].numel(), 0)
        self.assertEqual(next_token_ids[1].numel(), 0)
        self.assertIsNotNone(partial_state)
        self.assertEqual(partial_state["blocks"][0].tolist(), [10, 11, 12, 13])
        self.assertEqual(partial_state["blocks"][1].tolist(), [20, 21, 5, 99])
        self.assertEqual(
            partial_state["uncached_positions"].tolist(),
            [
                [False, False, False, False],
                [False, False, True, True],
            ],
        )

    def test_low_confidence_commits_rows_stable_at_mixed_pass_start(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.8},
            block_size=4,
            mask_id=99,
            max_running_requests=2,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=2,
            input_ids=torch.tensor([10, 11, 12, 13, 20, 21, 99, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([8, 8], dtype=torch.int32),
            out_cache_loc=torch.arange(8, dtype=torch.int64),
            seq_lens_sum=16,
            positions=torch.arange(8, dtype=torch.int64),
            extend_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([4, 4], dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            extend_seq_lens_cpu=[4, 4],
            extend_prefix_lens_cpu=[4, 4],
            extend_num_tokens=8,
            dllm_config=dllm_config,
            dllm_partial_start_offsets=torch.tensor([1, 2], dtype=torch.int32),
            dllm_partial_uncached_positions=torch.tensor(
                [
                    [False, False, False, False],
                    [False, False, True, True],
                ],
                dtype=torch.bool,
            ),
            dllm_partial_needs_warmup=torch.tensor([False, False], dtype=torch.bool),
            dllm_focus_token_sum=torch.tensor([2.0, 2.0], dtype=torch.float32),
            dllm_focus_steps=torch.tensor([1, 1], dtype=torch.int32),
            dllm_focus_block_progress=torch.tensor([3, 1], dtype=torch.int32),
        )
        logits = torch.zeros((6, 16), dtype=torch.float32)
        logits[4, 5] = 10.0
        runner = _RecordingModelRunner(
            LogitsProcessorOutput(
                next_token_logits=None,
                full_logits=logits,
            )
        )

        _, next_token_ids, _, partial_state = LowConfidence(dllm_config).run(
            runner, forward_batch
        )

        self.assertEqual(len(runner.forward_batches), 1)
        self.assertEqual(
            runner.forward_batches[0].input_ids.tolist(), [10, 11, 12, 13, 99, 99]
        )
        self.assertEqual(next_token_ids[0].tolist(), [11, 12, 13])
        self.assertEqual(next_token_ids[1].numel(), 0)
        self.assertIsNotNone(partial_state)
        self.assertIsNone(partial_state["blocks"][0])
        self.assertEqual(partial_state["blocks"][1].tolist(), [20, 21, 5, 99])
        self.assertEqual(
            partial_state["uncached_positions"].tolist(),
            [
                [False, False, False, False],
                [False, False, True, True],
            ],
        )

    def test_low_confidence_start_offset_sentinel_uses_mask_boundary(self):
        input_ids = torch.tensor([10, 11, 99, 99, 20, 21, 22, 99])

        self.assertEqual(
            _compute_start_from_masks(
                input_ids=input_ids,
                batch_size=2,
                block_size=4,
                mask_id=99,
                row=0,
            ),
            2,
        )
        self.assertEqual(
            _compute_start_from_masks(
                input_ids=input_ids,
                batch_size=2,
                block_size=4,
                mask_id=99,
                row=1,
            ),
            3,
        )

    def test_focus_importance_returns_one_score_per_masked_token(self):
        q = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        k = q.clone()
        scores = focus_importance(
            q=q,
            k=k,
            q_lens=[4],
            masked_positions=[torch.tensor([1, 3], dtype=torch.long)],
            num_q_heads=2,
            num_kv_heads=2,
            head_dim=2,
            scale=1.0,
        )

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].shape[0], 2)
        self.assertTrue(torch.all(scores[0] > 0))

    def test_focus_importance_uniform_matches_loop_with_padded_masks(self):
        torch.manual_seed(0)
        q = torch.randn(8, 4, dtype=torch.float32)
        k = torch.randn(8, 2, dtype=torch.float32)
        q_lens = [4, 4]
        masked_positions = [
            torch.tensor([0, 2, 3], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        ]

        batched_scores = focus_importance(
            q=q,
            k=k,
            q_lens=q_lens,
            masked_positions=masked_positions,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=2,
            scale=0.5,
        )
        loop_scores = _focus_importance_loop(
            q=q,
            k=k,
            q_lens=q_lens,
            masked_positions=masked_positions,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=2,
            scale=0.5,
        )

        self.assertEqual(len(batched_scores), len(loop_scores))
        for actual, expected in zip(batched_scores, loop_scores):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_focus_importance_cuda_matches_loop_ragged(self):
        torch.manual_seed(1)
        q_lens = [4, 3, 5]
        q = torch.randn(sum(q_lens), 4, dtype=torch.float32, device="cuda")
        k = torch.randn(sum(q_lens), 2, dtype=torch.float32, device="cuda")
        masked_positions = [
            torch.tensor([0, 2], dtype=torch.long, device="cuda"),
            torch.tensor([1], dtype=torch.long, device="cuda"),
            torch.tensor([0, 3, 4], dtype=torch.long, device="cuda"),
        ]

        triton_scores = focus_importance(
            q=q,
            k=k,
            q_lens=q_lens,
            masked_positions=masked_positions,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=2,
            scale=0.5,
        )
        loop_scores = _focus_importance_loop(
            q=q,
            k=k,
            q_lens=q_lens,
            masked_positions=masked_positions,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=2,
            scale=0.5,
        )

        for actual, expected in zip(triton_scores, loop_scores):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_focus_select_retain_positions_keeps_unmasked_and_top_masked(self):
        input_ids = torch.tensor([11, 99, 99, 14], dtype=torch.long)
        layer0_scores = [torch.tensor([0.1, 0.2], dtype=torch.float32)]
        layer1_scores = [torch.tensor([0.9, 0.3], dtype=torch.float32)]
        avg_tokens = torch.tensor([1.0], dtype=torch.float32)

        retained = focus_select_retain_positions(
            input_ids=input_ids,
            batch_size=1,
            block_size=4,
            mask_id=99,
            layer0_scores=layer0_scores,
            layer1_scores=layer1_scores,
            avg_tokens=avg_tokens,
            alpha=1.0,
            block_progress=torch.tensor([3], dtype=torch.int32),
        )

        self.assertEqual([int(v) for v in retained[0].tolist()], [0, 1, 3])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_focus_select_retain_positions_cuda_matches_cpu_processing_view(self):
        input_ids = torch.tensor([11, 99, 99, 14, 99, 12, 99], dtype=torch.long)
        processing_positions = [
            torch.tensor([0, 1, 3, 4], dtype=torch.long),
            torch.tensor([0, 2, 4], dtype=torch.long),
        ]
        layer0_scores = [
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            torch.tensor([0.0, 0.0], dtype=torch.float32),
        ]
        layer1_scores = [
            torch.tensor([5.1, 4.2], dtype=torch.float32),
            torch.tensor([1.0, 5.0], dtype=torch.float32),
        ]
        avg_tokens = torch.tensor([1.0, 1.0], dtype=torch.float32)
        progress = torch.tensor([-1, 2], dtype=torch.int32)

        expected = focus_select_retain_positions(
            input_ids=input_ids,
            batch_size=2,
            block_size=5,
            mask_id=99,
            layer0_scores=layer0_scores,
            layer1_scores=layer1_scores,
            avg_tokens=avg_tokens,
            alpha=1.0,
            block_progress=progress,
            processing_positions=processing_positions,
        )

        actual = focus_select_retain_positions(
            input_ids=input_ids.cuda(),
            batch_size=2,
            block_size=5,
            mask_id=99,
            layer0_scores=[score.cuda() for score in layer0_scores],
            layer1_scores=[score.cuda() for score in layer1_scores],
            avg_tokens=avg_tokens.cuda(),
            alpha=1.0,
            block_progress=progress.cuda(),
            processing_positions=[pos.cuda() for pos in processing_positions],
        )

        self.assertEqual(
            [retained.cpu().tolist() for retained in actual],
            [retained.tolist() for retained in expected],
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_focus_select_retain_metadata_matches_list_processing_view(self):
        input_ids = torch.tensor([11, 99, 99, 14, 99, 12, 99], dtype=torch.long)
        processing_positions = [
            torch.tensor([0, 1, 3, 4], dtype=torch.long),
            torch.tensor([0, 2, 4], dtype=torch.long),
        ]
        layer0_scores = [
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            torch.tensor([0.0, 0.0], dtype=torch.float32),
        ]
        layer1_scores = [
            torch.tensor([5.1, 4.2], dtype=torch.float32),
            torch.tensor([1.0, 5.0], dtype=torch.float32),
        ]
        avg_tokens = torch.tensor([1.0, 1.0], dtype=torch.float32)
        progress = torch.tensor([-1, 2], dtype=torch.int32)

        expected = focus_select_retain_positions(
            input_ids=input_ids,
            batch_size=2,
            block_size=5,
            mask_id=99,
            layer0_scores=layer0_scores,
            layer1_scores=layer1_scores,
            avg_tokens=avg_tokens,
            alpha=1.0,
            block_progress=progress,
            processing_positions=processing_positions,
        )

        metadata = {
            "q_lens": torch.tensor([4, 3], dtype=torch.int32, device="cuda"),
            "proc_indices": torch.tensor(
                [0, 1, 3, 4, 0, 2, 4], dtype=torch.long, device="cuda"
            ),
            "mask_indices": torch.tensor([1, 2, 4, 6], dtype=torch.long, device="cuda"),
            "mask_indptr": torch.tensor([0, 2, 4], dtype=torch.int64, device="cuda"),
            "mask_lengths": torch.tensor([2, 2], dtype=torch.int32, device="cuda"),
            "max_mask_len": torch.tensor(2, dtype=torch.int32, device="cuda"),
            "active_len": 7,
        }
        retained = focus_select_retain_metadata_from_importance(
            input_ids=input_ids.cuda(),
            batch_size=2,
            block_size=5,
            mask_id=99,
            metadata=metadata,
            layer0_scores=torch.cat(layer0_scores).cuda(),
            layer1_scores=torch.cat(layer1_scores).cuda(),
            avg_tokens=avg_tokens.cuda(),
            alpha=1.0,
            block_progress=progress.cuda(),
        )

        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=5,
            mask_id=99,
            max_running_requests=2,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=input_ids.cuda(),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([10, 10], dtype=torch.int32, device="cuda"),
            out_cache_loc=torch.arange(7, dtype=torch.int64, device="cuda"),
            seq_lens_sum=20,
            seq_lens_cpu=torch.tensor([10, 10], dtype=torch.int32),
            positions=torch.arange(7, dtype=torch.int64, device="cuda"),
            extend_seq_lens=torch.tensor([4, 3], dtype=torch.int32, device="cuda"),
            extend_prefix_lens=torch.tensor([5, 5], dtype=torch.int32, device="cuda"),
            extend_start_loc=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
            extend_seq_lens_cpu=[4, 3],
            extend_prefix_lens_cpu=[5, 5],
            extend_num_tokens=7,
            dllm_config=dllm_config,
            dllm_processing_positions=[pos.cuda() for pos in processing_positions],
        )
        expected_suffix, expected_kept_positions, expected_flat_indices = (
            focus_build_suffix_batch(
                forward_batch, [positions.cuda() for positions in expected]
            )
        )
        actual_suffix, actual_kept_positions, actual_flat_indices = (
            focus_build_suffix_batch(forward_batch, retained)
        )

        self.assertEqual(
            actual_kept_positions["positions"].cpu().tolist(),
            expected_kept_positions["positions"].cpu().tolist(),
        )
        self.assertEqual(
            actual_kept_positions["lengths"].cpu().tolist(),
            expected_kept_positions["lengths"].cpu().tolist(),
        )
        self.assertEqual(
            actual_kept_positions["rightmost_positions"].cpu().tolist(),
            expected_kept_positions["rightmost_positions"].cpu().tolist(),
        )
        self.assertEqual(
            actual_flat_indices.cpu().tolist(),
            expected_flat_indices.cpu().tolist(),
        )
        self.assertEqual(
            actual_suffix.input_ids.cpu().tolist(),
            expected_suffix.input_ids.cpu().tolist(),
        )
        self.assertEqual(
            actual_suffix.seq_lens.cpu().tolist(),
            expected_suffix.seq_lens.cpu().tolist(),
        )

    def test_focus_select_uses_threshold_fallback(self):
        input_ids = torch.tensor([99, 99, 99, 99], dtype=torch.long)
        layer0_scores = [torch.zeros(4, dtype=torch.float32)]
        layer1_scores = [torch.tensor([0.0, 0.0, 10.0, 10.0], dtype=torch.float32)]
        avg_tokens = torch.tensor([1.0], dtype=torch.float32)

        retained = focus_select_retain_positions(
            input_ids=input_ids,
            batch_size=1,
            block_size=4,
            mask_id=99,
            layer0_scores=layer0_scores,
            layer1_scores=layer1_scores,
            avg_tokens=avg_tokens,
            alpha=1.0,
            block_progress=torch.tensor([3], dtype=torch.int32),
        )

        self.assertEqual([int(v) for v in retained[0].tolist()], [1, 2, 3])

    def test_focus_select_keeps_unprocessed_before_rightmost(self):
        input_ids = torch.tensor([99, 99, 99, 99], dtype=torch.long)
        layer0_scores = [torch.zeros(4, dtype=torch.float32)]
        layer1_scores = [torch.tensor([0.0, 0.0, 0.0, 10.0], dtype=torch.float32)]
        avg_tokens = torch.tensor([1.0], dtype=torch.float32)

        retained = focus_select_retain_positions(
            input_ids=input_ids,
            batch_size=1,
            block_size=4,
            mask_id=99,
            layer0_scores=layer0_scores,
            layer1_scores=layer1_scores,
            avg_tokens=avg_tokens,
            alpha=1.0,
            block_progress=torch.tensor([-1], dtype=torch.int32),
        )

        self.assertEqual([int(v) for v in retained[0].tolist()], [0, 1, 2, 3])

    def test_focus_progress_updates_from_kept_positions(self):
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=2,
            input_ids=torch.tensor([99, 99, 99, 99, 99, 99, 99, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            out_cache_loc=torch.arange(8, dtype=torch.int64),
            seq_lens_sum=8,
        )

        progress = focus_init_block_progress(forward_batch)
        self.assertEqual(progress.tolist(), [-1, -1])

        focus_update_block_progress(forward_batch, [[0, 2], [1, 3]])
        self.assertEqual(forward_batch.dllm_focus_block_progress.tolist(), [2, 3])

        focus_update_block_progress(forward_batch, [[1], [2]])
        self.assertEqual(forward_batch.dllm_focus_block_progress.tolist(), [2, 3])

        focus_update_block_progress(
            forward_batch,
            {
                "positions": torch.tensor([3, 1], dtype=torch.long),
                "lengths": torch.tensor([1, 1], dtype=torch.int32),
                "rightmost_positions": torch.tensor([3, 1], dtype=torch.int32),
            },
        )
        self.assertEqual(forward_batch.dllm_focus_block_progress.tolist(), [3, 3])

    def test_focus_enabled_respects_disable_flag(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([99, 99, 99, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
            seq_lens_sum=4,
            dllm_config=dllm_config,
        )

        self.assertTrue(focus_enabled(forward_batch))
        forward_batch.dllm_focus_disabled = True
        self.assertFalse(focus_enabled(forward_batch))

    def test_delayed_cache_enabled_without_focus(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_delayed_cache=True,
            enable_focus=False,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([99, 99, 99, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
            seq_lens_sum=4,
            dllm_config=dllm_config,
        )

        self.assertTrue(delayed_cache_enabled(forward_batch))
        self.assertFalse(focus_enabled(forward_batch))
        forward_batch.dllm_focus_disabled = True
        self.assertFalse(delayed_cache_enabled(forward_batch))

    def test_focus_mark_cached_keeps_new_left_boundary_uncached(self):
        uncached = torch.ones((1, 4), dtype=torch.bool)
        input_ids = torch.tensor([11, 99, 12, 13], dtype=torch.long)

        focus_mark_cached_from_input_ids(
            uncached,
            input_ids=input_ids,
            batch_size=1,
            block_size=4,
            mask_id=99,
        )

        self.assertEqual(uncached.tolist(), [[True, True, False, False]])

    def test_focus_build_processing_batch_uses_rightmost_position_length(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([11, 99, 12, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            out_cache_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            seq_lens_sum=10,
            positions=torch.tensor([6, 7, 8, 9], dtype=torch.int64),
            dllm_config=dllm_config,
        )

        processing_batch = focus_build_processing_batch(
            forward_batch,
            [torch.tensor([1, 3], dtype=torch.long)],
        )

        self.assertEqual(processing_batch.forward_mode, ForwardMode.EXTEND)
        self.assertTrue(processing_batch.dllm_focus_active)
        self.assertEqual(processing_batch.input_ids.tolist(), [99, 99])
        self.assertEqual(processing_batch.positions.tolist(), [7, 9])
        self.assertEqual(processing_batch.out_cache_loc.tolist(), [101, 103])
        self.assertEqual(processing_batch.extend_seq_lens.tolist(), [2])
        self.assertEqual(processing_batch.extend_prefix_lens.tolist(), [6])
        self.assertEqual(processing_batch.seq_lens.tolist(), [10])
        self.assertEqual(processing_batch.dllm_processing_positions[0].tolist(), [1, 3])
        self.assertIsNone(processing_batch.cross_attention_custom_mask)
        self.assertIs(processing_batch.dllm_full_input_ids, forward_batch.input_ids)

    def test_processing_batch_uses_rightmost_position_length_for_left_sparse_query(
        self,
    ):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_delayed_cache=True,
            enable_focus=False,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([11, 99, 12, 13], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            out_cache_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            seq_lens_sum=10,
            positions=torch.tensor([6, 7, 8, 9], dtype=torch.int64),
            dllm_config=dllm_config,
        )

        processing_batch = focus_build_processing_batch(
            forward_batch,
            [torch.tensor([1], dtype=torch.long)],
            focus_active=False,
        )

        self.assertEqual(processing_batch.extend_seq_lens.tolist(), [1])
        self.assertEqual(processing_batch.extend_prefix_lens.tolist(), [6])
        self.assertEqual(processing_batch.seq_lens.tolist(), [8])
        self.assertEqual(processing_batch.seq_lens_sum, 8)
        self.assertIsNone(processing_batch.cross_attention_custom_mask)

    def test_delayed_cache_processing_batch_uses_dllm_extend(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_delayed_cache=True,
            enable_focus=False,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([11, 99, 12, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            out_cache_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            seq_lens_sum=10,
            positions=torch.tensor([6, 7, 8, 9], dtype=torch.int64),
            dllm_config=dllm_config,
        )

        processing_batch = focus_build_processing_batch(
            forward_batch,
            [torch.tensor([1, 3], dtype=torch.long)],
            focus_active=False,
        )

        self.assertEqual(processing_batch.forward_mode, ForwardMode.DLLM_EXTEND)
        self.assertFalse(processing_batch.dllm_focus_active)
        self.assertTrue(processing_batch.dllm_delayed_active)
        self.assertFalse(focus_enabled(processing_batch))
        self.assertEqual(processing_batch.input_ids.tolist(), [99, 99])
        self.assertEqual(processing_batch.extend_seq_lens.tolist(), [2])
        self.assertEqual(processing_batch.seq_lens.tolist(), [10])
        self.assertIsNone(processing_batch.cross_attention_custom_mask)

    def test_flashinfer_single_wrapper_receives_sparse_custom_mask(self):
        updater = FlashInferIndicesUpdaterPrefill.__new__(FlashInferIndicesUpdaterPrefill)
        updater.prefill_wrapper_ragged = object()
        updater.kv_indptr = [torch.zeros(3, dtype=torch.int32)]
        updater.qo_indptr = [torch.zeros(3, dtype=torch.int32)]

        calls = []

        def record_call(*args, **kwargs):
            calls.append((args, kwargs))

        updater.call_begin_forward = record_call
        custom_mask = torch.tensor([1, 0, 1, 1], dtype=torch.uint8)

        updater.update_single_wrapper(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
            seq_lens_sum=4,
            prefix_lens=torch.tensor([2], dtype=torch.int32),
            prefill_wrappers=[object()],
            use_ragged=False,
            encoder_lens=None,
            spec_info=None,
            cross_attention_custom_mask=custom_mask,
            query_lens=torch.tensor([1], dtype=torch.int32),
        )

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1]["cross_attention_custom_mask"], custom_mask)

    def test_focus_suffix_batch_maps_processing_positions_to_block_positions(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([11, 99, 12, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            out_cache_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            seq_lens_sum=10,
            positions=torch.tensor([6, 7, 8, 9], dtype=torch.int64),
            dllm_config=dllm_config,
        )
        processing_batch = focus_build_processing_batch(
            forward_batch,
            [torch.tensor([1, 3], dtype=torch.long)],
        )

        suffix_batch, kept_positions, flat_indices = focus_build_suffix_batch(
            processing_batch,
            [torch.tensor([0], dtype=torch.long)],
        )

        self.assertEqual(kept_positions["positions"].tolist(), [1])
        self.assertEqual(kept_positions["lengths"].tolist(), [1])
        self.assertEqual(kept_positions["rightmost_positions"].tolist(), [1])
        self.assertEqual(flat_indices.tolist(), [0])
        self.assertEqual(suffix_batch.input_ids.tolist(), [99])
        self.assertEqual(suffix_batch.positions.tolist(), [7])
        self.assertEqual(suffix_batch.out_cache_loc.tolist(), [101])
        self.assertEqual(suffix_batch.extend_seq_lens.tolist(), [1])
        self.assertEqual(suffix_batch.extend_prefix_lens.tolist(), [6])
        self.assertEqual(suffix_batch.seq_lens.tolist(), [8])

    def test_focus_suffix_batch_accepts_flat_metadata(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([11, 99, 12, 99], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            out_cache_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            seq_lens_sum=10,
            positions=torch.tensor([6, 7, 8, 9], dtype=torch.int64),
            dllm_config=dllm_config,
        )
        processing_batch = focus_build_processing_batch(
            forward_batch,
            [torch.tensor([1, 3], dtype=torch.long)],
        )

        suffix_batch, kept_positions, flat_indices = focus_build_suffix_batch(
            processing_batch,
            {
                "retain_flags": torch.tensor([True, False], dtype=torch.bool),
                "q_lens": torch.tensor([2], dtype=torch.int32),
                "proc_indices": torch.tensor([1, 3], dtype=torch.long),
            },
        )

        self.assertEqual(kept_positions["positions"].tolist(), [1])
        self.assertEqual(kept_positions["lengths"].tolist(), [1])
        self.assertEqual(kept_positions["rightmost_positions"].tolist(), [1])
        self.assertEqual(flat_indices.tolist(), [0])
        self.assertEqual(suffix_batch.input_ids.tolist(), [99])
        self.assertEqual(suffix_batch.positions.tolist(), [7])
        self.assertEqual(suffix_batch.out_cache_loc.tolist(), [101])
        self.assertEqual(suffix_batch.extend_seq_lens.tolist(), [1])
        self.assertEqual(suffix_batch.extend_prefix_lens.tolist(), [6])
        self.assertEqual(suffix_batch.seq_lens.tolist(), [8])

    def test_focus_build_suffix_batch_compacts_tokens_and_lengths(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([11, 99, 99, 14], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            out_cache_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            seq_lens_sum=10,
            positions=torch.tensor([6, 7, 8, 9], dtype=torch.int64),
            dllm_config=dllm_config,
        )

        suffix_batch, kept_positions, flat_indices = focus_build_suffix_batch(
            forward_batch,
            [torch.tensor([0, 2, 3], dtype=torch.long)],
        )

        self.assertEqual(suffix_batch.forward_mode, ForwardMode.EXTEND)
        self.assertEqual(kept_positions["positions"].tolist(), [0, 2, 3])
        self.assertEqual(kept_positions["lengths"].tolist(), [3])
        self.assertEqual(kept_positions["rightmost_positions"].tolist(), [3])
        self.assertEqual(flat_indices.tolist(), [0, 2, 3])
        self.assertEqual(suffix_batch.input_ids.tolist(), [11, 99, 14])
        self.assertEqual(suffix_batch.positions.tolist(), [6, 8, 9])
        self.assertEqual(suffix_batch.out_cache_loc.tolist(), [100, 102, 103])
        self.assertEqual(suffix_batch.extend_seq_lens.tolist(), [3])
        self.assertEqual(suffix_batch.extend_prefix_lens.tolist(), [6])
        self.assertEqual(suffix_batch.seq_lens.tolist(), [10])

    def test_focus_build_suffix_batch_metadata_uses_rightmost_position_length(self):
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=1,
            enable_focus=True,
            focus_alpha=1.0,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor([11, 99, 99, 14], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            out_cache_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            seq_lens_sum=10,
            positions=torch.tensor([6, 7, 8, 9], dtype=torch.int64),
            dllm_config=dllm_config,
        )

        suffix_batch, kept_positions, flat_indices = focus_build_suffix_batch(
            forward_batch,
            {
                "retain_flags": torch.tensor([True, False, True, True]),
                "q_lens": torch.tensor([4], dtype=torch.int32),
                "proc_indices": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            },
        )

        self.assertEqual(kept_positions["positions"].tolist(), [0, 2, 3])
        self.assertEqual(kept_positions["lengths"].tolist(), [3])
        self.assertEqual(kept_positions["rightmost_positions"].tolist(), [3])
        self.assertEqual(flat_indices.tolist(), [0, 2, 3])
        self.assertEqual(suffix_batch.extend_seq_lens.tolist(), [3])
        self.assertEqual(suffix_batch.extend_prefix_lens.tolist(), [6])
        self.assertEqual(suffix_batch.seq_lens.tolist(), [10])


if __name__ == "__main__":
    unittest.main()
