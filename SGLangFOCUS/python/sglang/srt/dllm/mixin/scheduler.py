from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, List, Optional, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)
_DLLM_SCHED_DEBUG = get_bool_env_var("SGLANG_DLLM_SCHED_DEBUG")
_DLLM_PARTIAL_KV_DEBUG = get_bool_env_var("SGLANG_DLLM_PARTIAL_KV_DEBUG")

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)

    def get_new_batch_dllm(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.enable_priority_preemption:
            running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill(running_batch=running_batch):
            return None

        running_bs = len(running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs, running_batch=running_batch)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder, running_batch=running_batch)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(
            can_run_list, forward_mode, adder=adder, running_batch=running_batch
        )
        return new_batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if self.dllm_config.enable_delayed_cache:
            return self._process_batch_result_dllm_delayed(batch, result)
        return self._process_batch_result_dllm_original(batch, result)

    def _process_batch_result_dllm_delayed(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.token_to_kv_pool_allocator.free_group_begin()
        try:
            if result.next_token_ids:
                committed_any = False
                if _DLLM_SCHED_DEBUG:
                    result_counts = {
                        "partial": 0,
                        "empty": 0,
                        "commit": 0,
                        "finished": 0,
                    }

                for idx in range(batch.batch_size()):
                    req = batch.reqs[idx]
                    partial_blocks = result.dllm_partial_blocks
                    if partial_blocks is not None and partial_blocks[idx] is not None:
                        if _DLLM_SCHED_DEBUG:
                            result_counts["partial"] += 1
                        block_size = req.dllm_config.block_size
                        block_offset = req.fill_len - block_size
                        if block_offset < 0:
                            raise RuntimeError(
                                "Unexpected DLLM partial block offset: "
                                f"fill_len={req.fill_len}, block_size={block_size}"
                            )
                        block = partial_blocks[idx].tolist()
                        if len(block) != block_size:
                            raise RuntimeError(
                                "Unexpected DLLM partial block length: "
                                f"{len(block)} != {block_size}"
                            )
                        req.dllm_block_offset = block_offset
                        req.dllm_partial_block = array("q", block)
                        req.dllm_partial_start = int(
                            result.dllm_partial_start_offsets[idx].item()
                        )
                        req.dllm_partial_uncached = result.dllm_partial_uncached_positions[
                            idx
                        ].tolist()
                        req.dllm_partial_needs_warmup = bool(
                            result.dllm_partial_needs_warmup[idx].item()
                        )
                        req.dllm_focus_token_sum = float(
                            result.dllm_focus_token_sum[idx].item()
                        )
                        req.dllm_focus_steps = int(result.dllm_focus_steps[idx].item())
                        req.dllm_focus_progress = int(
                            result.dllm_focus_block_progress[idx].item()
                        )
                        kv_indices = self.req_to_token_pool.req_to_token[
                            req.req_pool_idx,
                            block_offset : block_offset + block_size,
                        ]
                        req.dllm_partial_kv_indices = kv_indices.to(
                            dtype=kv_indices.dtype, copy=True
                        )
                        if block_offset > 0:
                            req.prefix_indices = self.req_to_token_pool.req_to_token[
                                req.req_pool_idx, :block_offset
                            ].to(dtype=req.prefix_indices.dtype, copy=True)
                        else:
                            req.prefix_indices = req.prefix_indices[:0]

                    next_token_ids = result.next_token_ids[idx].tolist()
                    raw_new_tokens = len(next_token_ids)
                    new_tokens = raw_new_tokens
                    if new_tokens == 0:
                        if _DLLM_SCHED_DEBUG:
                            result_counts["empty"] += 1
                        if not req.finished():
                            has_new_partial = (
                                partial_blocks is not None
                                and partial_blocks[idx] is not None
                            )
                            remaining_tokens = (
                                req.sampling_params.max_new_tokens - len(req.output_ids)
                            )
                            if remaining_tokens <= 0:
                                req.clear_dllm_partial_block()
                                req.update_finish_state(new_accepted_len=0)
                                if req.finished():
                                    if _DLLM_SCHED_DEBUG:
                                        result_counts["finished"] += 1
                                    if req.finished_len is not None:
                                        req.output_ids = req.output_ids[
                                            : req.finished_len
                                        ]
                                    req.kv_committed_len = min(
                                        req.kv_committed_len,
                                        len(req.origin_input_ids) + len(req.output_ids),
                                    )
                                    release_kv_cache(
                                        req, self.tree_cache, is_insert=False
                                    )
                                    req.time_stats.set_completion_time()
                                else:
                                    self.stash_chunked_request(req)
                                continue
                            if req.has_dllm_partial_block() and not has_new_partial:
                                req.clear_dllm_partial_block()
                            self.stash_chunked_request(req)
                        continue

                    req.clear_dllm_partial_block()
                    remaining_tokens = req.sampling_params.max_new_tokens - len(
                        req.output_ids
                    )
                    if remaining_tokens <= 0:
                        req.update_finish_state(new_accepted_len=0)
                        if req.finished():
                            release_kv_cache(req, self.tree_cache, is_insert=False)
                            req.time_stats.set_completion_time()
                        else:
                            self.stash_chunked_request(req)
                        continue

                    if new_tokens > remaining_tokens:
                        next_token_ids = next_token_ids[:remaining_tokens]
                        new_tokens = remaining_tokens

                    commit_start = req.fill_len - raw_new_tokens
                    req.full_untruncated_fill_ids[
                        commit_start : commit_start + new_tokens
                    ] = array("q", next_token_ids)
                    self.metrics_reporter.num_generated_tokens += new_tokens
                    committed_any = True

                    req.output_ids.extend(next_token_ids)
                    req.update_finish_state(new_accepted_len=new_tokens)
                    if _DLLM_SCHED_DEBUG:
                        result_counts["commit"] += 1

                    if req.finished():
                        if _DLLM_SCHED_DEBUG:
                            result_counts["finished"] += 1
                        if req.finished_len is not None:
                            req.output_ids = req.output_ids[: req.finished_len]
                        req.kv_committed_len = min(
                            req.kv_committed_len,
                            len(req.origin_input_ids) + len(req.output_ids),
                        )
                        release_kv_cache(req, self.tree_cache, is_insert=False)
                        req.time_stats.set_completion_time()
                    else:
                        req.promote_dllm_committed_prefix(self.req_to_token_pool)

                if committed_any:
                    self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
                if _DLLM_SCHED_DEBUG:
                    logger.info("DLLM result debug: %s", result_counts)
            else:
                for req in batch.reqs:
                    if not req.finished():
                        self.stash_chunked_request(req)
        finally:
            for req in batch.reqs:
                if req.inflight_middle_chunks > 0:
                    req.inflight_middle_chunks -= 1
                    req.time_stats.set_last_chunked_prefill_finish_time()
            self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=result.can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _process_batch_result_dllm_original(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        """Process the upstream synchronous/FDFO DLLM result contract."""
        if result.copy_done is not None:
            result.copy_done.synchronize()

        fdfo_mode = self.dllm_config.first_done_first_out_mode
        assert (
            not fdfo_mode or result.accept_length_per_req_cpu is not None
        ), "FDFO dLLM result is missing accept lengths."

        # FDFO also commits unresolved blocks so their KV can be reused.
        if fdfo_mode or result.next_token_ids:
            block_size = self.dllm_config.block_size
            algo_states = result.dllm_algo_state

            self.token_to_kv_pool_allocator.free_group_begin()
            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                if not fdfo_mode:
                    next_token_ids = result.next_token_ids[idx].tolist()
                    new_tokens = len(next_token_ids)
                    if new_tokens == 0:
                        continue

                    req.full_untruncated_fill_ids[
                        req.extend_range.end - new_tokens : req.extend_range.end
                    ] = array("q", next_token_ids)
                    self.metrics_reporter.num_generated_tokens += new_tokens

                    req.output_ids.extend(next_token_ids)
                    req.update_finish_state(new_accepted_len=new_tokens)

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    continue

                next_token_ids = result.next_token_ids[idx]
                assert len(next_token_ids) == block_size

                if result.accept_length_per_req_cpu[idx] == 0:
                    # Unresolved: keep partial state and KV for the next FDFO
                    # round.
                    req.dllm_incomplete_ids = array("q", next_token_ids)
                    req.dllm_algo_state = (
                        algo_states[idx] if algo_states is not None else None
                    )
                    continue

                req.dllm_incomplete_ids = array("q")
                req.dllm_algo_state = None

                req.full_untruncated_fill_ids[
                    req.extend_range.end - block_size : req.extend_range.end
                ] = array("q", next_token_ids)

                len_input = len(req.origin_input_ids)
                len_fill = req.extend_range.end
                if len_fill <= len_input:
                    continue

                if len_fill - len(next_token_ids) < len_input:
                    next_token_ids = next_token_ids[len_input - len_fill :]

                self.metrics_reporter.num_generated_tokens += len(next_token_ids)
                req.output_ids.extend(next_token_ids)
                req.update_finish_state(new_accepted_len=len(next_token_ids))

                if req.finished():
                    # FDFO advances a complete diffusion block, but a request
                    # can stop part-way through it (for example, due to the
                    # repeat-block detector).  Keep only the accepted prefix
                    # and release the speculative tail.  Finished DLLM
                    # requests should not populate the radix cache: their
                    # in-progress blocks are managed by the FDFO scheduler,
                    # not by the normal prefix-cache lifecycle.
                    if req.finished_len is not None:
                        req.output_ids = req.output_ids[: req.finished_len]
                    req.kv_committed_len = min(
                        req.kv_committed_len,
                        len(req.origin_input_ids) + len(req.output_ids),
                    )
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                    req.time_stats.set_completion_time()

            self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
            self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=result.can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))

        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_prefill(self: Scheduler, running_batch: ScheduleBatch) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(
        self: Scheduler, running_bs: int, running_batch: ScheduleBatch
    ) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(
        self: Scheduler, adder: PrefillAdder, running_batch: ScheduleBatch
    ) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        forward_mode = ForwardMode.DLLM_EXTEND

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
                running_batch=running_batch,
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
                running_batch=running_batch,
            )

        return forward_mode

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
        running_batch: ScheduleBatch,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(
                adder, incoming_reqs, running_batch=running_batch
            )

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_inflight_middle_chunks()

        if _DLLM_SCHED_DEBUG:
            running_bs = len(self.running_batch.reqs)
            phase_counts = {}
            phase_extend = {}
            phase_output = {}
            phase_prefix = {}
            for req in can_run_list:
                extend_len = (
                    req.extend_range.length if req.extend_range is not None else 0
                )
                phase_counts[req.dllm_phase.name] = (
                    phase_counts.get(req.dllm_phase.name, 0) + 1
                )
                phase_extend.setdefault(req.dllm_phase.name, {})
                phase_extend[req.dllm_phase.name][extend_len] = (
                    phase_extend[req.dllm_phase.name].get(extend_len, 0) + 1
                )
                phase_output.setdefault(req.dllm_phase.name, {})
                output_len = len(req.output_ids)
                phase_output[req.dllm_phase.name][output_len] = (
                    phase_output[req.dllm_phase.name].get(output_len, 0) + 1
                )
                phase_prefix.setdefault(req.dllm_phase.name, {})
                prefix_len = len(req.prefix_indices)
                phase_prefix[req.dllm_phase.name][prefix_len] = (
                    phase_prefix[req.dllm_phase.name].get(prefix_len, 0) + 1
                )
            logger.info(
                "DLLM schedule debug: can_run=%d phases=%s global_waiting=%d "
                "dllm_waiting=%d dllm_staging=%d running_bs=%d "
                "phase_extend=%s phase_prefix=%s phase_output=%s "
                "log_input_tokens=%d rem_input=%d rem_dllm=%d "
                "rem_total=%d cur_rem=%d req_pool_avail=%d",
                len(can_run_list),
                phase_counts,
                len(self.waiting_queue),
                len(self.dllm_manager.waiting_queue),
                len(self.dllm_manager.staging_queue),
                running_bs,
                phase_extend,
                phase_prefix,
                phase_output,
                adder.log_input_tokens,
                adder.rem_input_tokens,
                adder.rem_dllm_tokens,
                int(adder.rem_total_tokens),
                int(adder.cur_rem_tokens),
                self.req_to_token_pool.available_size(),
            )

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

    def _create_dllm_batch(
        self: Scheduler,
        can_run_list: List[Req],
        forward_mode: ForwardMode,
        adder: PrefillAdder,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        self._debug_check_dllm_kv_ownership()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def _debug_check_dllm_kv_ownership(self: Scheduler) -> None:
        if not _DLLM_PARTIAL_KV_DEBUG:
            return

        rows = []
        seen = set()
        for queue in (self.dllm_manager.waiting_queue, self.dllm_manager.staging_queue):
            for req in queue:
                if req.req_pool_idx is None or req.rid in seen:
                    continue
                seen.add(req.rid)
                rows.append(req)

        page_owner = {}
        page_size = self.page_size
        req_to_token = self.req_to_token_pool.req_to_token
        for req in rows:
            valid_len = min(max(req.kv_allocated_len, 0), req_to_token.shape[1])
            if valid_len <= 0:
                continue
            pages = torch.unique(
                req_to_token[req.req_pool_idx, :valid_len].to(torch.int64)
                // page_size
            ).detach().cpu().tolist()
            for page in pages:
                owner = page_owner.get(page)
                if owner is not None and owner != req.rid:
                    raise RuntimeError(
                        "DLLM KV page has multiple active owners: "
                        f"page={page} owner={owner} other={req.rid}"
                    )
                page_owner[page] = req.rid

        for req in rows:
            partial_kv = getattr(req, "dllm_partial_kv_indices", None)
            if partial_kv is None or partial_kv.numel() == 0:
                continue
            pages = torch.unique(
                partial_kv.to(device=req_to_token.device, dtype=torch.int64) // page_size
            ).detach().cpu().tolist()
            for page in pages:
                owner = page_owner.get(page)
                if owner != req.rid:
                    raise RuntimeError(
                        "DLLM partial KV page ownership mismatch: "
                        f"page={page} partial_owner={req.rid} active_owner={owner}"
                    )

    def process_dllm_incoming_reqs(
        self: Scheduler,
        adder: PrefillAdder,
        reqs: List[Req],
        running_batch: ScheduleBatch,
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            if self.dllm_config.enable_delayed_cache:
                running_bs = len(self.running_batch.reqs)
                if self._is_dllm_prefill_adder_full(adder, running_bs, req):
                    self.running_batch.batch_is_full = True
            else:
                running_bs = len(running_batch.reqs)
                if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                    running_bs
                ):
                    running_batch.batch_is_full = True

            # Try preemption if batch is full
            if running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if _DLLM_SCHED_DEBUG:
                    extend_len = (
                        req.extend_range.length
                        if req.extend_range is not None
                        else 0
                    )
                    logger.info(
                        "DLLM incoming stop: res=%s can_run=%d phase=%s "
                        "extend_input_len=%d output_len=%d max_new_tokens=%d "
                        "global_waiting=%d dllm_waiting=%d dllm_staging=%d "
                        "rem_input=%d rem_dllm=%d rem_total=%d cur_rem=%d "
                        "req_pool_avail=%d allocatable=%d",
                        res.name,
                        len(adder.can_run_list),
                        req.dllm_phase.name,
                        extend_len,
                        len(req.output_ids),
                        req.sampling_params.max_new_tokens,
                        len(self.waiting_queue),
                        len(self.dllm_manager.waiting_queue),
                        len(self.dllm_manager.staging_queue),
                        adder.rem_input_tokens,
                        adder.rem_dllm_tokens,
                        int(adder.rem_total_tokens),
                        int(adder.cur_rem_tokens),
                        self.req_to_token_pool.available_size(),
                        self.get_num_allocatable_reqs(running_bs),
                    )
                if res == AddReqResult.NO_TOKEN:
                    running_batch.batch_is_full = True
                break

        return res

    def _is_dllm_prefill_adder_full(
        self: Scheduler, adder: PrefillAdder, running_bs: int, req: Req
    ) -> bool:
        if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
            return True

        if req.req_pool_idx is not None:
            return False

        pending_req_slots = sum(r.req_pool_idx is None for r in adder.can_run_list)
        return pending_req_slots >= self.req_to_token_pool.available_size()

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                if _DLLM_SCHED_DEBUG:
                    extend_len = (
                        req.extend_range.length
                        if req.extend_range is not None
                        else 0
                    )
                    logger.info(
                        "DLLM staging stop: can_run=%d phase=%s "
                        "extend_input_len=%d output_len=%d max_new_tokens=%d "
                        "global_waiting=%d dllm_waiting=%d dllm_staging=%d "
                        "rem_input=%d rem_dllm=%d rem_total=%d cur_rem=%d "
                        "req_pool_avail=%d",
                        len(adder.can_run_list),
                        req.dllm_phase.name,
                        extend_len,
                        len(req.output_ids),
                        req.sampling_params.max_new_tokens,
                        len(self.waiting_queue),
                        len(self.dllm_manager.waiting_queue),
                        len(self.dllm_manager.staging_queue),
                        adder.rem_input_tokens,
                        adder.rem_dllm_tokens,
                        int(adder.rem_total_tokens),
                        int(adder.cur_rem_tokens),
                        self.req_to_token_pool.available_size(),
                    )
                return res

        return AddReqResult.CONTINUE


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by PrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by PrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if the active DLLM queues are empty."""
        if self.dllm_config is None:
            return True
        if self.dllm_config.enable_delayed_cache:
            return len(self.waiting_queue) == 0 and len(self.staging_queue) == 0
        return len(self.waiting_queue) == 0

    def increment_inflight_middle_chunks(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.inflight_middle_chunks += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def pop_aborted_reqs(self, abort_all: bool, rid: str) -> List[Req]:
        aborted_reqs: List[Req] = []
        seen: Set[int] = set()

        for queue_name in ("waiting_queue", "staging_queue"):
            queue = getattr(self, queue_name)
            kept_queue = []
            for req in queue:
                if abort_all or req.rid.startswith(rid):
                    req_id = id(req)
                    if req_id not in seen:
                        aborted_reqs.append(req)
                        seen.add(req_id)
                else:
                    kept_queue.append(req)
            setattr(self, queue_name, kept_queue)

        return aborted_reqs

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
