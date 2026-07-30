from __future__ import annotations

import enum
from array import array
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.dllm_partial_block: Optional[array] = None
        self.dllm_partial_start: int = 0
        self.dllm_partial_uncached: Optional[list[bool]] = None
        self.dllm_partial_needs_warmup: bool = True
        self.dllm_partial_kv_indices = None
        self.dllm_focus_token_sum: float = 0.0
        self.dllm_focus_steps: int = 0
        self.dllm_focus_progress: int = -1

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        if self.dllm_incomplete_ids:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.full_untruncated_fill_ids) < min_required_length:
            # still incoming stage
            return

        if (
            self.dllm_config.enable_delayed_cache
            and self.dllm_partial_block is not None
        ):
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        input_block = self.full_untruncated_fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if not self.dllm_config.enable_delayed_cache:
            if self.dllm_incomplete_ids:
                prefix_len = len(self.prefix_indices)
                assert len(self.dllm_incomplete_ids) == self.dllm_config.block_size
                self.full_untruncated_fill_ids = (
                    self.full_untruncated_fill_ids[:prefix_len]
                    + self.dllm_incomplete_ids
                )
                return

            self.dllm_block_offset = (
                0
                if not self.dllm_initialized
                else self.dllm_block_offset + self.dllm_config.block_size
            )
            self.full_untruncated_fill_ids = (
                self.origin_input_ids
                + self.output_ids
                + array("q", [self.dllm_config.mask_id] * self.dllm_config.block_size)
            )
            self.dllm_initialized = True
            return

        if self.dllm_partial_block is not None:
            committed_ids = self.origin_input_ids + self.output_ids
            if len(self.prefix_indices) > self.dllm_block_offset:
                self.dllm_block_offset = len(self.prefix_indices)
            prefix_ids = committed_ids[: self.dllm_block_offset]
            self.full_untruncated_fill_ids = prefix_ids + self.dllm_partial_block
            self.fill_len = len(self.full_untruncated_fill_ids)
            return

        self.dllm_block_offset = len(self.prefix_indices)
        self.full_untruncated_fill_ids = (
            self.origin_input_ids
            + self.output_ids
            + array("q", [self.dllm_config.mask_id] * self.dllm_config.block_size)
        )
        self.fill_len = len(self.full_untruncated_fill_ids)
        self.dllm_initialized = True

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        assert (
            prefix_len % self.dllm_config.block_size == 0
        ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len

    def has_dllm_partial_block(self: Req) -> bool:
        return self.dllm_partial_block is not None

    def promote_dllm_committed_prefix(
        self: Req, req_to_token_pool, safe_len=None
    ) -> int:
        if safe_len is None:
            safe_len = len(self.origin_input_ids) + len(self.output_ids)
            safe_len = (
                safe_len // self.dllm_config.block_size
            ) * self.dllm_config.block_size
            safe_len = min(safe_len, self.fill_len)

        if safe_len != len(self.prefix_indices):
            self.prefix_indices = req_to_token_pool.req_to_token[
                self.req_pool_idx, :safe_len
            ].to(dtype=self.prefix_indices.dtype, copy=True)

        self.dllm_block_offset = safe_len

        return safe_len

    def clear_dllm_partial_block(self: Req):
        self.dllm_partial_block = None
        self.dllm_partial_start = 0
        self.dllm_partial_uncached = None
        self.dllm_partial_needs_warmup = True
        self.dllm_partial_kv_indices = None
        self.dllm_focus_token_sum = 0.0
        self.dllm_focus_steps = 0
        self.dllm_focus_progress = -1
