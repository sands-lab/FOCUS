import logging
import os
from dataclasses import replace
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.focus import (
    FocusKeptPositions,
    delayed_cache_enabled,
    focus_build_processing_batch,
    focus_debug_summary_enabled,
    focus_enabled,
    focus_init_block_progress,
    focus_kept_positions_from_output,
    focus_log_debug_summary,
    focus_log_timing_summary,
    focus_mark_cached_from_input_ids,
    focus_processing_positions,
    focus_profile_range,
    focus_update_block_progress,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _trace_rows() -> Optional[set[int]]:
    raw = os.environ.get("SGLANG_DLLM_TRACE_ROWS")
    if raw is None:
        return None
    rows = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            rows.add(int(item))
    return rows


def _trace_row_enabled(rows: Optional[set[int]], row: int) -> bool:
    return rows is not None and row in rows


def _score_trace_rows() -> Optional[set[int]]:
    raw = os.environ.get("SGLANG_DLLM_SCORE_TRACE_ROWS")
    if raw is None:
        return None
    rows = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            rows.add(int(item))
    return rows


def _score_trace_limit() -> int:
    return int(os.environ.get("SGLANG_DLLM_SCORE_TRACE_LIMIT", "8") or 8)


def _trace_block_update(
    *,
    row: int,
    start: int,
    processing_positions: List[torch.Tensor],
    block_before: torch.Tensor,
    block_after: torch.Tensor,
    uncached_before: Optional[torch.Tensor],
    uncached_after: Optional[torch.Tensor],
    needs_warmup_before: Union[bool, torch.Tensor],
    decoded_count: int,
    threshold: float,
    partial: Optional[bool] = None,
    next_len: Optional[int] = None,
    mask_id: Optional[int] = None,
) -> None:
    before_cpu = block_before.detach().cpu()
    after_cpu = block_after.detach().cpu()
    decoded_positions = [
        idx
        for idx, (before_token, after_token) in enumerate(
            zip(before_cpu.tolist(), after_cpu.tolist())
        )
        if before_token != after_token
    ]
    proc = (
        [int(v) for v in processing_positions[row].detach().cpu().tolist()]
        if row < len(processing_positions)
        else []
    )
    if isinstance(needs_warmup_before, torch.Tensor):
        warmup = bool(needs_warmup_before[row].detach().cpu().item())
    else:
        warmup = bool(needs_warmup_before)
    uncached_before_count = (
        int(uncached_before[row].detach().to(torch.int32).sum().cpu().item())
        if uncached_before is not None
        else -1
    )
    uncached_after_count = (
        int(uncached_after[row].detach().to(torch.int32).sum().cpu().item())
        if uncached_after is not None
        else -1
    )
    decoded_tokens = [int(after_cpu[pos].item()) for pos in decoded_positions]
    msg = (
        "SGLang DLLM trace "
        f"row={row} start={start} threshold={threshold} "
        f"warmup={warmup} proc={proc} "
        f"uncached={uncached_before_count}->{uncached_after_count} "
        f"decoded_count={decoded_count} "
        f"decoded_positions={decoded_positions} "
        f"decoded_tokens={decoded_tokens} "
    )
    if partial is not None:
        msg += f"partial={partial} next_len={next_len} "
    if mask_id is not None:
        msg += f"mask_remaining={int((after_cpu == mask_id).sum().item())}"
    logger.info(msg)
    print(msg, flush=True)


def _trace_focus_kept_positions(
    *,
    row: int,
    kept_positions: FocusKeptPositions,
    avg_tokens: Optional[torch.Tensor],
    block_progress: Optional[torch.Tensor],
) -> None:
    if not isinstance(kept_positions, dict):
        if row >= len(kept_positions):
            return
        positions = [int(v) for v in kept_positions[row]]
        rightmost = max(positions) if positions else -1
    else:
        lengths = kept_positions["lengths"]
        if row < 0 or row >= int(lengths.numel()):
            return
        starts = torch.cumsum(lengths.to(dtype=torch.long), dim=0) - lengths.to(
            dtype=torch.long
        )
        row_start = int(starts[row].detach().cpu().item())
        row_len = int(lengths[row].detach().cpu().item())
        positions_tensor = kept_positions["positions"][row_start : row_start + row_len]
        positions = [int(v) for v in positions_tensor.detach().cpu().tolist()]
        rightmost_tensor = kept_positions.get("rightmost_positions")
        rightmost = (
            int(rightmost_tensor[row].detach().cpu().item())
            if rightmost_tensor is not None
            else (max(positions) if positions else -1)
        )

    avg = (
        float(avg_tokens[row].detach().cpu().item())
        if avg_tokens is not None and row < int(avg_tokens.numel())
        else float("nan")
    )
    progress = (
        int(block_progress[row].detach().cpu().item())
        if block_progress is not None and row < int(block_progress.numel())
        else -1
    )
    msg = (
        "SGLang FOCUS kept "
        f"row={row} avg={avg:.4f} progress={progress} "
        f"kept_len={len(positions)} rightmost={rightmost} positions={positions}"
    )
    logger.info(msg)
    print(msg, flush=True)


def _dense_compare_enabled() -> bool:
    return os.environ.get("SGLANG_DLLM_DELAYED_DENSE_COMPARE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _trunc_compare_enabled() -> bool:
    return os.environ.get("SGLANG_DLLM_DELAYED_TRUNC_COMPARE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _dense_compare_detail_enabled() -> bool:
    return os.environ.get("SGLANG_DLLM_DELAYED_DENSE_COMPARE_DETAIL", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _dense_compare_detail_limit() -> int:
    return int(os.environ.get("SGLANG_DLLM_DELAYED_DENSE_COMPARE_DETAIL_LIMIT", "12"))


def _force_full_delayed_processing() -> bool:
    return os.environ.get("SGLANG_DLLM_DELAYED_FORCE_FULL", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _lmdeploy_prefill_mimic_enabled() -> bool:
    return os.environ.get("SGLANG_DLLM_LMDEPLOY_PREFILL_MIMIC", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _snapshot_forward_batch(batch: ForwardBatch):
    attrs = (
        "batch_size",
        "input_ids",
        "positions",
        "out_cache_loc",
        "req_pool_indices",
        "seq_lens",
        "seq_lens_cpu",
        "seq_lens_sum",
        "extend_num_tokens",
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_start_loc",
        "extend_seq_lens_cpu",
        "extend_prefix_lens_cpu",
        "num_token_non_padded",
        "num_token_non_padded_cpu",
        "global_num_tokens_cpu",
    )
    return tuple((attr, getattr(batch, attr, None)) for attr in attrs)


def _restore_forward_batch(batch: ForwardBatch, snapshot):
    for attr, value in snapshot:
        setattr(batch, attr, value)


def _focus_processing_positions_from_state(
    uncached_positions: torch.Tensor,
    needs_warmup: Union[bool, torch.Tensor],
) -> List[torch.Tensor]:
    if isinstance(needs_warmup, bool):
        return focus_processing_positions(uncached_positions, needs_warmup)

    processing_mask = _focus_processing_mask_from_state(
        uncached_positions, needs_warmup
    )
    row_ids, local_positions = processing_mask.nonzero(as_tuple=True)
    counts = torch.bincount(row_ids, minlength=uncached_positions.shape[0]).to(
        dtype=torch.long
    )
    return list(local_positions.to(dtype=torch.long).split(counts.cpu().tolist()))


def _focus_processing_mask_from_state(
    uncached_positions: torch.Tensor,
    needs_warmup: Union[bool, torch.Tensor],
) -> torch.Tensor:
    batch_size = uncached_positions.shape[0]
    if isinstance(needs_warmup, bool):
        if needs_warmup:
            return torch.ones_like(uncached_positions, dtype=torch.bool)
        processing_mask = uncached_positions
    else:
        processing_mask = torch.where(
            needs_warmup.to(device=uncached_positions.device).view(batch_size, 1),
            torch.ones_like(uncached_positions, dtype=torch.bool),
            uncached_positions,
        )

    row_has_processing = processing_mask.any(dim=1)
    return torch.where(
        row_has_processing.view(batch_size, 1),
        processing_mask,
        torch.ones_like(uncached_positions, dtype=torch.bool),
    )


def _focus_processing_has_evictable_mask_from_state(
    block_tokens: torch.Tensor,
    uncached_positions: torch.Tensor,
    needs_warmup: Union[bool, torch.Tensor],
    mask_id: int,
    avg_tokens: torch.Tensor,
    alpha: float,
) -> bool:
    processing_mask = _focus_processing_mask_from_state(
        uncached_positions, needs_warmup
    )
    mask_lengths = ((block_tokens == mask_id) & processing_mask).sum(dim=1).to(
        device=avg_tokens.device, dtype=torch.int32
    )
    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.int32)
    targets = torch.minimum(mask_lengths, target_values)
    return bool(((targets > 0) & (mask_lengths > targets)).any().item())


def _kept_positions_from_processing_positions(
    processing_positions: List[torch.Tensor],
    device: torch.device,
) -> FocusKeptPositions:
    lengths = torch.tensor(
        [int(positions.numel()) for positions in processing_positions],
        dtype=torch.int32,
        device=device,
    )
    return {
        "positions": torch.cat(
            [
                positions.to(device=device, dtype=torch.long)
                for positions in processing_positions
            ],
            dim=0,
        ),
        "lengths": lengths,
        "rightmost_positions": torch.stack(
            [
                positions.to(device=device, dtype=torch.int32).max()
                for positions in processing_positions
            ],
            dim=0,
        ),
    }


def _build_partial_state(
    forward_batch: ForwardBatch,
    focus_uncached_positions: torch.Tensor,
    focus_needs_warmup: Union[bool, torch.Tensor],
    focus_token_sum: torch.Tensor,
    focus_steps: torch.Tensor,
    start_list: List[int],
    partial_rows: Optional[torch.Tensor] = None,
) -> dict:
    batch_size = forward_batch.batch_size
    block_size = forward_batch.dllm_config.block_size
    device = forward_batch.input_ids.device
    block_tokens = torch.reshape(
        forward_batch.input_ids[: batch_size * block_size],
        (batch_size, block_size),
    )
    if partial_rows is None:
        blocks = [block_tokens[i].detach().clone() for i in range(batch_size)]
    else:
        partial_rows_cpu = partial_rows.detach().cpu().tolist()
        blocks = [
            block_tokens[i].detach().clone() if partial_rows_cpu[i] else None
            for i in range(batch_size)
        ]
    return {
        "blocks": blocks,
        "uncached_positions": focus_uncached_positions.detach().clone(),
        "needs_warmup": (
            focus_needs_warmup.detach().clone()
            if isinstance(focus_needs_warmup, torch.Tensor)
            else torch.full(
                (batch_size,),
                bool(focus_needs_warmup),
                dtype=torch.bool,
                device=device,
            )
        ),
        "start_offsets": torch.tensor(start_list, dtype=torch.int32, device=device),
        "focus_token_sum": focus_token_sum.detach().clone(),
        "focus_steps": focus_steps.detach().clone(),
        "focus_block_progress": (
            focus_init_block_progress(forward_batch).detach().clone()
            if forward_batch.dllm_focus_block_progress is None
            else forward_batch.dllm_focus_block_progress.detach().clone()
        ),
    }


def _compute_start_from_masks(
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
    row: int,
) -> int:
    block_start = row * block_size
    block_end = block_start + block_size
    block_input_ids = input_ids[block_start:block_end]
    block_mask_index = block_input_ids == mask_id
    return block_size - int(torch.sum(block_mask_index).item())


def _compare_delayed_sparse_to_dense(
    model_runner: ModelRunner,
    forward_batch: ForwardBatch,
    dense_logits_output: LogitsProcessorOutput,
    sparse_logits_output: LogitsProcessorOutput,
    processing_positions: List[torch.Tensor],
) -> None:
    batch_size = forward_batch.batch_size
    block_size = forward_batch.dllm_config.block_size
    device = forward_batch.input_ids.device
    lengths = torch.tensor(
        [int(pos.numel()) for pos in processing_positions],
        dtype=torch.long,
        device=device,
    )
    if int(lengths.sum().item()) == 0:
        return

    local_positions = torch.cat(
        [pos.to(device=device, dtype=torch.long) for pos in processing_positions],
        dim=0,
    )
    row_ids = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long, device=device),
        lengths,
        output_size=int(local_positions.numel()),
    )
    dense_indices = row_ids * block_size + local_positions
    dense_logits = dense_logits_output.full_logits.index_select(0, dense_indices)
    sparse_logits = sparse_logits_output.full_logits[: local_positions.numel()]
    diff = (dense_logits.float() - sparse_logits.float()).abs()
    dense_top = torch.argmax(dense_logits, dim=-1)
    sparse_top = torch.argmax(sparse_logits, dim=-1)
    mismatch = dense_top != sparse_top
    logger.info(
        "DLLM delayed dense-compare rows=%d q_tokens=%d q_lens=%s "
        "max_abs=%.6g mean_abs=%.6g top_mismatch=%d/%d "
        "first_positions=%s dense_top=%s sparse_top=%s",
        batch_size,
        int(local_positions.numel()),
        [int(v) for v in lengths.detach().cpu().tolist()],
        float(diff.max().item()),
        float(diff.mean().item()),
        int(mismatch.sum().item()),
        int(local_positions.numel()),
        [int(v) for v in local_positions[:16].detach().cpu().tolist()],
        [int(v) for v in dense_top[:16].detach().cpu().tolist()],
        [int(v) for v in sparse_top[:16].detach().cpu().tolist()],
    )
    if _dense_compare_detail_enabled():
        msg = (
            "DLLM delayed dense-compare "
            f"rows={batch_size} q_tokens={int(local_positions.numel())} "
            f"q_lens={[int(v) for v in lengths.detach().cpu().tolist()]} "
            f"max_abs={float(diff.max().item()):.6g} "
            f"mean_abs={float(diff.mean().item()):.6g} "
            f"top_mismatch={int(mismatch.sum().item())}/{int(local_positions.numel())} "
            f"first_positions={[int(v) for v in local_positions[:16].detach().cpu().tolist()]} "
            f"dense_top={[int(v) for v in dense_top[:16].detach().cpu().tolist()]} "
            f"sparse_top={[int(v) for v in sparse_top[:16].detach().cpu().tolist()]}"
        )
        print(msg, flush=True)
    if _dense_compare_detail_enabled() and bool(mismatch.any().item()):
        row_rightmost = local_positions.new_empty((batch_size,))
        start = 0
        for row, length in enumerate(lengths.detach().cpu().tolist()):
            q_len = int(length)
            row_rightmost[row] = local_positions[start + q_len - 1] if q_len > 0 else -1
            start += q_len
        mismatch_rows = row_ids[mismatch]
        mismatch_pos = local_positions[mismatch]
        mismatch_rightmost = row_rightmost.index_select(0, mismatch_rows)
        at_rightmost = mismatch_pos == mismatch_rightmost
        before_rightmost = mismatch_pos < mismatch_rightmost
        after_rightmost = mismatch_pos > mismatch_rightmost
        mismatch_diff = diff.max(dim=-1).values[mismatch]
        limit = _dense_compare_detail_limit()
        detail_parts = []
        for i in range(min(limit, int(mismatch_rows.numel()))):
            detail_parts.append(
                {
                    "row": int(mismatch_rows[i].item()),
                    "pos": int(mismatch_pos[i].item()),
                    "rightmost": int(mismatch_rightmost[i].item()),
                    "dense_top": int(dense_top[mismatch][i].item()),
                    "sparse_top": int(sparse_top[mismatch][i].item()),
                    "max_abs": float(mismatch_diff[i].item()),
                }
            )
        print(
            "DLLM delayed dense-compare detail "
            f"at_rightmost={int(at_rightmost.sum().item())} "
            f"before_rightmost={int(before_rightmost.sum().item())} "
            f"after_rightmost={int(after_rightmost.sum().item())} "
            f"row_rightmost={[int(v) for v in row_rightmost.detach().cpu().tolist()]} "
            f"mismatches={detail_parts}",
            flush=True,
        )


def _processing_positions_to_rightmost(
    processing_positions: List[torch.Tensor],
    block_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    full_positions = []
    for local_positions in processing_positions:
        rightmost = int(local_positions.max().detach().cpu().item())
        full_positions.append(
            torch.arange(rightmost + 1, dtype=torch.long, device=device)
        )
    return full_positions


def _compare_delayed_sparse_to_truncated_recompute(
    recompute_logits_output: LogitsProcessorOutput,
    sparse_logits_output: LogitsProcessorOutput,
    processing_positions: List[torch.Tensor],
    recompute_positions: List[torch.Tensor],
) -> None:
    device = sparse_logits_output.full_logits.device
    sparse_lengths = torch.tensor(
        [int(pos.numel()) for pos in processing_positions],
        dtype=torch.long,
        device=device,
    )
    recompute_lengths = torch.tensor(
        [int(pos.numel()) for pos in recompute_positions],
        dtype=torch.long,
        device=device,
    )
    if int(sparse_lengths.sum().item()) == 0:
        return

    sparse_offsets = torch.cumsum(sparse_lengths, dim=0) - sparse_lengths
    recompute_offsets = torch.cumsum(recompute_lengths, dim=0) - recompute_lengths
    local_positions = torch.cat(
        [pos.to(device=device, dtype=torch.long) for pos in processing_positions],
        dim=0,
    )
    row_ids = torch.repeat_interleave(
        torch.arange(len(processing_positions), dtype=torch.long, device=device),
        sparse_lengths,
        output_size=int(local_positions.numel()),
    )
    row_sparse_offsets = sparse_offsets.index_select(0, row_ids)
    row_recompute_offsets = recompute_offsets.index_select(0, row_ids)
    sparse_indices = row_sparse_offsets + (
        torch.arange(local_positions.numel(), dtype=torch.long, device=device)
        - row_sparse_offsets
    )
    recompute_indices = row_recompute_offsets + local_positions

    recompute_logits = recompute_logits_output.full_logits.index_select(
        0, recompute_indices
    )
    sparse_logits = sparse_logits_output.full_logits.index_select(0, sparse_indices)
    diff = (recompute_logits.float() - sparse_logits.float()).abs()
    recompute_top = torch.argmax(recompute_logits, dim=-1)
    sparse_top = torch.argmax(sparse_logits, dim=-1)
    mismatch = recompute_top != sparse_top
    msg = (
        "DLLM delayed trunc-compare "
        f"rows={len(processing_positions)} q_tokens={int(local_positions.numel())} "
        f"q_lens={[int(v) for v in sparse_lengths.detach().cpu().tolist()]} "
        f"rightmost={[int(v.numel()) - 1 for v in recompute_positions]} "
        f"max_abs={float(diff.max().item()):.6g} "
        f"mean_abs={float(diff.mean().item()):.6g} "
        f"top_mismatch={int(mismatch.sum().item())}/{int(local_positions.numel())} "
        f"first_positions={[int(v) for v in local_positions[:16].detach().cpu().tolist()]} "
        f"recompute_top={[int(v) for v in recompute_top[:16].detach().cpu().tolist()]} "
        f"sparse_top={[int(v) for v in sparse_top[:16].detach().cpu().tolist()]}"
    )
    logger.info(msg)
    print(msg, flush=True)

    if _dense_compare_detail_enabled() and bool(mismatch.any().item()):
        mismatch_rows = row_ids[mismatch]
        mismatch_pos = local_positions[mismatch]
        mismatch_diff = diff.max(dim=-1).values[mismatch]
        limit = _dense_compare_detail_limit()
        detail_parts = []
        for i in range(min(limit, int(mismatch_rows.numel()))):
            detail_parts.append(
                {
                    "row": int(mismatch_rows[i].item()),
                    "pos": int(mismatch_pos[i].item()),
                    "rightmost": int(recompute_lengths[mismatch_rows[i]].item()) - 1,
                    "recompute_top": int(recompute_top[mismatch][i].item()),
                    "sparse_top": int(sparse_top[mismatch][i].item()),
                    "max_abs": float(mismatch_diff[i].item()),
                }
            )
        print(
            "DLLM delayed trunc-compare detail "
            f"mismatches={detail_parts}",
            flush=True,
        )


class LowConfidence(DllmAlgorithm):
    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self._score_trace_calls = 0

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """Run the original synchronous/FDFO low-confidence step."""
        batch_size = forward_batch.batch_size
        block_token_indices = self._block_token_indices(forward_batch)
        if block_token_indices is None:
            logits = full_logits.view(batch_size, self.block_size, -1)
        else:
            logits = full_logits.index_select(0, block_token_indices).view(
                batch_size, self.block_size, -1
            )
        input_ids = self._block_input_ids(forward_batch, block_token_indices)
        block_mask_index = input_ids == self.mask_id
        done = block_mask_index.sum(dim=1) == 0

        x = torch.argmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(block_mask_index, confidence, -float("inf"))

        transfer_index = confidence > self.threshold
        has_transfer = transfer_index.sum(dim=1) > 0
        top1_indices = torch.argmax(confidence, dim=1)
        batch_indices = torch.arange(batch_size, device=top1_indices.device)
        top1_mask = torch.zeros_like(transfer_index, dtype=torch.bool)
        top1_mask[batch_indices, top1_indices] = True
        transfer_index = torch.where(
            has_transfer.unsqueeze(-1), transfer_index, top1_mask
        )

        x = torch.where(block_mask_index, x, input_ids)
        new_input_ids = torch.where(transfer_index, x, input_ids)
        # Preserve the input_ids tensor identity. Ragged initial FDFO prefill
        # includes prompt tokens before each block, so update only the gathered
        # trailing block positions in that case.
        if block_token_indices is None:
            forward_batch.input_ids.copy_(new_input_ids.view(-1))
        else:
            forward_batch.input_ids.index_copy_(
                0, block_token_indices, new_input_ids.view(-1)
            )

        return done.tolist()

    def _apply_logits(
        self,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        kept_positions: Optional[FocusKeptPositions],
        focus_token_sum: Optional[torch.Tensor],
        focus_steps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = forward_batch.batch_size
        real_num_tokens = batch_size * self.block_size
        block_input_ids = forward_batch.input_ids[:real_num_tokens].view(
            batch_size, self.block_size
        )
        block_mask_index = block_input_ids == self.mask_id
        full_logits = logits_output.full_logits
        device = block_input_ids.device

        if kept_positions is None:
            curr_logits = full_logits[:real_num_tokens].reshape(
                batch_size, self.block_size, -1
            )
            pred = torch.argmax(curr_logits, dim=-1)
            prob = torch.gather(
                F.softmax(curr_logits, dim=-1),
                dim=-1,
                index=pred.unsqueeze(-1),
            ).squeeze(-1)
            confidence = torch.where(
                block_mask_index,
                prob,
                torch.full_like(prob, float("-inf")),
            )
        else:
            pred = block_input_ids.clone()
            confidence = torch.full(
                (batch_size, self.block_size),
                float("-inf"),
                dtype=full_logits.dtype,
                device=device,
            )
            if isinstance(kept_positions, dict):
                kept_lengths = kept_positions["lengths"].to(
                    device=device, dtype=torch.long
                )
                local_positions = kept_positions["positions"].to(
                    device=device, dtype=torch.long
                )
                total_kept = local_positions.numel()
            else:
                kept_lengths_cpu = [len(positions) for positions in kept_positions]
                total_kept = sum(kept_lengths_cpu)
                kept_lengths = torch.tensor(
                    kept_lengths_cpu, dtype=torch.long, device=device
                )
                local_positions = torch.tensor(
                    [pos for positions in kept_positions for pos in positions],
                    dtype=torch.long,
                    device=device,
                )
            if total_kept > 0:
                row_ids = torch.repeat_interleave(
                    torch.arange(batch_size, dtype=torch.long, device=device),
                    kept_lengths,
                    output_size=total_kept,
                )
                curr_logits = full_logits[:total_kept]
                pred_flat = torch.argmax(curr_logits, dim=-1)
                prob_flat = torch.gather(
                    F.softmax(curr_logits, dim=-1),
                    dim=-1,
                    index=pred_flat.unsqueeze(-1),
                ).squeeze(-1)
                kept_mask_index = block_mask_index[row_ids, local_positions]
                row_ids = row_ids[kept_mask_index]
                local_positions = local_positions[kept_mask_index]
                pred[row_ids, local_positions] = pred_flat[kept_mask_index]
                confidence[row_ids, local_positions] = prob_flat[kept_mask_index]

        transfer_index = confidence > self.threshold
        candidate_rows = torch.isfinite(confidence).any(dim=1)
        fallback_rows = (
            block_mask_index.any(dim=1) & candidate_rows & ~transfer_index.any(dim=1)
        )
        fallback_index = torch.argmax(confidence, dim=1)
        transfer_index[fallback_rows, fallback_index[fallback_rows]] = True
        score_rows = _score_trace_rows()
        if score_rows is not None and self._score_trace_calls < _score_trace_limit():
            self._score_trace_calls += 1
            for row in range(batch_size):
                if not _trace_row_enabled(score_rows, row):
                    continue
                row_conf = confidence[row].detach().float().cpu()
                row_pred = pred[row].detach().cpu()
                row_mask = block_mask_index[row].detach().cpu()
                row_transfer = transfer_index[row].detach().cpu()
                items = []
                for pos in range(self.block_size):
                    if bool(row_mask[pos]):
                        items.append(
                            (
                                pos,
                                int(row_pred[pos].item()),
                                float(row_conf[pos].item()),
                                bool(row_transfer[pos]),
                            )
                        )
                msg = (
                    "SGLang DLLM score trace "
                    f"call={self._score_trace_calls} row={row} "
                    f"threshold={float(self.threshold)} "
                    f"logits_dtype={full_logits.dtype} "
                    f"confidence_dtype={confidence.dtype} items={items}"
                )
                logger.info(msg)
                print(msg, flush=True)
        block_input_ids[transfer_index] = pred[transfer_index]

        decoded_counts = transfer_index.sum(dim=1)
        if focus_token_sum is not None and focus_steps is not None:
            has_decoded = decoded_counts > 0
            focus_token_sum += decoded_counts.to(dtype=focus_token_sum.dtype)
            focus_steps += has_decoded.to(dtype=focus_steps.dtype)
        return decoded_counts

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> Union[DllmRunOutput, Tuple[
        Union[LogitsProcessorOutput, torch.Tensor],
        List[torch.Tensor],
        bool,
        Optional[dict],
    ]]:
        if not self.uses_delayed_cache_scheduler:
            return super().run(model_runner, forward_batch, algo_states)

        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        focus_active = focus_enabled(forward_batch)
        delayed_active = delayed_cache_enabled(forward_batch)
        delayed_state_active = (
            delayed_active and forward_batch.dllm_partial_uncached_positions is not None
        )
        focus_token_sum = (
            forward_batch.dllm_focus_token_sum.clone()
            if forward_batch.dllm_focus_token_sum is not None
            else torch.zeros(batch_size, dtype=torch.float32, device=device)
        )
        focus_steps = (
            forward_batch.dllm_focus_steps.clone()
            if forward_batch.dllm_focus_steps is not None
            else torch.zeros(batch_size, dtype=torch.int32, device=device)
        )
        focus_uncached_positions = None
        focus_needs_warmup: Union[bool, torch.Tensor] = True
        debug_summary = focus_active and focus_debug_summary_enabled()
        dense_compare = delayed_active and _dense_compare_enabled()
        trunc_compare = delayed_active and _trunc_compare_enabled()
        debug_iterations = 0
        debug_processing_tokens = 0
        debug_decoded_total = torch.zeros((), dtype=torch.int32, device=device)
        debug_decoded_rows = torch.zeros((), dtype=torch.int32, device=device)
        trace_rows = _trace_rows()
        trace_state = None
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id
        debug_masks_start = int(mask_index.sum().item()) if debug_summary else 0
        start_offsets = forward_batch.dllm_partial_start_offsets

        # Fast path: if there is no mask token, forward and save kv cache.
        # A completed DLLM block still needs a full-block pass so the stable KV
        # cache is materialized before the decoded tokens are emitted.
        if torch.sum(mask_index).item() == 0:
            fast_forward_batch = (
                replace(
                    forward_batch,
                    dllm_focus_disabled=True,
                    dllm_nomask_forward=True,
                )
                if focus_active
                else replace(forward_batch, dllm_nomask_forward=True)
            )
            profile_name = "low_confidence.forward_nomask"
            with focus_profile_range(profile_name):
                fast_forward_snapshot = _snapshot_forward_batch(fast_forward_batch)
                out = model_runner.forward(fast_forward_batch, pp_proxy_tensors=None)
                _restore_forward_batch(fast_forward_batch, fast_forward_snapshot)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            if start_offsets is None:
                next_token_ids = []
            else:
                block_tokens = torch.reshape(
                    forward_batch.input_ids[: batch_size * self.block_size],
                    (batch_size, self.block_size),
                )
                next_token_ids = []
                for i in range(batch_size):
                    start = int(start_offsets[i].item())
                    next_token_ids.append(
                        block_tokens[i, start:]
                        if start >= 0
                        else block_tokens.new_empty((0,), dtype=block_tokens.dtype)
                    )
            focus_log_timing_summary()
            return logits_output, next_token_ids, can_run_cuda_graph, None

        # Calculate start positions for each block
        if start_offsets is not None:
            start_list = [
                (
                    int(start_offsets[i].item())
                    if int(start_offsets[i].item()) >= 0
                    else _compute_start_from_masks(
                        forward_batch.input_ids,
                        batch_size,
                        self.block_size,
                        self.mask_id,
                        i,
                    )
                )
                for i in range(batch_size)
            ]
        else:
            for block_id in range(batch_size):
                start_list.append(
                    _compute_start_from_masks(
                        forward_batch.input_ids,
                        batch_size,
                        self.block_size,
                        self.mask_id,
                        block_id,
                    )
                )

        if (
            delayed_active
            and not delayed_state_active
            and _lmdeploy_prefill_mimic_enabled()
            and any(start > 0 for start in start_list)
        ):
            dense_forward_batch = replace(
                forward_batch,
                dllm_focus_disabled=True,
            )
            with focus_profile_range("low_confidence.forward_lmdeploy_prefill_mimic"):
                dense_forward_snapshot = _snapshot_forward_batch(dense_forward_batch)
                out = model_runner.forward(dense_forward_batch, pp_proxy_tensors=None)
                _restore_forward_batch(dense_forward_batch, dense_forward_snapshot)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            block_tokens_before = forward_batch.input_ids[
                : batch_size * self.block_size
            ].view(batch_size, self.block_size).detach().clone()
            decoded_counts = self._apply_logits(
                forward_batch=forward_batch,
                logits_output=logits_output,
                kept_positions=None,
                focus_token_sum=None,
                focus_steps=None,
            )
            block_tokens = forward_batch.input_ids[
                : batch_size * self.block_size
            ].view(batch_size, self.block_size)
            partial_rows = (block_tokens == self.mask_id).any(dim=1)
            next_token_ids = []
            partial_rows_cpu = partial_rows.detach().cpu().tolist()
            for i in range(batch_size):
                if partial_rows_cpu[i]:
                    next_token_ids.append(
                        forward_batch.input_ids.new_empty(
                            (0,), dtype=forward_batch.input_ids.dtype
                        )
                    )
                else:
                    start = start_list[i]
                    next_token_ids.append(
                        block_tokens[i, start:]
                        if start >= 0
                        else block_tokens.new_empty((0,), dtype=block_tokens.dtype)
                    )

            if trace_rows is not None:
                full_positions = [
                    torch.arange(self.block_size, dtype=torch.long, device=device)
                    for _ in range(batch_size)
                ]
                uncached_before = torch.ones(
                    (batch_size, self.block_size), dtype=torch.bool, device=device
                )
                for i in range(batch_size):
                    if not _trace_row_enabled(trace_rows, i):
                        continue
                    _trace_block_update(
                        row=i,
                        start=start_list[i],
                        processing_positions=full_positions,
                        block_before=block_tokens_before[i],
                        block_after=block_tokens[i].detach().clone(),
                        uncached_before=uncached_before,
                        uncached_after=uncached_before,
                        needs_warmup_before=True,
                        decoded_count=int(decoded_counts[i].detach().cpu().item()),
                        threshold=float(self.threshold),
                        partial=partial_rows_cpu[i],
                        next_len=int(next_token_ids[i].numel()),
                        mask_id=self.mask_id,
                    )

            if bool(partial_rows.any().item()):
                partial_state = _build_partial_state(
                    forward_batch=forward_batch,
                    focus_uncached_positions=torch.ones(
                        (batch_size, self.block_size),
                        dtype=torch.bool,
                        device=device,
                    ),
                    focus_needs_warmup=torch.ones(
                        (batch_size,), dtype=torch.bool, device=device
                    ),
                    focus_token_sum=torch.zeros(
                        batch_size, dtype=torch.float32, device=device
                    ),
                    focus_steps=torch.zeros(
                        batch_size, dtype=torch.int32, device=device
                    ),
                    start_list=start_list,
                    partial_rows=partial_rows,
                )
                return logits_output, next_token_ids, can_run_cuda_graph, partial_state

            return logits_output, next_token_ids, can_run_cuda_graph, None

        if delayed_active:
            if forward_batch.dllm_focus_block_progress is None:
                focus_init_block_progress(forward_batch)
            if delayed_state_active:
                focus_uncached_positions = (
                    forward_batch.dllm_partial_uncached_positions.clone()
                )
                focus_needs_warmup = forward_batch.dllm_partial_needs_warmup.clone()
            else:
                focus_uncached_positions = torch.ones(
                    (batch_size, self.block_size), dtype=torch.bool, device=device
                )
                focus_needs_warmup = True
            block_tokens = forward_batch.input_ids[: batch_size * self.block_size].view(
                batch_size, self.block_size
            )
            focus_uncached_positions |= block_tokens == self.mask_id
            row_had_partial_state = (
                start_offsets.to(device=device) >= 0
                if start_offsets is not None
                else torch.zeros(batch_size, dtype=torch.bool, device=device)
            )

        max_iterations = 1 if delayed_active else self.block_size
        row_started_incomplete = None
        for _ in range(max_iterations):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            iteration_focus_active = focus_active
            if delayed_active:
                block_tokens_before = forward_batch.input_ids[
                    : batch_size * self.block_size
                ].view(batch_size, self.block_size)
                row_started_incomplete = (
                    (block_tokens_before == self.mask_id).any(dim=1)
                    | (
                        row_had_partial_state
                        & focus_uncached_positions.any(dim=1)
                    )
                )
                with focus_profile_range("low_confidence.prepare_processing"):
                    if focus_active:
                        denom = torch.clamp(focus_steps, min=1).to(torch.float32)
                        forward_batch.dllm_focus_avg_tokens = torch.where(
                            focus_steps > 0,
                            focus_token_sum / denom,
                            torch.ones_like(focus_token_sum),
                        )
                    processing_positions = _focus_processing_positions_from_state(
                        focus_uncached_positions, focus_needs_warmup
                    )
                    if _force_full_delayed_processing():
                        full_positions = torch.arange(
                            self.block_size, dtype=torch.long, device=device
                        )
                        processing_positions = [
                            full_positions for _ in range(batch_size)
                        ]
                    if focus_active:
                        iteration_focus_active = (
                            _focus_processing_has_evictable_mask_from_state(
                                block_tokens_before,
                                focus_uncached_positions,
                                focus_needs_warmup,
                                mask_id=self.mask_id,
                                avg_tokens=forward_batch.dllm_focus_avg_tokens,
                                alpha=forward_batch.dllm_config.focus_alpha,
                            )
                        )
                    if trace_rows is not None and focus_active:
                        avg_tokens = forward_batch.dllm_focus_avg_tokens
                        progress = forward_batch.dllm_focus_block_progress
                        for row in range(batch_size):
                            if not _trace_row_enabled(trace_rows, row):
                                continue
                            row_positions = processing_positions[row].to(device=device)
                            row_tokens = block_tokens_before[row].index_select(
                                0, row_positions
                            )
                            row_mask_len = int((row_tokens == self.mask_id).sum().item())
                            row_target = int(
                                min(
                                    row_mask_len,
                                    max(
                                        1,
                                        int(
                                            torch.ceil(
                                                torch.clamp(
                                                    avg_tokens[row].float(), min=1.0
                                                )
                                                * float(
                                                    forward_batch.dllm_config.focus_alpha
                                                )
                                            ).item()
                                        ),
                                    ),
                                )
                                if row_mask_len > 0
                                else 0
                            )
                            msg = (
                                "SGLang FOCUS gate "
                                f"row={row} focus_active={focus_active} "
                                f"iteration_focus_active={iteration_focus_active} "
                                f"q_len={int(row_positions.numel())} "
                                f"mask_len={row_mask_len} target={row_target} "
                                f"avg={float(avg_tokens[row].item()):.4f} "
                                f"progress={int(progress[row].item()) if progress is not None else -1} "
                                f"positions={[int(v) for v in row_positions.detach().cpu().tolist()]}"
                            )
                            logger.info(msg)
                            print(msg, flush=True)
                    active_forward_batch = focus_build_processing_batch(
                        forward_batch,
                        processing_positions,
                        focus_active=iteration_focus_active,
                    )
                    if trace_rows is not None:
                        trace_block_before = block_tokens_before.detach().clone()
                        trace_uncached_before = focus_uncached_positions.detach().clone()
                        trace_needs_warmup_before = (
                            focus_needs_warmup.detach().clone()
                            if isinstance(focus_needs_warmup, torch.Tensor)
                            else bool(focus_needs_warmup)
                        )
                if isinstance(focus_needs_warmup, torch.Tensor):
                    focus_needs_warmup = torch.zeros_like(focus_needs_warmup)
                else:
                    focus_needs_warmup = False
            else:
                active_forward_batch = forward_batch
            if debug_summary:
                debug_iterations += 1
                debug_processing_tokens += int(active_forward_batch.input_ids.numel())

            dense_compare_logits = None
            if dense_compare and delayed_active:
                kv_slots = forward_batch.out_cache_loc[
                    : batch_size * self.block_size
                ].detach()
                kv_pool = model_runner.token_to_kv_pool
                kv_snapshot = kv_pool.get_cpu_copy(kv_slots)
                dense_forward_batch = replace(
                    forward_batch,
                    dllm_focus_disabled=True,
                    dllm_focus_active=False,
                    dllm_delayed_active=False,
                    dllm_processing_positions=None,
                    dllm_full_input_ids=None,
                    forward_metadata_ready=False,
                    forward_metadata_planned_bs=None,
                    forward_metadata_planned_num_tokens=None,
                    forward_metadata_replan_equivalent=False,
                )
                try:
                    with focus_profile_range("low_confidence.forward_dense_debug"):
                        dense_forward_snapshot = _snapshot_forward_batch(
                            dense_forward_batch
                        )
                        dense_out = model_runner.forward(
                            dense_forward_batch, pp_proxy_tensors=None
                        )
                        _restore_forward_batch(
                            dense_forward_batch, dense_forward_snapshot
                        )
                    dense_compare_logits = dense_out.logits_output
                finally:
                    kv_pool.load_cpu_copy(kv_snapshot, kv_slots)

            trunc_compare_logits = None
            trunc_compare_positions = None
            if trunc_compare and delayed_active and not iteration_focus_active:
                kv_slots = forward_batch.out_cache_loc[
                    : batch_size * self.block_size
                ].detach()
                kv_pool = model_runner.token_to_kv_pool
                kv_snapshot = kv_pool.get_cpu_copy(kv_slots)
                trunc_compare_positions = _processing_positions_to_rightmost(
                    processing_positions,
                    self.block_size,
                    device,
                )
                trunc_forward_batch = focus_build_processing_batch(
                    forward_batch,
                    trunc_compare_positions,
                    focus_active=False,
                )
                try:
                    with focus_profile_range("low_confidence.forward_trunc_debug"):
                        trunc_forward_snapshot = _snapshot_forward_batch(
                            trunc_forward_batch
                        )
                        trunc_out = model_runner.forward(
                            trunc_forward_batch, pp_proxy_tensors=None
                        )
                        _restore_forward_batch(
                            trunc_forward_batch, trunc_forward_snapshot
                        )
                    trunc_compare_logits = trunc_out.logits_output
                finally:
                    kv_pool.load_cpu_copy(kv_snapshot, kv_slots)

            with focus_profile_range("low_confidence.forward_processing"):
                active_forward_snapshot = _snapshot_forward_batch(active_forward_batch)
                out = model_runner.forward(active_forward_batch, pp_proxy_tensors=None)
                _restore_forward_batch(active_forward_batch, active_forward_snapshot)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            if dense_compare_logits is not None:
                _compare_delayed_sparse_to_dense(
                    model_runner=model_runner,
                    forward_batch=forward_batch,
                    dense_logits_output=dense_compare_logits,
                    sparse_logits_output=logits_output,
                    processing_positions=processing_positions,
                )
            if trunc_compare_logits is not None:
                _compare_delayed_sparse_to_truncated_recompute(
                    recompute_logits_output=trunc_compare_logits,
                    sparse_logits_output=logits_output,
                    processing_positions=processing_positions,
                    recompute_positions=trunc_compare_positions,
                )
            kept_positions = (
                focus_kept_positions_from_output(
                    logits_output.customized_info, batch_size
                )
                if iteration_focus_active
                else (
                    _kept_positions_from_processing_positions(
                        processing_positions, device
                    )
                    if delayed_active
                    else None
                )
            )
            if (
                delayed_active
                and iteration_focus_active
                and trace_rows is not None
                and kept_positions is not None
            ):
                for i in range(batch_size):
                    if _trace_row_enabled(trace_rows, i):
                        _trace_focus_kept_positions(
                            row=i,
                            kept_positions=kept_positions,
                            avg_tokens=forward_batch.dllm_focus_avg_tokens,
                            block_progress=forward_batch.dllm_focus_block_progress,
                        )
            if delayed_active:
                with focus_profile_range("low_confidence.update_focus_state"):
                    if focus_active:
                        focus_update_block_progress(forward_batch, kept_positions)
                    focus_mark_cached_from_input_ids(
                        focus_uncached_positions,
                        forward_batch.input_ids,
                        batch_size,
                        self.block_size,
                        self.mask_id,
                    )
            if forward_batch.input_ids.shape[0] < batch_size * self.block_size:
                raise RuntimeError(
                    "DLLM processing batch has fewer input tokens than expected: "
                    f"batch_size={batch_size}, block_size={self.block_size}, "
                    f"num_input_ids={forward_batch.input_ids.shape[0]}"
                )
            with focus_profile_range("low_confidence.apply_logits"):
                decoded_counts = self._apply_logits(
                    forward_batch=forward_batch,
                    logits_output=logits_output,
                    kept_positions=kept_positions,
                    focus_token_sum=focus_token_sum if delayed_active else None,
                    focus_steps=focus_steps if delayed_active else None,
                )
            if delayed_active and trace_rows is not None:
                trace_state = (
                    processing_positions,
                    trace_block_before,
                    forward_batch.input_ids[: batch_size * self.block_size]
                    .view(batch_size, self.block_size)
                    .detach()
                    .clone(),
                    trace_uncached_before,
                    focus_uncached_positions.detach().clone(),
                    trace_needs_warmup_before,
                    decoded_counts.detach().clone(),
                )
            if debug_summary:
                debug_decoded_total += decoded_counts.to(torch.int32).sum()
                debug_decoded_rows += (decoded_counts > 0).to(torch.int32).sum()

        if delayed_active:
            if focus_active:
                denom = torch.clamp(focus_steps, min=1).to(torch.float32)
                forward_batch.dllm_focus_avg_tokens = torch.where(
                    focus_steps > 0,
                    focus_token_sum / denom,
                    torch.ones_like(focus_token_sum),
                )
            block_tokens = torch.reshape(
                forward_batch.input_ids[: batch_size * self.block_size],
                (batch_size, self.block_size),
            )

            if debug_summary and focus_active:
                masks_end = int((forward_batch.input_ids == self.mask_id).sum().item())
                decoded_total = int(debug_decoded_total.item())
                decoded_rows = int(debug_decoded_rows.item())
                focus_step_total = int(focus_steps.to(torch.int32).sum().item())
                avg_tokens_mean = (
                    float(forward_batch.dllm_focus_avg_tokens.float().mean().item())
                    if forward_batch.dllm_focus_avg_tokens is not None
                    else 0.0
                )
                print_row = (
                    "FOCUS loop summary "
                    f"batch={batch_size} iterations={debug_iterations} "
                    f"processing_tokens={debug_processing_tokens} "
                    f"masks={debug_masks_start}->{masks_end} "
                    f"decoded_total={decoded_total} decoded_rows={decoded_rows} "
                    f"focus_steps={focus_step_total} "
                    f"avg_tokens_mean={avg_tokens_mean:.3f}"
                )
                logger.info(print_row)
                focus_log_debug_summary()

            row_has_masks = (block_tokens == self.mask_id).any(dim=1)
            row_has_uncached = focus_uncached_positions.any(dim=1)
            row_needs_stable_full = (
                row_started_incomplete
                if row_started_incomplete is not None
                else torch.zeros_like(row_has_masks)
            ) & ~row_has_masks & ~row_has_uncached
            partial_rows = row_has_masks | row_has_uncached | row_needs_stable_full
            next_token_ids = []
            partial_rows_cpu = partial_rows.detach().cpu().tolist()
            for i in range(batch_size):
                if partial_rows_cpu[i]:
                    next_token_ids.append(
                        forward_batch.input_ids.new_empty(
                            (0,), dtype=forward_batch.input_ids.dtype
                        )
                    )
                    continue

                start = start_list[i]
                next_token_ids.append(
                    block_tokens[i, start:]
                    if start >= 0
                    else block_tokens.new_empty((0,), dtype=block_tokens.dtype)
                )

            if trace_state is not None:
                (
                    trace_processing_positions,
                    trace_block_before,
                    trace_block_after,
                    trace_uncached_before,
                    trace_uncached_after,
                    trace_needs_warmup_before,
                    trace_decoded_counts,
                ) = trace_state
                for i in range(batch_size):
                    if not _trace_row_enabled(trace_rows, i):
                        continue
                    _trace_block_update(
                        row=i,
                        start=start_list[i],
                        processing_positions=trace_processing_positions,
                        block_before=trace_block_before[i],
                        block_after=trace_block_after[i],
                        uncached_before=trace_uncached_before,
                        uncached_after=trace_uncached_after,
                        needs_warmup_before=trace_needs_warmup_before,
                        decoded_count=int(trace_decoded_counts[i].detach().cpu().item()),
                        threshold=float(self.threshold),
                        partial=partial_rows_cpu[i],
                        next_len=int(next_token_ids[i].numel()),
                        mask_id=self.mask_id,
                    )

            if bool(partial_rows.any().item()):
                focus_log_timing_summary()
                partial_state = _build_partial_state(
                    forward_batch=forward_batch,
                    focus_uncached_positions=focus_uncached_positions,
                    focus_needs_warmup=focus_needs_warmup,
                    focus_token_sum=focus_token_sum,
                    focus_steps=focus_steps,
                    start_list=start_list,
                    partial_rows=partial_rows,
                )
                return logits_output, next_token_ids, can_run_cuda_graph, partial_state

            focus_log_timing_summary()
            return logits_output, next_token_ids, can_run_cuda_graph, None

        final_forward_batch = (
            replace(
                forward_batch,
                dllm_focus_disabled=True,
                dllm_final_forward=True,
            )
            if focus_active
            else replace(forward_batch, dllm_final_forward=True)
        )
        with focus_profile_range("low_confidence.forward_final"):
            out = model_runner.forward(final_forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        focus_log_timing_summary()
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(
            forward_batch.input_ids[: batch_size * self.block_size],
            (batch_size, self.block_size),
        )
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph, None


Algorithm = LowConfidence
