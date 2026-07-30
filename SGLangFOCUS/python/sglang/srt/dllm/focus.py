from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from sglang.srt.dllm.focus_kernels import (
    focus_importance_ragged_triton,
    focus_select_and_enforce_ragged_triton,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)
_focus_debug_step = 0
_focus_timing_stats = defaultdict(lambda: [0, 0.0, 0.0])
_focus_timing_contexts = 0
_focus_debug_stats = defaultdict(float)
FocusKeptPositions = Union[List[List[int]], Dict[str, torch.Tensor]]
FocusRetainSelection = Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]]


def focus_profile_ranges_enabled() -> bool:
    return os.environ.get("SGLANG_FOCUS_PROFILE_RANGES", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def focus_timing_enabled() -> bool:
    return os.environ.get("SGLANG_FOCUS_TIMING", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def focus_debug_summary_enabled() -> bool:
    return os.environ.get("SGLANG_FOCUS_DEBUG_SUMMARY", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def dllm_disable_delayed_on_extend() -> bool:
    return os.environ.get("SGLANG_DLLM_DISABLE_DELAYED_ON_EXTEND", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def focus_debug_add(name: str, value: float = 1.0) -> None:
    if focus_debug_summary_enabled():
        _focus_debug_stats[name] += float(value)


def focus_log_debug_summary(limit: int = 32) -> None:
    if not focus_debug_summary_enabled() or not _focus_debug_stats:
        return
    rows = sorted(_focus_debug_stats.items(), key=lambda item: item[0])[:limit]
    logger.info(
        "FOCUS debug cumulative %s",
        " ".join(f"{name}={value:.4g}" for name, value in rows),
    )
    active_tokens = _focus_debug_stats.get("eviction_active_tokens", 0.0)
    if active_tokens > 0:
        kept_tokens = _focus_debug_stats.get("eviction_kept_tokens", 0.0)
        evicted_tokens = _focus_debug_stats.get("eviction_evicted_tokens", 0.0)
        calls = _focus_debug_stats.get("eviction_calls", 0.0)
        reduced_calls = _focus_debug_stats.get("eviction_reduced_calls", 0.0)
        logger.info(
            "FOCUS debug eviction_summary calls=%.0f reduced_calls=%.0f "
            "active_tokens=%.0f kept_tokens=%.0f evicted_tokens=%.0f "
            "kept_ratio=%.4f",
            calls,
            reduced_calls,
            active_tokens,
            kept_tokens,
            evicted_tokens,
            kept_tokens / max(active_tokens, 1.0),
        )


def focus_profile_range(name: str):
    return _FocusProfileRange(name)


def focus_detail_profile_range(name: str):
    if os.environ.get("SGLANG_FOCUS_TIMING_DETAIL", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return _FocusProfileRange(name)
    return nullcontext()


class _FocusProfileRange:
    def __init__(self, name: str):
        self.name = name
        self.record_ctx = None
        self.cpu_start = None
        self.cuda_start = None
        self.cuda_end = None

    def __enter__(self):
        if focus_profile_ranges_enabled():
            self.record_ctx = torch.profiler.record_function(f"focus.{self.name}")
            self.record_ctx.__enter__()
        if focus_timing_enabled():
            self.cpu_start = time.perf_counter()
            if torch.cuda.is_available():
                self.cuda_start = torch.cuda.Event(enable_timing=True)
                self.cuda_end = torch.cuda.Event(enable_timing=True)
                self.cuda_start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        global _focus_timing_contexts
        if self.cuda_end is not None:
            self.cuda_end.record()
            self.cuda_end.synchronize()
            cuda_ms = float(self.cuda_start.elapsed_time(self.cuda_end))
        else:
            cuda_ms = 0.0
        if self.cpu_start is not None:
            cpu_ms = (time.perf_counter() - self.cpu_start) * 1000.0
            stat = _focus_timing_stats[self.name]
            stat[0] += 1
            stat[1] += cpu_ms
            stat[2] += cuda_ms
            _focus_timing_contexts += 1
            interval = int(os.environ.get("SGLANG_FOCUS_TIMING_INTERVAL", "0") or 0)
            if interval > 0 and _focus_timing_contexts % interval == 0:
                focus_log_timing_summary()
        if self.record_ctx is not None:
            return self.record_ctx.__exit__(exc_type, exc, tb)
        return False


def focus_log_timing_summary(limit: int = 24) -> None:
    if not focus_timing_enabled() or not _focus_timing_stats:
        return
    limit = int(os.environ.get("SGLANG_FOCUS_TIMING_LIMIT", str(limit)) or limit)
    rows = sorted(
        (
            (name, count, cpu_ms, cuda_ms)
            for name, (count, cpu_ms, cuda_ms) in _focus_timing_stats.items()
        ),
        key=lambda row: row[3] if row[3] > 0 else row[2],
        reverse=True,
    )[:limit]
    logger.info("FOCUS timing summary: name count cpu_ms cpu_avg cuda_ms cuda_avg")
    for name, count, cpu_ms, cuda_ms in rows:
        logger.info(
            "FOCUS timing %-44s %6d %10.2f %8.3f %10.2f %8.3f",
            name,
            count,
            cpu_ms,
            cpu_ms / max(count, 1),
            cuda_ms,
            cuda_ms / max(count, 1),
        )


def focus_enabled(forward_batch: ForwardBatch) -> bool:
    cfg = getattr(forward_batch, "dllm_config", None)
    processing_positions = getattr(forward_batch, "dllm_processing_positions", None)
    allow_extend = (
        forward_batch.forward_mode.is_dllm_extend()
        and processing_positions is None
        and not dllm_disable_delayed_on_extend()
    )
    return bool(
        cfg is not None
        and getattr(cfg, "enable_focus", False)
        and not getattr(forward_batch, "dllm_focus_disabled", False)
        and (getattr(forward_batch, "dllm_focus_active", False) or allow_extend)
    )


def delayed_cache_enabled(forward_batch: ForwardBatch) -> bool:
    cfg = getattr(forward_batch, "dllm_config", None)
    allow_extend = (
        forward_batch.forward_mode.is_dllm_extend()
        and not dllm_disable_delayed_on_extend()
    )
    return bool(
        cfg is not None
        and getattr(cfg, "enable_delayed_cache", False)
        and not getattr(forward_batch, "dllm_focus_disabled", False)
        and (
            allow_extend
            or getattr(forward_batch, "dllm_focus_active", False)
            or getattr(forward_batch, "dllm_delayed_active", False)
        )
    )


def focus_avg_tokens(forward_batch: ForwardBatch) -> torch.Tensor:
    if forward_batch.dllm_focus_avg_tokens is not None:
        return forward_batch.dllm_focus_avg_tokens
    return torch.ones(
        forward_batch.batch_size,
        dtype=torch.float32,
        device=forward_batch.input_ids.device,
    )


def focus_init_block_progress(forward_batch: ForwardBatch) -> torch.Tensor:
    progress = torch.full(
        (forward_batch.batch_size,),
        -1,
        dtype=torch.int32,
        device=forward_batch.input_ids.device,
    )
    forward_batch.dllm_focus_block_progress = progress
    return progress


def focus_block_progress(forward_batch: ForwardBatch) -> torch.Tensor:
    if forward_batch.dllm_focus_block_progress is not None:
        return forward_batch.dllm_focus_block_progress
    return focus_init_block_progress(forward_batch)


def focus_update_block_progress(
    forward_batch: ForwardBatch, kept_positions: Optional[FocusKeptPositions]
) -> None:
    if kept_positions is None:
        return
    progress = focus_block_progress(forward_batch)

    if isinstance(kept_positions, dict):
        rightmost = kept_positions.get("rightmost_positions")
        if rightmost is None:
            return
        progress.copy_(
            torch.maximum(
                progress,
                rightmost.to(device=progress.device, dtype=progress.dtype),
            )
        )
        return

    rightmost_positions = [-1] * len(kept_positions)
    for seq_idx, positions in enumerate(kept_positions):
        if len(positions) == 0:
            continue
        rightmost_positions[seq_idx] = max(int(pos) for pos in positions)
    rightmost = torch.tensor(
        rightmost_positions, dtype=progress.dtype, device=progress.device
    )
    progress.copy_(torch.maximum(progress, rightmost))


def focus_kept_positions_from_output(
    customized_info: Optional[dict], batch_size: int
) -> Optional[FocusKeptPositions]:
    if customized_info is None:
        return None

    values = customized_info.get("focus_kept_positions")
    if values is None:
        return None

    if isinstance(values, dict):
        positions = values.get("positions")
        lengths = values.get("lengths")
        rightmost = values.get("rightmost_positions")
        if positions is None or lengths is None or rightmost is None:
            raise ValueError("Invalid FOCUS tensor metadata.")
        if lengths.numel() != batch_size or rightmost.numel() != batch_size:
            raise ValueError(
                "Invalid FOCUS metadata length: "
                f"expected {batch_size}, got lengths={lengths.numel()} "
                f"rightmost={rightmost.numel()}"
            )
        return values

    if len(values) != batch_size:
        raise ValueError(
            f"Invalid FOCUS metadata length: expected {batch_size}, got {len(values)}"
        )

    ret: List[List[int]] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            ret.append([int(v) for v in value.tolist()])
        else:
            ret.append([int(v) for v in value])
    return ret


def _focus_compute_target(mask_len: int, avg_tokens: torch.Tensor, alpha: float) -> int:
    if mask_len <= 0:
        return 0
    avg = torch.maximum(avg_tokens.float(), avg_tokens.new_tensor(1.0))
    retain = torch.ceil(avg * float(alpha)).to(dtype=torch.long)
    return int(min(mask_len, max(1, int(retain.item()))))


def _focus_select_mask(delta: torch.Tensor, target: int) -> torch.Tensor:
    if delta.numel() == 0 or target <= 0:
        return torch.zeros_like(delta, dtype=torch.bool)

    target = min(target, delta.numel())
    if target >= delta.numel():
        return torch.ones_like(delta, dtype=torch.bool)

    order = torch.argsort(delta, descending=True)
    base_selection = torch.zeros_like(delta, dtype=torch.bool)
    base_selection[order[:target]] = True

    mean = delta.float().mean()
    std = delta.float().std(unbiased=False)
    threshold_selection = delta.float() >= (mean + std)
    use_threshold = threshold_selection.sum() >= target
    return torch.where(use_threshold, threshold_selection, base_selection)


def _focus_enforce_rules(
    positions: torch.Tensor,
    retain_mask: torch.Tensor,
    block_progress: torch.Tensor,
) -> torch.Tensor:
    if positions.numel() == 0:
        return retain_mask

    retain = retain_mask.clone()

    if positions.numel() > 1:
        adjacency = (positions[1:] - positions[:-1]) == 1
        adjust = adjacency & retain[1:] & ~retain[:-1]
        retain[:-1] |= adjust

    retain |= ~retain.any()

    rightmost = positions[retain].max()
    evicted_before = (positions < rightmost) & ~retain
    is_unprocessed = positions > block_progress.to(device=positions.device)
    retain |= evicted_before & is_unprocessed
    return retain


def focus_mask_positions(
    input_ids: torch.Tensor, batch_size: int, block_size: int, mask_id: int
) -> List[torch.Tensor]:
    active_len = batch_size * block_size
    if input_ids.shape[0] < active_len:
        raise RuntimeError(
            "FOCUS mask-position batch has fewer input tokens than expected: "
            f"batch_size={batch_size}, block_size={block_size}, "
            f"num_input_ids={input_ids.shape[0]}"
        )
    block_input_ids = input_ids[:active_len].view(batch_size, block_size)
    row_ids, local_positions = (block_input_ids == mask_id).nonzero(as_tuple=True)
    return _split_local_positions(row_ids, local_positions, batch_size)


def _split_local_positions(
    row_ids: torch.Tensor,
    local_positions: torch.Tensor,
    batch_size: int,
) -> List[torch.Tensor]:
    if row_ids.numel() == 0:
        return [
            local_positions.new_empty((0,), dtype=torch.long) for _ in range(batch_size)
        ]

    counts = torch.bincount(row_ids, minlength=batch_size).to(dtype=torch.long)
    return list(
        torch.split(local_positions.to(dtype=torch.long), counts.cpu().tolist())
    )


def _ragged_mask_positions(
    input_ids: torch.Tensor,
    q_lens: Sequence[int],
    mask_id: int,
) -> List[torch.Tensor]:
    return _ragged_true_local_positions(input_ids == mask_id, q_lens)


def _ragged_true_local_positions(
    flags: torch.Tensor,
    q_lens: Sequence[int],
) -> List[torch.Tensor]:
    batch_size = len(q_lens)
    flat_offsets = flags.nonzero(as_tuple=False).flatten()
    if flat_offsets.numel() == 0:
        return [
            flat_offsets.new_empty((0,), dtype=torch.long) for _ in range(batch_size)
        ]

    q_lens_tensor = torch.tensor(q_lens, dtype=torch.long, device=flags.device)
    seq_ends = torch.cumsum(q_lens_tensor, dim=0)
    row_ids = torch.searchsorted(seq_ends, flat_offsets, right=True)
    seq_starts = seq_ends - q_lens_tensor
    local_positions = flat_offsets - seq_starts.index_select(0, row_ids)
    return _split_local_positions(row_ids, local_positions, batch_size)


def focus_processing_positions(
    uncached_positions: torch.Tensor,
    needs_warmup: bool,
) -> List[torch.Tensor]:
    batch_size, block_size = uncached_positions.shape
    device = uncached_positions.device
    full_positions = torch.arange(block_size, dtype=torch.long, device=device)
    if needs_warmup:
        return [full_positions for _ in range(batch_size)]

    row_ids, local_positions = uncached_positions.nonzero(as_tuple=True)
    split_positions = _split_local_positions(row_ids, local_positions, batch_size)
    return [
        positions if positions.numel() > 0 else full_positions
        for positions in split_positions
    ]


def focus_mark_cached_from_input_ids(
    uncached_positions: torch.Tensor,
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
) -> None:
    active_len = batch_size * block_size
    if input_ids.shape[0] < active_len:
        raise RuntimeError(
            "FOCUS cache-mark batch has fewer input tokens than expected: "
            f"batch_size={batch_size}, block_size={block_size}, "
            f"num_input_ids={input_ids.shape[0]}"
        )
    block_input_ids = input_ids[:active_len].view(batch_size, block_size)
    non_mask = block_input_ids != mask_id
    right_neighbor = torch.roll(non_mask, shifts=-1, dims=1)
    right_neighbor[:, -1] = True
    ready = non_mask & right_neighbor
    uncached_positions &= ~ready


def focus_q_lens_and_mask_positions(
    forward_batch: ForwardBatch,
) -> Tuple[List[int], List[torch.Tensor]]:
    cfg = forward_batch.dllm_config
    if cfg is None:
        raise ValueError("FOCUS mask positions require a DLLM config.")

    processing_positions = forward_batch.dllm_processing_positions
    if processing_positions is None:
        q_lens = [cfg.block_size] * forward_batch.batch_size
        return q_lens, focus_mask_positions(
            forward_batch.input_ids,
            forward_batch.batch_size,
            cfg.block_size,
            cfg.mask_id,
        )

    q_lens = [int(pos.numel()) for pos in processing_positions]
    return q_lens, _ragged_mask_positions(forward_batch.input_ids, q_lens, cfg.mask_id)


def focus_has_evictable_mask(
    masked_positions: Sequence[torch.Tensor],
    avg_tokens: torch.Tensor,
    alpha: float,
) -> bool:
    mask_lens = [int(pos.numel()) for pos in masked_positions]
    if max(mask_lens, default=0) <= 0:
        return False

    mask_lengths = torch.tensor(mask_lens, dtype=torch.int32, device=avg_tokens.device)
    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.int32)
    targets = torch.minimum(mask_lengths, target_values)
    return bool(((targets > 0) & (mask_lengths > targets)).any().item())


def focus_mask_metadata(forward_batch: ForwardBatch) -> Dict[str, torch.Tensor]:
    cfg = forward_batch.dllm_config
    if cfg is None:
        raise ValueError("FOCUS mask metadata requires a DLLM config.")

    device = forward_batch.input_ids.device
    batch_size = forward_batch.batch_size
    processing_positions = forward_batch.dllm_processing_positions
    if processing_positions is None:
        q_lens = torch.full(
            (batch_size,), cfg.block_size, dtype=torch.int32, device=device
        )
        max_q_len_host = cfg.block_size
        active_len = batch_size * cfg.block_size
        proc_indices = torch.arange(
            cfg.block_size, dtype=torch.long, device=device
        ).repeat(batch_size)
    else:
        q_lens_values = [int(pos.numel()) for pos in processing_positions]
        q_lens = torch.tensor(
            q_lens_values,
            dtype=torch.int32,
            device=device,
        )
        max_q_len_host = max(q_lens_values, default=0)
        active_len = sum(q_lens_values)
        proc_indices = torch.cat(
            [pos.to(device=device) for pos in processing_positions]
        ).to(dtype=torch.long)

    flags = forward_batch.input_ids[:active_len] == cfg.mask_id
    mask_indices = flags.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
    if mask_indices.numel() == 0:
        mask_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        mask_indptr = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
        return {
            "q_lens": q_lens,
            "proc_indices": proc_indices,
            "mask_indices": mask_indices,
            "mask_indptr": mask_indptr,
            "mask_lengths": mask_lengths,
            "max_mask_len": torch.zeros((), dtype=torch.int32, device=device),
            "max_mask_len_host": 0,
            "active_len": active_len,
        }

    q_lens_long = q_lens.to(dtype=torch.long)
    seq_ends = torch.cumsum(q_lens_long, dim=0)
    row_ids = torch.searchsorted(seq_ends, mask_indices, right=True)
    mask_lengths = torch.bincount(row_ids, minlength=batch_size).to(
        device=device, dtype=torch.int32
    )
    mask_indptr = torch.empty((batch_size + 1,), dtype=torch.int64, device=device)
    mask_indptr[0] = 0
    mask_indptr[1:] = torch.cumsum(mask_lengths.to(dtype=torch.int64), dim=0)
    max_mask_len = mask_lengths.max()
    return {
        "q_lens": q_lens,
        "proc_indices": proc_indices,
        "mask_indices": mask_indices,
        "mask_indptr": mask_indptr,
        "mask_lengths": mask_lengths,
        "max_mask_len": max_mask_len,
        # Triton only needs a constexpr upper bound for the ragged row width.
        # Use the host-side max query length to avoid synchronizing on
        # mask_lengths.max() in every FOCUS processing step.
        "max_mask_len_host": max_q_len_host,
        "active_len": active_len,
    }


def focus_has_evictable_mask_metadata(
    metadata: Dict[str, torch.Tensor],
    avg_tokens: torch.Tensor,
    alpha: float,
) -> bool:
    mask_lengths = metadata["mask_lengths"].to(device=avg_tokens.device)
    if bool((mask_lengths <= 0).all().item()):
        return False
    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.int32)
    targets = torch.minimum(mask_lengths, target_values)
    return bool(((targets > 0) & (mask_lengths > targets)).any().item())


def focus_processing_has_evictable_mask(
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
    processing_positions: Sequence[torch.Tensor],
    avg_tokens: torch.Tensor,
    alpha: float,
) -> bool:
    block_input_ids = input_ids[: batch_size * block_size].view(batch_size, block_size)
    mask_lengths = torch.stack(
        [
            (
                block_input_ids[seq_idx].index_select(0, positions.to(input_ids.device))
                == mask_id
            ).sum()
            for seq_idx, positions in enumerate(processing_positions)
        ]
    ).to(device=avg_tokens.device, dtype=torch.int32)
    if bool((mask_lengths <= 0).all().item()):
        return False

    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.int32)
    targets = torch.minimum(mask_lengths, target_values)
    return bool(((targets > 0) & (mask_lengths > targets)).any().item())


def focus_all_processing_kept_positions(
    forward_batch: ForwardBatch,
) -> Dict[str, torch.Tensor]:
    cfg = forward_batch.dllm_config
    if cfg is None:
        raise ValueError("FOCUS kept positions require a DLLM config.")

    device = forward_batch.input_ids.device
    processing_positions = forward_batch.dllm_processing_positions
    if processing_positions is None:
        positions = torch.arange(cfg.block_size, dtype=torch.long, device=device)
        return {
            "positions": positions.repeat(forward_batch.batch_size),
            "lengths": torch.full(
                (forward_batch.batch_size,),
                cfg.block_size,
                dtype=torch.int32,
                device=device,
            ),
            "rightmost_positions": torch.full(
                (forward_batch.batch_size,),
                cfg.block_size - 1,
                dtype=torch.int32,
                device=device,
            ),
        }

    lengths = torch.tensor(
        [int(pos.numel()) for pos in processing_positions],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.cat([pos.to(device=device) for pos in processing_positions]).to(
        dtype=torch.long
    )
    rightmost_positions = torch.stack(
        [pos.max().to(device=device, dtype=torch.int32) for pos in processing_positions]
    )
    return {
        "positions": positions,
        "lengths": lengths,
        "rightmost_positions": rightmost_positions,
    }


def focus_importance(
    q: torch.Tensor,
    k: torch.Tensor,
    q_lens: Sequence[int],
    masked_positions: Sequence[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> List[torch.Tensor]:
    if q.is_cuda and k.is_cuda:
        triton_scores = _focus_importance_triton(
            q=q,
            k=k,
            q_lens=q_lens,
            masked_positions=masked_positions,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
        )
        if triton_scores is not None:
            return triton_scores

    if _focus_has_uniform_lens(q, q_lens):
        return _focus_importance_uniform(
            q=q,
            k=k,
            q_lens=q_lens,
            masked_positions=masked_positions,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
        )

    return _focus_importance_loop(
        q=q,
        k=k,
        q_lens=q_lens,
        masked_positions=masked_positions,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
    )


def focus_importance_from_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    metadata: Dict[str, torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    if not (q.is_cuda and k.is_cuda):
        q_lens = metadata["q_lens"].detach().cpu().tolist()
        mask_indices = metadata["mask_indices"]
        mask_indptr = metadata["mask_indptr"]
        masked_positions = []
        q_lens_long = metadata["q_lens"].to(dtype=torch.long)
        seq_ends = torch.cumsum(q_lens_long, dim=0)
        seq_starts = seq_ends - q_lens_long
        for seq_idx in range(len(q_lens)):
            start = int(mask_indptr[seq_idx].item())
            end = int(mask_indptr[seq_idx + 1].item())
            positions = mask_indices[start:end] - seq_starts[seq_idx]
            masked_positions.append(positions)
        scores = focus_importance(
            q=q,
            k=k,
            q_lens=q_lens,
            masked_positions=masked_positions,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
        )
        return torch.cat(scores) if scores else q.new_empty((0,), dtype=q.dtype)

    max_mask_len = int(metadata.get("max_mask_len_host", 0))
    if max_mask_len <= 0 and "max_mask_len" in metadata:
        max_mask_len = int(metadata["max_mask_len"].item())
    if max_mask_len <= 0:
        return q.new_empty((0,), dtype=q.dtype)
    q_view = q.view(-1, num_q_heads, head_dim)
    k_view = k.view(-1, num_kv_heads, head_dim)
    return focus_importance_ragged_triton(
        q=q_view,
        k=k_view,
        mask_indices=metadata["mask_indices"],
        mask_indptr=metadata["mask_indptr"],
        max_mask_len=max_mask_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
    )


def _focus_importance_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    q_lens: Sequence[int],
    masked_positions: Sequence[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> Optional[List[torch.Tensor]]:
    metadata = _focus_ragged_mask_metadata(q_lens, masked_positions, q.device)
    if metadata is None:
        return [
            torch.empty(0, dtype=q.dtype, device=q.device) for _ in range(len(q_lens))
        ]
    mask_indices, mask_indptr, mask_lens, max_mask_len = metadata

    q_view = q.view(-1, num_q_heads, head_dim)
    k_view = k.view(-1, num_kv_heads, head_dim)
    flat_scores = focus_importance_ragged_triton(
        q=q_view,
        k=k_view,
        mask_indices=mask_indices,
        mask_indptr=mask_indptr,
        max_mask_len=max_mask_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
    )
    return list(flat_scores.split(mask_lens))


def _focus_has_uniform_lens(q: torch.Tensor, q_lens: Sequence[int]) -> bool:
    if len(q_lens) == 0:
        return False
    seq_len = int(q_lens[0])
    if seq_len <= 0 or q.shape[0] != len(q_lens) * seq_len:
        return False
    return all(int(length) == seq_len for length in q_lens)


def _focus_importance_uniform(
    q: torch.Tensor,
    k: torch.Tensor,
    q_lens: Sequence[int],
    masked_positions: Sequence[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> List[torch.Tensor]:
    batch_size = len(q_lens)
    block_size = int(q_lens[0])
    mask_lens = [int(pos.numel()) for pos in masked_positions]
    max_mask_len = max(mask_lens, default=0)
    if max_mask_len == 0:
        return [
            torch.empty(0, dtype=q.dtype, device=q.device) for _ in range(batch_size)
        ]

    nonempty_indices = [
        seq_idx for seq_idx, mask_len in enumerate(mask_lens) if mask_len > 0
    ]
    active_batch = len(nonempty_indices)
    device = q.device

    pos_padded = torch.zeros(
        (active_batch, max_mask_len), dtype=torch.long, device=device
    )
    valid_mask = torch.zeros(
        (active_batch, max_mask_len), dtype=torch.bool, device=device
    )
    for active_idx, seq_idx in enumerate(nonempty_indices):
        mask_len = mask_lens[seq_idx]
        pos_padded[active_idx, :mask_len] = masked_positions[seq_idx]
        valid_mask[active_idx, :mask_len] = True

    kv_group_size = max(1, num_q_heads // max(1, num_kv_heads))
    kv_head_index = (
        torch.arange(num_q_heads, device=device, dtype=torch.long) // kv_group_size
    ).clamp_(max=max(0, num_kv_heads - 1))

    active_seq_indices = torch.tensor(nonempty_indices, dtype=torch.long, device=device)
    q_view = q.view(batch_size, block_size, num_q_heads, head_dim).index_select(
        0, active_seq_indices
    )
    k_view = k.view(batch_size, block_size, num_kv_heads, head_dim).index_select(
        0, active_seq_indices
    )
    k_view = k_view.index_select(2, kv_head_index)

    q_gather_index = pos_padded[:, :, None, None].expand(
        active_batch, max_mask_len, num_q_heads, head_dim
    )
    seq_q = q_view.gather(1, q_gather_index)
    seq_k = k_view.gather(1, q_gather_index)

    scores = torch.einsum("bmhd,bnhd->bmhn", seq_q.float(), seq_k.float()) * float(
        scale
    )
    key_mask = valid_mask[:, None, None, :]
    scores = scores.masked_fill(~key_mask, float("-inf"))

    neg_inf = torch.full_like(scores[..., :1], float("-inf"))
    prev_scores = torch.cat((neg_inf, scores[..., :-1]), dim=-1)
    next_scores = torch.cat((scores[..., 1:], neg_inf), dim=-1)
    pooled_scores = torch.maximum(scores, prev_scores)
    pooled_scores = torch.maximum(pooled_scores, next_scores)
    pooled_scores = pooled_scores.masked_fill(~key_mask, float("-inf"))

    weights = torch.softmax(pooled_scores, dim=-1)
    weights = weights * valid_mask[:, :, None, None]
    batched_scores = weights.sum(dim=(1, 2)).to(dtype=q.dtype)

    ret: List[torch.Tensor] = [
        torch.empty(0, dtype=q.dtype, device=device) for _ in range(batch_size)
    ]
    for active_idx, seq_idx in enumerate(nonempty_indices):
        ret[seq_idx] = batched_scores[active_idx, : mask_lens[seq_idx]]
    return ret


def _focus_importance_loop(
    q: torch.Tensor,
    k: torch.Tensor,
    q_lens: Sequence[int],
    masked_positions: Sequence[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> List[torch.Tensor]:
    kv_group_size = max(1, num_q_heads // max(1, num_kv_heads))
    kv_head_index = (
        torch.arange(num_q_heads, device=q.device, dtype=torch.long) // kv_group_size
    ).clamp_(max=max(0, num_kv_heads - 1))

    q = q.view(-1, num_q_heads, head_dim)
    k = k.view(-1, num_kv_heads, head_dim)

    ret: List[torch.Tensor] = []
    start = 0
    for seq_len, seq_mask_pos in zip(q_lens, masked_positions):
        end = start + seq_len
        if seq_mask_pos.numel() == 0:
            ret.append(torch.empty(0, dtype=q.dtype, device=q.device))
            start = end
            continue

        seq_q = q[start:end].index_select(0, seq_mask_pos)
        seq_k = k[start:end].index_select(0, seq_mask_pos)
        seq_k = seq_k.index_select(1, kv_head_index)

        scores = torch.einsum("qhd,khd->qhk", seq_q.float(), seq_k.float()) * float(
            scale
        )
        neg_inf = torch.full_like(scores[..., :1], float("-inf"))
        prev_scores = torch.cat((neg_inf, scores[..., :-1]), dim=-1)
        next_scores = torch.cat((scores[..., 1:], neg_inf), dim=-1)
        pooled_scores = torch.maximum(scores, prev_scores)
        pooled_scores = torch.maximum(pooled_scores, next_scores)
        weights = torch.softmax(pooled_scores, dim=-1)
        ret.append(weights.sum(dim=(0, 1)).to(dtype=q.dtype))
        start = end

    return ret


def focus_select_retain_positions(
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
    layer0_scores: Sequence[torch.Tensor],
    layer1_scores: Sequence[torch.Tensor],
    avg_tokens: torch.Tensor,
    alpha: float,
    block_progress: Optional[torch.Tensor] = None,
    processing_positions: Optional[Sequence[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    masked_positions = _focus_mask_positions_for_view(
        input_ids=input_ids,
        batch_size=batch_size,
        block_size=block_size,
        mask_id=mask_id,
        processing_positions=processing_positions,
    )

    retain_positions: List[torch.Tensor] = []
    device = input_ids.device
    if block_progress is None:
        block_progress = torch.full((batch_size,), -1, dtype=torch.int32, device=device)

    if input_ids.is_cuda:
        triton_retained = _focus_select_retain_positions_triton(
            input_ids=input_ids,
            batch_size=batch_size,
            block_size=block_size,
            mask_id=mask_id,
            masked_positions=masked_positions,
            layer0_scores=layer0_scores,
            layer1_scores=layer1_scores,
            avg_tokens=avg_tokens,
            alpha=alpha,
            block_progress=block_progress,
            processing_positions=processing_positions,
        )
        if triton_retained is not None:
            return triton_retained

    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.long)
    target_values_cpu = target_values.detach().cpu().tolist()
    processing_start_locs = None
    if processing_positions is not None:
        processing_start_locs = [0]
        for positions in processing_positions[:-1]:
            processing_start_locs.append(
                processing_start_locs[-1] + int(positions.numel())
            )

    for seq_idx in range(batch_size):
        if processing_positions is None:
            start = seq_idx * block_size
            end = start + block_size
            block_input_ids = input_ids[start:end]
            original_mask_pos = masked_positions[seq_idx]
            seq_len = block_size
        else:
            seq_positions = processing_positions[seq_idx]
            seq_len = int(seq_positions.numel())
            start = processing_start_locs[seq_idx]
            end = start + seq_len
            block_input_ids = input_ids[start:end]
            original_mask_pos = seq_positions.index_select(0, masked_positions[seq_idx])
        keep_unmasked = (block_input_ids != mask_id).nonzero(as_tuple=False).flatten()
        mask_pos = masked_positions[seq_idx]

        if mask_pos.numel() == 0:
            retain_positions.append(
                torch.arange(seq_len, dtype=torch.long, device=device)
            )
            continue

        delta = layer1_scores[seq_idx].float() - layer0_scores[seq_idx].float()
        mask_len = int(mask_pos.numel())
        target = min(mask_len, max(1, int(target_values_cpu[seq_idx])))
        should_evict = mask_len > target
        if should_evict:
            retain_mask = _focus_select_mask(delta, target)
        else:
            retain_mask = torch.ones_like(mask_pos, dtype=torch.bool)
        retain_mask = _focus_enforce_rules(
            original_mask_pos, retain_mask, block_progress[seq_idx]
        )
        keep_masked = mask_pos[retain_mask]

        retain_flags = torch.zeros(seq_len, dtype=torch.bool, device=device)
        retain_flags[keep_unmasked] = True
        retain_flags[keep_masked] = True
        retain = retain_flags.nonzero(as_tuple=False).flatten()
        retain_positions.append(retain)

    return retain_positions


def focus_select_retain_metadata(
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
    layer0_scores: Sequence[torch.Tensor],
    layer1_scores: Sequence[torch.Tensor],
    avg_tokens: torch.Tensor,
    alpha: float,
    block_progress: Optional[torch.Tensor] = None,
    processing_positions: Optional[Sequence[torch.Tensor]] = None,
) -> FocusRetainSelection:
    if not input_ids.is_cuda:
        return focus_select_retain_positions(
            input_ids=input_ids,
            batch_size=batch_size,
            block_size=block_size,
            mask_id=mask_id,
            layer0_scores=layer0_scores,
            layer1_scores=layer1_scores,
            avg_tokens=avg_tokens,
            alpha=alpha,
            block_progress=block_progress,
            processing_positions=processing_positions,
        )

    if block_progress is None:
        block_progress = torch.full(
            (batch_size,), -1, dtype=torch.int32, device=input_ids.device
        )

    masked_positions = _focus_mask_positions_for_view(
        input_ids=input_ids,
        batch_size=batch_size,
        block_size=block_size,
        mask_id=mask_id,
        processing_positions=processing_positions,
    )
    metadata = _focus_select_retain_metadata_triton(
        input_ids=input_ids,
        batch_size=batch_size,
        block_size=block_size,
        mask_id=mask_id,
        masked_positions=masked_positions,
        layer0_scores=layer0_scores,
        layer1_scores=layer1_scores,
        avg_tokens=avg_tokens,
        alpha=alpha,
        block_progress=block_progress,
        processing_positions=processing_positions,
    )
    if metadata is not None:
        return metadata

    return focus_select_retain_positions(
        input_ids=input_ids,
        batch_size=batch_size,
        block_size=block_size,
        mask_id=mask_id,
        layer0_scores=layer0_scores,
        layer1_scores=layer1_scores,
        avg_tokens=avg_tokens,
        alpha=alpha,
        block_progress=block_progress,
        processing_positions=processing_positions,
    )


def focus_select_retain_metadata_from_importance(
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
    metadata: Dict[str, torch.Tensor],
    layer0_scores: torch.Tensor,
    layer1_scores: torch.Tensor,
    avg_tokens: torch.Tensor,
    alpha: float,
    block_progress: Optional[torch.Tensor] = None,
) -> FocusRetainSelection:
    if not input_ids.is_cuda:
        q_lens = metadata["q_lens"].detach().cpu().tolist()
        mask_indices = metadata["mask_indices"]
        mask_indptr = metadata["mask_indptr"]
        processing_positions = None
        if q_lens != [block_size] * batch_size:
            proc_indices = metadata["proc_indices"]
            processing_positions = [
                proc_indices[sum(q_lens[:seq_idx]) : sum(q_lens[: seq_idx + 1])].to(
                    dtype=torch.long
                )
                for seq_idx in range(batch_size)
            ]
        q_lens_long = metadata["q_lens"].to(dtype=torch.long)
        seq_ends = torch.cumsum(q_lens_long, dim=0)
        seq_starts = seq_ends - q_lens_long
        layer0_split = []
        layer1_split = []
        masked_positions = []
        for seq_idx in range(batch_size):
            start = int(mask_indptr[seq_idx].item())
            end = int(mask_indptr[seq_idx + 1].item())
            masked_positions.append(mask_indices[start:end] - seq_starts[seq_idx])
            layer0_split.append(layer0_scores[start:end])
            layer1_split.append(layer1_scores[start:end])
        return focus_select_retain_positions(
            input_ids=input_ids,
            batch_size=batch_size,
            block_size=block_size,
            mask_id=mask_id,
            layer0_scores=layer0_split,
            layer1_scores=layer1_split,
            avg_tokens=avg_tokens,
            alpha=alpha,
            block_progress=block_progress,
            processing_positions=processing_positions,
        )

    if block_progress is None:
        block_progress = torch.full(
            (batch_size,), -1, dtype=torch.int32, device=input_ids.device
        )

    mask_indices = metadata["mask_indices"]
    if mask_indices.numel() == 0:
        active_len = int(metadata["active_len"])
        return {
            "retain_flags": torch.ones(
                (active_len,), dtype=torch.bool, device=input_ids.device
            ),
            "q_lens": metadata["q_lens"],
            "proc_indices": metadata["proc_indices"],
        }
    if (
        layer0_scores.numel() != mask_indices.numel()
        or layer1_scores.numel() != mask_indices.numel()
    ):
        raise RuntimeError(
            "FOCUS flat importance score length mismatch: "
            f"mask={mask_indices.numel()} layer0={layer0_scores.numel()} "
            f"layer1={layer1_scores.numel()}"
        )

    active_len = int(metadata["active_len"])
    prev_scores = torch.zeros(
        (active_len,), dtype=layer0_scores.dtype, device=input_ids.device
    )
    prev_scores.index_copy_(0, mask_indices, layer0_scores)

    mask_lengths = metadata["mask_lengths"]
    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.int32)
    targets = torch.minimum(mask_lengths, target_values)
    targets = torch.where(mask_lengths > 0, torch.clamp(targets, min=1), targets)
    should_evict = (targets > 0) & (mask_lengths > targets)

    max_mask_len = int(metadata.get("max_mask_len_host", 0))
    if max_mask_len <= 0 and "max_mask_len" in metadata:
        max_mask_len = int(metadata["max_mask_len"].item())

    retain_mask_flat = focus_select_and_enforce_ragged_triton(
        importance=layer1_scores,
        prev_scores=prev_scores,
        mask_indices=mask_indices,
        proc_indices=metadata["proc_indices"],
        mask_indptr=metadata["mask_indptr"],
        targets=targets,
        should_evict=should_evict,
        block_progress=block_progress,
        max_mask_len=max_mask_len,
    )

    retain_flags = input_ids[:active_len] != mask_id
    retain_flags[mask_indices] = retain_mask_flat
    return {
        "retain_flags": retain_flags,
        "q_lens": metadata["q_lens"],
        "proc_indices": metadata["proc_indices"],
    }


def _focus_mask_positions_for_view(
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
    processing_positions: Optional[Sequence[torch.Tensor]],
) -> List[torch.Tensor]:
    if processing_positions is None:
        return focus_mask_positions(input_ids, batch_size, block_size, mask_id)

    q_lens = [int(pos.numel()) for pos in processing_positions]
    return _ragged_mask_positions(input_ids, q_lens, mask_id)


def _focus_ragged_mask_metadata(
    q_lens: Sequence[int],
    masked_positions: Sequence[torch.Tensor],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int], int]]:
    mask_lens = [int(pos.numel()) for pos in masked_positions]
    max_mask_len = max(mask_lens, default=0)
    if max_mask_len == 0:
        return None

    indices: List[torch.Tensor] = []
    indptr_values = [0]
    seq_start = 0
    mask_start = 0
    for q_len, seq_mask_pos in zip(q_lens, masked_positions):
        mask_len = int(seq_mask_pos.numel())
        if mask_len > 0:
            indices.append(seq_mask_pos.to(device=device) + seq_start)
        mask_start += mask_len
        indptr_values.append(mask_start)
        seq_start += int(q_len)

    mask_indices = torch.cat(indices, dim=0)
    mask_indptr = torch.tensor(indptr_values, dtype=torch.int64, device=device)
    return mask_indices, mask_indptr, mask_lens, max_mask_len


def _focus_select_retain_positions_triton(
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
    masked_positions: Sequence[torch.Tensor],
    layer0_scores: Sequence[torch.Tensor],
    layer1_scores: Sequence[torch.Tensor],
    avg_tokens: torch.Tensor,
    alpha: float,
    block_progress: torch.Tensor,
    processing_positions: Optional[Sequence[torch.Tensor]],
) -> Optional[List[torch.Tensor]]:
    if len(layer0_scores) != batch_size or len(layer1_scores) != batch_size:
        return None

    if processing_positions is None:
        q_lens = [block_size] * batch_size
    else:
        q_lens = [int(pos.numel()) for pos in processing_positions]

    metadata = _focus_ragged_mask_metadata(q_lens, masked_positions, input_ids.device)
    if metadata is None:
        return [
            torch.arange(q_len, dtype=torch.long, device=input_ids.device)
            for q_len in q_lens
        ]
    mask_indices, mask_indptr, mask_lens, max_mask_len = metadata

    if any(
        int(layer0_scores[idx].numel()) != mask_lens[idx]
        or int(layer1_scores[idx].numel()) != mask_lens[idx]
        for idx in range(batch_size)
    ):
        return None

    layer1_flat = torch.cat([score for score in layer1_scores if score.numel() > 0])
    layer0_flat = torch.cat([score for score in layer0_scores if score.numel() > 0])
    prev_scores = torch.zeros(
        (input_ids.numel(),), dtype=layer0_flat.dtype, device=input_ids.device
    )
    prev_scores.index_copy_(0, mask_indices, layer0_flat)

    mask_lengths = torch.tensor(mask_lens, dtype=torch.int32, device=input_ids.device)
    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.int32)
    targets = torch.minimum(mask_lengths, target_values)
    targets = torch.where(mask_lengths > 0, torch.clamp(targets, min=1), targets)
    should_evict = (targets > 0) & (mask_lengths > targets)

    if processing_positions is None:
        proc_indices = torch.arange(
            block_size, dtype=torch.long, device=input_ids.device
        ).repeat(batch_size)
    else:
        proc_indices = torch.cat(
            [pos.to(device=input_ids.device) for pos in processing_positions]
        )

    retain_mask_flat = focus_select_and_enforce_ragged_triton(
        importance=layer1_flat,
        prev_scores=prev_scores,
        mask_indices=mask_indices,
        proc_indices=proc_indices,
        mask_indptr=mask_indptr,
        targets=targets,
        should_evict=should_evict,
        block_progress=block_progress,
        max_mask_len=max_mask_len,
    )

    retain_flags = input_ids != mask_id
    retain_flags[mask_indices] = retain_mask_flat

    return _ragged_true_local_positions(retain_flags, q_lens)


def _focus_select_retain_metadata_triton(
    input_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    mask_id: int,
    masked_positions: Sequence[torch.Tensor],
    layer0_scores: Sequence[torch.Tensor],
    layer1_scores: Sequence[torch.Tensor],
    avg_tokens: torch.Tensor,
    alpha: float,
    block_progress: torch.Tensor,
    processing_positions: Optional[Sequence[torch.Tensor]],
) -> Optional[Dict[str, torch.Tensor]]:
    if len(layer0_scores) != batch_size or len(layer1_scores) != batch_size:
        return None

    if processing_positions is None:
        q_lens_values = [block_size] * batch_size
        proc_indices = torch.arange(
            block_size, dtype=torch.long, device=input_ids.device
        ).repeat(batch_size)
    else:
        q_lens_values = [int(pos.numel()) for pos in processing_positions]
        proc_indices = torch.cat(
            [pos.to(device=input_ids.device) for pos in processing_positions]
        ).to(dtype=torch.long)

    metadata = _focus_ragged_mask_metadata(
        q_lens_values, masked_positions, input_ids.device
    )
    active_len = int(sum(q_lens_values))
    q_lens = torch.tensor(q_lens_values, dtype=torch.int32, device=input_ids.device)
    if metadata is None:
        return {
            "retain_flags": torch.ones(
                (active_len,), dtype=torch.bool, device=input_ids.device
            ),
            "q_lens": q_lens,
            "proc_indices": proc_indices,
        }

    mask_indices, mask_indptr, mask_lens, max_mask_len = metadata
    if any(
        int(layer0_scores[idx].numel()) != mask_lens[idx]
        or int(layer1_scores[idx].numel()) != mask_lens[idx]
        for idx in range(batch_size)
    ):
        return None

    layer1_flat = torch.cat([score for score in layer1_scores if score.numel() > 0])
    layer0_flat = torch.cat([score for score in layer0_scores if score.numel() > 0])
    prev_scores = torch.zeros(
        (active_len,), dtype=layer0_flat.dtype, device=input_ids.device
    )
    prev_scores.index_copy_(0, mask_indices, layer0_flat)

    mask_lengths = torch.tensor(mask_lens, dtype=torch.int32, device=input_ids.device)
    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.int32)
    targets = torch.minimum(mask_lengths, target_values)
    targets = torch.where(mask_lengths > 0, torch.clamp(targets, min=1), targets)
    should_evict = (targets > 0) & (mask_lengths > targets)

    retain_mask_flat = focus_select_and_enforce_ragged_triton(
        importance=layer1_flat,
        prev_scores=prev_scores,
        mask_indices=mask_indices,
        proc_indices=proc_indices,
        mask_indptr=mask_indptr,
        targets=targets,
        should_evict=should_evict,
        block_progress=block_progress,
        max_mask_len=max_mask_len,
    )

    retain_flags = input_ids[:active_len] != mask_id
    retain_flags[mask_indices] = retain_mask_flat
    return {
        "retain_flags": retain_flags,
        "q_lens": q_lens,
        "proc_indices": proc_indices,
    }


def _focus_debug_interval() -> int:
    value = os.environ.get("SGLANG_FOCUS_EVICTION_DEBUG_INTERVAL")
    if value is None:
        value = os.environ.get("SGLANG_FOCUS_EVICTION_DEBUG")
    if value is None:
        value = os.environ.get("SGLANG_FOCUS_DEBUG_INTERVAL")
    if value is None:
        value = os.environ.get("SGLANG_FOCUS_DEBUG")
    if value is None:
        return 0

    normalized = value.lower()
    if normalized in ("1", "true", "yes", "on"):
        return 1
    if normalized in ("0", "false", "no", "off"):
        return 0

    try:
        return max(0, int(value))
    except ValueError:
        logger.warning("Invalid SGLANG_FOCUS_DEBUG_INTERVAL=%r; disabling.", value)
        return 0


def _focus_log_suffix_stats(
    forward_batch: ForwardBatch,
    flat_indices: torch.Tensor,
    kept_q_lens: torch.Tensor,
    active_q_lens: Optional[torch.Tensor] = None,
) -> None:
    interval = _focus_debug_interval()
    if not focus_debug_summary_enabled() and interval <= 0:
        return

    if active_q_lens is None:
        active_q_lens = _focus_active_q_lens(forward_batch).to(
            device=kept_q_lens.device, dtype=kept_q_lens.dtype
        )
    else:
        active_q_lens = active_q_lens.to(
            device=kept_q_lens.device, dtype=kept_q_lens.dtype
        )

    active_tokens = int(active_q_lens.sum().item())
    kept_tokens = int(flat_indices.numel())
    evicted_tokens = max(active_tokens - kept_tokens, 0)
    focus_debug_add("eviction_calls")
    focus_debug_add("eviction_active_tokens", active_tokens)
    focus_debug_add("eviction_kept_tokens", kept_tokens)
    focus_debug_add("eviction_evicted_tokens", evicted_tokens)
    if kept_tokens < active_tokens:
        focus_debug_add("eviction_reduced_calls")

    if interval <= 0:
        return

    global _focus_debug_step
    _focus_debug_step += 1
    if _focus_debug_step > 10 and _focus_debug_step % interval != 0:
        return

    cfg = forward_batch.dllm_config
    if cfg is None:
        return

    input_ids = forward_batch.input_ids
    masked_tokens = int((input_ids == cfg.mask_id).sum().item())
    kept_masked_tokens = int(
        (input_ids.index_select(0, flat_indices) == cfg.mask_id).sum().item()
    )

    avg_tokens = forward_batch.dllm_focus_avg_tokens
    avg_tokens_mean = None
    if avg_tokens is not None and avg_tokens.numel() > 0:
        avg_tokens_mean = float(avg_tokens.float().mean().item())

    progress = forward_batch.dllm_focus_block_progress
    progress_min = None
    progress_max = None
    if progress is not None and progress.numel() > 0:
        progress_min = int(progress.min().item())
        progress_max = int(progress.max().item())

    active_q_lens_float = active_q_lens.float()
    kept_q_lens_float = kept_q_lens.float()
    logger.info(
        "FOCUS eviction stats step=%d batch=%d kept=%d/%d evicted=%d "
        "kept_ratio=%.4f masked_kept=%d/%d "
        "active_q_len_mean=%.2f active_q_len_min=%d active_q_len_max=%d "
        "kept_q_len_mean=%.2f kept_q_len_min=%d kept_q_len_max=%d "
        "avg_tokens=%s progress=[%s,%s]",
        _focus_debug_step,
        forward_batch.batch_size,
        kept_tokens,
        active_tokens,
        evicted_tokens,
        kept_tokens / max(active_tokens, 1),
        kept_masked_tokens,
        masked_tokens,
        float(active_q_lens_float.mean().item()),
        int(active_q_lens.min().item()),
        int(active_q_lens.max().item()),
        float(kept_q_lens_float.mean().item()),
        int(kept_q_lens.min().item()),
        int(kept_q_lens.max().item()),
        "none" if avg_tokens_mean is None else f"{avg_tokens_mean:.2f}",
        "none" if progress_min is None else str(progress_min),
        "none" if progress_max is None else str(progress_max),
    )


def _focus_active_q_lens(forward_batch: ForwardBatch) -> torch.Tensor:
    device = forward_batch.input_ids.device
    processing_positions = forward_batch.dllm_processing_positions
    if processing_positions is not None:
        return torch.tensor(
            [int(pos.numel()) for pos in processing_positions],
            dtype=torch.int32,
            device=device,
        )
    if forward_batch.extend_seq_lens is not None:
        return forward_batch.extend_seq_lens.to(device=device, dtype=torch.int32)
    cfg = forward_batch.dllm_config
    if cfg is not None:
        return torch.full(
            (forward_batch.batch_size,),
            cfg.block_size,
            dtype=torch.int32,
            device=device,
        )
    return torch.tensor(
        [int(forward_batch.input_ids.numel())],
        dtype=torch.int32,
        device=device,
    )


def _focus_prefix_lens(forward_batch: ForwardBatch, block_size: int) -> torch.Tensor:
    if forward_batch.extend_prefix_lens is not None:
        return forward_batch.extend_prefix_lens.to(torch.int32)
    return (forward_batch.seq_lens - block_size).to(torch.int32)


def _focus_seq_lens_from_rightmost(
    prefix_lens: torch.Tensor,
    rightmost_positions: Union[Sequence[int], torch.Tensor],
) -> torch.Tensor:
    if isinstance(rightmost_positions, torch.Tensor):
        rightmost = rightmost_positions.to(device=prefix_lens.device, dtype=torch.int32)
    else:
        rightmost = torch.tensor(
            rightmost_positions, dtype=torch.int32, device=prefix_lens.device
        )
    return prefix_lens + rightmost + 1


def _focus_prefix_lens_cpu(forward_batch: ForwardBatch, prefix_lens: torch.Tensor):
    if forward_batch.extend_prefix_lens_cpu is not None:
        return forward_batch.extend_prefix_lens_cpu
    return [int(v) for v in prefix_lens.tolist()]


def _focus_suffix_host_metadata(
    forward_batch: ForwardBatch,
    seq_lens: torch.Tensor,
    *,
    exact_extend_lens_cpu: Optional[List[int]] = None,
    extend_lens: Optional[torch.Tensor] = None,
):
    if seq_lens.is_cuda and not forward_batch.return_logprob:
        return None, int(forward_batch.seq_lens_sum), exact_extend_lens_cpu

    seq_lens_cpu = seq_lens.detach().cpu()
    if exact_extend_lens_cpu is None:
        if extend_lens is None:
            extend_lens_cpu = None
        else:
            extend_lens_cpu = extend_lens.detach().cpu().tolist()
    else:
        extend_lens_cpu = exact_extend_lens_cpu
    return seq_lens_cpu, int(seq_lens_cpu.sum().item()), extend_lens_cpu


def focus_build_processing_batch(
    forward_batch: ForwardBatch,
    processing_positions: Sequence[torch.Tensor],
    focus_active: bool = True,
) -> ForwardBatch:
    cfg = forward_batch.dllm_config
    if cfg is None:
        raise ValueError("DLLM processing batch requires a DLLM config.")

    block_size = cfg.block_size
    device = forward_batch.input_ids.device
    flat_indices: List[torch.Tensor] = []
    q_lens_values: List[int] = []
    rightmost_position_tensors: List[torch.Tensor] = []
    for seq_idx, local_positions in enumerate(processing_positions):
        if local_positions.numel() == 0:
            raise ValueError("DLLM processing positions must be non-empty.")
        flat_indices.append(local_positions + seq_idx * block_size)
        q_lens_values.append(int(local_positions.numel()))
        rightmost_position_tensors.append(
            local_positions.max().to(device=device, dtype=torch.int32)
        )

    flat_indices_tensor = torch.cat(flat_indices, dim=0)
    q_lens = torch.tensor(q_lens_values, dtype=torch.int32, device=device)
    prefix_lens = _focus_prefix_lens(forward_batch, block_size)
    rightmost_positions = torch.stack(rightmost_position_tensors, dim=0)
    seq_lens = _focus_seq_lens_from_rightmost(prefix_lens, rightmost_positions)
    seq_lens_cpu = seq_lens.detach().cpu()
    prefix_lens_cpu = _focus_prefix_lens_cpu(forward_batch, prefix_lens)

    num_token_non_padded = None
    if forward_batch.num_token_non_padded is not None:
        num_token_non_padded = torch.tensor(
            int(flat_indices_tensor.numel()), dtype=torch.int32, device=device
        )

    input_embeds = None
    if forward_batch.input_embeds is not None:
        input_embeds = forward_batch.input_embeds.index_select(0, flat_indices_tensor)

    return replace(
        forward_batch,
        forward_mode=ForwardMode.EXTEND if focus_active else ForwardMode.DLLM_EXTEND,
        input_ids=forward_batch.input_ids.index_select(0, flat_indices_tensor),
        positions=forward_batch.positions.index_select(0, flat_indices_tensor),
        out_cache_loc=forward_batch.out_cache_loc.index_select(0, flat_indices_tensor),
        seq_lens=seq_lens,
        seq_lens_sum=int(seq_lens_cpu.sum().item()),
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=int(flat_indices_tensor.numel()),
        extend_seq_lens=q_lens,
        extend_prefix_lens=prefix_lens,
        extend_start_loc=torch.cumsum(q_lens, dim=0) - q_lens,
        extend_seq_lens_cpu=q_lens_values,
        extend_prefix_lens_cpu=prefix_lens_cpu,
        num_token_non_padded=num_token_non_padded,
        num_token_non_padded_cpu=int(flat_indices_tensor.numel()),
        cross_attention_custom_mask=None,
        input_embeds=input_embeds,
        forward_metadata_ready=False,
        forward_metadata_planned_bs=None,
        forward_metadata_planned_num_tokens=None,
        forward_metadata_replan_equivalent=False,
        dllm_focus_active=focus_active,
        dllm_delayed_active=not focus_active,
        dllm_processing_positions=[
            pos.to(device=device) for pos in processing_positions
        ],
        dllm_full_input_ids=forward_batch.input_ids,
    )


def focus_build_suffix_batch(
    forward_batch: ForwardBatch,
    retain_positions: FocusRetainSelection,
) -> Tuple[ForwardBatch, Dict[str, torch.Tensor], torch.Tensor]:
    cfg = forward_batch.dllm_config
    if cfg is None:
        raise ValueError("FOCUS suffix batch requires a DLLM config.")

    block_size = cfg.block_size
    device = forward_batch.input_ids.device
    if isinstance(retain_positions, dict) and "retain_flags" in retain_positions:
        return _focus_build_suffix_batch_from_metadata(forward_batch, retain_positions)

    processing_positions = forward_batch.dllm_processing_positions
    flat_indices: List[torch.Tensor] = []
    kept_position_tensors: List[torch.Tensor] = []
    rightmost_position_tensors: List[torch.Tensor] = []
    q_lens_values: List[int] = []
    start_locs = forward_batch.extend_start_loc
    for seq_idx, local_positions in enumerate(retain_positions):
        q_lens_values.append(int(local_positions.numel()))
        if processing_positions is None:
            original_positions = local_positions
            flat_indices.append(local_positions + seq_idx * block_size)
        else:
            original_positions = processing_positions[seq_idx].index_select(
                0, local_positions
            )
            start = (
                start_locs[seq_idx]
                if start_locs is not None
                else local_positions.new_tensor(0)
            )
            flat_indices.append(local_positions + start)

        kept_position_tensors.append(original_positions)
        rightmost_position_tensors.append(
            original_positions.max().to(device=device, dtype=torch.int32)
        )

    flat_indices_tensor = torch.cat(flat_indices, dim=0)
    focus_debug_add("suffix_batch_calls")
    focus_debug_add("suffix_batch_tokens", int(flat_indices_tensor.numel()))
    focus_debug_add("suffix_batch_q_len_min_sum", min(q_lens_values, default=0))
    focus_debug_add("suffix_batch_q_len_max_sum", max(q_lens_values, default=0))
    q_lens = torch.tensor(q_lens_values, dtype=torch.int32, device=device)
    kept_positions = {
        "positions": torch.cat(kept_position_tensors, dim=0).to(
            device=device, dtype=torch.long
        ),
        "lengths": q_lens,
        "rightmost_positions": torch.stack(rightmost_position_tensors, dim=0).to(
            device=device, dtype=torch.int32
        ),
    }
    _focus_log_suffix_stats(
        forward_batch,
        flat_indices_tensor,
        kept_q_lens=q_lens,
        active_q_lens=_focus_active_q_lens(forward_batch),
    )
    prefix_lens = _focus_prefix_lens(forward_batch, block_size)
    seq_lens = _focus_seq_lens_from_rightmost(
        prefix_lens, kept_positions["rightmost_positions"]
    )
    seq_lens_cpu, seq_lens_sum, extend_seq_lens_cpu = _focus_suffix_host_metadata(
        forward_batch,
        seq_lens,
        exact_extend_lens_cpu=q_lens_values,
    )
    prefix_lens_cpu = _focus_prefix_lens_cpu(forward_batch, prefix_lens)

    num_token_non_padded = None
    if forward_batch.num_token_non_padded is not None:
        num_token_non_padded = torch.tensor(
            int(flat_indices_tensor.numel()), dtype=torch.int32, device=device
        )

    input_embeds = None
    if forward_batch.input_embeds is not None:
        input_embeds = forward_batch.input_embeds.index_select(0, flat_indices_tensor)

    suffix_batch = replace(
        forward_batch,
        forward_mode=ForwardMode.EXTEND,
        input_ids=forward_batch.input_ids.index_select(0, flat_indices_tensor),
        positions=forward_batch.positions.index_select(0, flat_indices_tensor),
        out_cache_loc=forward_batch.out_cache_loc.index_select(0, flat_indices_tensor),
        seq_lens=seq_lens,
        seq_lens_sum=seq_lens_sum,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=int(flat_indices_tensor.numel()),
        extend_seq_lens=q_lens,
        extend_prefix_lens=prefix_lens,
        extend_start_loc=torch.cumsum(q_lens, dim=0) - q_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens_cpu=prefix_lens_cpu,
        num_token_non_padded=num_token_non_padded,
        num_token_non_padded_cpu=int(flat_indices_tensor.numel()),
        input_embeds=input_embeds,
        forward_metadata_ready=False,
        forward_metadata_planned_bs=None,
        forward_metadata_planned_num_tokens=None,
        forward_metadata_replan_equivalent=False,
    )

    return suffix_batch, kept_positions, flat_indices_tensor


def _focus_build_suffix_batch_from_metadata(
    forward_batch: ForwardBatch,
    retain_metadata: Dict[str, torch.Tensor],
) -> Tuple[ForwardBatch, Dict[str, torch.Tensor], torch.Tensor]:
    cfg = forward_batch.dllm_config
    if cfg is None:
        raise ValueError("FOCUS suffix batch requires a DLLM config.")

    block_size = cfg.block_size
    device = forward_batch.input_ids.device
    retain_flags = retain_metadata["retain_flags"].to(device=device, dtype=torch.bool)
    q_lens = retain_metadata["q_lens"].to(device=device, dtype=torch.int32)
    proc_indices = retain_metadata.get("proc_indices")
    if proc_indices is not None:
        proc_indices = proc_indices.to(device=device, dtype=torch.long)

    flat_indices_tensor = (
        retain_flags.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
    )
    if flat_indices_tensor.numel() == 0:
        raise RuntimeError("FOCUS suffix batch cannot retain zero tokens.")

    seq_ends = torch.cumsum(q_lens.to(dtype=torch.long), dim=0)
    row_ids = torch.searchsorted(seq_ends, flat_indices_tensor, right=True)
    seq_starts = seq_ends - q_lens.to(dtype=torch.long)
    local_positions = flat_indices_tensor - seq_starts.index_select(0, row_ids)

    kept_lengths = torch.bincount(row_ids, minlength=forward_batch.batch_size).to(
        device=device, dtype=torch.int32
    )
    if proc_indices is None:
        original_positions = local_positions
    else:
        original_positions = proc_indices.index_select(0, flat_indices_tensor)

    rightmost_positions = torch.full(
        (forward_batch.batch_size,), -1, dtype=torch.int32, device=device
    )
    rightmost_positions.scatter_reduce_(
        0,
        row_ids.to(dtype=torch.long),
        original_positions.to(dtype=torch.int32),
        reduce="amax",
        include_self=True,
    )
    if not rightmost_positions.is_cuda and torch.any(rightmost_positions < 0):
        raise RuntimeError("FOCUS suffix batch has an empty retained sequence.")

    kept_positions = {
        "positions": original_positions.to(dtype=torch.long),
        "lengths": kept_lengths,
        "rightmost_positions": rightmost_positions,
    }

    focus_debug_add("suffix_batch_calls")
    focus_debug_add("suffix_batch_tokens", int(flat_indices_tensor.numel()))
    kept_lengths_cpu = (
        kept_lengths.detach().cpu().tolist()
        if focus_debug_summary_enabled() or not kept_lengths.is_cuda
        else None
    )
    if kept_lengths_cpu is not None:
        focus_debug_add("suffix_batch_q_len_min_sum", min(kept_lengths_cpu))
        focus_debug_add("suffix_batch_q_len_max_sum", max(kept_lengths_cpu))
    _focus_log_suffix_stats(
        forward_batch,
        flat_indices_tensor,
        kept_q_lens=kept_lengths,
        active_q_lens=q_lens,
    )

    prefix_lens = _focus_prefix_lens(forward_batch, block_size)
    seq_lens = _focus_seq_lens_from_rightmost(prefix_lens, rightmost_positions)
    seq_lens_cpu, seq_lens_sum, extend_seq_lens_cpu = _focus_suffix_host_metadata(
        forward_batch,
        seq_lens,
        exact_extend_lens_cpu=kept_lengths_cpu,
        extend_lens=kept_lengths,
    )
    prefix_lens_cpu = _focus_prefix_lens_cpu(forward_batch, prefix_lens)

    num_token_non_padded = None
    if forward_batch.num_token_non_padded is not None:
        num_token_non_padded = torch.tensor(
            int(flat_indices_tensor.numel()), dtype=torch.int32, device=device
        )

    input_embeds = None
    if forward_batch.input_embeds is not None:
        input_embeds = forward_batch.input_embeds.index_select(0, flat_indices_tensor)

    suffix_batch = replace(
        forward_batch,
        forward_mode=ForwardMode.EXTEND,
        input_ids=forward_batch.input_ids.index_select(0, flat_indices_tensor),
        positions=forward_batch.positions.index_select(0, flat_indices_tensor),
        out_cache_loc=forward_batch.out_cache_loc.index_select(0, flat_indices_tensor),
        seq_lens=seq_lens,
        seq_lens_sum=seq_lens_sum,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=int(flat_indices_tensor.numel()),
        extend_seq_lens=kept_lengths,
        extend_prefix_lens=prefix_lens,
        extend_start_loc=torch.cumsum(kept_lengths, dim=0) - kept_lengths,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens_cpu=prefix_lens_cpu,
        num_token_non_padded=num_token_non_padded,
        num_token_non_padded_cpu=int(flat_indices_tensor.numel()),
        input_embeds=input_embeds,
        forward_metadata_ready=False,
        forward_metadata_planned_bs=None,
        forward_metadata_planned_num_tokens=None,
        forward_metadata_replan_equivalent=False,
    )

    return suffix_batch, kept_positions, flat_indices_tensor
