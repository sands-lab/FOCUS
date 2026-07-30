from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _focus_importance_ragged_kernel(
    q,
    k,
    indices,
    indptr,
    workspace,
    importance,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_ws,
    max_seq,
    head_dim,
    scale,
    rows_per_seq,
    BLOCK_D: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    kv_group_size: tl.constexpr,
):
    pid = tl.program_id(0)

    seq_idx = pid // rows_per_seq
    head_row = pid % rows_per_seq
    head_idx = head_row // max_seq
    query_idx = head_row % max_seq

    seq_start = tl.load(indptr + seq_idx).to(tl.int64)
    seq_end = tl.load(indptr + seq_idx + 1).to(tl.int64)
    seq_len = seq_end - seq_start
    if (seq_len <= 0) | (query_idx >= seq_len):
        return

    kv_head_idx = head_idx // kv_group_size

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    q_token_idx = tl.load(indices + seq_start + query_idx).to(tl.int64)
    q_ptr = q + q_token_idx * stride_qt + head_idx * stride_qh
    q_vec = tl.load(q_ptr + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)

    neg_inf = -float("inf")
    row_workspace_ptr = workspace + pid * stride_ws

    prev_score = neg_inf
    pos = 0
    cond = pos < seq_len
    key_idx = tl.load(indices + seq_start + pos, mask=cond, other=0).to(tl.int64)
    k_ptr = k + key_idx * stride_kt + kv_head_idx * stride_kh
    k_vec = tl.load(k_ptr + offs_d * stride_kd, mask=mask_d & cond, other=0.0).to(
        tl.float32
    )
    score = tl.sum(q_vec * k_vec, axis=0)
    curr_score = tl.where(cond, score * scale, neg_inf)

    pos = 1
    cond = pos < seq_len
    key_idx = tl.load(indices + seq_start + pos, mask=cond, other=0).to(tl.int64)
    k_ptr = k + key_idx * stride_kt + kv_head_idx * stride_kh
    k_vec = tl.load(k_ptr + offs_d * stride_kd, mask=mask_d & cond, other=0.0).to(
        tl.float32
    )
    score = tl.sum(q_vec * k_vec, axis=0)
    next_score = tl.where(cond, score * scale, neg_inf)

    for key_pos in tl.range(0, max_seq):
        key_valid = key_pos < seq_len
        next_valid = (key_pos + 1) < seq_len
        pooled = tl.maximum(prev_score, curr_score)
        pooled = tl.maximum(pooled, tl.where(next_valid, next_score, neg_inf))
        tl.store(row_workspace_ptr + key_pos, tl.where(key_valid, pooled, neg_inf))
        prev_score = tl.where(key_valid, curr_score, prev_score)
        curr_score = tl.where(next_valid, next_score, curr_score)
        future_idx = key_pos + 2
        cond_future = future_idx < seq_len
        key_idx = tl.load(
            indices + seq_start + future_idx, mask=cond_future, other=0
        ).to(tl.int64)
        k_ptr = k + key_idx * stride_kt + kv_head_idx * stride_kh
        k_vec = tl.load(
            k_ptr + offs_d * stride_kd, mask=mask_d & cond_future, other=0.0
        ).to(tl.float32)
        score = tl.sum(q_vec * k_vec, axis=0)
        next_score = tl.where(cond_future, score * scale, neg_inf)

    row_max = neg_inf
    offs_block = tl.arange(0, BLOCK_ROW)
    for start in tl.range(0, max_seq, BLOCK_ROW):
        block_offsets = start + offs_block
        mask = block_offsets < max_seq
        block_vals = tl.load(
            row_workspace_ptr + block_offsets, mask=mask, other=neg_inf
        )
        row_max = tl.maximum(row_max, tl.max(block_vals, axis=0))

    row_sum = tl.zeros([1], dtype=tl.float32)
    for start in tl.range(0, max_seq, BLOCK_ROW):
        block_offsets = start + offs_block
        mask = block_offsets < max_seq
        block_vals = tl.load(
            row_workspace_ptr + block_offsets, mask=mask, other=neg_inf
        )
        row_sum += tl.sum(tl.exp(block_vals - row_max), axis=0)

    row_sum = tl.where(row_sum > 0, row_sum, 1.0)
    inv_row_sum = 1.0 / row_sum
    importance_row_ptr = importance + seq_start
    for start in tl.range(0, max_seq, BLOCK_ROW):
        block_offsets = start + offs_block
        mask = block_offsets < seq_len
        block_vals = tl.load(
            row_workspace_ptr + block_offsets, mask=mask, other=neg_inf
        )
        weights = tl.exp(block_vals - row_max) * inv_row_sum
        tl.atomic_add(importance_row_ptr + block_offsets, weights, mask=mask)


@triton.jit
def _focus_select_enforce_ragged_kernel(
    importance,
    prev_scores,
    mask_indices,
    proc_indices,
    indptr,
    targets,
    should_evict,
    block_progress,
    output,
    stride_prog,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    start = tl.load(indptr + seq).to(tl.int64)
    end = tl.load(indptr + seq + 1).to(tl.int64)
    seq_len = end - start
    offs = tl.arange(0, BLOCK)
    in_bounds = offs < seq_len

    token_indices = tl.load(mask_indices + start + offs, mask=in_bounds, other=0).to(
        tl.int64
    )
    curr_scores = tl.load(importance + start + offs, mask=in_bounds, other=0.0).to(
        tl.float32
    )
    prev = tl.load(prev_scores + token_indices, mask=in_bounds, other=0.0).to(
        tl.float32
    )
    scores_row = curr_scores - prev
    scores_rank = scores_row

    valid_row = in_bounds.to(tl.int1)
    valid_f32 = valid_row.to(tl.float32)
    counts = tl.sum(valid_f32, axis=0).to(tl.float32)

    target = tl.load(targets + seq).to(tl.int32)
    evict = tl.load(should_evict + seq).to(tl.int1)
    target = tl.where(evict, target, 0)
    target = tl.maximum(target, 0)
    max_counts = counts.to(tl.int32)
    target = tl.minimum(target, max_counts)
    positive = target > 0
    target_clamped = tl.where(positive, tl.maximum(target, 1), target)

    selected = tl.zeros([BLOCK], dtype=tl.int1)
    remaining = target_clamped
    filler = float("-inf")
    for _ in range(0, BLOCK):
        available = valid_row & (~selected) & in_bounds
        available_count = tl.sum(available.to(tl.int32), axis=0)
        work = (available_count > 0) & (remaining > 0)
        masked_scores = tl.where(available, scores_rank, filler)
        best_val = tl.max(masked_scores, axis=0)
        select_mask = available & (masked_scores == best_val)
        prefix = tl.cumsum(select_mask.to(tl.int32))
        take = select_mask & (prefix <= remaining) & (prefix > 0)
        take = tl.where(work, take, tl.zeros_like(take))
        selected = selected | take
        remaining -= tl.sum(take.to(tl.int32), axis=0)
        scores_rank = tl.where(take, filler, scores_rank)

    masked_scores = scores_row * valid_f32
    denom = tl.maximum(counts, 1.0)
    mean = tl.sum(masked_scores, axis=0) / denom
    diff = (scores_row - mean) * valid_f32
    variance = tl.sum(diff * diff, axis=0) / denom
    std = tl.sqrt(variance)
    threshold = mean + std
    candidate_mask = (scores_row >= threshold) & valid_row
    candidate_counts = tl.sum(candidate_mask.to(tl.int32), axis=0)
    use_threshold = (target_clamped > 0) & (candidate_counts >= target_clamped)
    selection = tl.where(use_threshold, candidate_mask, selected)
    selection = selection & valid_row & in_bounds

    base_retain = tl.where(evict, selection, valid_row & in_bounds)
    retain_ptr = output + start
    tl.store(retain_ptr + offs, base_retain, mask=in_bounds)

    retain = base_retain
    positions = tl.load(proc_indices + token_indices, mask=in_bounds, other=-1).to(
        tl.int32
    )

    has_next = (offs + 1) < seq_len
    next_token_idx = tl.load(
        mask_indices + start + offs + 1, mask=has_next, other=0
    ).to(tl.int64)
    next_pos = tl.load(proc_indices + next_token_idx, mask=has_next, other=-2).to(
        tl.int32
    )
    next_valid = valid_row & has_next
    next_retain = tl.load(retain_ptr + offs + 1, mask=has_next, other=0).to(tl.int1)

    adjacency = ((next_pos - positions) == 1) & has_next & valid_row & next_valid
    adjust = adjacency & next_retain & (retain == 0)
    retain = retain | adjust

    retain_valid = retain & valid_row
    keep_count = tl.sum(retain_valid.to(tl.int32), axis=0)
    retain = tl.where(keep_count == 0, valid_row & in_bounds, retain)
    retain_valid = retain & valid_row

    rightmost = tl.max(tl.where(retain_valid, positions, -1), axis=0)
    evicted_before = (positions < rightmost) & (retain == 0) & valid_row
    progress = tl.load(block_progress + seq * stride_prog).to(tl.int32)
    retain = retain | (evicted_before & (positions > progress))

    tl.store(retain_ptr + offs, retain, mask=in_bounds)


def focus_importance_ragged_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_indptr: torch.Tensor,
    max_mask_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    if max_mask_len <= 0:
        return q.new_empty((0,), dtype=q.dtype)

    batch_size = mask_indptr.numel() - 1
    rows_per_seq = num_q_heads * max_mask_len
    total_rows = rows_per_seq * batch_size
    importance = torch.zeros(
        (mask_indices.numel(),), dtype=torch.float32, device=q.device
    )
    workspace = torch.empty(
        (total_rows, max_mask_len), dtype=torch.float32, device=q.device
    )
    block_d = triton.next_power_of_2(head_dim)
    block_row = triton.next_power_of_2(max(16, min(max_mask_len, 128)))
    num_warps, num_stages = _pick_focus_importance_meta(block_d, block_row)
    kv_group_size = max(1, num_q_heads // max(1, num_kv_heads))
    _focus_importance_ragged_kernel[(total_rows,)](
        q,
        k,
        mask_indices,
        mask_indptr,
        workspace,
        importance,
        *q.stride(),
        *k.stride(),
        workspace.stride(0),
        max_mask_len,
        head_dim,
        float(scale),
        rows_per_seq,
        BLOCK_D=block_d,
        BLOCK_ROW=block_row,
        kv_group_size=kv_group_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return importance.to(dtype=q.dtype)


def focus_select_and_enforce_ragged_triton(
    importance: torch.Tensor,
    prev_scores: torch.Tensor,
    mask_indices: torch.Tensor,
    proc_indices: torch.Tensor,
    mask_indptr: torch.Tensor,
    targets: torch.Tensor,
    should_evict: torch.Tensor,
    block_progress: torch.Tensor,
    max_mask_len: int,
) -> torch.Tensor:
    output = torch.zeros_like(importance, dtype=torch.bool)
    if max_mask_len <= 0:
        return output

    block = triton.next_power_of_2(max_mask_len)
    num_warps, num_stages = _pick_focus_select_meta(max_mask_len)
    _focus_select_enforce_ragged_kernel[(mask_indptr.numel() - 1,)](
        importance,
        prev_scores,
        mask_indices,
        proc_indices,
        mask_indptr,
        targets,
        should_evict,
        block_progress,
        output,
        block_progress.stride(0),
        BLOCK=block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _pick_focus_select_meta(width: int) -> tuple[int, int]:
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        if width >= 256:
            return 8, 3
        if width >= 128:
            return 4, 3
        if width >= 64:
            return 4, 2
        return 2, 2
    if major >= 8:
        if width >= 256:
            return 4, 2
        if width >= 128:
            return 2, 2
        return 1, 2
    return 1, 1


def _pick_focus_importance_meta(block_d: int, block_row: int) -> tuple[int, int]:
    major, _ = torch.cuda.get_device_capability()
    area = block_d * block_row
    if block_row <= 32 and block_d <= 128:
        if major >= 8:
            return 2, 2
        return 1, 1
    if major >= 9:
        if area >= 16384:
            return 8, 4
        if area >= 8192:
            return 8, 3
        return 4, 3
    if major >= 8:
        if area >= 16384:
            return 4, 3
        if area >= 8192:
            return 4, 2
        return 2, 2
    return 1, 1
