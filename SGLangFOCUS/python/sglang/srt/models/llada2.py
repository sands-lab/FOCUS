# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang LLaDA2MoeModelLM model."""

import logging
import os
from dataclasses import replace
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.dllm.focus import (
    FocusKeptPositions,
    focus_all_processing_kept_positions,
    focus_avg_tokens,
    focus_block_progress,
    focus_build_suffix_batch,
    focus_detail_profile_range,
    focus_enabled,
    focus_importance,
    focus_importance_from_metadata,
    focus_mask_metadata,
    focus_profile_range,
    focus_q_lens_and_mask_positions,
    focus_select_retain_metadata_from_importance,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.runtime_context import (
    get_forward,
    get_parallel,
    get_server_args,
    get_stream,
)
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    is_cuda,
    is_hip,
    is_non_idle_and_non_empty,
    is_npu,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

LoraConfig = None
logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_hip = is_hip()
_LLADA2_ACTIVE_TRACE: Optional[Tuple[int, int]] = None


def _normalize_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    if _is_cuda and not _is_hip and positions.dtype != torch.int32:
        return positions.to(torch.int32)
    return positions


def _create_llada2_fused_set_kv_buffer_arg(
    value: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    positions: torch.Tensor,
):
    fused_arg = create_fused_set_kv_buffer_arg(
        value=value,
        layer=layer,
        forward_batch=forward_batch,
    )
    if _is_cuda and not _is_hip and fused_arg.cache_loc.dtype != positions.dtype:
        fused_arg = replace(
            fused_arg, cache_loc=fused_arg.cache_loc.to(positions.dtype)
        )
    return fused_arg


def _llada2_dense_timing_prefix(forward_batch: ForwardBatch) -> str:
    if getattr(forward_batch, "dllm_final_forward", False):
        return "llada2.final"
    if getattr(forward_batch, "dllm_nomask_forward", False):
        return "llada2.nomask"
    if getattr(forward_batch, "dllm_focus_active", False):
        return "llada2.focus_suffix"
    if forward_batch.forward_mode.is_dllm_extend():
        return "llada2.baseline"
    return "llada2.prefill"


def _llada2_timing_layer_name(layer_id: int) -> str:
    if layer_id == 0:
        return "layer0"
    if layer_id == 1:
        return "layer1"
    return "suffix_layers"


def _llada2_dense_layer_timing_prefix(
    forward_batch: ForwardBatch, layer_id: int
) -> str:
    return f"{_llada2_dense_timing_prefix(forward_batch)}.{_llada2_timing_layer_name(layer_id)}"


def _llada2_focus_trace_rows() -> Optional[set[int]]:
    raw = os.getenv("SGLANG_FOCUS_TRACE_ROWS")
    if raw is None:
        raw = os.getenv("SGLANG_DLLM_TRACE_ROWS")
    if raw is None:
        return None
    rows = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            rows.add(int(item))
    return rows


def _llada2_tensor_ints(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.detach().cpu().tolist()]


def _llada2_trace_focus_retain(
    tag: str,
    forward_batch: ForwardBatch,
    metadata: Dict[str, torch.Tensor],
    retain_positions: Dict[str, torch.Tensor],
    avg_tokens: torch.Tensor,
    alpha: float,
) -> None:
    trace_rows = _llada2_focus_trace_rows()
    if trace_rows is None:
        return

    q_lens = metadata["q_lens"]
    proc_indices = metadata["proc_indices"]
    mask_indices = metadata["mask_indices"]
    mask_indptr = metadata["mask_indptr"]
    mask_lengths = metadata["mask_lengths"]
    retain_flags = retain_positions["retain_flags"].to(dtype=torch.bool)
    target_values = torch.ceil(
        torch.clamp(avg_tokens.float(), min=1.0) * float(alpha)
    ).to(dtype=torch.int32)
    targets = torch.minimum(mask_lengths, target_values)
    targets = torch.where(mask_lengths > 0, torch.clamp(targets, min=1), targets)
    should_evict = (targets > 0) & (mask_lengths > targets)
    q_starts = torch.cumsum(q_lens.to(dtype=torch.long), dim=0) - q_lens.to(
        dtype=torch.long
    )

    for row in sorted(trace_rows):
        if row < 0 or row >= forward_batch.batch_size:
            continue
        row_start = int(q_starts[row].item())
        row_end = row_start + int(q_lens[row].item())
        mask_start = int(mask_indptr[row].item())
        mask_end = int(mask_indptr[row + 1].item())
        row_active = proc_indices[row_start:row_end]
        row_mask_flat = mask_indices[mask_start:mask_end]
        row_mask_positions = proc_indices.index_select(0, row_mask_flat)
        row_keep_flags = retain_flags[row_start:row_end]
        row_kept = row_active[row_keep_flags]
        print(
            "SGLANG FOCUS retain "
            f"{tag} row={row} q_len={row_end - row_start} "
            f"mask_len={mask_end - mask_start} avg={float(avg_tokens[row].item()):.4f} "
            f"target={int(targets[row].item())} "
            f"should_evict={bool(should_evict[row].item())} "
            f"progress={int(focus_block_progress(forward_batch)[row].item())} "
            f"active={_llada2_tensor_ints(row_active)} "
            f"mask={_llada2_tensor_ints(row_mask_positions)} "
            f"kept={_llada2_tensor_ints(row_kept)}",
            flush=True,
        )


def _llada2_trace_focus_kept(
    tag: str,
    forward_batch: ForwardBatch,
    kept_positions: Dict[str, torch.Tensor],
) -> None:
    trace_rows = _llada2_focus_trace_rows()
    if trace_rows is None:
        return

    lengths = kept_positions["lengths"]
    positions = kept_positions["positions"]
    starts = torch.cumsum(lengths.to(dtype=torch.long), dim=0) - lengths.to(
        dtype=torch.long
    )
    rightmost = kept_positions["rightmost_positions"]
    for row in sorted(trace_rows):
        if row < 0 or row >= forward_batch.batch_size:
            continue
        row_start = int(starts[row].item())
        row_end = row_start + int(lengths[row].item())
        row_positions = positions[row_start:row_end]
        print(
            "SGLANG FOCUS kept "
            f"{tag} row={row} kept_len={row_end - row_start} "
            f"rightmost={int(rightmost[row].item())} "
            f"positions={_llada2_tensor_ints(row_positions)}",
            flush=True,
        )


def _llada2_state_trace_target() -> Optional[Tuple[int, int]]:
    row = os.getenv("SGLANG_LLADA2_STATE_TRACE_ROW")
    pos = os.getenv("SGLANG_LLADA2_STATE_TRACE_POS")
    if row is None or pos is None:
        return None
    return int(row), int(pos)


def _llada2_state_trace_calls() -> Optional[set[int]]:
    raw = os.getenv("SGLANG_LLADA2_STATE_TRACE_CALLS")
    if raw is None:
        return None
    calls = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            calls.add(int(item))
    return calls


def _llada2_boundary_trace_layers() -> Optional[set[int]]:
    raw = os.getenv("SGLANG_LLADA2_BOUNDARY_TRACE_LAYERS")
    if raw is None:
        return None
    layers = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            layers.add(int(item))
    return layers


def _llada2_state_trace_index(
    forward_batch: ForwardBatch,
    row: int,
    pos: int,
) -> Optional[int]:
    if row < 0 or row >= forward_batch.batch_size:
        return None

    processing_positions = getattr(forward_batch, "dllm_processing_positions", None)
    if processing_positions is not None:
        if row >= len(processing_positions):
            return None
        row_positions = processing_positions[row].detach()
        matches = torch.nonzero(row_positions == pos, as_tuple=False).flatten()
        if matches.numel() == 0:
            return None
        offset = sum(int(p.numel()) for p in processing_positions[:row])
        return offset + int(matches[0].item())

    cfg = getattr(forward_batch, "dllm_config", None)
    if cfg is None:
        return None
    idx = row * cfg.block_size + pos
    if idx >= int(forward_batch.input_ids.numel()):
        return None
    return idx


def _llada2_trace_state_tensor(
    *,
    call: int,
    stage: str,
    tensor: Optional[torch.Tensor],
    index: Optional[int],
) -> None:
    if tensor is None or index is None:
        return
    flat = tensor.reshape(-1, tensor.shape[-1])
    if index < 0 or index >= int(flat.shape[0]):
        return
    vec = flat[index].detach().float()
    sample = [float(v) for v in vec[:6].cpu().tolist()]
    print(
        "SGLang LLaDA2 state "
        f"call={call} stage={stage} idx={index} dtype={tensor.dtype} "
        f"norm={float(vec.norm().item()):.8g} "
        f"mean={float(vec.mean().item()):.8g} "
        f"maxabs={float(vec.abs().max().item()):.8g} "
        f"first6={sample}",
        flush=True,
    )


def _llada2_trace_state_meta(
    *,
    call: int,
    forward_batch: ForwardBatch,
    row: int,
    pos: int,
    index: int,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    flat_ids = input_ids.reshape(-1)
    flat_positions = positions.reshape(-1)
    if index < 0 or index >= int(flat_ids.numel()):
        return

    row_positions_text = "none"
    processing_positions = getattr(forward_batch, "dllm_processing_positions", None)
    if processing_positions is not None and 0 <= row < len(processing_positions):
        row_positions_text = _llada2_tensor_ints(processing_positions[row].detach())

    print(
        "SGLang LLaDA2 state-meta "
        f"call={call} row={row} pos={pos} idx={index} "
        f"token_id={int(flat_ids[index].item())} "
        f"position_id={int(flat_positions[index].item())} "
        f"forward_mode={forward_batch.forward_mode.name} "
        f"focus_active={bool(getattr(forward_batch, 'dllm_focus_active', False))} "
        f"delayed={getattr(forward_batch, 'dllm_partial_uncached_positions', None) is not None} "
        f"seq_lens={_llada2_tensor_ints(forward_batch.seq_lens.detach())} "
        f"processing_positions={row_positions_text}",
        flush=True,
    )


def _llada2_enable_fused_set_kv_buffer(forward_batch: ForwardBatch) -> bool:
    if not enable_fused_set_kv_buffer(forward_batch):
        return False

    cfg = getattr(forward_batch, "dllm_config", None)
    if cfg is None or not cfg.enable_delayed_cache:
        return True

    if os.getenv("SGLANG_LLADA2_DISABLE_FUSED_SET_KV_DLLM", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False

    return forward_batch.input_ids.numel() == forward_batch.batch_size * cfg.block_size

split_qkv_rmsnorm_rope_pos_cache_half_npu = None
if _is_npu:
    try:
        from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu import (
            split_qkv_rmsnorm_rope_pos_cache_half_npu,
        )
    except (ImportError, OSError):
        pass


class LLaDA2MoeMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        if config.hidden_act != "silu":
            raise ValueError("Unsupported activation. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if (self.tp_size == 1) and hidden_states.shape[0] == 0:
            return hidden_states

        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class LLaDA2MoeGate(nn.Module):
    def __init__(
        self,
        config,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.router_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None)
        if (
            self.router_dtype == torch.float32
            or os.getenv("SGLANG_LLADA2_ROUTER_FP32", "0") == "1"
        ):
            return logits
        return logits.to(hidden_states.dtype)


class LLaDA2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.tp_size = get_parallel().tp_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", None)

        # fused_topk_npu() conducting norm before scale with routed_scaling_factor by default
        # norm_topk_prob=True will renorm the routed_scaling_factor thus need to keep norm_topk_prob=False
        if _is_npu:
            self.norm_topk_prob = False

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # Gate always runs at half / full precision for now.
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype = None
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        # TODO global_server_args.ep_num_redundant_experts is used for eplb, not supported now
        assert get_server_args().ep_num_redundant_experts == 0
        # check group topk
        self.num_expert_group = getattr(config, "n_group", 0)
        self.topk_group = getattr(config, "topk_group", 0)
        if self.num_expert_group > 0 or self.topk_group > 0:
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True
        else:
            self.num_expert_group = self.topk_group = None
            self.use_grouped_topk = False

        self.num_experts = (
            config.num_experts + get_server_args().ep_num_redundant_experts
        )

        self.gate = LLaDA2MoeGate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.correction_bias = (
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), (
                "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"
            )

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            # num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=self.topk_group,
            scoring_func=self.score_function or "softmax",
            correction_bias=self.correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=_is_npu,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )
        # shared expert
        if config.num_shared_experts is not None:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = LLaDA2MoeMLP(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        # dispatcher
        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_parallel().tp_size

            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.num_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=get_deepep_mode(),
                async_finish=True,  # TODO
                return_recv_hook=True,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if not get_moe_a2a_backend().is_deepep():
            return self.forward_normal(hidden_states)
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        shared_output = None
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)
            self._trace_moe_tensor("shared", shared_output)
        return shared_output

    def _trace_moe_enabled(self) -> bool:
        trace_ctx = _LLADA2_ACTIVE_TRACE
        trace_layers = _llada2_boundary_trace_layers()
        return trace_ctx is not None and (
            trace_layers is not None and self.layer_id in trace_layers
        )

    def _trace_moe_tensor(self, stage: str, tensor: Optional[torch.Tensor]) -> None:
        if not self._trace_moe_enabled() or tensor is None:
            return
        call, index = _LLADA2_ACTIVE_TRACE
        _llada2_trace_state_tensor(
            call=call,
            stage=f"layer{self.layer_id}.moe_{stage}",
            tensor=tensor,
            index=index,
        )

    def _trace_moe_topk(self, topk_output) -> None:
        if not self._trace_moe_enabled():
            return
        topk_ids = getattr(topk_output, "topk_ids", None)
        topk_weights = getattr(topk_output, "topk_weights", None)
        if topk_ids is None or topk_weights is None:
            return
        call, index = _LLADA2_ACTIVE_TRACE
        if index < 0 or index >= int(topk_ids.shape[0]):
            return
        ids = _llada2_tensor_ints(topk_ids[index].detach())
        weights = [float(v) for v in topk_weights[index].detach().float().cpu().tolist()]
        print(
            "SGLang LLaDA2 moe-topk "
            f"call={call} layer={self.layer_id} idx={index} "
            f"ids={ids} weights={weights}",
            flush=True,
        )

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        self._trace_moe_topk(topk_output)
        routed_output = self.experts(hidden_states, topk_output)
        self._trace_moe_tensor("routed", routed_output)
        return routed_output

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states.clone())

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(hidden_states)
        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            final_hidden_states = self._forward_router_experts(hidden_states)

        if self.num_shared_experts > 0:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None
        forward_mode = forward_batch.forward_mode
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            router_logits = self.gate(hidden_states)
            if self.num_shared_experts > 0:
                shared_output = self.shared_experts(hidden_states)

            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None:
            final_hidden_states += shared_output
        return final_hidden_states


class LLaDA2MoeAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        self.dp_size = get_parallel().attn_dp_size
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        if self.total_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_kv_heads == 0
        assert self.total_num_heads >= self.total_kv_heads

        self.num_heads = self.total_num_heads // attn_tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.head_dim * self.num_heads

        self.num_kv_heads = max(1, self.total_kv_heads // attn_tp_size)
        self.kv_size = max(1, self.num_kv_heads * self.head_dim)

        self.scale = self.head_dim**-0.5

        self.use_qk_norm = getattr(config, "use_qk_norm", True)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        else:
            self.rotary_dim = self.head_dim
        rope_theta, rope_scaling = get_rope_config(config)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )

        self.alt_stream = alt_stream

    def forward_prepare_native(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        allow_fused_set_kv: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = _normalize_rope_positions(positions)
        qkv, _ = self.query_key_value(hidden_states)
        if _is_npu and split_qkv_rmsnorm_rope_pos_cache_half_npu is not None:
            q, k, v = split_qkv_rmsnorm_rope_pos_cache_half_npu(
                qkv,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.q_size,
                self.kv_size,
                self.head_dim,
                eps=self.query_layernorm.variance_epsilon if self.use_qk_norm else None,
                q_weight=self.query_layernorm.weight if self.use_qk_norm else None,
                k_weight=self.key_layernorm.weight if self.use_qk_norm else None,
                q_bias=(
                    getattr(self.query_layernorm, "bias", None)
                    if self.use_qk_norm
                    else None
                ),
                k_bias=(
                    getattr(self.key_layernorm, "bias", None)
                    if self.use_qk_norm
                    else None
                ),
                rope_dim=self.rotary_dim,
            )
        else:
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            if self.use_qk_norm:
                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=self.query_layernorm,
                    k_norm=self.key_layernorm,
                    head_dim=self.head_dim,
                    alt_stream=self.alt_stream,
                )
        can_fuse_set_kv = (
            allow_fused_set_kv
            and self.head_dim == self.rotary_emb.rotary_dim
            and _llada2_enable_fused_set_kv_buffer(forward_batch)
        )
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                _create_llada2_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                    positions=positions,
                )
                if can_fuse_set_kv
                else None
            ),
        )
        return q, k, v

    def compute_focus_importance(
        self, q: torch.Tensor, k: torch.Tensor, forward_batch: ForwardBatch
    ):
        q_lens, masked_positions = focus_q_lens_and_mask_positions(forward_batch)
        return focus_importance(
            q=q,
            k=k,
            q_lens=q_lens,
            masked_positions=masked_positions,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.scale,
        )

    def compute_focus_importance_from_metadata(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return focus_importance_from_metadata(
            q=q,
            k=k,
            metadata=metadata,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.scale,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        timing_prefix = _llada2_dense_layer_timing_prefix(forward_batch, self.layer_id)
        can_fuse_set_kv = (
            self.head_dim == self.rotary_emb.rotary_dim
            and _llada2_enable_fused_set_kv_buffer(forward_batch)
        )
        with focus_detail_profile_range(f"{timing_prefix}.qkv"):
            q, k, v = self.forward_prepare_native(
                positions,
                hidden_states,
                forward_batch,
            )
        with focus_detail_profile_range(f"{timing_prefix}.attn"):
            context_layer = self.attn(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=not can_fuse_set_kv,
            )
        with focus_detail_profile_range(f"{timing_prefix}.attn_dense"):
            attn_output, _ = self.dense(context_layer)
        return attn_output


class LLaDA2MoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.dp_size = get_parallel().attn_dp_size
        self.attention = LLaDA2MoeAttention(
            config,
            layer_id,
            quant_config,
            reduce_results=False,
            prefix=add_prefix("attention", prefix),
            alt_stream=alt_stream,
        )
        self.layer_id = layer_id
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank

        self.is_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id)
        is_previous_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id - 1)
        is_next_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id + 1)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        if self.is_layer_sparse:
            self.mlp = LLaDA2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = LLaDA2MoeMLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def _is_layer_sparse(self, config: PretrainedConfig, layer_id: int) -> bool:
        return (
            config.num_experts is not None and layer_id >= config.first_k_dense_replace
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        trace_ctx = _LLADA2_ACTIVE_TRACE
        trace_layers = _llada2_boundary_trace_layers()
        trace_boundary = trace_ctx is not None and (
            trace_layers is not None and self.layer_id in trace_layers
        )

        def emit_boundary(stage: str, tensor: torch.Tensor) -> None:
            if trace_boundary:
                _llada2_trace_state_tensor(
                    call=trace_ctx[0],
                    stage=f"layer{self.layer_id}.{stage}",
                    tensor=tensor,
                    index=trace_ctx[1],
                )

        timing_prefix = _llada2_dense_layer_timing_prefix(forward_batch, self.layer_id)
        with focus_detail_profile_range(f"{timing_prefix}.prepare_attn"):
            hidden_states, residual = self.layer_communicator.prepare_attn(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        emit_boundary("prepare_attn", hidden_states)

        with focus_detail_profile_range(f"{timing_prefix}.attention"):
            hidden_states = self.attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        emit_boundary("attention", hidden_states)

        with focus_detail_profile_range(f"{timing_prefix}.prepare_mlp"):
            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        emit_boundary("prepare_mlp", hidden_states)

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        with focus_detail_profile_range(f"{timing_prefix}.mlp"):
            with get_forward().scoped(mlp_reduce_scatter=mlp_reduce_scatter):
                hidden_states = self.mlp(hidden_states, forward_batch)
        emit_boundary("mlp", hidden_states)

        with focus_detail_profile_range(f"{timing_prefix}.postprocess"):
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        emit_boundary("postprocess", hidden_states)

        return hidden_states, residual

    def forward_focus_layer0(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]
    ]:
        timing_prefix = "llada2.focus.layer0"
        with focus_detail_profile_range(f"{timing_prefix}.prepare_attn"):
            hidden_states, residual = self.layer_communicator.prepare_attn(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        with focus_detail_profile_range(f"{timing_prefix}.qkv"):
            q, k, v = self.attention.forward_prepare_native(
                positions,
                hidden_states,
                forward_batch,
            )
        mask_metadata = focus_mask_metadata(forward_batch)
        with focus_detail_profile_range(f"{timing_prefix}.importance"):
            layer0_scores = self.attention.compute_focus_importance_from_metadata(
                q, k, mask_metadata
            )
        with focus_detail_profile_range(f"{timing_prefix}.attn"):
            hidden_states = self.attention.attn(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=not _llada2_enable_fused_set_kv_buffer(forward_batch),
            )
        with focus_detail_profile_range(f"{timing_prefix}.attn_dense"):
            hidden_states, _ = self.attention.dense(hidden_states)

        with focus_detail_profile_range(f"{timing_prefix}.prepare_mlp"):
            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        with focus_detail_profile_range(f"{timing_prefix}.mlp"):
            with get_forward().scoped(mlp_reduce_scatter=use_reduce_scatter):
                hidden_states = self.mlp(hidden_states, forward_batch)
        with focus_detail_profile_range(f"{timing_prefix}.postprocess"):
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        return hidden_states, residual, layer0_scores, mask_metadata

    def forward_focus_layer1(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        layer0_scores: Optional[torch.Tensor],
        mask_metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, ForwardBatch, FocusKeptPositions]:
        cfg = forward_batch.dllm_config
        assert cfg is not None

        timing_prefix = "llada2.focus.layer1"
        with focus_detail_profile_range(f"{timing_prefix}.prepare_attn"):
            hidden_states, residual = self.layer_communicator.prepare_attn(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        can_fuse_set_kv = (
            self.attention.head_dim == self.attention.rotary_emb.rotary_dim
            and _llada2_enable_fused_set_kv_buffer(forward_batch)
        )
        with focus_profile_range("llada2.layer1.qkv_importance"):
            with focus_detail_profile_range(f"{timing_prefix}.qkv"):
                q, k, v = self.attention.forward_prepare_native(
                    positions,
                    hidden_states,
                    forward_batch,
                    allow_fused_set_kv=True,
                )
            if layer0_scores is not None:
                with focus_detail_profile_range(f"{timing_prefix}.importance"):
                    layer1_scores = self.attention.compute_focus_importance_from_metadata(
                        q, k, mask_metadata
                    )
            else:
                layer1_scores = None
        if not can_fuse_set_kv:
            with focus_profile_range("llada2.layer1.fill_kv"):
                get_token_to_kv_pool().set_kv_buffer(
                    self.attention.attn,
                    forward_batch.out_cache_loc,
                    k.view(-1, self.attention.num_kv_heads, self.attention.head_dim),
                    v.view(-1, self.attention.num_kv_heads, self.attention.head_dim),
                    self.attention.attn.k_scale,
                    self.attention.attn.v_scale,
                )
        if layer0_scores is None or layer1_scores is None:
            kept_positions = focus_all_processing_kept_positions(forward_batch)
            with focus_profile_range("llada2.layer1.full_attn_dense"):
                with focus_detail_profile_range(f"{timing_prefix}.full_attn"):
                    hidden_states = self.attention.attn(
                        q,
                        k,
                        v,
                        forward_batch,
                        save_kv_cache=False,
                    )
                with focus_detail_profile_range(f"{timing_prefix}.full_attn_dense"):
                    hidden_states, _ = self.attention.dense(hidden_states)

            with focus_profile_range("llada2.layer1.full_mlp"):
                with focus_detail_profile_range(f"{timing_prefix}.full_prepare_mlp"):
                    hidden_states, residual = self.layer_communicator.prepare_mlp(
                        hidden_states=hidden_states,
                        residual=residual,
                        forward_batch=forward_batch,
                    )
                use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
                    forward_batch
                )
                with focus_detail_profile_range(f"{timing_prefix}.full_mlp"):
                    with get_forward().scoped(
                        mlp_reduce_scatter=use_reduce_scatter
                    ):
                        hidden_states = self.mlp(hidden_states, forward_batch)
                with focus_detail_profile_range(f"{timing_prefix}.full_postprocess"):
                    hidden_states, residual = self.layer_communicator.postprocess_layer(
                        hidden_states=hidden_states,
                        residual=residual,
                        forward_batch=forward_batch,
                    )
            return hidden_states, residual, forward_batch, kept_positions

        with focus_profile_range("llada2.layer1.select_retain"):
            retain_positions = focus_select_retain_metadata_from_importance(
                input_ids=forward_batch.input_ids,
                batch_size=forward_batch.batch_size,
                block_size=cfg.block_size,
                mask_id=cfg.mask_id,
                metadata=mask_metadata,
                layer0_scores=layer0_scores,
                layer1_scores=layer1_scores,
                avg_tokens=focus_avg_tokens(forward_batch),
                alpha=cfg.focus_alpha,
                block_progress=focus_block_progress(forward_batch),
            )
        _llada2_trace_focus_retain(
            "layer1",
            forward_batch,
            mask_metadata,
            retain_positions,
            focus_avg_tokens(forward_batch),
            cfg.focus_alpha,
        )
        with focus_profile_range("llada2.layer1.build_suffix_batch"):
            suffix_batch, kept_positions, flat_indices = focus_build_suffix_batch(
                forward_batch, retain_positions
            )
        _llada2_trace_focus_kept("layer1", forward_batch, kept_positions)
        with focus_profile_range("llada2.layer1.init_suffix_metadata"):
            get_attn_backend().init_forward_metadata(suffix_batch)

        with focus_profile_range("llada2.layer1.gather_suffix"):
            q = q.index_select(0, flat_indices)
            k = k.index_select(0, flat_indices)
            v = v.index_select(0, flat_indices)
            if residual is not None:
                residual = residual.index_select(0, flat_indices)

        with focus_profile_range("llada2.layer1.suffix_attn_dense"):
            with focus_detail_profile_range(f"{timing_prefix}.suffix_attn"):
                hidden_states = self.attention.attn(
                    q,
                    k,
                    v,
                    suffix_batch,
                    save_kv_cache=False,
                )
            with focus_detail_profile_range(f"{timing_prefix}.suffix_attn_dense"):
                hidden_states, _ = self.attention.dense(hidden_states)

        with focus_profile_range("llada2.layer1.suffix_mlp"):
            with focus_detail_profile_range(f"{timing_prefix}.suffix_prepare_mlp"):
                hidden_states, residual = self.layer_communicator.prepare_mlp(
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=suffix_batch,
                )
            use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
                suffix_batch
            )
            with focus_detail_profile_range(f"{timing_prefix}.suffix_mlp"):
                with get_forward().scoped(mlp_reduce_scatter=use_reduce_scatter):
                    hidden_states = self.mlp(hidden_states, suffix_batch)
            with focus_detail_profile_range(f"{timing_prefix}.suffix_postprocess"):
                hidden_states, residual = self.layer_communicator.postprocess_layer(
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=suffix_batch,
                )
        return hidden_states, residual, suffix_batch, kept_positions


class LLaDA2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        if self.pp_group.is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=add_prefix("word_embeddings", prefix),
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout)

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LLaDA2MoeBlock(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self._state_trace_call = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        global _LLADA2_ACTIVE_TRACE
        timing_prefix = _llada2_dense_timing_prefix(forward_batch)
        trace_target = _llada2_state_trace_target()
        trace_idx = None
        trace_call = None
        trace_active = False
        if trace_target is not None:
            trace_idx = _llada2_state_trace_index(forward_batch, *trace_target)
            if trace_idx is not None:
                self._state_trace_call += 1
                trace_call = self._state_trace_call
                trace_calls = _llada2_state_trace_calls()
                trace_active = trace_calls is None or trace_call in trace_calls
        with focus_profile_range(f"{timing_prefix}.embed"):
            if self.pp_group.is_first_rank:
                if input_embeds is None:
                    hidden_states = self.word_embeddings(input_ids)
                else:
                    hidden_states = input_embeds
                residual = None
            else:
                assert pp_proxy_tensors is not None
                hidden_states = pp_proxy_tensors["hidden_states"]
                residual = pp_proxy_tensors["residual"]
        if trace_active:
            _llada2_trace_state_meta(
                call=trace_call,
                forward_batch=forward_batch,
                row=trace_target[0],
                pos=trace_target[1],
                index=trace_idx,
                input_ids=input_ids,
                positions=positions,
            )
            _llada2_trace_state_tensor(
                call=trace_call, stage="embed", tensor=hidden_states, index=trace_idx
            )

        _LLADA2_ACTIVE_TRACE = (trace_call, trace_idx) if trace_active else None
        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                with focus_profile_range(
                    f"{timing_prefix}.{_llada2_timing_layer_name(i)}"
                ):
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                    )
                if trace_active:
                    _llada2_trace_state_tensor(
                        call=trace_call,
                        stage=f"layer{i}",
                        tensor=hidden_states,
                        index=trace_idx,
                    )
        _LLADA2_ACTIVE_TRACE = None
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                with focus_profile_range(f"{timing_prefix}.norm"):
                    if residual is None:
                        hidden_states = self.norm(hidden_states)
                    else:
                        hidden_states, _ = self.norm(hidden_states, residual)
                if trace_active:
                    _llada2_trace_state_tensor(
                        call=trace_call,
                        stage="norm",
                        tensor=hidden_states,
                        index=trace_idx,
                    )
            return hidden_states

    def forward_focus(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Tuple[torch.Tensor, ForwardBatch, FocusKeptPositions]:
        if self.pp_group.world_size != 1:
            raise RuntimeError("FOCUS currently requires pp_size=1 for LLaDA2.")

        with focus_profile_range("llada2.focus.embed"):
            if input_embeds is None:
                hidden_states = self.word_embeddings(input_ids)
            else:
                hidden_states = input_embeds
        residual = None

        active_batch = forward_batch
        active_positions = positions
        kept_positions = [
            list(range(forward_batch.dllm_config.block_size))
            for _ in range(forward_batch.batch_size)
        ]
        layer0_scores: Optional[torch.Tensor] = None
        mask_metadata: Optional[Dict[str, torch.Tensor]] = None

        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                if i == 0:
                    with focus_profile_range("llada2.focus.layer0"):
                        hidden_states, residual, layer0_scores, mask_metadata = (
                            layer.forward_focus_layer0(
                                active_positions, hidden_states, active_batch, residual
                            )
                        )
                elif i == 1:
                    with focus_profile_range("llada2.focus.layer1"):
                        assert mask_metadata is not None
                        hidden_states, residual, active_batch, kept_positions = (
                            layer.forward_focus_layer1(
                                active_positions,
                                hidden_states,
                                active_batch,
                                residual,
                                layer0_scores,
                                mask_metadata,
                            )
                        )
                    active_positions = active_batch.positions
                else:
                    with focus_profile_range("llada2.focus_suffix.suffix_layers"):
                        hidden_states, residual = layer(
                            active_positions,
                            hidden_states,
                            active_batch,
                            residual,
                        )

        with focus_profile_range("llada2.focus_suffix.norm"):
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, active_batch, kept_positions


class LLaDA2MoeModelLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = get_stream("alt") if _is_cuda else None

        self.model = LLaDA2MoeModel(
            config,
            quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("model", ""),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.word_embeddings
        else:
            # TODO something wrong with ParallelLMHead with DP attention enabled
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        """Used by the eagle_worker."""
        return self.model.word_embeddings.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker."""
        del self.model.word_embeddings.weight
        del self.lm_head.weight
        self.model.word_embeddings.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if focus_enabled(forward_batch):
            hidden_states, logits_batch, kept_positions = self.model.forward_focus(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            logits_batch = replace(logits_batch, forward_mode=ForwardMode.DLLM_EXTEND)
            with focus_profile_range("llada2.logits"):
                ret = self.logits_processor(
                    logits_batch.input_ids, hidden_states, self.lm_head, logits_batch
                )
            customized_info = {} if ret.customized_info is None else ret.customized_info
            customized_info["focus_kept_positions"] = kept_positions
            ret.customized_info = customized_info
            return ret

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            timing_prefix = _llada2_dense_timing_prefix(forward_batch)
            with focus_profile_range(f"{timing_prefix}.logits"):
                ret = self.logits_processor(
                    input_ids, hidden_states, self.lm_head, forward_batch
                )
                trace_target = _llada2_state_trace_target()
                if trace_target is not None and ret.full_logits is not None:
                    trace_idx = _llada2_state_trace_index(forward_batch, *trace_target)
                    trace_call = getattr(self.model, "_state_trace_call", 0)
                    trace_calls = _llada2_state_trace_calls()
                    if trace_idx is not None and (
                        trace_calls is None or trace_call in trace_calls
                    ):
                        _llada2_trace_state_tensor(
                            call=trace_call,
                            stage="logits",
                            tensor=ret.full_logits,
                            index=trace_idx,
                        )
                return ret
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if (
                ("v_head" in name)
                or ("inv_freq" in name)
                or (self.config.tie_word_embeddings and "lm_head" in name)
            ):
                continue

            if (
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                import torch.nn.functional as F

                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        # Lazy: get_moe_weights() snapshots x.data, and building the map here would
        # pin every expert weight's pre-process_weights_after_loading storage.
        if not hasattr(self, "routed_experts_weights_of_layer"):
            self.routed_experts_weights_of_layer = LazyValue(
                lambda: {
                    layer_id: layer.mlp.get_moe_weights()
                    for layer_id, layer in enumerate(self.model.layers)
                    if not isinstance(layer, PPMissingLayer)
                    and isinstance(layer.mlp, LLaDA2MoeSparseMoeBlock)
                }
            )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        num_groups = getattr(config, "n_group", 0)
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None if num_groups == 0 else num_groups,
        )


EntryClass = LLaDA2MoeModelLM
