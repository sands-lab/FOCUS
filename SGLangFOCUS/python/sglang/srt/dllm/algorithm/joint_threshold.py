from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class JointThreshold(DllmAlgorithm):
    """Joint-threshold denoising with mask-to-token and token-to-token edits."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)

    def max_steps(self, block_size: int) -> int:
        return block_size + self.max_post_edit_steps + 1

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        batch_size = forward_batch.batch_size
        input_ids = self._block_input_ids(forward_batch)
        prompt_mask = input_ids != self.mask_id
        return [
            {
                "post_edit_steps": 0,
                "finished": False,
                "prompt_mask": prompt_mask[i],
            }
            for i in range(batch_size)
        ]

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        block_token_indices = self._block_token_indices(forward_batch)
        if block_token_indices is None:
            block_input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
            block_logits = full_logits.view(batch_size, self.block_size, -1)
        else:
            block_input_ids = forward_batch.input_ids.index_select(
                0, block_token_indices
            ).view(batch_size, self.block_size)
            block_logits = full_logits.index_select(0, block_token_indices).view(
                batch_size, self.block_size, -1
            )
        done: List[bool] = []

        for i in range(batch_size):
            state = states[i]
            if state["finished"]:
                done.append(True)
                continue

            curr_input_ids = block_input_ids[i]
            curr_logits = block_logits[i]
            curr_prompt_mask = state["prompt_mask"]

            if self.penalty_lambda > 0:
                prev_ids = curr_input_ids[:-1]
                curr_logits[1:, :].scatter_(
                    1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                )

            x = torch.argmax(curr_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(curr_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )

            mask_index = curr_input_ids == self.mask_id
            has_mask = mask_index.any()
            mask_transfer_index = torch.zeros_like(mask_index)
            budget_exhausted = False
            if has_mask:
                confidence = torch.where(mask_index, p, -np.inf)
                mask_transfer_index = confidence > self.threshold
                if not mask_transfer_index.any():
                    _, select_index = torch.topk(confidence, k=1)
                    mask_transfer_index[select_index] = True
            else:
                state["post_edit_steps"] += 1
                if state["post_edit_steps"] > self.max_post_edit_steps:
                    state["finished"] = True
                    budget_exhausted = True

            if not budget_exhausted:
                edit_mask = ~mask_index & ~curr_prompt_mask
                edit_transfer_index = (
                    (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                )
                transfer_index = mask_transfer_index | edit_transfer_index
                if transfer_index.any():
                    curr_input_ids[transfer_index] = x[transfer_index]
                else:
                    state["finished"] = True

            done.append(state["finished"])

        if block_token_indices is not None:
            forward_batch.input_ids.index_copy_(
                0, block_token_indices, block_input_ids.reshape(-1)
            )

        return done


Algorithm = JointThreshold
