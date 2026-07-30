from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def infer_sdar_block_size_from_model_path(model_path: str) -> int | None:
    match = re.search(r"(?:^|[-_/])b(\d+)(?:$|[-_/])", model_path.lower())
    if match is None:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        first_done_first_out_mode: bool = False,
        enable_delayed_cache: bool = False,
        enable_focus: bool = False,
        focus_alpha: float = 1.0,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.first_done_first_out_mode = first_done_first_out_mode
        self.enable_focus = enable_focus
        self.enable_delayed_cache = bool(enable_delayed_cache or enable_focus)
        self.focus_alpha = focus_alpha

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        from sglang.srt.configs.model_config import ModelConfig

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
            if arch in ("SDARForCausalLM", "SDARMoeForCausalLM"):
                block_size = (
                    infer_sdar_block_size_from_model_path(server_args.model_path)
                    or block_size
                )
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)
        enable_focus = algorithm_config.get(
            "enable_focus", server_args.dllm_enable_focus
        )
        enable_delayed_cache = bool(
            algorithm_config.get(
                "enable_delayed_cache", server_args.dllm_enable_delayed_cache
            )
            or enable_focus
        )
        focus_alpha = float(
            algorithm_config.get("focus_alpha", server_args.dllm_focus_alpha)
        )
        if enable_focus and focus_alpha < 1.0:
            raise ValueError("FOCUS requires dllm_focus_alpha >= 1.0.")

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            first_done_first_out_mode=server_args.dllm_fdfo,
            enable_delayed_cache=enable_delayed_cache,
            enable_focus=enable_focus,
            focus_alpha=focus_alpha,
        )
