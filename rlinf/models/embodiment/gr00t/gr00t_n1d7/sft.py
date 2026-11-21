# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Independent-window SFT with the native NVIDIA GR00T N1.7 training head."""

import torch


def dependency_revision(directory) -> str | None:
    """Return a source checkout revision, or None for a packaged installation."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def window_loss(
    action_loss: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Average native masked errors over real action values, then over windows."""
    if action_loss.shape != action_mask.shape or action_loss.ndim != 3:
        raise ValueError("Action loss and mask must have matching [B,H,A] shapes")
    if action_loss.shape[0] == 0:
        raise ValueError("At least one action window is required")
    counts = action_mask.float().sum(dim=(1, 2))
    if torch.any(counts <= 0):
        raise ValueError("Every window must contain valid actions")
    # The native head has already masked its elementwise squared errors.
    per_window = action_loss.float().sum(dim=(1, 2)) / counts
    return per_window.mean()


def native_sft_forward(model, forward_type, data: dict) -> dict:
    """Route independent windows through the native supervised training head."""
    from rlinf.models.embodiment.base_policy import ForwardType

    if forward_type != ForwardType.SFT:
        raise ValueError("This adapter accepts only ForwardType.SFT")
    # Preserve the processor's concatenated image patches and [B,H,A] actions.
    inputs = {
        key: value for key, value in data.items() if isinstance(value, torch.Tensor)
    }
    backbone_inputs, action_inputs = model.prepare_input(inputs)
    output = model.action_head(model.backbone(backbone_inputs), action_inputs)
    return {"loss": window_loss(output["action_loss"], output["action_mask"])}


def sft_no_split_modules(gradient_checkpointing: bool) -> list[str]:
    """Select FSDP boundaries for the native SFT policy."""
    return [
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
        # FSDP must enclose activation checkpointing to retain canonical Adam names.
        "CheckpointWrapper" if gradient_checkpointing else "BasicTransformerBlock",
    ]


def build_model(cfg, torch_dtype=torch.float32):
    """Load native GR00T weights strictly; keep original parameter names."""
    import json
    from pathlib import Path

    import gr00t
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from safetensors.torch import load_file

    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        redirect_qwen3_backbone_to_local,
    )

    if torch_dtype != torch.float32:
        raise ValueError("Native GR00T SFT requires FP32 master parameters")
    dependency = Path(gr00t.__file__).resolve().parent.parent
    if cfg.get("dependency_path") and dependency != Path(cfg.dependency_path).resolve():
        raise ValueError(
            f"Expected GR00T dependency at {cfg.dependency_path}, loaded {dependency}"
        )
    commit = dependency_revision(dependency)
    if cfg.get("dependency_commit") and commit != cfg.dependency_commit:
        raise ValueError(f"Expected GR00T commit {cfg.dependency_commit}, got {commit}")

    class NativeSftPolicy(Gr00tN1d7):
        _no_split_modules = sft_no_split_modules(cfg.gradient_checkpointing)

        def forward(self, forward_type=ForwardType.SFT, data=None, **kwargs):
            return native_sft_forward(self, forward_type, data)

    root = Path(cfg.model_path)
    config = Gr00tN1d7Config.from_pretrained(root, local_files_only=True)
    if config.action_horizon != 40 or config.max_action_dim != 132:
        raise ValueError("Expected native H40, 132-dimensional padded action head")
    config.load_bf16 = False
    config.backbone_trainable_params_fp32 = True
    config.no_split_modules = NativeSftPolicy._no_split_modules
    config._no_split_modules = NativeSftPolicy._no_split_modules
    # Eager construction leaves the native Beta sampler on real CPU tensors.
    with redirect_qwen3_backbone_to_local(config.model_name, cfg.backbone_model_path):
        model = NativeSftPolicy(
            config, transformers_loading_kwargs={"local_files_only": True}
        )
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weights = {}
    for name in sorted(set(index["weight_map"].values())):
        shard = load_file(str(root / name), device="cpu")
        if weights.keys() & shard.keys():
            raise ValueError("Duplicate tensor names in base checkpoint shards")
        weights.update(shard)
    model.load_state_dict(weights, strict=True)
    del weights
    model.float()
    if cfg.get("disable_dropout", True):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    if cfg.gradient_checkpointing:
        from functools import partial

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        apply_activation_checkpointing(
            model.action_head,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda module: type(module).__name__ == "BasicTransformerBlock",
        )
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise TypeError("Native GR00T SFT master parameters must all be FP32")
    return model
