# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared in-process policy backend for SO-101 Pi05 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def build_so101_model_config(
    checkpoint: Path, norm_stats: Path, *, num_steps: int = 5
) -> Any:
    """Build the RLinf OpenPI configuration used by local and gRPC inference."""
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[4]
    cfg = OmegaConf.load(root / "examples/sft/config/model/pi0_5.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.model_path = str(checkpoint)
    cfg.num_action_chunks = 20
    cfg.action_dim = 6
    cfg.add_value_head = False
    cfg.openpi.config_name = "pi05_so101"
    cfg.openpi.norm_stats_path = str(norm_stats)
    cfg.openpi.asset_id = norm_stats.parent.name
    cfg.openpi.num_images_in_input = 1
    cfg.openpi.action_chunk = 20
    cfg.openpi.action_env_dim = 6
    cfg.openpi.num_steps = num_steps
    cfg.openpi.add_value_head = False
    cfg.openpi.train_expert_only = False
    return cfg


class SO101LocalPolicyBackend:
    """Load Pi05 locally and return normalized ``[20, 6]`` action chunks."""

    def __init__(
        self,
        checkpoint: Path,
        norm_stats: Path,
        device: str,
        *,
        num_steps: int = 5,
    ) -> None:
        import torch

        from rlinf.models import get_model

        checkpoint = checkpoint.expanduser().resolve()
        norm_stats = norm_stats.expanduser().resolve()
        if not (checkpoint / "model_state_dict/full_weights.pt").is_file():
            raise FileNotFoundError(f"Incomplete RLinf actor checkpoint: {checkpoint}")
        if not norm_stats.is_file():
            raise FileNotFoundError(f"Missing norm_stats: {norm_stats}")
        self.device = torch.device(device)
        self.model = (
            get_model(
                build_so101_model_config(checkpoint, norm_stats, num_steps=num_steps)
            )
            .to(self.device)
            .eval()
        )

    def predict(self, payload: dict[str, Any]) -> np.ndarray:
        """Return one normalized action chunk for a canonical SO-101 payload."""
        import torch

        image = np.asarray(payload["observation.images.wrist"], dtype=np.uint8)
        state = np.asarray(payload["observation.state"], dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")
        if state.shape != (6,):
            raise ValueError(f"Expected SO-101 state shape (6,), got {state.shape}")
        # Robot-facing gripper values use [-1, 1]. Pi05's SO-101
        # normalization statistics follow LeRobot's [0, 1] convention.
        model_state = state.copy()
        model_state[-1] = (model_state[-1] + 1.0) / 2.0
        env_obs = {
            "main_images": torch.from_numpy(image[None]),
            "wrist_images": None,
            "extra_view_images": None,
            "states": torch.from_numpy(model_state[None]),
            "task_descriptions": [str(payload.get("task", ""))],
        }
        with torch.no_grad():
            actions, _ = self.model.predict_action_batch(
                env_obs, mode="eval", compute_values=False
            )
        result = actions[0].detach().float().cpu().numpy()
        if result.shape != (20, 6):
            raise ValueError(f"SO-101 Pi05 must return (20, 6), got {result.shape}")
        result[:, -1] = 2.0 * result[:, -1] - 1.0
        return np.clip(result, -1.0, 1.0)
