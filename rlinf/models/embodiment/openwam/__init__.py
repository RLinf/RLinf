# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenWAM model loading for RLinf SFT training and embodied evaluation."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.openwam.openwam_policy import OpenWAMPolicy


def get_model(cfg: DictConfig, torch_dtype: torch.dtype | None = None) -> OpenWAMPolicy:
    """Load an OpenWAM policy from a self-contained checkpoint directory."""
    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("OpenWAM requires model_path to be set in the model config.")
    openwam_cfg = cfg.get("openwam", {}) or {}
    # Rollout/eval recipes load straight onto ``device``. The FSDP SFT recipe
    # sets ``load_to_device: false``: the policy is then built on the CPU and
    # FSDP moves each rank's shard to its GPU while wrapping, instead of every
    # rank first materialising the whole model on its own device.
    load_to_device = bool(cfg.get("load_to_device", True))
    device = str(cfg.get("device", "cuda")) if load_to_device else "cpu"
    model = OpenWAMPolicy.from_checkpoint(
        model_path=str(model_path),
        ckpt_name=cfg.get("ckpt_name"),
        device=device,
        torch_dtype=torch_dtype,
        num_frames=int(cfg.get("num_frames", 49)),
        height=int(cfg.get("height", 384)),
        width=int(cfg.get("width", 320)),
        denoise_steps=int(cfg.get("denoise_steps", 10)),
        lambda_video=float(cfg.get("lambda_video", 1.0)),
        lambda_action=float(cfg.get("lambda_action", 1.0)),
        inference_horizon=openwam_cfg.get("inference_horizon"),
        encoder_model_path=(
            str(cfg.get("encoder_model_path"))
            if cfg.get("encoder_model_path") is not None
            else None
        ),
        runtime_dtype=(
            torch_dtype_from_precision(cfg.get("runtime_dtype"))
            if cfg.get("runtime_dtype") is not None
            else None
        ),
    )
    if not load_to_device:
        # FSDP moves parameters and buffers to the accelerator while wrapping,
        # but OpenWAM prepares its inputs (text ids, frames) on the device it
        # cached at load time. Point that cache at the accelerator now.
        model.retarget_runtime_device(torch.device(str(cfg.get("device", "cuda"))))
    return model


__all__ = ["OpenWAMPolicy", "get_model"]
