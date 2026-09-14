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

"""Cosmos3 (NVIDIA ``OmniMoTModel``) action-policy wrapper for RLinf SFT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger

logger = get_logger()


class Cosmos3Policy(nn.Module, BasePolicy):
    """RLinf SFT adapter around a cosmos ``OmniMoTModel``."""

    _no_split_modules: list[str] = [
        "MoTDecoderLayer",
        "PackedAttentionMoT",
        "Cosmos3VFMNetwork",
    ]

    def __init__(
        self,
        model: nn.Module,
        *,
        lr_multipliers: dict[str, float] | None = None,
        ema_enabled: bool = True,
    ):
        super().__init__()
        self.omni = model
        self.lr_multipliers: dict[str, float] = dict(lr_multipliers or {})
        self._ema_enabled = bool(ema_enabled) and bool(
            getattr(getattr(model, "config", None), "ema", None)
            and model.config.ema.enabled
        )
        self._global_step: int = 0

    @property
    def device(self) -> torch.device:
        return next(self.omni.parameters()).device

    def set_global_step(self, step: int) -> None:
        """Called once per global (optimizer) step by ``FSDPSftWorker``."""
        step = int(step)
        if self._ema_enabled and step == self._global_step + 1:
            self._update_ema(self._global_step)
        self._global_step = step

    @torch.no_grad()
    def _update_ema(self, iteration: int) -> None:
        model = self.omni
        if not getattr(model, "config", None) or not model.config.ema.enabled:
            return
        net = getattr(model, "net", None)
        net_ema = getattr(model, "net_ema", None)
        worker = getattr(model, "net_ema_worker", None)
        if net is None or net_ema is None or worker is None:
            return
        if getattr(model, "_uses_aux_loss_free_load_balancing", False) or getattr(
            model, "_uses_ema_router_bias", False
        ):
            for sync_fn in ("sync_expert_biases_to_ema", "sync_router_biases_to_ema"):
                fn = getattr(model, sync_fn, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001 - best-effort buffer sync
                        pass
        beta = model.ema_beta(iteration)
        worker.update_average(net, net_ema, beta=beta)

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            f"Cosmos3Policy supports ForwardType.SFT only, got {forward_type}."
        )

    def sft_forward(self, data: Any = None, **kwargs) -> dict[str, torch.Tensor]:
        """Compute the cosmos flow-matching SFT loss for one batch."""
        torch.compiler.cudagraph_mark_step_begin()

        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError(
                "Cosmos3 sft_forward requires `data` from the SFT dataloader."
            )

        from cosmos_framework.utils.misc import to as _cosmos_to

        data = _cosmos_to(data, "cuda")

        if not getattr(self, "_vae_patch_applied", False):
            self._vae_patch_applied = True
            _orig_fn = type(self.omni)._encode_vision_item

            def _patched(omni_self, state, *args, **kwargs):
                if hasattr(state, "device") and state.device.type == "cpu":
                    state = state.to("cuda")
                return _orig_fn(omni_self, state, *args, **kwargs)

            import functools

            self.omni._encode_vision_item = functools.partial(_patched, self.omni)
        output_batch, loss = self.omni.training_step(data, self._global_step)

        result: dict[str, Any] = {"loss": loss}
        if isinstance(output_batch, dict):
            for key, value in output_batch.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value) and value.numel() == 1:
                    result[key] = value.detach()
        return result

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        return None

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "Cosmos3Policy.default_forward is not implemented; rollout/eval runs "
            "through the sglang /v1/actions/generations adapter (Cosmos3SGLangAdapter)."
        )

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError(
            "Cosmos3Policy.predict_action_batch is not implemented; rollout/eval runs "
            "through the sglang /v1/actions/generations adapter (Cosmos3SGLangAdapter)."
        )
