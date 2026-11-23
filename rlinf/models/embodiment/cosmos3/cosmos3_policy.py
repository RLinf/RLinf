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

"""Cosmos3 action-policy wrapper for RLinf SFT and native rollout."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.cosmos3.env_io import Cosmos3EnvIO
from rlinf.models.embodiment.cosmos3.transforms.lerobot import (
    prepare_lerobot_batch,
    prepare_sft_observation_batch,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


class Cosmos3Policy(Cosmos3EnvIO, nn.Module, BasePolicy):
    """RLinf adapter around a cosmos ``OmniMoTModel``."""

    _no_split_modules: list[str] = [
        "MoTDecoderLayer",
        "PackedAttentionMoT",
        "Cosmos3VFMNetwork",
    ]
    # cosmos DomainAwareLinear reads its embedding weights directly (grouped_mm).
    _fsdp_wrap_embeddings: bool = False

    def __init__(
        self,
        model: nn.Module,
        *,
        lr_multipliers: dict[str, float] | None = None,
        ema_enabled: bool = True,
        rollout_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.omni = model
        self.lr_multipliers: dict[str, float] = dict(lr_multipliers or {})
        self._ema_enabled = bool(ema_enabled) and bool(
            getattr(getattr(model, "config", None), "ema", None)
            and model.config.ema.enabled
        )
        if not self._ema_enabled:
            self.register_load_state_dict_pre_hook(self._exclude_ema_weights)
        self._global_step: int = 0
        self._rollout_config = dict(rollout_config or {})
        self._action_normalizer = None

    @staticmethod
    def _exclude_ema_weights(_module, state_dict, prefix, *_args) -> None:
        """Allow a training checkpoint to initialize a policy built without EMA."""
        ema_prefix = prefix + "omni.net_ema."
        ema_keys = [key for key in state_dict if key.startswith(ema_prefix)]
        for key in ema_keys:
            del state_dict[key]
        if ema_keys:
            logger.info("Cosmos3: excluded %d EMA checkpoint keys", len(ema_keys))

    @staticmethod
    def get_weight_sync_param_names(module: nn.Module) -> list[str]:
        """Resolve Cosmos checkpoint-wrapper names in the outermost model state dict."""
        # Checkpoint wrappers rewrite parameter names. Resolve their serialized keys
        # by tensor identity, including every alias of a tied weight or buffer.
        state = module.state_dict(keep_vars=True)
        names_by_id: dict[int, list[str]] = {}
        for name, tensor in state.items():
            names_by_id.setdefault(id(tensor), []).append(name)

        trainable_names = set()
        for param in module.parameters():
            if param.requires_grad:
                trainable_names.update(names_by_id[id(param)])

        buffer_names = set()
        for submodule in module.modules():
            for name, buffer in submodule.named_buffers(
                recurse=False, remove_duplicate=False
            ):
                if name not in submodule._non_persistent_buffers_set:
                    buffer_names.update(names_by_id[id(buffer)])

        buffer_names.difference_update(trainable_names)
        return [name for name in state if name in trainable_names] + [
            name for name in state if name in buffer_names
        ]

    @property
    def device(self) -> torch.device:
        return next(self.omni.parameters()).device

    def set_global_step(self, step: int, *, update_ema: bool = True) -> None:
        """Track optimizer steps; restoration and rollout versions can skip EMA."""
        step = int(step)
        if update_ema and self._ema_enabled and step == self._global_step + 1:
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

    def prepare_lerobot_sft_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare expert actions and optional observed video for online SFT."""
        return prepare_lerobot_batch(
            batch,
            max_action_dim=int(self.omni.config.max_action_dim),
            normalizer=self._get_action_normalizer(),
            prepare_observation_batch=partial(
                prepare_sft_observation_batch,
                cfg=self._rollout_config,
                device=self.device,
                tokenize_text=self.omni.tokenize_text,
                input_video_key=self.omni.input_video_key,
            ),
            include_vision=self._rollout_config.get("sft_objective") == "joint",
        )

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            f"Cosmos3Policy supports ForwardType.SFT only, got {forward_type}."
        )

    def sft_forward(self, data: Any = None, **kwargs) -> dict[str, torch.Tensor]:
        """Compute the native SFT loss for one batch."""
        torch.compiler.cudagraph_mark_step_begin()

        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError(
                "Cosmos3 sft_forward requires `data` from the SFT dataloader."
            )

        from cosmos_framework.utils.misc import to as _cosmos_to

        # Native packing reads these fields as Python integers and token lists.
        data = {
            key: value
            if key in {"text_token_ids", "image_size"}
            else _cosmos_to(value, self.device)
            for key, value in data.items()
        }

        if not getattr(self, "_vae_patch_applied", False):
            self._vae_patch_applied = True
            _orig_fn = type(self.omni)._encode_vision_item

            def _patched(omni_self, state, *args, **kwargs):
                if hasattr(state, "device") and state.device != self.device:
                    state = state.to(self.device)
                return _orig_fn(omni_self, state, *args, **kwargs)

            self.omni._encode_vision_item = partial(_patched, self.omni)
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
            "Cosmos3Policy.default_forward is not implemented; native rollout "
            "uses predict_action_batch()."
        )
