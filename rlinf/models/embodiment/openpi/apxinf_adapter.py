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

"""Adapter from RLinf's OpenPI observation contract to ApxInf's PI0.5 policy.

The LIBERO environment already owns embodiment-specific processing: camera
selection/orientation and the 8-D state layout.  This adapter deliberately does
not repeat those transforms.  ApxInf owns model-specific processing (resize,
tokenization, optional state normalization, flow inference and action
unnormalization) through ``AutoPolicy``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


class OpenPIApxInfAdapter:
    """Run an ApxInf PI0.5 policy behind RLinf's batched eval interface."""

    IMAGE_KEYS = ("observation/image", "observation/wrist_image")

    def __init__(
        self,
        model_cfg: DictConfig | Mapping[str, Any],
        device: str,
        *,
        policy: Any | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.apxinf_cfg = model_cfg.get("apxinf", {})
        self.action_dim = int(model_cfg.get("action_dim", 7))
        self.num_action_chunks = int(model_cfg.get("num_action_chunks"))
        self.action_horizon = int(self.apxinf_cfg.get("action_horizon", 10))
        self.num_flow_steps = int(
            self.apxinf_cfg.get(
                "num_flow_steps", model_cfg.get("openpi", {}).get("num_steps", 10)
            )
        )
        self.noise_source = str(self.apxinf_cfg.get("noise_source", "apxinf"))
        self.seed = int(self.apxinf_cfg.get("seed", 0))
        self.device = str(device)

        self._validate_config()
        self._noise_generator = None
        if self.noise_source == "torch":
            self._noise_generator = torch.Generator(device=self.device)
            self._noise_generator.manual_seed(self.seed)

        self.policy = policy if policy is not None else self._load_policy()
        self._validate_policy_contract()

    def _validate_config(self) -> None:
        if self.num_action_chunks <= 0:
            raise ValueError("rollout.model.num_action_chunks must be positive")
        if self.action_horizon < self.num_action_chunks:
            raise ValueError(
                "ApxInf action_horizon must be at least num_action_chunks; got "
                f"{self.action_horizon} < {self.num_action_chunks}"
            )
        if self.action_dim <= 0:
            raise ValueError("rollout.model.action_dim must be positive")
        if self.num_flow_steps <= 0:
            raise ValueError("ApxInf num_flow_steps must be positive")
        if self.noise_source not in {"apxinf", "torch", "observation"}:
            raise ValueError(
                "ApxInf noise_source must be one of: apxinf, torch, observation"
            )

        openpi_cfg = self.model_cfg.get("openpi", {})
        config_name = str(openpi_cfg.get("config_name", "pi05_libero"))
        if config_name != "pi05_libero":
            raise ValueError(
                "The initial ApxInf backend supports only OpenPI pi05_libero; "
                f"got {config_name!r}"
            )
        configured_steps = int(openpi_cfg.get("num_steps", self.num_flow_steps))
        if configured_steps != self.num_flow_steps:
            raise ValueError(
                "ApxInf num_flow_steps must match OpenPI num_steps for parity; "
                f"got {self.num_flow_steps} and {configured_steps}"
            )
        noise_method = str(openpi_cfg.get("noise_method", "flow_ode"))
        if noise_method not in {"flow_ode", "flow_sde"}:
            raise ValueError(
                "ApxInf PI0.5 currently implements flow ODE inference; only "
                "flow_ode, or RLinf eval-mode flow_sde (which resolves to ODE), "
                f"is supported, got {noise_method!r}"
            )

    def _load_policy(self):
        try:
            from apxinf import AutoPolicy
        except ImportError as error:
            raise ImportError(
                "ApxInf is not importable in the rollout worker. Install the "
                "official infinigence/ApxInf Python package and apxinf_py binding "
                "in the RLinf runtime environment."
            ) from error

        model_path = self.model_cfg.get("model_path")
        if not model_path:
            raise ValueError("rollout.model.model_path is required for ApxInf")

        kwargs: dict[str, Any] = {
            "model_type": str(self.apxinf_cfg.get("model_type", "pi05")),
            "device": self.device,
            "precision": str(self.apxinf_cfg.get("precision", "bf16")),
            "norm_key": str(self.apxinf_cfg.get("norm_key", "actions")),
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "num_flow_steps": self.num_flow_steps,
            "flow_start_time": float(self.apxinf_cfg.get("flow_start_time", 1.0)),
            "seed": self.seed,
            "discrete_state": bool(self.apxinf_cfg.get("discrete_state", False)),
            "image_keys": self.IMAGE_KEYS,
            "state_key": "observation/state",
            "prompt_key": "prompt",
            "metadata": {"backend": "rlinf-apxinf"},
        }
        for key in ("checkpoint", "calibration", "tactics", "tokenizer_path"):
            value = self.apxinf_cfg.get(key, None)
            if value:
                kwargs[key] = str(value)
        if self.apxinf_cfg.get("num_views", None) is not None:
            kwargs["num_views"] = int(self.apxinf_cfg.get("num_views"))
        if bool(self.apxinf_cfg.get("autotune", False)):
            kwargs["autotune"] = True
        return AutoPolicy.from_pretrained(str(model_path), **kwargs)

    def _validate_policy_contract(self) -> None:
        actual_horizon = int(self.policy.action_horizon)
        actual_dim = int(self.policy.action_dim)
        if actual_horizon != self.action_horizon:
            raise ValueError(
                f"ApxInf loaded action_horizon={actual_horizon}, expected "
                f"{self.action_horizon}"
            )
        if actual_dim != self.action_dim:
            raise ValueError(
                f"ApxInf loaded action_dim={actual_dim}, expected {self.action_dim}"
            )

    @staticmethod
    def _batch_size(env_obs: Mapping[str, Any]) -> int:
        for value in env_obs.values():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        raise ValueError("Cannot infer batch size from RLinf observation")

    @staticmethod
    def _item(value: Any, index: int) -> Any:
        if torch.is_tensor(value):
            return _to_numpy(value[index])
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value[index])
        if isinstance(value, (list, tuple)):
            return value[index]
        raise TypeError(f"Unsupported batched observation value: {type(value)!r}")

    def _sample_noise(self, batch_size: int) -> np.ndarray | None:
        if self.noise_source != "torch":
            return None
        model_dim = int(self.policy.metadata["model_action_dim"])
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, self.action_horizon, model_dim),
            generator=self._noise_generator,
            device=self.device,
            dtype=torch.float32,
        )
        return noise.cpu().numpy()

    def predict_action_batch(
        self, env_obs: Mapping[str, Any], *, mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convert and infer a batch without duplicating embodiment transforms."""
        if mode != "eval":
            raise ValueError("OpenPIApxInfAdapter is eval-only")
        required = ("main_images", "wrist_images", "states", "task_descriptions")
        missing = [key for key in required if key not in env_obs]
        if missing:
            raise KeyError(f"RLinf observation is missing keys: {missing}")
        if env_obs.get("extra_view_images") is not None:
            raise ValueError("pi05_libero ApxInf expects exactly two camera views")

        batch_size = self._batch_size(env_obs)
        for key in required:
            value = env_obs[key]
            if self._batch_size({key: value}) != batch_size:
                raise ValueError(
                    f"Observation field {key!r} has a mismatched batch size"
                )

        explicit_noise = env_obs.get("noise")
        if self.noise_source == "observation" and explicit_noise is None:
            raise ValueError("noise_source=observation requires env_obs['noise']")
        sampled_noise = (
            None if explicit_noise is not None else self._sample_noise(batch_size)
        )

        action_rows = []
        timings = []
        for index in range(batch_size):
            observation = {
                # LIBERO already rotated both images by 180 degrees. ApxInf's
                # policy receives them unchanged and performs only model-level
                # parse/resize processing.
                "observation/image": self._item(env_obs["main_images"], index),
                "observation/wrist_image": self._item(env_obs["wrist_images"], index),
                "observation/state": self._item(env_obs["states"], index),
                "prompt": self._item(env_obs["task_descriptions"], index),
            }
            noise = None
            if explicit_noise is not None:
                noise = self._item(explicit_noise, index)
            elif sampled_noise is not None:
                noise = sampled_noise[index]
            result = self.policy.infer(observation, noise=noise)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise ValueError(
                    f"ApxInf actions must have shape [horizon, dim], got {actions.shape}"
                )
            if actions.shape[0] < self.num_action_chunks:
                raise ValueError(
                    f"ApxInf returned {actions.shape[0]} actions, but RLinf must "
                    f"execute {self.num_action_chunks}"
                )
            if actions.shape[1] != self.action_dim:
                raise ValueError(
                    f"ApxInf returned action_dim={actions.shape[1]}, expected "
                    f"{self.action_dim}"
                )
            action_rows.append(actions[: self.num_action_chunks])
            timings.append(result.get("timing", {}))

        actions = torch.from_numpy(np.stack(action_rows, axis=0))
        return actions, {"apxinf_timing": timings}

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()
