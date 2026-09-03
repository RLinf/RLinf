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

"""OpenPI-specific adapter for the ApxInf bare-model rollout backend.

RLinf owns the complete OpenPI input/output transform chain. ApxInf receives
only resized uint8 RGB views, token ids and optional flow noise through its L1
``Model.infer_rgb`` API, and returns normalized model-space actions.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


def _batch_size(env_obs: Mapping[str, Any]) -> int:
    for value in env_obs.values():
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Cannot infer batch size from RLinf observation")


def _item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return _to_numpy(value[index])
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value[index])
    if isinstance(value, (list, tuple)):
        return value[index]
    raise TypeError(f"Unsupported batched observation value: {type(value)!r}")


def _active_token_ids(transformed: Mapping[str, Any]) -> np.ndarray:
    padded_token_ids = np.asarray(transformed["tokenized_prompt"])
    token_mask = np.asarray(transformed["tokenized_prompt_mask"], dtype=np.bool_)
    if padded_token_ids.shape != token_mask.shape:
        raise ValueError(
            "RLinf OpenPI token ids and mask have different shapes: "
            f"{padded_token_ids.shape} and {token_mask.shape}"
        )
    # OpenPI pads prompts to max_token_len and passes a separate mask to its
    # PyTorch model. ApxInf L1 has no mask argument, so give it only the active
    # prefix; otherwise padding would become real text tokens.
    token_ids = np.ascontiguousarray(padded_token_ids[token_mask], dtype=np.uint32)
    if token_ids.size == 0:
        raise ValueError("RLinf OpenPI produced an empty token sequence")
    return token_ids


class _RLInfOpenPITransforms:
    """Build and run RLinf's native OpenPI transforms without PyTorch weights."""

    def __init__(self, model_cfg: DictConfig | Mapping[str, Any]) -> None:
        import openpi.shared.download as download
        import openpi.transforms as transforms
        from openpi.models.model import IMAGE_KEYS
        from openpi.shared import normalize as normalize_lib
        from openpi.training import checkpoints

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        self.model_cfg = model_cfg
        self.num_action_chunks = int(model_cfg.get("num_action_chunks"))
        self.action_dim = int(model_cfg.get("action_dim"))
        openpi_cfg = model_cfg.get("openpi", {})
        config_name = str(openpi_cfg.get("config_name", "pi05_libero"))
        data_kwargs = model_cfg.get("openpi_data", None)
        train_config = get_openpi_config(
            config_name,
            model_path=model_cfg.get("model_path"),
            data_kwargs=data_kwargs,
        )

        # Apply only genuine OpenPI model-config overrides. RLinf-only sampling
        # fields (for example noise_method) are intentionally ignored here.
        model_config = train_config.model
        model_fields = {field.name for field in dataclasses.fields(model_config)}
        overrides = {
            key: value for key, value in openpi_cfg.items() if key in model_fields
        }
        if overrides:
            model_config = dataclasses.replace(model_config, **overrides)

        checkpoint_dir = pathlib.Path(
            download.maybe_download(str(model_cfg.get("model_path")))
        )
        data_config = train_config.data.create(
            train_config.assets_dirs,
            model_config,
        )

        norm_stats_path = (
            data_kwargs.get("norm_stats_path", None)
            if data_kwargs is not None
            else None
        )
        if norm_stats_path:
            norm_dir = pathlib.Path(str(norm_stats_path)).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = normalize_lib.load(norm_dir)
        elif (checkpoint_dir / "norm_stats.json").is_file():
            # Converted LeRobot/ApxInf checkpoints keep stats at the root.
            norm_stats = normalize_lib.load(checkpoint_dir)
        elif data_config.norm_stats is not None:
            norm_stats = data_config.norm_stats
        else:
            if data_config.asset_id is None:
                raise ValueError("OpenPI asset_id is required to load norm stats")
            norm_stats = checkpoints.load_norm_stats(
                checkpoint_dir,
                data_config.asset_id,
            )

        self._input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.data_transforms.outputs,
            ]
        )
        self._image_keys = tuple(IMAGE_KEYS)
        self._config_name = config_name
        state_indices = openpi_cfg.get("state_indices", None)
        self._state_indices = list(state_indices) if state_indices else None

    def _select_state(self, states: Any) -> Any:
        if not self._state_indices:
            return states
        state_dim = int(states.shape[-1])
        if state_dim == len(self._state_indices):
            return states
        if state_dim <= max(self._state_indices):
            raise ValueError(
                f"Cannot select state_indices={self._state_indices} "
                f"from state dim {state_dim}"
            )
        if torch.is_tensor(states):
            indices = torch.as_tensor(self._state_indices, device=states.device)
            return states.index_select(-1, indices)
        return np.asarray(states)[..., self._state_indices]

    def _policy_observation(self, env_obs: Mapping[str, Any]) -> dict[str, Any]:
        required = ("main_images", "states", "task_descriptions")
        missing = [key for key in required if key not in env_obs]
        if missing:
            raise KeyError(f"RLinf observation is missing keys: {missing}")

        states = self._select_state(env_obs["states"])
        observation: dict[str, Any] = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self._config_name:
            observation["observation/state_ee_pos"] = states[:, :3]
            observation["observation/state_ee_rot"] = states[:, 3:6]
            observation["observation/state_gripper"] = states[:, 6:7]
        else:
            observation["observation/state"] = states
        if env_obs.get("wrist_images") is not None:
            observation["observation/wrist_image"] = env_obs["wrist_images"]
        if env_obs.get("extra_view_images") is not None:
            observation["observation/extra_view_image"] = env_obs["extra_view_images"]
        return observation

    def preprocess_batch(
        self,
        env_obs: Mapping[str, Any],
        *,
        num_views: int,
        image_size: int,
    ) -> list[dict[str, np.ndarray]]:
        """Return one L1 ApxInf request and postprocess context per env."""
        policy_obs = self._policy_observation(env_obs)
        batch_size = _batch_size(policy_obs)
        for key, value in policy_obs.items():
            if _batch_size({key: value}) != batch_size:
                raise ValueError(f"Observation field {key!r} has mismatched batch size")

        prepared = []
        for index in range(batch_size):
            sample = {key: _item(value, index) for key, value in policy_obs.items()}
            transformed = self._input_transform(sample)
            images = transformed["image"]
            masks = transformed["image_mask"]
            active_keys = [key for key in self._image_keys if bool(masks[key])]
            if len(active_keys) != num_views:
                raise ValueError(
                    "RLinf OpenPI transforms produced "
                    f"{len(active_keys)} active camera views {active_keys}, but "
                    f"the ApxInf checkpoint expects {num_views}"
                )
            rgb_u8 = np.ascontiguousarray(
                np.stack([images[key] for key in active_keys]),
                dtype=np.uint8,
            )
            expected_rgb_shape = (num_views, image_size, image_size, 3)
            if rgb_u8.shape != expected_rgb_shape:
                raise ValueError(
                    f"RLinf OpenPI RGB shape {rgb_u8.shape}, expected "
                    f"{expected_rgb_shape}"
                )
            token_ids = _active_token_ids(transformed)
            prepared.append(
                {
                    "rgb_u8": rgb_u8,
                    "token_ids": token_ids,
                    "state": np.ascontiguousarray(transformed["state"]),
                }
            )
        return prepared

    def postprocess_batch(
        self,
        normalized_actions: np.ndarray,
        prepared: list[dict[str, np.ndarray]],
    ) -> torch.Tensor:
        """Run RLinf's native OpenPI output transforms on normalized actions."""
        rows = []
        for actions, context in zip(normalized_actions, prepared, strict=True):
            output = self._output_transform(
                {
                    "actions": np.asarray(actions, dtype=np.float32),
                    "state": context["state"],
                }
            )
            row = np.asarray(output["actions"], dtype=np.float32)
            if row.ndim != 2:
                raise ValueError(
                    f"RLinf OpenPI actions must be rank 2, got {row.shape}"
                )
            if row.shape[0] < self.num_action_chunks:
                raise ValueError(
                    f"RLinf OpenPI postprocess returned {row.shape[0]} actions, "
                    f"but {self.num_action_chunks} must be executed"
                )
            if row.shape[1] != self.action_dim:
                raise ValueError(
                    f"RLinf OpenPI postprocess returned action_dim={row.shape[1]}, "
                    f"expected {self.action_dim}"
                )
            rows.append(row[: self.num_action_chunks])
        return torch.from_numpy(np.stack(rows))


class OpenPIApxInfAdapter:
    """Run ApxInf bare PI0.5 inference behind RLinf's OpenPI transforms."""

    def __init__(
        self,
        model_cfg: DictConfig | Mapping[str, Any],
        device: str,
        *,
        model: Any | None = None,
        processor: Any | None = None,
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
        self.model = model if model is not None else self._load_model()
        self.processor = (
            processor if processor is not None else _RLInfOpenPITransforms(model_cfg)
        )
        self._validate_model_contract()

        self._noise_generator = None
        if self.noise_source == "torch":
            self._noise_generator = torch.Generator(device=self.device)
            self._noise_generator.manual_seed(self.seed)

        self.metadata = {
            "backend": "rlinf-apxinf-l1",
            "model_type": "pi05",
            "action_horizon": int(self.model.action_horizon),
            "model_action_dim": int(self.model.action_dim),
            "num_views": int(self.model.num_views),
            "image_size": int(self.model.image_size),
        }

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
        if not config_name.startswith("pi05_"):
            raise ValueError(
                "The ApxInf bare-model adapter currently supports PI0.5 OpenPI "
                f"configs; got {config_name!r}"
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
                "ApxInf PI0.5 implements flow ODE inference; only flow_ode, or "
                "RLinf eval-mode flow_sde (which resolves to ODE), is supported; "
                f"got {noise_method!r}"
            )

    def _load_model(self):
        try:
            import apxinf_py
        except ImportError as error:
            raise ImportError(
                "apxinf_py is not importable in the rollout worker. Install the "
                "official infinigence/ApxInf CUDA Python binding."
            ) from error

        model_path = self.model_cfg.get("model_path")
        if not model_path:
            raise ValueError("rollout.model.model_path is required for ApxInf")
        model_dir = pathlib.Path(str(model_path))
        checkpoint = self.apxinf_cfg.get("checkpoint", None)
        checkpoint_path = (
            pathlib.Path(str(checkpoint))
            if checkpoint
            else model_dir / "model.safetensors"
        )

        tactics = self.apxinf_cfg.get("tactics", None)
        if tactics is None:
            try:
                from apxinf._tactics import resolve_pi05_tactics

                tactics = resolve_pi05_tactics(
                    self.device,
                    str(self.apxinf_cfg.get("precision", "bf16")),
                    model_dir=model_dir,
                    override=None,
                    allow_missing=bool(self.apxinf_cfg.get("autotune", False)),
                )
            except ImportError:
                tactics = None

        kwargs: dict[str, Any] = {
            "device": self.device,
            "precision": str(self.apxinf_cfg.get("precision", "bf16")),
            "autotune": bool(self.apxinf_cfg.get("autotune", False)),
            "action_horizon": self.action_horizon,
            "num_flow_steps": self.num_flow_steps,
            "flow_start_time": float(self.apxinf_cfg.get("flow_start_time", 1.0)),
            "sampling_seed": self.seed,
        }
        calibration = self.apxinf_cfg.get("calibration", None)
        if calibration:
            kwargs["calibration"] = str(calibration)
        if tactics:
            kwargs["tactics"] = str(tactics)
        if self.apxinf_cfg.get("num_views", None) is not None:
            kwargs["num_views"] = int(self.apxinf_cfg.get("num_views"))
        return apxinf_py.Model.load("pi05", str(checkpoint_path), **kwargs)

    def _validate_model_contract(self) -> None:
        if int(self.model.action_horizon) != self.action_horizon:
            raise ValueError(
                f"ApxInf loaded action_horizon={self.model.action_horizon}, "
                f"expected {self.action_horizon}"
            )
        if int(self.model.action_dim) <= 0:
            raise ValueError("ApxInf loaded an invalid model action dimension")
        if int(self.model.num_views) <= 0:
            raise ValueError("ApxInf loaded an invalid camera-view count")

    def _sample_noise(self, batch_size: int) -> np.ndarray | None:
        if self.noise_source != "torch":
            return None
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, self.action_horizon, int(self.model.action_dim)),
            generator=self._noise_generator,
            device=self.device,
            dtype=torch.float32,
        )
        return noise.cpu().numpy()

    def predict_action_batch(
        self, env_obs: Mapping[str, Any], *, mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run RLinf transforms -> ApxInf L1 -> RLinf output transforms."""
        if mode != "eval":
            raise ValueError("OpenPIApxInfAdapter is eval-only")

        prepared = self.processor.preprocess_batch(
            env_obs,
            num_views=int(self.model.num_views),
            image_size=int(self.model.image_size),
        )
        batch_size = len(prepared)
        explicit_noise = env_obs.get("noise")
        if self.noise_source == "observation" and explicit_noise is None:
            raise ValueError("noise_source=observation requires env_obs['noise']")
        sampled_noise = (
            None if explicit_noise is not None else self._sample_noise(batch_size)
        )

        normalized_rows = []
        timings = []
        for index, model_input in enumerate(prepared):
            noise = None
            if explicit_noise is not None:
                noise = _item(explicit_noise, index)
            elif sampled_noise is not None:
                noise = sampled_noise[index]

            started = time.perf_counter()
            normalized = np.asarray(
                self.model.infer_rgb(
                    model_input["rgb_u8"],
                    "nhwc",
                    model_input["token_ids"],
                    noise=noise,
                ),
                dtype=np.float32,
            )
            model_ms = (time.perf_counter() - started) * 1000.0
            expected_shape = (self.action_horizon, int(self.model.action_dim))
            if normalized.shape != expected_shape:
                raise ValueError(
                    f"ApxInf normalized actions have shape {normalized.shape}, "
                    f"expected {expected_shape}"
                )
            if not np.isfinite(normalized).all():
                raise FloatingPointError("ApxInf returned non-finite actions")
            normalized_rows.append(normalized)
            timings.append({"model_ms": model_ms})

        actions = self.processor.postprocess_batch(
            np.stack(normalized_rows),
            prepared,
        )
        return actions, {"apxinf_timing": timings}

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
