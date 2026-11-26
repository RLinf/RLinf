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

"""Cosmos3 embodied SGLang adapter for action evaluation and RL sampling.

Cosmos3's action policy is served synchronously over SGLang's action endpoint:
a ``msgpack`` ``POST /v1/actions/generations`` carrying N envs in one body
returns the N action chunks in one response
(``data[0].action.values`` shape ``[N, horizon, raw_action_dim]``). The worker
supplies the HTTP client and delegates request handling to this adapter.

Training uses JSON ``POST /rollout/generate`` and a msgpack response carrying
joint endpoints for NFT resampling.

The server only batches prompts of equal token length, up to
``--batching-max-size``; :meth:`Cosmos3SGLangAdapter.request_groups` groups
envs accordingly.

Cosmos3 LIBERO SFT trains with ``action_space=frame_wise_relative``,
``rotation_space="6d"`` (rot6d 10-D) and ``action_normalization="quantile_rot"``.
The model therefore emits a **normalized, padded, rot6d** action chunk. The
RLinf LIBERO env expects a 7-D ``[xyz, axisangle(3), gripper]`` delta. So this
adapter must:

1. take the first ``raw_action_dim`` (=10) channels of the padded output;
2. denormalize them with the LIBERO quantile stats (server-side policy mode
   does *not* denormalize, so the client owns it);
3. split ``[trans3, rot6d6, gripper1]`` and convert rot6d -> axis-angle
   (rot6d -> rotation matrix -> rotvec via scipy);
4. re-assemble ``[trans3, axisangle3, gripper1]`` (7-D) and clip/pad the chunk
   count to ``num_action_chunks``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from rlinf.models.embodiment.cosmos3.action_codec import (
    Cosmos3ActionCodec,
    encode_prompts,
    shifted_timesteps,
)
from rlinf.models.embodiment.sglang_adapter import SGLangEmbodiedAdapter


class Cosmos3SGLangAdapter(SGLangEmbodiedAdapter):
    """Adapt Cosmos3 observations, actions, and denoising replay for SGLang."""

    action_path = "/v1/actions/generations"
    supports_training = True

    def __init__(self, cfg: Any, rank: int):
        self.cfg_rollout = cfg.rollout
        self.model_cfg = cfg.rollout.model
        self.rank = rank

        c = self.model_cfg
        self._codec = Cosmos3ActionCodec(c)
        self._action_mode = str(c.get("action_mode", "policy"))
        self._num_frames = int(c.get("num_frames", 17))
        self._size = self._codec.size
        self._target_w = self._codec.target_w
        self._target_h = self._codec.target_h
        self._action_fps = float(c.get("action_fps", 30.0))
        self._action_view_point = str(c.get("action_view_point", "ego_view"))
        # (LIBERO fps=20, native --num_steps 30). sglang defaults fps=5,
        self._fps = int(c.get("fps", 20))
        # num_inference_steps=35 -> OOD prompt caption + different denoise.
        self._num_inference_steps = int(c.get("num_inference_steps", 30))
        self._guidance = float(c.get("guidance", 1.5))
        self._shift = float(c.get("shift", 10.0))

        # Cosmos3 LIBERO recipe: rot6d 10-D, frame_wise_relative, quantile_rot.
        # raw_action_dim is the MODEL width 10, distinct from the env's 7.
        self._raw_action_dim = self._codec.raw_action_dim
        self._action_normalization = self._codec.action_normalization
        self._action_stats_path = self._codec.action_stats_path
        self._stats = self._codec.stats

        env_cfg = cfg.env.get("train", None) or cfg.env.get("eval", None) or {}
        env_type = c.get("domain_name", None) or env_cfg.get("env_type", None)
        self._domain_name = str(env_type) if env_type else None
        self._domain_id = c.get("domain_id", None)
        self._num_action_chunks = self._codec.num_action_chunks
        # sglang's own default for --batching-max-size is 1.
        server_cfg = cfg.rollout.get("sglang", {}).get("server", {}) or {}
        self._max_request_batch = max(
            1, int(server_cfg.get("batching_max_size", 1) or 1)
        )

    def validate_training(self, cfg) -> None:
        if str(self.model_cfg.model_type) != "cosmos3":
            raise ValueError("SGLang embodied RL currently requires model_type=cosmos3")
        if (
            cfg.algorithm.get("adv_type") != "grpo"
            or cfg.algorithm.get("loss_type") != "embodied_nft"
        ):
            raise ValueError(
                "Cosmos3 SGLang training requires GRPO advantages with embodied_nft loss"
            )
        for key, expected in (
            ("nft_xcur_source", "resample"),
            ("nft_target_space", "x0"),
        ):
            if cfg.algorithm.get(key) != expected:
                raise ValueError(
                    f"Cosmos3 NFT requires {key}={expected!r}, "
                    f"got {cfg.algorithm.get(key)!r}"
                )
        if (
            cfg.algorithm.get("entropy_bonus", 0) != 0
            or cfg.algorithm.get("kl_beta", 0) != 0
        ):
            raise ValueError("Cosmos3 action RL requires entropy_bonus=0 and kl_beta=0")
        replay_keys = (
            "model_type",
            "model_variant",
            "num_action_chunks",
            "action_dim",
            "raw_action_dim",
            "action_mode",
            "domain_name",
            "domain_id",
            "action_view_point",
            "action_normalization",
            "action_stats_path",
            "size",
            "resolution",
            "num_frames",
            "fps",
            "action_fps",
            "num_inference_steps",
            "shift",
            "noise_level",
            "prompt_bytes",
            "tokenizer_path",
        )
        for key in replay_keys:
            if cfg.actor.model.get(key) != cfg.rollout.model.get(key):
                raise ValueError(
                    f"Cosmos3 actor and rollout must use the same model.{key}"
                )
        if self.model_cfg.get("ema_enabled", False):
            raise ValueError("Cosmos3 SGLang RL requires ema_enabled=false")
        if self.model_cfg.get("domain_id") is not None:
            raise ValueError(
                "Cosmos3 native replay selects the action head by domain_name; leave domain_id unset"
            )
        if float(self.model_cfg.action_fps) != float(self.model_cfg.fps):
            raise ValueError("Cosmos3 native replay requires action_fps=fps")
        if (
            int(self.model_cfg.action_dim) != 7
            or int(self.model_cfg.raw_action_dim) != 10
        ):
            raise ValueError(
                "Cosmos3 SGLang RL currently requires LIBERO rot6d actions (raw=10, env=7)"
            )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
            Cosmos3DenoisingStage,
        )

        if not getattr(Cosmos3DenoisingStage, "supports_action_rollout", False):
            raise RuntimeError(
                "Cosmos3 training requires a SGLang checkout with action-rollout support. "
                "Install the modified SGLang 0.5.19 source in the serving environment."
            )

    @staticmethod
    def map_weight_name(name: str) -> str | None:
        from rlinf.models.embodiment.cosmos3.weight_mapping import actor_to_sglang_key

        return actor_to_sglang_key(name)

    def create_weight_target(self, load_weights, model_path: str):
        from pathlib import Path

        from rlinf.hybrid_engines.sglang.common.weight_target import (
            SGLangWeightTarget,
        )

        return SGLangWeightTarget(
            load_weights,
            checkpoint_dir=Path(model_path) / "transformer",
            target_module="transformer",
            name_mapper=self.map_weight_name,
        )

    def request_groups(self, env_obs: dict) -> list[list[int]]:
        """Group env indices by prompt, each group at most the server batch size.

        Args:
            env_obs: The env observation batch passed to :meth:`build_request`.

        Returns:
            Lists of env indices that together cover every env once.
        """
        tasks = self._extract_tasks(env_obs)
        n_cam = 2 if env_obs.get("wrist_images") is not None else 1
        by_prompt: dict[str, list[int]] = {}
        for index, task in enumerate(tasks):
            by_prompt.setdefault(self._augment_prompt(task, n_cam), []).append(index)
        size = self._max_request_batch
        return [
            indices[start : start + size]
            for indices in by_prompt.values()
            for start in range(0, len(indices), size)
        ]

    @staticmethod
    def _parse_size(size: str) -> tuple[int, int]:
        return Cosmos3ActionCodec.parse_size(size)

    @staticmethod
    def _load_stats(path: Any, normalization: str) -> dict[str, np.ndarray] | None:
        return Cosmos3ActionCodec.load_stats(path, normalization)

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        return self._codec.denormalize(action)

    def _rot6d_to_axisangle(self, rot6d: np.ndarray) -> np.ndarray:
        return self._codec.rot6d_to_axisangle(rot6d)

    def prepare_request_context(
        self, env_obs: dict, mode: str = "eval"
    ) -> dict[str, Any]:
        del env_obs
        if mode == "train":
            return {"seed": int(torch.randint(2**31 - 1, (1,)).item())}
        return {}

    def build_request_with_context(
        self,
        env_obs: dict,
        mode: str = "eval",
        request_context: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        if (
            mode == "train"
            and request_context is not None
            and "seed" in request_context
        ):
            return self.build_rollout_request(env_obs, seed=request_context["seed"])
        return self.build_request(env_obs, mode=mode)

    def build_request(
        self, env_obs: dict, mode: Literal["train", "eval"] = "eval"
    ) -> tuple[dict, dict]:
        """Build an action evaluation or training rollout request and local state."""
        if mode == "train":
            return self.build_rollout_request(env_obs)
        if mode != "eval":
            raise ValueError(f"Unsupported Cosmos3 rollout mode: {mode}")
        tasks = self._extract_tasks(env_obs)
        image_uris = self._encode_images(
            env_obs, len(tasks), self._target_w, self._target_h
        )
        n_cam = 2 if env_obs.get("wrist_images") is not None else 1
        seed = int(self.cfg_rollout.get("sglang", {}).get("seed", 1140) or 1140)
        prompts = [self._augment_prompt(t, n_cam) for t in tasks]
        input_payload: dict[str, Any] = {"prompt": prompts}
        if all(image_uris):
            input_payload["input_reference"] = image_uris
        elif any(image_uris):
            raise ValueError(
                "Cosmos3 single-POST batching requires an image for every env."
            )
        parameters: dict[str, Any] = {
            "action_mode": self._action_mode,
            "raw_action_dim": self._raw_action_dim,
            "num_frames": self._num_frames,
            "action_fps": self._action_fps,
            "action_view_point": self._action_view_point,
            "fps": self._fps,
            "num_inference_steps": self._num_inference_steps,
            "guidance_scale": self._guidance,
            "flow_shift": self._shift,
            "seed": seed,
            "height": self._target_h,
            "width": self._target_w,
        }
        if self._domain_id is not None:
            parameters["domain_id"] = int(self._domain_id)
        elif self._domain_name is not None:
            parameters["domain_name"] = self._domain_name
        else:
            raise ValueError(
                "Cosmos3 needs domain_id or env_type to select the action head"
            )
        payload = {
            "input": input_payload,
            "parameters": parameters,
            "runtime": {"response_format": "envelope", "output_format": "numpy"},
        }
        return payload, {}

    def build_rollout_request(
        self, env_obs: dict, *, seed: int | None = None
    ) -> tuple[dict, dict]:
        """Build a JSON /rollout/generate request and retain actor observations."""
        tasks = self._extract_tasks(env_obs)
        n_cam = 2 if env_obs.get("wrist_images") is not None else 1
        prompts = [self._augment_prompt(task, n_cam) for task in tasks]
        if not prompts or len(set(prompts)) != 1:
            raise ValueError("Cosmos3 RL requests must group identical prompts")
        if (
            self._action_mode != "policy"
            or self._num_frames != self._num_action_chunks + 1
        ):
            raise ValueError("Cosmos3 RL requires policy mode and num_frames=horizon+1")
        images = self.prepare_images(
            env_obs, len(tasks), self._target_w, self._target_h
        )
        if images is None:
            raise ValueError("Cosmos3 RL requires camera observations")
        noise_level = float(self.model_cfg.get("noise_level", 0.5))
        if noise_level <= 0:
            raise ValueError("Cosmos3 RL noise_level must be positive")
        seed = int(torch.randint(2**31 - 1, (1,)).item()) if seed is None else int(seed)
        payload = {
            "prompt": prompts[0],
            "image_path": self._encode_prepared_images(images, len(tasks)),
            "seed": seed,
            "height": self._target_h,
            "width": self._target_w,
            "num_frames": self._num_frames,
            "fps": self._fps,
            "num_inference_steps": self._num_inference_steps,
            # Native actor replay uses the conditional velocity, without CFG.
            "guidance_scale": 1.0,
            "rollout": True,
            "rollout_sde_type": "sde",
            "rollout_noise_level": noise_level,
            "rollout_debug_mode": False,
            "rollout_return_dit_trajectory": True,
            "rollout_return_step_indices": [self._num_inference_steps],
            "extra_sampling_params": {
                "action_mode": "policy",
                "domain_name": self._domain_name,
                "domain_id": self._domain_id,
                "raw_action_dim": self._raw_action_dim,
                "action_fps": self._action_fps,
                "action_view_point": self._action_view_point,
                "flow_shift": self._shift,
                "use_duration_template": False,
                "use_system_prompt": False,
                "use_guardrails": False,
            },
        }
        return payload, {
            "images": torch.from_numpy(images.copy()),
            "prompts": encode_prompts(
                prompts, int(self.model_cfg.get("prompt_bytes", 512))
            ),
            "nft_noise_seed": seed,
        }

    def parse_rollout_response(
        self, response: list[dict], state: dict
    ) -> tuple[torch.Tensor, dict]:
        """Restore NFT actions and joint endpoints in environment order."""
        from safetensors.torch import load

        def tensor(value):
            if not isinstance(value, dict) or not value.get("__tensor__"):
                raise ValueError(
                    "Cosmos3 rollout response is missing a serialized tensor"
                )
            return load(value["data"])["t"]

        if not isinstance(response, list) or len(response) != len(state["images"]):
            raise ValueError("Cosmos3 rollout response batch size differs from request")
        actions = []
        nft_actions, nft_endpoints = [], []
        times = shifted_timesteps(
            self._num_inference_steps, self._shift, torch.device("cpu")
        )
        for item in response:
            trajectory = item.get("dit_trajectory")
            if not trajectory:
                raise ValueError(
                    "SGLang did not return Cosmos3 action replay data; install the modified SGLang 0.5.19 source in the serving environment"
                )
            endpoint_times = tensor(trajectory["timesteps"]).float()
            expected_times = times[-1:] * 1000.0
            if endpoint_times.shape != expected_times.shape or not torch.allclose(
                endpoint_times, expected_times
            ):
                raise ValueError("Cosmos3 server did not return the final NFT endpoint")
            sigmas = tensor(trajectory["sigmas"]).float()
            if sigmas.shape != times.shape or not torch.allclose(sigmas, times):
                raise ValueError(
                    "Cosmos3 server and actor use different flow schedules"
                )
            endpoint_state = tensor(trajectory["latents"]).float()
            if endpoint_state.ndim != 2 or endpoint_state.shape[0] != 1:
                raise ValueError(
                    f"Invalid Cosmos3 NFT endpoint shape: {endpoint_state.shape}"
                )
            action = tensor(item["generated_output"]).float()
            if tuple(action.shape) != (self._num_action_chunks, self._raw_action_dim):
                raise ValueError(f"Invalid Cosmos3 action shape: {action.shape}")
            if not all(torch.isfinite(t).all() for t in (endpoint_state, action)):
                raise ValueError(
                    "Cosmos3 rollout contains nonfinite actions or replay tensors"
                )
            actions.append(self._action_to_env(action.numpy()))
            nft_actions.append(action)
            nft_endpoints.append(endpoint_state[0])
        endpoint = torch.stack(nft_endpoints)
        generator = torch.Generator().manual_seed(state["nft_noise_seed"])
        return torch.from_numpy(np.stack(actions)).float(), {
            "prev_logprobs": torch.zeros(len(actions), self._num_action_chunks, 7),
            "prev_values": torch.zeros(len(actions), 1),
            "forward_inputs": {
                "images": state["images"],
                "prompts": state["prompts"],
                "nft_x0": torch.stack(nft_actions),
                "nft_joint_x0": endpoint,
                "nft_joint_noise": torch.randn(endpoint.shape, generator=generator),
                # Resampled linear flow matching has no SDE drift correction.
                "nft_noise_level": torch.zeros(len(actions)),
            },
        }

    @staticmethod
    def _augment_prompt(task: str, n_cam: int) -> str:
        return Cosmos3ActionCodec.augment_prompt(task, n_cam)

    def parse_response(
        self, resp: dict, state: dict[str, Any]
    ) -> tuple[torch.Tensor, dict]:
        """/v1/actions/generations response -> [N, num_action_chunks, action_dim]."""
        del state
        values = self._action_data(resp)
        arr = np.asarray(values, dtype=np.float32)  # [N, horizon, D]
        if arr.ndim != 3:
            raise RuntimeError(
                f"Cosmos3 action values have unexpected ndim={arr.ndim} "
                f"(expected one [horizon, D] array per env); shape={arr.shape}"
            )
        chunks = [self._action_to_env(arr[i]) for i in range(arr.shape[0])]
        if not chunks:
            raise ValueError("Cosmos3 returned no action chunks to parse.")
        actions = (
            torch.tensor(np.stack(chunks, axis=0), dtype=torch.float32)
            .detach()
            .cpu()
            .contiguous()
        )  # [N, num_action_chunks, 7]
        flat = actions.reshape(actions.shape[0], -1)
        info = {
            "prev_logprobs": torch.zeros_like(flat),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": {"action": flat.cpu()},
        }
        return actions, info

    @staticmethod
    def _extract_tasks(env_obs: dict) -> list[str]:
        return Cosmos3ActionCodec.extract_tasks(env_obs)

    @staticmethod
    def _encode_prepared_images(images: np.ndarray | None, n: int) -> list[str | None]:
        return Cosmos3ActionCodec.encode_prepared_images(images, n)

    @staticmethod
    def _encode_images(
        env_obs: dict, n: int, target_w: int = 832, target_h: int = 480
    ) -> list[str | None]:
        return Cosmos3ActionCodec.encode_images(env_obs, n, target_w, target_h)

    @staticmethod
    def prepare_images(
        env_obs: dict, n: int, target_w: int, target_h: int
    ) -> np.ndarray | None:
        return Cosmos3ActionCodec.prepare_images(env_obs, n, target_w, target_h)

    @staticmethod
    def _resize_reflect_pad(
        img_hwc: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        return Cosmos3ActionCodec.resize_reflect_pad(img_hwc, target_w, target_h)

    @staticmethod
    def _action_data(resp: dict) -> list[Any]:
        """Return each env's ``[horizon, D]`` action values, in ``input_index`` order."""
        data = [item for item in resp.get("data") or [] if isinstance(item, dict)]
        data.sort(key=lambda item: item.get("input_index", item.get("index", 0)))
        values = [(item.get("action") or {}).get("values") for item in data]
        if not values or any(v is None for v in values):
            raise RuntimeError(
                f"Cosmos3 /v1/actions response carried no 'data[i].action.values' "
                f"— check action_gen=true and action_mode=policy. resp={resp}"
            )
        return values

    def _action_to_env(self, action: Any) -> np.ndarray:
        return self._codec.action_to_env(action)
