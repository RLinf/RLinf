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
action denoising transition logprobs and replay latents for the native actor.

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

import base64
import io
import json
from typing import Any, Literal

import numpy as np
import torch

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
        self._action_mode = str(c.get("action_mode", "policy"))
        self._num_frames = int(c.get("num_frames", 17))
        self._size = str(c.get("size", "832x480"))
        self._target_w, self._target_h = self._parse_size(self._size)
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
        self._raw_action_dim = int(c.get("raw_action_dim", 10))
        self._action_normalization = str(c.get("action_normalization", "quantile"))
        self._action_stats_path = c.get("action_stats_path", None)
        self._stats = self._load_stats(
            self._action_stats_path, self._action_normalization
        )

        env_cfg = cfg.env.get("train", None) or cfg.env.get("eval", None) or {}
        env_type = c.get("domain_name", None) or env_cfg.get("env_type", None)
        self._domain_name = str(env_type) if env_type else None
        self._domain_id = c.get("domain_id", None)
        self._num_action_chunks = int(self.model_cfg.num_action_chunks)
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
            or cfg.algorithm.get("loss_type") != "actor"
        ):
            raise ValueError("Cosmos3 SGLang RL currently supports GRPO actor loss")
        if cfg.algorithm.get("logprob_type") != "chunk_level":
            raise ValueError(
                "Cosmos3 action denoising logprobs require chunk_level aggregation"
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
        server = cfg.rollout.sglang.server
        sp_degree = int(server.get("sp_degree", 1))
        if sp_degree > 1 and (
            int(server.get("ulysses_degree", 1)) != sp_degree
            or int(server.get("ring_degree", 1)) != 1
            or int(server.get("kv_gather_degree", 1) or 1) != 1
        ):
            raise ValueError(
                "Cosmos3 action SP requires ulysses_degree=sp_degree and ring_degree=1"
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
        """Parse ``"WxH"`` (e.g. ``"320x192"``) into (target_w, target_h)."""
        try:
            w_s, h_s = str(size).lower().replace(",", "x").split("x")
            return int(w_s), int(h_s)
        except Exception:
            return 832, 480

    @staticmethod
    def _load_stats(path: Any, normalization: str) -> dict[str, np.ndarray] | None:
        # Per-channel stats keys loaded from the LIBERO quantile stats JSON.
        stat_keys = ("mean", "std", "min", "max", "q01", "q99")
        if not path:
            return None
        with open(path, encoding="utf-8") as f:  # noqa: PTH123
            raw = json.load(f)
        key = "global_raw" if normalization == "quantile_rot" else "global"
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
        return {
            k: np.asarray(v, dtype=np.float32) for k, v in raw.items() if k in stat_keys
        }

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        """Invert the action normalization for the first ``raw_action_dim`` channels."""
        d = self._raw_action_dim
        norm = action[..., :d]
        if self._stats is None:
            return norm
        method = self._action_normalization
        if method in ("quantile", "quantile_rot"):
            q01 = self._stats["q01"]
            q99 = self._stats["q99"]
            return (norm + 1.0) / 2.0 * (q99 - q01) + q01
        if method == "meanstd":
            mean = self._stats["mean"]
            std = self._stats["std"]
            return norm * std + mean
        if method == "minmax":
            lo = self._stats["min"]
            hi = self._stats["max"]
            return (norm + 1.0) / 2.0 * (hi - lo) + lo
        return norm

    def _rot6d_to_axisangle(self, rot6d: np.ndarray) -> np.ndarray:
        """rot6d ``[N, 6]`` -> axis-angle (rotvec) ``[N, 3]``."""
        from scipy.spatial.transform import Rotation as R

        rot6d = np.asarray(rot6d, dtype=np.float32).reshape(-1, 6)
        col0 = rot6d[:, :3]
        col1 = rot6d[:, 3:6]
        # Re-orthogonalize (the network output is only approximately rot6d).
        col0 = col0 / np.clip(np.linalg.norm(col0, axis=-1, keepdims=True), 1e-8, None)
        col1 = col1 - np.sum(col1 * col0, axis=-1, keepdims=True) * col0
        col1 = col1 / np.clip(np.linalg.norm(col1, axis=-1, keepdims=True), 1e-8, None)
        col2 = np.cross(col0, col1)
        mat = np.stack([col0, col1, col2], axis=-1)  # [N, 3, 3] (columns)
        # scipy wants [..., 3, 3] with rows as basis -> transpose (columns->rows).
        return R.from_matrix(mat).as_rotvec().astype(np.float32, copy=False)

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

    def build_rollout_request(self, env_obs: dict) -> tuple[dict, dict]:
        """Build a JSON /rollout/generate request and retain actor observations."""
        from rlinf.models.embodiment.cosmos3.cosmos3_policy import encode_prompts

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
        step = int(torch.randint(self._num_inference_steps, (1,)).item())
        seed = int(torch.randint(2**31 - 1, (1,)).item())
        payload = {
            "prompt": prompts[0],
            "image_path": self._encode_images(
                env_obs, len(tasks), self._target_w, self._target_h
            ),
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
            "rollout_log_prob_no_const": False,
            "rollout_debug_mode": False,
            "rollout_return_dit_trajectory": True,
            "rollout_return_step_indices": [step, step + 1],
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
            "step": step,
        }

    def parse_rollout_response(
        self, response: list[dict], state: dict
    ) -> tuple[torch.Tensor, dict]:
        """Restore the denoising transition used for the rollout logprobs."""
        from safetensors.torch import load

        from rlinf.models.embodiment.cosmos3.cosmos3_action_model import (
            shifted_timesteps,
        )

        def tensor(value):
            if not isinstance(value, dict) or not value.get("__tensor__"):
                raise ValueError(
                    "Cosmos3 rollout response is missing a serialized tensor"
                )
            return load(value["data"])["t"]

        if not isinstance(response, list) or len(response) != len(state["images"]):
            raise ValueError("Cosmos3 rollout response batch size differs from request")
        actions, logprobs, latents, next_latents = [], [], [], []
        step = state["step"]
        times = shifted_timesteps(
            self._num_inference_steps, self._shift, torch.device("cpu")
        )
        for item in response:
            trajectory = item.get("dit_trajectory")
            if not trajectory:
                raise ValueError(
                    "SGLang did not return Cosmos3 action replay data; install the modified SGLang 0.5.19 source in the serving environment"
                )
            replay_times = tensor(trajectory["timesteps"]).float()
            expected_times = times[step : step + 2] * 1000.0
            if replay_times.shape != expected_times.shape or not torch.allclose(
                replay_times, expected_times
            ):
                raise ValueError("Cosmos3 server returned the wrong replay step")
            sigmas = tensor(trajectory["sigmas"]).float()
            if sigmas.shape != times.shape or not torch.allclose(sigmas, times):
                raise ValueError(
                    "Cosmos3 server and actor use different flow schedules"
                )
            pair = tensor(trajectory["latents"]).float()
            lp = tensor(item.get("rollout_log_probs")).float()
            expected = (self._num_inference_steps,)
            if pair.ndim != 2 or pair.shape[0] != 2 or tuple(lp.shape) != expected:
                raise ValueError(
                    f"Invalid Cosmos3 replay shapes: latents={pair.shape}, logprobs={lp.shape}"
                )
            action = tensor(item["generated_output"]).float()
            if tuple(action.shape) != (self._num_action_chunks, self._raw_action_dim):
                raise ValueError(f"Invalid Cosmos3 action shape: {action.shape}")
            if not all(torch.isfinite(t).all() for t in (pair, lp, action)):
                raise ValueError(
                    "Cosmos3 rollout contains nonfinite actions or replay tensors"
                )
            actions.append(self._action_to_env(action.numpy()))
            # SGLang reports a mean over the unmasked action dimensions at each
            # step. RLinf's chunk-level loss sums this [horizon, action_dim] layout.
            chunk_lp = torch.zeros(self._num_action_chunks, 7)
            chunk_lp[0, 0] = lp[step] * self._num_action_chunks * self._raw_action_dim
            logprobs.append(chunk_lp)
            latents.append(pair[0])
            next_latents.append(pair[1])
        return torch.from_numpy(np.stack(actions)).float(), {
            "prev_logprobs": torch.stack(logprobs),
            "prev_values": torch.zeros(len(actions), 1),
            "forward_inputs": {
                "images": state["images"],
                "prompts": state["prompts"],
                "latent": torch.stack(latents),
                "next_latent": torch.stack(next_latents),
                "step": torch.full((len(actions),), step, dtype=torch.long),
            },
        }

    @staticmethod
    def _augment_prompt(task: str, n_cam: int) -> str:
        """Append the concat-view viewpoint+layout sentences the SFT recipe used."""
        if n_cam < 2:
            return task
        p = task.rstrip()
        if not p.endswith("."):
            p += "."
        p += (
            " This video contains concatenated views from multiple camera perspectives."
        )
        p += (
            " The left half shows the third-person view; the right half shows"
            " the wrist-mounted camera."
        )
        return p

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
        for key in ("task_descriptions", "task", "language", "language_instruction"):
            tasks = env_obs.get(key)
            if tasks is not None:
                if isinstance(tasks, torch.Tensor):
                    tasks = tasks.tolist()
                return [str(t) for t in tasks]
        raise ValueError(
            "env_obs has no task descriptions "
            "(task_descriptions/task/language/language_instruction)"
        )

    @staticmethod
    def _encode_images(
        env_obs: dict, n: int, target_w: int = 832, target_h: int = 480
    ) -> list[str | None]:
        """Encode each env's camera views as a PNG base64 data URI."""
        images = Cosmos3SGLangAdapter.prepare_images(env_obs, n, target_w, target_h)
        if images is None:
            return [None] * n
        from PIL import Image

        uris: list[str | None] = [None] * n
        for i, img in enumerate(images):
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            uris[i] = f"data:image/png;base64,{b64}"
        return uris

    @staticmethod
    def prepare_images(
        env_obs: dict, n: int, target_w: int, target_h: int
    ) -> np.ndarray | None:
        """Prepare camera pixels shared by local and SGLang inference."""
        main = env_obs.get("main_images")
        if main is None:
            main = env_obs.get("images")
        wrist = env_obs.get("wrist_images")
        if main is None:
            return None

        def _to_uint8_hwc(x: Any) -> np.ndarray:
            arr = torch.as_tensor(
                x
                if not isinstance(x, list)
                else torch.stack([torch.as_tensor(a) for a in x])
            )
            if arr.ndim == 5:  # [N, 1, H, W, 3] -> [N, H, W, 3]
                arr = arr[:, 0]
            return arr.clamp(0, 255).to(torch.uint8).cpu().numpy().astype(np.uint8)

        main_arr = _to_uint8_hwc(main)  # [N, H, W, 3]
        if wrist is not None:
            wrist_arr = _to_uint8_hwc(wrist)
            # h-concat: left = agentview (third-person), right = wrist
            if main_arr.shape[1:] == wrist_arr.shape[1:]:
                concat = np.concatenate([main_arr, wrist_arr], axis=2)
            else:
                concat = main_arr  # shape mismatch fallback
        else:
            concat = main_arr  # single-view fallback

        if concat.shape[0] != n:
            raise ValueError(f"Expected {n} Cosmos3 images, got {concat.shape[0]}")
        return np.stack(
            [
                Cosmos3SGLangAdapter._resize_reflect_pad(concat[i], target_w, target_h)
                for i in range(n)
            ]
        )

    @staticmethod
    def _resize_reflect_pad(
        img_hwc: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Aspect-preserving resize (no upscale) + reflection/edge pad bottom+right."""
        from PIL import Image

        h, w = img_hwc.shape[:2]
        scale = min(target_w / w, target_h / h, 1.0)
        nh = int(scale * h + 0.5)
        nw = int(scale * w + 0.5)
        nh = max(1, min(nh, target_h))
        nw = max(1, min(nw, target_w))
        if nh != h or nw != w:
            pil = Image.fromarray(img_hwc).resize((nw, nh), resample=Image.BICUBIC)
            arr = np.asarray(pil, dtype=np.uint8)
        else:
            arr = img_hwc
        pad_w = target_w - nw
        pad_h = target_h - nh
        if pad_w > 0 or pad_h > 0:
            # reflect if BOTH pads are smaller than the content dim, else edge.
            if (pad_w > 0 and pad_w >= nw) or (pad_h > 0 and pad_h >= nh):
                mode = "edge"
            else:
                mode = "reflect"
            if pad_h > 0:
                arr = np.pad(arr, ((0, pad_h), (0, 0), (0, 0)), mode=mode)
            if pad_w > 0:
                arr = np.pad(arr, ((0, 0), (0, pad_w), (0, 0)), mode=mode)
        return arr.astype(np.uint8)

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
        """Model action chunk -> env-scale ``[num_action_chunks, action_dim]``."""
        a = action
        while (
            isinstance(a, list)
            and len(a) == 1
            and isinstance(a[0], list)
            and a[0]
            and isinstance(a[0][0], (list, tuple))
        ):
            a = a[0]
        if isinstance(a, (list, tuple)) and a and not isinstance(a[0], (list, tuple)):
            a = [list(a)]
        rows = [[float(v) for v in r] for r in a]
        if not rows:
            raise ValueError("Cosmos3 returned an empty action array")
        arr = np.asarray(rows, dtype=np.float32)  # [T, D] (D may be padded > raw)
        raw_norm = arr[..., : self._raw_action_dim]

        arr = self._denormalize(raw_norm)  # [T, 10]
        trans = arr[:, :3]
        rot6d = arr[:, 3:9]
        gripper = arr[:, 9:10]
        axisangle = self._rot6d_to_axisangle(rot6d)  # [T, 3]
        env = np.concatenate([trans, axisangle, gripper], axis=-1)  # [T, 7]

        if env.shape[0] < self._num_action_chunks:
            pad = np.repeat(env[-1:], self._num_action_chunks - env.shape[0], axis=0)
            env = np.concatenate([env, pad], axis=0)
        elif env.shape[0] > self._num_action_chunks:
            env = env[: self._num_action_chunks]
        return env
