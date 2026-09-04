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

"""Cosmos3 embodied sglang adapter for the ``/v1/actions/generations`` endpoint.

Cosmos3's action policy is served synchronously over sglang's action endpoint:
a single ``msgpack`` ``POST /v1/actions/generations`` carrying all N envs in
one body returns the N action chunks in one response
(``data[0].action.values`` shape ``[N, horizon, raw_action_dim]``). The worker
owns the HTTP round-trip; this adapter only builds the request and parses the
response.

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


class Cosmos3SGLangAdapter:
    """Cosmos3 env-obs <-> action-chunks adapter for the ``/v1/actions/generations`` path."""

    action_path = "/v1/actions/generations"

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

        # Cosmos3 LIBERO recipe: rot6d 10-D, frame_wise_relative, quantile_rot.
        # raw_action_dim is the MODEL width 10, distinct from the env's 7.
        self._raw_action_dim = int(c.get("raw_action_dim", 10))
        self._action_normalization = str(c.get("action_normalization", "quantile"))
        self._action_stats_path = c.get("action_stats_path", None)
        self._stats = self._load_stats(
            self._action_stats_path, self._action_normalization
        )

        eval_env_cfg = cfg.env.get("eval", None) or {}
        env_type = eval_env_cfg.get("env_type", None)
        self._domain_name = str(env_type) if env_type else None
        self._domain_id = c.get("domain_id", None)
        self._num_action_chunks = int(self.model_cfg.num_action_chunks)

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
        """env_obs -> (ONE batched /v1/actions/generations payload, state)."""
        if mode != "eval":
            raise NotImplementedError("Cosmos3 sglang adapter supports eval only.")
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
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 2:  # [horizon, D] — single env (B=1)
            arr = arr[None]  # -> [1, horizon, D]
        if arr.ndim != 3:
            raise RuntimeError(
                f"Cosmos3 action values have unexpected ndim={arr.ndim} "
                f"(expected [N, horizon, D] or [horizon, D]); shape={arr.shape}"
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
        main = env_obs.get("main_images")
        if main is None:
            main = env_obs.get("images")
        wrist = env_obs.get("wrist_images")
        if main is None:
            return [None] * n

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

        uris: list[str | None] = [None] * n
        from PIL import Image  # lazy: keep import cost off non-eval paths

        for i in range(min(n, concat.shape[0])):
            img = Cosmos3SGLangAdapter._resize_reflect_pad(
                concat[i], target_w, target_h
            )
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            uris[i] = f"data:image/png;base64,{b64}"
        return uris

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
    def _action_data(resp: dict) -> Any:
        """Read action values from a ``/v1/actions/generations`` response."""
        data = resp.get("data") or []
        action = data[0].get("action") if data and isinstance(data[0], dict) else None
        values = action.get("values") if isinstance(action, dict) else None
        if values is None:
            raise RuntimeError(
                f"Cosmos3 /v1/actions response carried no 'data[0].action.values' "
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
        raw_norm = arr[..., : self._raw_action_dim]  # server output (normalized?)

        arr = self._denormalize(raw_norm)  # [T, 10]
        trans = arr[:, :3]
        rot6d = arr[:, 3:9]
        gripper = arr[:, 9:10]
        axisangle = self._rot6d_to_axisangle(rot6d)  # [T, 3]
        env = np.concatenate([trans, axisangle, gripper], axis=-1)  # [T, 7]

        if not getattr(self, "_dbg_printed", False):
            self._dbg_printed = True
            import sys

            print(
                f"[C3DBG] raw_norm shape={raw_norm.shape} "
                f"min={raw_norm.min():.4f} max={raw_norm.max():.4f} "
                f"mean={raw_norm.mean():.4f}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[C3DBG] denorm shape={arr.shape} min={arr.min():.4f} "
                f"max={arr.max():.4f} | trans[min/max]={trans.min():.4f}/"
                f"{trans.max():.4f} rot6d[min/max]={rot6d.min():.4f}/"
                f"{rot6d.max():.4f} gripper[min/max]={gripper.min():.4f}/"
                f"{gripper.max():.4f}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[C3DBG] axisangle min/max={axisangle.min():.4f}/"
                f"{axisangle.max():.4f}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[C3DBG] env chunk0={env[0].tolist()}",
                file=sys.stderr,
                flush=True,
            )

        if env.shape[0] < self._num_action_chunks:
            pad = np.repeat(env[-1:], self._num_action_chunks - env.shape[0], axis=0)
            env = np.concatenate([env, pad], axis=0)
        elif env.shape[0] > self._num_action_chunks:
            env = env[: self._num_action_chunks]
        return env
