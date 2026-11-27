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

"""Shared observation, prompt, and action transforms for Cosmos3 policies."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def shifted_timesteps(
    num_steps: int, shift: float, device: torch.device
) -> torch.Tensor:
    """Return the descending flow schedule used for stochastic actor sampling."""
    if num_steps < 1 or shift <= 0:
        raise ValueError(f"Expected num_steps>=1 and shift>0, got {num_steps}, {shift}")
    base = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return shift * base / (1.0 + (shift - 1.0) * base)


def encode_prompts(prompts: list[str], width: int) -> torch.Tensor:
    """Pack environment instructions into fixed-width trajectory tensors."""
    encoded = torch.zeros((len(prompts), width), dtype=torch.uint8)
    for index, prompt in enumerate(prompts):
        data = prompt.encode("utf-8")
        if len(data) > width:
            raise ValueError(
                f"Cosmos3 prompt needs {len(data)} bytes; prompt_bytes={width}."
            )
        encoded[index, : len(data)] = torch.tensor(list(data), dtype=torch.uint8)
    return encoded


def decode_prompts(encoded: torch.Tensor) -> list[str]:
    """Decode instructions saved in an RLinf trajectory."""
    return [bytes(row[row != 0].cpu().tolist()).decode("utf-8") for row in encoded]


class Cosmos3ActionCodec:
    """Observation preprocessing and action unnormalization shared by HF and SGLang."""

    def __init__(self, model_cfg: Mapping[str, Any] | None = None) -> None:
        cfg = model_cfg if model_cfg is not None else {}
        self.model_cfg = cfg
        self.size = str(cfg.get("size", "832x480"))
        self.target_w, self.target_h = self.parse_size(self.size)
        self.raw_action_dim = int(cfg.get("raw_action_dim", 10))
        self.action_normalization = str(cfg.get("action_normalization", "quantile"))
        self.action_stats_path = cfg.get("action_stats_path", None)
        self.stats = self.load_stats(self.action_stats_path, self.action_normalization)
        self.num_action_chunks = int(cfg.get("num_action_chunks", 16))
        self.prompt_bytes = int(cfg.get("prompt_bytes", 512))

    @staticmethod
    def parse_size(size: str) -> tuple[int, int]:
        """Parse ``"WxH"`` (e.g. ``"320x192"``) into positive (target_w, target_h)."""
        parts = str(size).lower().replace(",", "x").split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid Cosmos3 size {size!r}; expected 'WxH'")
        try:
            target_w, target_h = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid Cosmos3 size {size!r}; expected integer 'WxH'"
            ) from exc
        if target_w <= 0 or target_h <= 0:
            raise ValueError(
                f"Cosmos3 size must be positive, got {size!r} ({target_w}x{target_h})"
            )
        return target_w, target_h

    @staticmethod
    def load_stats(path: Any, normalization: str) -> dict[str, np.ndarray] | None:
        """Load per-channel normalization stats JSON when configured."""
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

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """Invert action normalization for the first ``raw_action_dim`` channels."""
        norm = action[..., : self.raw_action_dim]
        if self.stats is None:
            return norm
        method = self.action_normalization
        if method in ("quantile", "quantile_rot"):
            q01 = self.stats["q01"]
            q99 = self.stats["q99"]
            return (norm + 1.0) / 2.0 * (q99 - q01) + q01
        if method == "meanstd":
            mean = self.stats["mean"]
            std = self.stats["std"]
            return norm * std + mean
        if method == "minmax":
            lo = self.stats["min"]
            hi = self.stats["max"]
            return (norm + 1.0) / 2.0 * (hi - lo) + lo
        raise ValueError(
            f"Unsupported Cosmos3 action_normalization {method!r}; "
            "expected 'quantile', 'quantile_rot', 'meanstd', or 'minmax'"
        )

    @staticmethod
    def rot6d_to_axisangle(rot6d: np.ndarray) -> np.ndarray:
        """Convert rot6d ``[N, 6]`` to axis-angle (rotvec) ``[N, 3]``."""
        from scipy.spatial.transform import Rotation as R

        rot6d = np.asarray(rot6d, dtype=np.float32).reshape(-1, 6)
        col0 = rot6d[:, :3]
        col1 = rot6d[:, 3:6]
        # Re-orthogonalize (network outputs are only approximately orthonormal).
        col0 = col0 / np.clip(np.linalg.norm(col0, axis=-1, keepdims=True), 1e-8, None)
        col1 = col1 - np.sum(col1 * col0, axis=-1, keepdims=True) * col0
        col1 = col1 / np.clip(np.linalg.norm(col1, axis=-1, keepdims=True), 1e-8, None)
        col2 = np.cross(col0, col1)
        mat = np.stack([col0, col1, col2], axis=-1)  # [N, 3, 3] (columns)
        return R.from_matrix(mat).as_rotvec().astype(np.float32, copy=False)

    @staticmethod
    def augment_prompt(task: str, n_cam: int) -> str:
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

    @staticmethod
    def extract_tasks(env_obs: Mapping[str, Any]) -> list[str]:
        """Extract instruction strings from an env observation dict."""
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
    def encode_prepared_images(images: np.ndarray | None, n: int) -> list[str | None]:
        """Encode prepared camera frames ``[N, H, W, 3]`` as PNG data URIs."""
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

    @classmethod
    def encode_images(
        cls,
        env_obs: Mapping[str, Any],
        n: int,
        target_w: int = 832,
        target_h: int = 480,
    ) -> list[str | None]:
        """Encode each env's camera views as a PNG base64 data URI."""
        images = cls.prepare_images(env_obs, n, target_w, target_h)
        return cls.encode_prepared_images(images, n)

    @classmethod
    def prepare_images(
        cls,
        env_obs: Mapping[str, Any],
        n: int,
        target_w: int,
        target_h: int,
    ) -> np.ndarray | None:
        """Prepare camera pixels shared by local HF and SGLang inference."""
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
            if main_arr.shape != wrist_arr.shape:
                raise ValueError(
                    "Cosmos3 main_images and wrist_images must have matching shapes, "
                    f"got {main_arr.shape} vs {wrist_arr.shape}"
                )
            # h-concat: left = agentview (third-person), right = wrist
            concat = np.concatenate([main_arr, wrist_arr], axis=2)
        else:
            concat = main_arr

        if concat.shape[0] != n:
            raise ValueError(f"Expected {n} Cosmos3 images, got {concat.shape[0]}")
        return np.stack(
            [cls.resize_reflect_pad(concat[i], target_w, target_h) for i in range(n)]
        )

    @staticmethod
    def resize_reflect_pad(
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
            mode = (
                "edge"
                if (pad_w > 0 and pad_w >= nw) or (pad_h > 0 and pad_h >= nh)
                else "reflect"
            )
            if pad_h > 0:
                arr = np.pad(arr, ((0, pad_h), (0, 0), (0, 0)), mode=mode)
            if pad_w > 0:
                arr = np.pad(arr, ((0, 0), (0, pad_w), (0, 0)), mode=mode)
        return arr.astype(np.uint8)

    def encode_observations(
        self, env_obs: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare image and prompt tensors for local HF rollout."""
        tasks = self.extract_tasks(env_obs)
        n_cam = 2 if env_obs.get("wrist_images") is not None else 1
        prompts = [self.augment_prompt(task, n_cam) for task in tasks]
        images = self.prepare_images(env_obs, len(tasks), self.target_w, self.target_h)
        if images is None:
            raise ValueError("Cosmos3 local inference requires camera images")
        return torch.from_numpy(images.copy()), encode_prompts(
            prompts, self.prompt_bytes
        )

    def action_to_env(self, action: Any) -> np.ndarray:
        """Convert a model action chunk to env-scale ``[num_action_chunks, 7]``."""
        if self.raw_action_dim != 10:
            raise ValueError(
                f"Cosmos3 rot6d action decoding requires raw_action_dim=10, "
                f"got {self.raw_action_dim}"
            )
        a = action
        if isinstance(a, np.ndarray):
            arr = np.asarray(a, dtype=np.float32)
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr[0]
            arr = np.atleast_2d(arr)
        else:
            while (
                isinstance(a, list)
                and len(a) == 1
                and isinstance(a[0], list)
                and a[0]
                and isinstance(a[0][0], (list, tuple))
            ):
                a = a[0]
            if (
                isinstance(a, (list, tuple))
                and a
                and not isinstance(a[0], (list, tuple))
            ):
                a = [list(a)]
            rows = [[float(v) for v in r] for r in a]
            if not rows:
                raise ValueError("Cosmos3 returned an empty action array")
            arr = np.asarray(rows, dtype=np.float32)

        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty Cosmos3 action chunk, got shape {arr.shape}"
            )
        if arr.shape[-1] < self.raw_action_dim:
            raise ValueError(
                f"Expected at least {self.raw_action_dim} action channels, "
                f"got shape {arr.shape}"
            )
        raw_norm = arr[..., : self.raw_action_dim]
        if not np.isfinite(raw_norm).all():
            raise ValueError("Cosmos3 action contains non-finite values")

        arr = self.denormalize(raw_norm)
        if not np.isfinite(arr).all():
            raise ValueError("Denormalized Cosmos3 action contains non-finite values")
        trans = arr[:, :3]
        rot6d = arr[:, 3:9]
        gripper = arr[:, 9:10]
        axisangle = self.rot6d_to_axisangle(rot6d)
        env = np.concatenate([trans, axisangle, gripper], axis=-1)

        if env.shape[0] < self.num_action_chunks:
            pad = np.repeat(env[-1:], self.num_action_chunks - env.shape[0], axis=0)
            env = np.concatenate([env, pad], axis=0)
        elif env.shape[0] > self.num_action_chunks:
            env = env[: self.num_action_chunks]
        return env
