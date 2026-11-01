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

"""Encoded control observations shared by Dream environments."""

import base64
import io
from typing import Any

import torch
from PIL import Image, ImageOps


class DreamObservationBuffer:
    """Collect one environment's control observations before worker transport."""

    def __init__(self) -> None:
        self.episode_id = 0
        self.samples: list[dict[str, Any]] = []
        self.last_sample: dict[str, Any] | None = None

    def reset(self) -> None:
        """Start an episode with an empty observation history."""
        self.episode_id += 1
        self.samples = []
        self.last_sample = None

    def clear(self) -> None:
        """Clear the chunk history while retaining the last captured sample."""
        self.samples = []

    def append(
        self,
        observation: dict[str, Any],
        timestamp: float,
    ) -> None:
        """Encode the three camera views with their measured state and time."""
        main = observation["main_images"]
        wrists = observation.get("extra_view_images", observation.get("wrist_images"))
        if (
            main.ndim != 4
            or main.shape[0] != 1
            or main.shape[-1] != 3
            or wrists is None
            or wrists.ndim != 5
            or wrists.shape[:2] != (1, 2)
            or wrists.shape[-1] != 3
            or main.dtype != torch.uint8
            or wrists.dtype != torch.uint8
        ):
            raise ValueError(
                "Dream requires one uint8 RGB top view and two wrist views."
            )
        frames = {}
        views = (main[0], wrists[0, 0], wrists[0, 1])
        for role, image in zip(
            ("overhead", "wrist_l", "wrist_r"),
            views,
            strict=True,
        ):
            buffer = io.BytesIO()
            ImageOps.pad(
                Image.fromarray(image.cpu().numpy()),
                (320, 240),
                method=Image.Resampling.BILINEAR,
                color=(0, 0, 0),
            ).save(buffer, format="JPEG", quality=85)
            frames[f"video.{role}"] = base64.b64encode(buffer.getvalue()).decode(
                "ascii"
            )
        sample = {
            "frames": frames,
            "state": observation["states"].detach().cpu().clone(),
            "timestamp": timestamp,
        }
        self.samples.append(sample)
        self.last_sample = sample

    def pack(
        self,
        task_descriptions: list[str],
        commanded: torch.Tensor | None,
        *,
        finished: bool,
        active_arms: tuple[str, ...] = ("left", "right"),
    ) -> dict[str, Any]:
        """Return a single-environment batch with an explicit control time axis."""
        if self.last_sample is None:
            raise RuntimeError(
                "Reset the Dream environment before reading observations."
            )
        samples = self.samples or [self.last_sample]
        return {
            "frames": [[sample["frames"] for sample in samples]],
            "states": torch.stack([sample["state"] for sample in samples], dim=1),
            "timestamps": torch.tensor(
                [[sample["timestamp"] for sample in samples]], dtype=torch.float64
            ),
            "task_descriptions": task_descriptions,
            "episode_id": torch.tensor([self.episode_id]),
            "finished": torch.tensor([finished]),
            "commanded": commanded,
            "active_arms": [active_arms],
        }
