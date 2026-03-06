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

"""Ray actor that serves visual reward classifier inference on a GPU node.

The server is placed on a node with GPU resources and provides a ``predict``
method so that env workers on CPU-only nodes can request classifier inference
remotely.

Usage (from the head node or driver script)::

    server = ClassifierRewardServer.options(
        name="ClassifierRewardServer",
        num_gpus=0.05,
    ).remote(
        checkpoint_path="/path/to/reward_classifier.pt",
        device="cuda",
    )
    ray.get(server.ready.remote())  # wait for model to load

Env workers can then retrieve the handle via::

    server = ray.get_actor("ClassifierRewardServer")
    logit = ray.get(server.predict.remote(frames_dict))
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import ray
import torch


@ray.remote
class ClassifierRewardServer:
    """Remote classifier inference server.

    Args:
        checkpoint_path: Path to the trained ``reward_classifier.pt``.
        image_keys: Camera keys.  If ``None``, inferred from checkpoint.
        device: Torch device string (e.g. ``"cuda"``).
    """

    def __init__(
        self,
        checkpoint_path: str,
        image_keys: Optional[list[str]] = None,
        device: str = "cuda",
    ) -> None:
        from rlinf.envs.realworld.common.reward_classifier.classifier import (
            load_reward_classifier,
        )

        self._device = device
        self._model = load_reward_classifier(
            checkpoint_path,
            image_keys=image_keys,
            device=device,
        )
        self._image_keys = self._model.image_keys
        print(
            f"[ClassifierRewardServer] Loaded model on {device}, "
            f"image_keys={self._image_keys}",
            flush=True,
        )

    def ready(self) -> bool:
        """Health-check / wait-for-init probe."""
        return True

    @torch.no_grad()
    def predict(self, frames: dict[str, tuple[bytes, tuple, str]]) -> float:
        """Run classifier on camera frames serialized as raw bytes.

        Args:
            frames: ``{camera_key: (raw_bytes, shape_tuple, dtype_str)}``.
                Raw-bytes encoding avoids numpy pickle version mismatch
                between nodes running different numpy versions.

        Returns:
            Classifier logit (scalar float).
        """
        batch: dict[str, torch.Tensor] = {}
        for key in self._image_keys:
            raw_bytes, shape, dtype_str = frames[key]
            img = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape)
            img = torch.from_numpy(img.copy())
            if img.ndim == 3:
                img = img.unsqueeze(0)
            batch[key] = img.to(self._device)
        logit = self._model(batch)
        return float(logit.item())
