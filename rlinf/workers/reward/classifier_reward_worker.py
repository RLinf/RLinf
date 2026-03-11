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

"""ClassifierRewardServerWorker - Worker wrapper for visual reward classifier.

This module provides a Worker-based wrapper for the ClassifierRewardServer,
allowing it to be placed via component_placement configuration.

Usage in YAML config::

    cluster:
      component_placement:
        reward_server:
          node_group: "4090"
          placement: 0

    reward_server:
      checkpoint_path: "/path/to/reward_classifier.pt"
      image_keys: null  # auto-detect from checkpoint
      device: "cuda"
      server_name: "ClassifierRewardServer"

The env workers can then connect to it via Ray named actor::

    server = ray.get_actor("ClassifierRewardServer")
    logit = ray.get(server.predict.remote(frames_dict))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from rlinf.scheduler import Worker

if TYPE_CHECKING:
    pass


class ClassifierRewardServerWorker(Worker):
    """Worker wrapper for visual reward classifier inference server.

    This Worker loads a reward classifier model and exposes it as a named
    Ray actor that env workers can connect to for remote inference.

    Args:
        cfg: Configuration containing reward_server settings:
            - checkpoint_path: Path to the trained reward_classifier.pt
            - image_keys: Camera keys (optional, auto-detect if None)
            - device: Torch device string (default "cuda")
            - server_name: Name for the Ray actor (default "ClassifierRewardServer")
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._model = None
        self._device = None
        self._image_keys = None

    def init_worker(self):
        """Initialize the classifier model on the worker's GPU."""
        from rlinf.envs.realworld.common.reward_classifier.classifier import (
            load_reward_classifier,
        )

        reward_server_cfg = self.cfg.reward_server
        checkpoint_path = reward_server_cfg.checkpoint_path
        image_keys = reward_server_cfg.get("image_keys", None)
        self._device = reward_server_cfg.get("device", "cuda")

        self._model = load_reward_classifier(
            checkpoint_path,
            image_keys=image_keys,
            device=self._device,
        )
        self._image_keys = self._model.image_keys

        self.log_info(
            f"ClassifierRewardServerWorker loaded model on {self._device}, "
            f"image_keys={self._image_keys}"
        )

    def ready(self) -> bool:
        """Health-check / wait-for-init probe."""
        return self._model is not None

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


def launch_classifier_reward_server(
    cfg: DictConfig,
    cluster,
    placement_strategy,
) -> ClassifierRewardServerWorker:
    """Launch ClassifierRewardServerWorker with the given placement.

    Args:
        cfg: Full configuration containing reward_server settings.
        cluster: The Cluster instance.
        placement_strategy: PlacementStrategy for the reward server.

    Returns:
        The launched worker group.
    """
    reward_server_cfg = cfg.reward_server
    server_name = reward_server_cfg.get("server_name", "ClassifierRewardServer")

    worker_group = ClassifierRewardServerWorker.create_group(cfg).launch(
        cluster,
        name=server_name,
        placement_strategy=placement_strategy,
        # High concurrency to handle multiple env workers
        max_concurrency=128,
    )

    # Wait for initialization
    worker_group.init_workers().wait()

    # Verify the server is ready
    ready_results = worker_group.ready().wait()
    assert all(ready_results), (
        f"ClassifierRewardServerWorker failed to initialize: {ready_results}"
    )

    return worker_group
