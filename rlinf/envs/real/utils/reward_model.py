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

"""A learned reward scored from what the policy sees, instead of the task's."""

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np


class RewardModel:
    """Score one step from a camera frame, through a reward worker.

    Args:
        worker: A launched reward worker, whose ``compute_reward`` takes a
            batch of ``main_images`` and returns a handle to wait on.
        image_key: The camera the model scores. ``None`` takes the first
            camera by name.
    """

    def __init__(self, worker: Any, image_key: Optional[str] = None) -> None:
        self.worker = worker
        self.image_key = image_key

    @classmethod
    def launch(
        cls,
        worker_cfg: Optional[Mapping[str, Any]],
        *,
        image_key: Optional[str],
        node_rank: int,
        node_group: Optional[str],
        hardware_rank: Optional[int],
        env_idx: int,
        worker_rank: int,
    ) -> "RewardModel":
        """Start a reward worker beside this env and wait for it to load.

        Raises:
            ValueError: If no worker config was given.
        """
        if worker_cfg is None:
            raise ValueError(
                "use_reward_model=True but reward_worker_cfg is not provided in "
                "env override_cfg."
            )
        from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

        worker = EmbodiedRewardWorker.launch_for_realworld(
            reward_cfg=worker_cfg,
            node_rank=node_rank,
            node_group_label=node_group,
            hardware_rank=hardware_rank,
            env_idx=env_idx,
            worker_rank=worker_rank,
        )
        worker.init_worker().wait()
        return cls(worker, image_key)

    def __call__(self, frames: Mapping[str, np.ndarray]) -> float:
        """The reward for one step's policy frames.

        Raises:
            ValueError: If there is no frame to score.
            KeyError: If the configured camera is not among the frames.
        """
        if not frames:
            raise ValueError("No frames available for reward model inference.")
        key = self.image_key if self.image_key is not None else sorted(frames)[0]
        if key not in frames:
            raise KeyError(
                f"reward_image_key '{key}' not found in frames. "
                f"Available keys: {list(frames)}"
            )
        output = self.worker.compute_reward(
            {"main_images": np.expand_dims(frames[key], axis=0)}
        ).wait()[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        return float(np.asarray(output).reshape(-1)[0])
