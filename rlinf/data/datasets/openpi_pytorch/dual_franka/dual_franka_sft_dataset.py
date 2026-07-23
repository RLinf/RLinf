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

"""Standard LeRobot dataset wrapper for dual-Franka tcp_rot6d SFT.

Unlike the BEHAVIOR streaming dataset, dual-Franka demos use a normal
``LeRobotDataset`` with ``delta_timestamps`` for the action horizon and a
``DistributedSampler`` in the dataloader (rank-partitioned frames).
"""

from __future__ import annotations

from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root
from rlinf.utils.logging import get_logger

logger = get_logger()


class DualFrankaSftDataset(LeRobotDataset):
    """LeRobot dual-Franka frames with an action chunk of horizon ``H``.

    Feature keys match classic dual-franka tcp_rot6d LeRobot exports::

        image, extra_view_image-0, extra_view_image-1, state, actions[, task]
    """

    def __init__(
        self,
        *,
        data_path: str,
        action_horizon: int,
        fps: int | None = None,
        tolerance_s: float = 1e-4,
    ) -> None:
        root = resolve_lerobot_dataset_root(data_path)
        repo_id = root.name
        # When train_data_paths points at the dataset root itself, LeRobot still
        # wants a repo_id; use the directory name. Prefer meta fps when unset.
        meta_fps = None
        info_path = root / "meta" / "info.json"
        if info_path.is_file():
            import json

            with info_path.open("r", encoding="utf-8") as f:
                meta_fps = int(json.load(f).get("fps", 10))
        resolved_fps = int(fps) if fps is not None else (meta_fps or 10)

        delta_timestamps = {
            "actions": [t / float(resolved_fps) for t in range(action_horizon)]
        }
        logger.info(
            "DualFrankaSftDataset root=%s repo_id=%s fps=%d horizon=%d",
            root,
            repo_id,
            resolved_fps,
            action_horizon,
        )
        super().__init__(
            repo_id=repo_id,
            root=str(root),
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
        )
        self._dataset_root = Path(root)
        self._resolved_fps = resolved_fps

    @property
    def dataset_root(self) -> Path:
        """Return the resolved local LeRobot dataset root."""
        return self._dataset_root

    @property
    def resolved_fps(self) -> int:
        """Return the frame rate used to build action timestamps."""
        return self._resolved_fps
