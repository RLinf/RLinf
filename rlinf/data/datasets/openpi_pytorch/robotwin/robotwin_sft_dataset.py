# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LeRobot dataset wrapper for RoboTwin ALOHA demonstrations."""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root
from rlinf.utils.logging import get_logger

logger = get_logger()


class RobotwinSftDataset(LeRobotDataset):
    """RoboTwin frames with a future ``action`` chunk for Pi0 SFT.

    The expected feature names are the ones used by the canonical
    ``pi0_aloha_robotwin`` OpenPI data config:

    ``observation.images.cam_high``, ``observation.images.cam_left_wrist``,
    ``observation.images.cam_right_wrist``, ``observation.state``, ``action``
    and ``task``/``prompt``.
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
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                "RoboTwin openpi_pytorch SFT expects a LeRobot dataset root "
                f"containing meta/info.json; got {root}"
            )

        with info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
        dataset_fps = info.get("fps")
        resolved_fps = int(fps) if fps is not None else dataset_fps
        if resolved_fps is None or int(resolved_fps) <= 0:
            raise ValueError(
                f"RoboTwin dataset {root} has no positive fps in meta/info.json; "
                "set data.fps explicitly."
            )
        resolved_fps = int(resolved_fps)

        delta_timestamps = {
            "action": [t / float(resolved_fps) for t in range(action_horizon)]
        }
        logger.info(
            "RoboTwinSftDataset root=%s repo_id=%s fps=%d horizon=%d",
            root,
            root.name,
            resolved_fps,
            action_horizon,
        )
        super().__init__(
            repo_id=root.name,
            root=str(root),
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
        )
        self._dataset_root = Path(root)
        self._resolved_fps = resolved_fps

    @property
    def dataset_root(self) -> Path:
        """Return the resolved on-disk dataset root."""
        return self._dataset_root

    @property
    def resolved_fps(self) -> int:
        """Return the frame rate used for action chunk timestamps."""
        return self._resolved_fps
