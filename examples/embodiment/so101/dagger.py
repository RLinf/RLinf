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

"""DAgger state machine and episode recorder for the SO-101 example."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.storage.lerobot.compat import add_frame_to_dataset


class DaggerState(str, Enum):
    """DAgger session states."""

    READY = "ready"
    POLICY_RECORDING = "policy_recording"
    HANDOVER = "handover"
    EXPERT_RECORDING = "expert_recording"
    SAVED = "saved"
    DISCARDED = "discarded"
    RESETTING = "resetting"
    EXIT = "exit"


class DaggerSessionController:
    """Validate DAgger keys and maintain episode state."""

    def __init__(self) -> None:
        self.state = DaggerState.READY

    def handle(self, key: str) -> str:
        """Handle a key and return the operation for the caller."""
        key = key.lower()
        if key == "s" and self.state is DaggerState.READY:
            self.state = DaggerState.POLICY_RECORDING
            return "start_episode"
        if key == " " and self.state is DaggerState.POLICY_RECORDING:
            self.state = DaggerState.HANDOVER
            return "start_handover"
        if key == "c" and self.state in {
            DaggerState.POLICY_RECORDING,
            DaggerState.EXPERT_RECORDING,
        }:
            self.state = DaggerState.SAVED
            return "save_episode"
        if key == "a" and self.state is DaggerState.EXPERT_RECORDING:
            self.state = DaggerState.DISCARDED
            return "discard_episode_after_hold"
        if key == "a" and self.state in {
            DaggerState.POLICY_RECORDING,
            DaggerState.HANDOVER,
        }:
            self.state = DaggerState.DISCARDED
            return "discard_episode"
        if key == "r" and self.state in {
            DaggerState.SAVED,
            DaggerState.DISCARDED,
        }:
            self.state = DaggerState.RESETTING
            return "reset_only"
        if key == "q" and self.state is not DaggerState.EXIT:
            self.state = DaggerState.EXIT
            return "fold_and_exit"
        return "ignored"

    def handover_finished(self) -> None:
        """Enter expert recording after alignment and hold countdown."""
        if self.state is not DaggerState.HANDOVER:
            raise RuntimeError(f"Cannot finish handover in {self.state.value}.")
        self.state = DaggerState.EXPERT_RECORDING

    def reset_finished(self) -> None:
        """Return to READY after reset and scene arrangement."""
        if self.state is not DaggerState.RESETTING:
            raise RuntimeError(f"Cannot finish reset in {self.state.value}.")
        self.state = DaggerState.READY


class DaggerEpisodeRecorder:
    """Write policy/expert frames to a LeRobot v3 episode."""

    def __init__(
        self, root: Path, task: str, image_shape: tuple[int, int, int], fps: int
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DAgger recording requires LeRobot 0.4 or newer."
            ) from exc
        self._dataset = LeRobotDataset.create(
            repo_id=root.expanduser().name,
            root=root.expanduser(),
            robot_type="so101",
            fps=fps,
            use_videos=False,
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["joint"],
                },
                "observation.images.wrist": {
                    "dtype": "image",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                },
                "action": {"dtype": "float32", "shape": (6,), "names": ["joint"]},
                "policy_action": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["joint"],
                },
                "expert_action": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["joint"],
                },
                "executed_action": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["joint"],
                },
                "intervene_flag": {"dtype": "bool", "shape": (1,), "names": ["flag"]},
                "is_success": {"dtype": "bool", "shape": (1,), "names": ["success"]},
                "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
            },
        )
        self._task = task
        self._frames: list[dict[str, Any]] = []
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Start an episode."""
        if self._frames:
            raise RuntimeError("Previous DAgger episode is not finalized.")
        self._active = True

    def append(
        self,
        observation: dict[str, Any],
        policy_action: np.ndarray,
        expert_action: np.ndarray,
        executed_action: np.ndarray,
        intervened: bool,
    ) -> None:
        """Append one frame with policy, expert, and executed actions."""
        if not self._active:
            return
        state = np.asarray(observation["observation.state"], dtype=np.float32).reshape(
            -1
        )
        if state.shape != (6,):
            raise ValueError(
                f"SO-101 observation state must have shape (6,), got {state.shape}"
            )
        # The client sends normalized RLinf values, while canonical LeRobot
        # SO-101 datasets use degrees and a 0..100 gripper scale.
        native_state = state.copy()
        native_state[:-1] *= 100.0
        native_state[-1] = (native_state[-1] + 1.0) * 50.0
        self._frames.append(
            {
                "observation.state": native_state,
                "observation.images.wrist": np.asarray(
                    observation["observation.images.wrist"], dtype=np.uint8
                ),
                "action": np.asarray(executed_action, dtype=np.float32),
                "policy_action": np.asarray(policy_action, dtype=np.float32),
                "expert_action": np.asarray(expert_action, dtype=np.float32),
                "executed_action": np.asarray(executed_action, dtype=np.float32),
                "intervene_flag": np.asarray([intervened], dtype=bool),
                "is_success": np.asarray([False], dtype=bool),
                "done": np.asarray([False], dtype=bool),
                "task": self._task,
            }
        )

    def save(self, success: bool = True) -> int:
        """Save the active episode and return its frame count."""
        if not self._active or not self._frames:
            raise RuntimeError("No active DAgger episode to save.")
        self._frames[-1]["done"] = np.asarray([True], dtype=bool)
        self._frames[-1]["is_success"] = np.asarray([success], dtype=bool)
        for frame in self._frames:
            add_frame_to_dataset(self._dataset, frame)
        count = len(self._frames)
        self._dataset.save_episode()
        self._frames.clear()
        self._active = False
        return count

    def discard(self) -> None:
        """Discard the active episode."""
        self._frames.clear()
        self._active = False

    def close(self) -> None:
        """Stop the image writer and release dataset resources."""
        if getattr(self._dataset, "image_writer", None) is not None:
            self._dataset.image_writer.wait_until_done()
            self._dataset.image_writer.stop()
            self._dataset.image_writer = None
