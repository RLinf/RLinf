# Copyright 2025 The RLinf Authors.
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

"""LeRobot dataset writer for saving rollout data."""

import gc
from typing import Any

from rlinf.utils.logging import get_logger


class LeRobotDatasetWriter:
    """
    Wrapper for LeRobotDataset that provides a simplified interface for writing episodes.

    Usage:
        writer = LeRobotDatasetWriter()
        writer.create(
            repo_id="my-dataset",
            robot_type="franka_panda",
            fps=5,
            features={...}
        )

        for episode_data in episodes:
            writer.add_episode(episode_data)

        writer.finalize(push_to_hub=False)
    """

    def __init__(self):
        """Initialize the writer."""
        self.dataset = None
        self.logger = get_logger()

    def create(
        self,
        repo_id: str,
        robot_type: str = "franka_panda",
        fps: int = 5,
        features: dict[str, dict[str, Any]] | None = None,
        image_writer_threads: int = 10,
        image_writer_processes: int = 5,
        image_shape: tuple[int, int, int] = (256, 256, 3),
        state_dim: int = 8,
        action_dim: int = 7,
        has_image: bool = True,
        wrist_image_keys: dict[str, tuple[int, ...]] | None = None,
        extra_view_image_keys: dict[str, tuple[int, ...]] | None = None,
        has_intervene_flag: bool = True,
    ) -> None:
        """
        Create a new LeRobot dataset.

        Args:
            repo_id: The identifier for the new LeRobot dataset
            robot_type: Robot type (default "franka_panda")
            fps: Frame rate (default 5)
            features: Feature schema dictionary defining the dataset structure.
                If None, auto-generated from dimensions.
            image_writer_threads: Number of threads for image writing
            image_writer_processes: Number of processes for image writing
            image_shape: Image shape (H, W, C) for the main ``image`` feature.
            state_dim: State dimension for auto-generated features
            action_dim: Action dimension for auto-generated features
            has_image: Whether to include the main ``image`` feature.
            wrist_image_keys: Mapping of wrist-camera image key names to their
                ``(H, W, C)`` shapes.  A single view produces
                ``{"wrist_image": (H, W, C)}``; multiple views produce
                ``{"wrist_image/0": …, "wrist_image/1": …, …}``.
            extra_view_image_keys: Same as *wrist_image_keys* but for the
                extra-view camera(s).
            has_intervene_flag: Whether to include per-frame human-intervention
                flag (bool, shape ``(1,)``) in auto-generated features.

        """
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        if features is None:
            features = {
                "state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": ["actions"],
                },
                "done": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["done"],
                },
                "is_success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["is_success"],
                },
            }
            if has_intervene_flag:
                features["intervene_flag"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["intervene_flag"],
                }
            if has_image:
                features["image"] = {
                    "dtype": "image",
                    "shape": list(image_shape),
                    "names": ["height", "width", "channel"],
                }
            for keys in (wrist_image_keys, extra_view_image_keys):
                if keys:
                    for key, shape in keys.items():
                        features[key] = {
                            "dtype": "image",
                            "shape": list(shape),
                            "names": ["height", "width", "channel"],
                        }

        self.logger.info(
            f"Creating LeRobot dataset: repo_id={repo_id}, robot_type={robot_type}, fps={fps}"
        )
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )

    def add_episode(self, episode_data: list[dict[str, Any]]) -> None:
        """
        Add an episode to the dataset.

        Args:
            episode_data: List of frame dictionaries, where each frame contains:
                - image: np.ndarray [H, W, C]
                - wrist_image: np.ndarray [H, W, C] (optional)
                - state: np.ndarray [state_dim]
                - actions: np.ndarray [action_dim]
                - task: str (task instruction)
                - intervene_flag: np.ndarray [1] of bool (optional; matches schema)
                - Any other fields defined in the features schema

        The frames will be automatically processed to include both the original
        image format and the observation.images format (transposed to [C, H, W]).
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not created. Call create() first.")

        if not episode_data:
            self.logger.warning("Empty episode_data provided, skipping.")
            return
        for frame_data in episode_data:
            self.dataset.add_frame(frame_data)

        self.dataset.save_episode()
        self.logger.info(
            f"Saved episode with {len(episode_data)} frames, task: '{episode_data[0].get('task', 'N/A')}'"
        )

    def finalize(self) -> None:
        """Finalize the dataset and properly clean up all resources."""
        if self.dataset is None:
            raise RuntimeError("Dataset not created. Call create() first.")

        if (
            hasattr(self.dataset, "image_writer")
            and self.dataset.image_writer is not None
        ):
            self.dataset.image_writer.wait_until_done()

        if (
            hasattr(self.dataset, "image_writer")
            and self.dataset.image_writer is not None
        ):
            self.dataset.image_writer.stop()
            self.dataset.image_writer = None

        if hasattr(self.dataset, "episode_buffer"):
            self.dataset.episode_buffer = None

        if hasattr(self.dataset, "hf_dataset"):
            self.dataset.hf_dataset = None

        del self.dataset
        self.dataset = None
        gc.collect()
        self.logger.info("Dataset finalized.")
