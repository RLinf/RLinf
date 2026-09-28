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
from types import MethodType
from typing import Any, Optional

import numpy as np

from rlinf.data.storage.lerobot.compat import add_frame_to_dataset
from rlinf.utils.logging import get_logger


def _save_image(dataset, image, fpath) -> None:
    """Write lossless RGB PNG in the caller's existing export thread."""
    import cv2

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Floating-point images must be in [0, 1].")
        image = (image * 255).astype(np.uint8)
    # PNG remains lossless; disabling compression avoids spending the control
    # machine's CPU on temporary images that are later embedded into parquet.
    if not cv2.imwrite(
        str(fpath),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    ):
        raise OSError(f"Failed to write recording image: {fpath}")


def _save_episode_table(dataset, episode_buffer: dict, episode_index: int) -> None:
    """Write LeRobot v2 rows in small batches without retaining past episodes."""
    import datasets
    import pyarrow.parquet as pq
    from datasets.table import embed_table_storage

    features = dataset.hf_features
    path = dataset.root / dataset.meta.get_data_file_path(ep_index=episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".parquet.tmp")
    try:
        with pq.ParquetWriter(temporary_path, features.arrow_schema) as writer:
            # Bound embedded image bytes, including incompressible RGB frames.
            for start in range(0, len(episode_buffer["index"]), 16):
                batch = datasets.Dataset.from_dict(
                    {key: episode_buffer[key][start : start + 16] for key in features},
                    features=features,
                )
                table = embed_table_storage(batch.data.table)
                writer.write_table(table)
                del table, batch
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _silence_hf_datasets_progress_bars() -> None:
    # Disable HF ``datasets`` Map / parquet tqdm bars; called from
    # ``create()`` so importing this module doesn't affect other consumers.
    try:
        import datasets as _hf_datasets

        _hf_datasets.disable_progress_bar()
    except ImportError:
        pass


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
        has_segment_id: bool = False,
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
                ``{"wrist_image-0": …, "wrist_image-1": …, …}``.
            extra_view_image_keys: Same as *wrist_image_keys* but for the
                extra-view camera(s).
            has_intervene_flag: Whether to include per-frame human-intervention
                flag (bool, shape ``(1,)``) in auto-generated features.
            has_segment_id: Whether to include per-frame ``segment_id``
                (uint8, shape ``(1,)``) in auto-generated features. Used for
                in-episode sub-task boundaries set by KeyboardStartEndWrapper.

        """

        try:  # lerobot >= 0.2 layout
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError:  # lerobot < 0.2
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        _silence_hf_datasets_progress_bars()

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
            if has_segment_id:
                features["segment_id"] = {
                    "dtype": "uint8",
                    "shape": (1,),
                    "names": ["segment_id"],
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
        if hasattr(self.dataset, "_save_episode_table"):
            # LeRobot v2 normally concatenates all embedded episodes in RAM.
            # This instance only records; keep upstream metadata/stats handling
            # but replace its table writer. Other dataset instances are untouched.
            #
            # These are LeRobot-internal hooks, pinned by LEROBOT_COMMIT in
            # requirements/install.sh. Re-verify them (and the guards below)
            # whenever that pin moves.
            self.dataset._save_episode_table = MethodType(
                _save_episode_table, self.dataset
            )
            self.dataset.hf_dataset = None
            if image_writer_threads == 0 and image_writer_processes == 0:
                self.dataset._save_image = MethodType(_save_image, self.dataset)

    def add_frame(self, frame_data: dict[str, Any]) -> None:
        """Stream one frame into the in-progress episode buffer.

        Images are written to disk immediately by the dataset's async image
        writer; only small per-frame fields stay in memory. Pair with
        :meth:`save_episode` to close out the episode.

        Args:
            frame_data: Same per-frame dict as the entries of the list taken
                by :meth:`add_episode`.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not created. Call create() first.")
        add_frame_to_dataset(self.dataset, frame_data)
        if self.dataset.image_writer is not None:
            # The upstream image queue is unbounded. Drain this frame's views
            # before accepting another; CollectEpisode supplies bounded async I/O.
            self.dataset.image_writer.wait_until_done()

    def save_episode(self, is_success: Optional[bool] = None) -> int:
        """Save the episode accumulated through :meth:`add_frame` calls.

        Args:
            is_success: When given, overwrite every frame's ``is_success``
                entry with this episode-level value and mark the last frame
                ``done``. This lets streaming callers stamp the outcome that
                only becomes known after the final frame was added.

        Returns:
            The number of frames saved; ``0`` when no episode was in progress.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not created. Call create() first.")
        episode_buffer = getattr(self.dataset, "episode_buffer", None)
        if not episode_buffer or not episode_buffer.get("size"):
            self.logger.warning("save_episode called with no frames; skipping.")
            return 0
        size = episode_buffer["size"]
        if is_success is not None:
            if "is_success" in episode_buffer:
                episode_buffer["is_success"] = [
                    np.array([is_success], dtype=bool) for _ in range(size)
                ]
            if "done" in episode_buffer:
                episode_buffer["done"][-1] = np.array([True], dtype=bool)
        self.dataset.save_episode()
        self.logger.info(f"Saved streaming episode with {size} frames.")
        return size

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
            add_frame_to_dataset(self.dataset, frame_data)

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
