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

"""
Rollout data collector for saving rollout data to LeRobot format.

This collector is designed to be used in the environment's step function,
directly collecting data at each timestep to ensure continuous data.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from rlinf.data.lerobot_writer import LeRobotDatasetWriter

# Debug log directory (override via DATA_COLLECTOR_LOG_DIR env var)
DEBUG_LOG_DIR = os.environ.get(
    "DATA_COLLECTOR_LOG_DIR",
    os.path.join(os.getcwd(), "logs", "data_collector_logs"),
)
print(f"[EnvDataCollector] Log directory: {DEBUG_LOG_DIR}")
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
_debug_logger = logging.getLogger("EnvDataCollector")
_debug_logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler(
    os.path.join(DEBUG_LOG_DIR, "data_collector_debug.log")
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
_debug_logger.addHandler(_file_handler)
_debug_logger.info(f"Data collector debug log initialized. Log dir: {DEBUG_LOG_DIR}")


class RolloutEpisodeData:
    """Stores data for a single episode during rollout."""

    def __init__(self, env_id: int, task_description: str):
        self.env_id = env_id
        self.task_description = task_description
        self.images: list[np.ndarray] = []
        self.wrist_images: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.dones: list[bool] = []  # Record the done status at each step
        self.is_success: bool = False
        self._completed: bool = False
        # Record the step number of the first termination (1-indexed)
        # In libero, terminated=True means the success condition is reached, but the env continues running
        # When saving, only save up to the first terminated position
        self._first_terminated_step: Optional[int] = None

    def add_step(
        self,
        image: np.ndarray,
        wrist_image: Optional[np.ndarray],
        state: np.ndarray,
        action: np.ndarray,
        done: bool = False,
    ) -> None:
        """Add a single timestep of data."""
        self.images.append(image)
        if wrist_image is not None:
            self.wrist_images.append(wrist_image)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)

    def mark_first_terminated(self, step: int) -> None:
        """
        Record the step number of the first termination.

        In libero, terminated=True means the success condition is reached,
        but the environment continues running. When saving data, only save
        up to the first terminated position.

        Args:
            step: Current step number (1-indexed)
        """
        if self._first_terminated_step is None:
            self._first_terminated_step = step
            _debug_logger.info(
                f"[mark_first_terminated] env_id={self.env_id}, first_terminated_step={step}"
            )

    def mark_success(self, success: bool) -> None:
        """Mark the episode as successful or failed."""
        self.is_success = success

    def mark_completed(self) -> None:
        """Mark the episode as completed (done or truncated)."""
        self._completed = True

    @property
    def is_completed(self) -> bool:
        return self._completed

    @property
    def length(self) -> int:
        return len(self.images)

    def mark_last_step_done(self) -> None:
        """Mark the last step as done (episode boundary)."""
        if self.dones:
            self.dones[-1] = True

    def to_export_dict(self) -> dict[str, Any]:
        """
        Convert to export format for LeRobotDatasetWriter.

        If the episode succeeded (has first_terminated_step), only export up to the first terminated position.
        If the episode failed (no first_terminated_step), export all data.
        """
        if self.length == 0:
            return None

        # Determine the export data length
        # If succeeded (has first_terminated_step), only export up to the first termination
        # If failed, export all data
        end_step = (
            self._first_terminated_step
            if self._first_terminated_step is not None
            else self.length
        )

        # Truncate data
        images = self.images[:end_step]
        wrist_images = self.wrist_images[:end_step] if self.wrist_images else []
        states = self.states[:end_step]
        actions = self.actions[:end_step]
        dones = self.dones[:end_step]

        # Ensure the last step's done is True (marks episode boundary)
        if dones:
            dones[-1] = True

        return {
            "images": np.stack(images, axis=0),
            "wrist_images": np.stack(wrist_images, axis=0) if wrist_images else None,
            "states": np.stack(states, axis=0),
            "actions": np.stack(actions, axis=0),
            "dones": np.array(dones, dtype=bool),
            "task": self.task_description,
            "is_success": self.is_success,
        }


class EnvDataCollector:
    """
    Collects rollout data at the environment level for continuous timestep data.

    This collector is designed to be integrated into the environment's step function,
    collecting data at each timestep (not chunk level) to ensure data continuity.

    Key features:
    - Collects data at every step, not every chunk
    - Handles auto-reset scenarios
    - Tracks success based on termination signal
    - Supports incremental write to avoid OOM (optional)

    Usage in LiberoEnv:
        # In __init__:
        self.data_collector = EnvDataCollector(num_envs, save_dir, ...)

        # In step():
        self.data_collector.collect_step(
            env_id, image, wrist_image, state, action,
            is_terminated, is_truncated, task_description
        )

        # When needed:
        self.data_collector.save_to_lerobot()

    Incremental Write Mode:
        # Enable incremental write to avoid OOM with large datasets
        self.data_collector = EnvDataCollector(
            num_envs, save_dir,
            incremental_write_enabled=True,
            flush_interval=100,  # Write to disk every 100 episodes
        )
        # Episodes are automatically written to disk when flush_interval is reached
        # Memory is released after each flush
    """

    def __init__(
        self,
        num_envs: int,
        save_dir: Optional[str] = None,
        robot_type: str = "panda",
        fps: int = 10,
        only_successful: bool = False,
        # Incremental write parameters
        incremental_write_enabled: bool = True,  # Enable incremental write by default to avoid OOM
        flush_interval: int = 100,
        stats_sample_ratio: float = 0.1,
    ):
        """
        Initialize the environment data collector.

        Args:
            num_envs: Number of parallel environments
            save_dir: Directory to save LeRobot dataset (None to disable saving)
            robot_type: Robot type for LeRobot metadata
            fps: Frame rate for LeRobot metadata
            only_successful: If True, only save successful episodes
            incremental_write_enabled: If True, write episodes to disk incrementally
                                       to avoid OOM with large datasets
            flush_interval: Number of episodes to accumulate before writing to disk
                           (only used when incremental_write_enabled=True)
            stats_sample_ratio: Ratio of data to sample for statistics computation
                               (only used when incremental_write_enabled=True)
        """
        self.num_envs = num_envs
        self.save_dir = save_dir
        self.robot_type = robot_type
        self.fps = fps
        self.only_successful = only_successful

        # Incremental write parameters
        self._incremental_write_enabled = incremental_write_enabled
        self._flush_interval = flush_interval
        self._stats_sample_ratio = stats_sample_ratio

        # Current episode being collected for each env
        self._current_episodes: list[Optional[RolloutEpisodeData]] = [
            None for _ in range(num_envs)
        ]

        # Completed episodes ready for export
        self._completed_episodes: list[RolloutEpisodeData] = []

        # Track if collector is enabled
        self._enabled = save_dir is not None

        # Incremental write state
        self._writer: Optional["LeRobotDatasetWriter"] = None
        self._total_episodes_written = 0
        self._writer_initialized = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, save_dir: str) -> None:
        """Enable data collection with the specified save directory."""
        self.save_dir = save_dir
        self._enabled = True

    def disable(self) -> None:
        """Disable data collection."""
        self._enabled = False

    def start_episode(self, env_id: int, task_description: str) -> None:
        """Start a new episode for the given environment."""
        if not self._enabled:
            return

        # If there's an existing incomplete episode, finalize it as failure
        if self._current_episodes[env_id] is not None:
            _debug_logger.warning(
                f"[start_episode] env_id={env_id} has incomplete episode, finalizing as failure"
            )
            self._finalize_episode(env_id, is_success=False)

        self._current_episodes[env_id] = RolloutEpisodeData(
            env_id=env_id,
            task_description=task_description,
        )
        _debug_logger.info(
            f"[start_episode] env_id={env_id}, task='{task_description}'"
        )

    def collect_step(
        self,
        env_id: int,
        image: np.ndarray,
        wrist_image: Optional[np.ndarray],
        state: np.ndarray,
        action: np.ndarray,
        is_terminated: bool,
        is_truncated: bool,
        task_description: str,
    ) -> None:
        """
        Collect data from a single step for a single environment.

        This should be called in the environment's step() function for each env.

        Args:
            env_id: Environment index
            image: Main camera image [H, W, C]
            wrist_image: Wrist camera image [H, W, C] or None
            state: State vector [state_dim]
            action: Action vector [action_dim]
            is_terminated: Whether the episode terminated (success in libero)
            is_truncated: Whether the episode was truncated (timeout)
            task_description: Task description string
        """
        if not self._enabled:
            return

        # In libero:
        # - terminated=True means the success condition is reached, but the env continues running until max_episode_length
        # - truncated=True means max_episode_length is reached, the episode truly ends
        # - terminated=True may occur multiple times within a single episode
        # - When saving data, only save up to the first terminated position

        # Start episode if not already started
        if self._current_episodes[env_id] is None:
            # If truncated=True but no active episode exists, this is a residual signal, skip it
            if is_truncated:
                _debug_logger.debug(
                    f"[collect_step] Skipping residual truncated signal for env_id={env_id}"
                )
                return
            self.start_episode(env_id, task_description)

        episode = self._current_episodes[env_id]

        # Process and add step data
        processed_image = self._process_image(image)
        processed_wrist = (
            self._process_image(wrist_image) if wrist_image is not None else None
        )
        processed_state = state.astype(np.float32) if state is not None else None
        processed_action = action.astype(np.float32) if action is not None else None

        if (
            processed_image is not None
            and processed_state is not None
            and processed_action is not None
        ):
            episode.add_step(
                processed_image,
                processed_wrist,
                processed_state,
                processed_action,
                done=is_truncated,
            )
            # Log detailed info every 50 steps, or when truncated
            if episode.length % 50 == 0 or is_truncated:
                _debug_logger.debug(
                    f"[collect_step] env_id={env_id}, step={episode.length}, "
                    f"image_shape={processed_image.shape}, "
                    f"wrist_shape={processed_wrist.shape if processed_wrist is not None else None}, "
                    f"state_shape={processed_state.shape}, action_shape={processed_action.shape}, "
                    f"terminated={is_terminated}, truncated={is_truncated}, "
                    f"first_terminated_step={episode._first_terminated_step}"
                )

        # Record the step number of the first termination (success condition)
        if is_terminated:
            episode.mark_first_terminated(episode.length)

        # Episode truly ends only when truncated (reaching max_episode_length)
        if is_truncated:
            # Success check: whether terminated has ever occurred
            is_success = episode._first_terminated_step is not None
            _debug_logger.info(
                f"[collect_step] Episode TRUNCATED! env_id={env_id}, total_length={episode.length}, "
                f"first_terminated_step={episode._first_terminated_step}, is_success={is_success}"
            )
            self._finalize_episode(env_id, is_success=is_success)

    def collect_step_batch(
        self,
        images: np.ndarray,
        wrist_images: Optional[np.ndarray],
        states: np.ndarray,
        actions: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        task_descriptions: list[str],
    ) -> None:
        """
        Collect data from a single step across all environments (batch version).

        Args:
            images: Main camera images [num_envs, H, W, C]
            wrist_images: Wrist camera images [num_envs, H, W, C] or None
            states: State vectors [num_envs, state_dim]
            actions: Action vectors [num_envs, action_dim]
            terminations: Termination flags [num_envs]
            truncations: Truncation flags [num_envs]
            task_descriptions: Task descriptions [num_envs]
        """
        if not self._enabled:
            return

        for env_id in range(self.num_envs):
            image = images[env_id] if images is not None else None
            wrist_image = wrist_images[env_id] if wrist_images is not None else None
            state = states[env_id] if states is not None else None
            action = actions[env_id] if actions is not None else None
            is_terminated = (
                bool(terminations[env_id]) if terminations is not None else False
            )
            is_truncated = (
                bool(truncations[env_id]) if truncations is not None else False
            )
            task_desc = (
                task_descriptions[env_id]
                if env_id < len(task_descriptions)
                else "unknown task"
            )

            self.collect_step(
                env_id=env_id,
                image=image,
                wrist_image=wrist_image,
                state=state,
                action=action,
                is_terminated=is_terminated,
                is_truncated=is_truncated,
                task_description=task_desc,
            )

    def on_reset(self, env_id: int, task_description: str) -> None:
        """
        Called when an environment resets.

        This starts a new episode for the environment.

        Args:
            env_id: Environment index
            task_description: Task description for the new episode
        """
        if not self._enabled:
            return

        # Start a new episode
        self.start_episode(env_id, task_description)

    def on_reset_batch(self, env_ids: np.ndarray, task_descriptions: list[str]) -> None:
        """
        Called when multiple environments reset (batch version).

        Args:
            env_ids: Array of environment indices that reset
            task_descriptions: Task descriptions for all environments
        """
        if not self._enabled:
            return

        for env_id in env_ids:
            task_desc = (
                task_descriptions[env_id]
                if env_id < len(task_descriptions)
                else "unknown task"
            )
            self.on_reset(env_id, task_desc)

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """Process image to uint8 format."""
        if image is None:
            return None

        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return image

    def _finalize_episode(self, env_id: int, is_success: bool) -> None:
        """Finalize the current episode for the given environment."""
        episode = self._current_episodes[env_id]
        if episode is None or episode.length == 0:
            _debug_logger.warning(
                f"[_finalize_episode] env_id={env_id} has no data, skipping"
            )
            self._current_episodes[env_id] = None
            return

        episode.mark_success(is_success)
        episode.mark_completed()
        self._completed_episodes.append(episode)

        # Calculate the actual export length (truncate to first termination on success, export all on failure)
        export_length = (
            episode._first_terminated_step
            if episode._first_terminated_step is not None
            else episode.length
        )
        _debug_logger.info(
            f"[_finalize_episode] env_id={env_id}, total_length={episode.length}, "
            f"export_length={export_length}, first_terminated_step={episode._first_terminated_step}, "
            f"is_success={is_success}, task='{episode.task_description}'"
        )
        self._current_episodes[env_id] = None

        # Check if incremental write is needed
        self._maybe_flush_episodes()

    # ==================== Incremental write methods ====================

    def _maybe_flush_episodes(self) -> int:
        """Check if flush_interval is reached; if so, write to disk."""
        if not self._incremental_write_enabled:
            return 0
        if len(self._completed_episodes) < self._flush_interval:
            return 0
        return self._flush_episodes()

    def _flush_episodes(self) -> int:
        """Write accumulated episodes to disk and release memory."""
        if not self._completed_episodes:
            return 0

        # Lazily initialize writer
        if self._writer is None:
            self._init_writer()

        if self._writer is None:
            _debug_logger.warning(
                "[_flush_episodes] Writer not initialized, skipping flush"
            )
            return 0

        num_written = 0
        for episode in self._completed_episodes:
            ep_data = episode.to_export_dict()
            if ep_data is None:
                continue
            if self.only_successful and not ep_data["is_success"]:
                continue

            self._writer.add_episode(
                images=ep_data["images"],
                wrist_images=ep_data["wrist_images"],
                states=ep_data["states"],
                actions=ep_data["actions"],
                task=ep_data["task"],
                is_success=ep_data["is_success"],
                dones=ep_data["dones"],
            )
            num_written += 1

        self._total_episodes_written += num_written
        _debug_logger.info(
            f"[_flush_episodes] Flushed {num_written} episodes to disk. "
            f"Total written: {self._total_episodes_written}"
        )

        # Release memory!
        self._completed_episodes.clear()
        return num_written

    def _init_writer(self) -> None:
        """Lazily initialize the LeRobotDatasetWriter."""
        if self._writer_initialized:
            return

        if self.save_dir is None:
            _debug_logger.warning(
                "[_init_writer] No save_dir specified, cannot initialize writer"
            )
            return

        # Need to get data dimensions from the first episode
        if not self._completed_episodes:
            _debug_logger.warning(
                "[_init_writer] No completed episodes to determine dimensions"
            )
            return

        first_ep = self._completed_episodes[0].to_export_dict()
        if first_ep is None:
            _debug_logger.warning("[_init_writer] First episode has no data")
            return

        image_shape = first_ep["images"].shape[1:]  # [H, W, C]
        state_dim = first_ep["states"].shape[1]
        action_dim = first_ep["actions"].shape[1]

        _debug_logger.info(
            f"[_init_writer] Initializing writer with: "
            f"image_shape={image_shape}, state_dim={state_dim}, action_dim={action_dim}, "
            f"use_incremental_stats=True, stats_sample_ratio={self._stats_sample_ratio}"
        )

        from rlinf.data.lerobot_writer import LeRobotDatasetWriter

        self._writer = LeRobotDatasetWriter(
            root_dir=self.save_dir,
            robot_type=self.robot_type,
            fps=self.fps,
            image_shape=image_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            use_incremental_stats=True,  # Use incremental statistics
            stats_sample_ratio=self._stats_sample_ratio,
        )
        self._writer_initialized = True

    def finalize_all(self) -> None:
        """Finalize all current episodes (call at end of rollout)."""
        if not self._enabled:
            return

        _debug_logger.info("[finalize_all] Finalizing all incomplete episodes...")
        for env_id in range(self.num_envs):
            if self._current_episodes[env_id] is not None:
                ep_len = self._current_episodes[env_id].length
                _debug_logger.warning(
                    f"[finalize_all] env_id={env_id} has incomplete episode with {ep_len} steps, marking as failure"
                )
                # Mark incomplete episodes as failures
                self._finalize_episode(env_id, is_success=False)

    def save_to_lerobot(
        self,
        save_dir: Optional[str] = None,
        robot_type: Optional[str] = None,
        fps: Optional[int] = None,
        only_successful: Optional[bool] = None,
    ) -> int:
        """
        Save collected episodes to LeRobot format.

        Args:
            save_dir: Override save directory
            robot_type: Override robot type
            fps: Override fps
            only_successful: Override only_successful filter

        Returns:
            Number of episodes saved
        """
        save_dir = save_dir or self.save_dir
        robot_type = robot_type or self.robot_type
        fps = fps or self.fps
        only_successful = (
            only_successful if only_successful is not None else self.only_successful
        )

        if save_dir is None:
            _debug_logger.warning(
                "[save_to_lerobot] No save_dir specified, skipping save"
            )
            return 0

        # Critical fix: update self.save_dir to ensure incremental write uses the correct directory
        # This is important in distributed settings, as env_worker passes a save_dir with rank suffix
        if save_dir != self.save_dir:
            _debug_logger.info(
                f"[save_to_lerobot] Updating save_dir from '{self.save_dir}' to '{save_dir}'"
            )
            self.save_dir = save_dir
            # If writer was already initialized with a different directory, it needs to be reset
            if self._writer is not None and self._writer_initialized:
                _debug_logger.warning(
                    "[save_to_lerobot] Writer was initialized with different save_dir, "
                    "this may cause issues in distributed mode"
                )

        _debug_logger.info(f"[save_to_lerobot] Starting save to {save_dir}")
        _debug_logger.info(
            f"[save_to_lerobot] Total completed episodes in memory: {len(self._completed_episodes)}"
        )
        _debug_logger.info(
            f"[save_to_lerobot] Incremental write enabled: {self._incremental_write_enabled}"
        )
        _debug_logger.info(
            f"[save_to_lerobot] Total episodes already written: {self._total_episodes_written}"
        )
        _debug_logger.info(
            f"[save_to_lerobot] Writer initialized: {self._writer_initialized}, Writer is not None: {self._writer is not None}"
        )

        # Finalize any remaining episodes
        self.finalize_all()

        if self._incremental_write_enabled:
            # Incremental write mode: write remaining episodes and finalize
            _debug_logger.info(
                f"[save_to_lerobot] After finalize_all, episodes in memory: {len(self._completed_episodes)}"
            )
            num_remaining = self._flush_episodes()
            _debug_logger.info(
                f"[save_to_lerobot] After flush_episodes: num_remaining={num_remaining}, writer is not None: {self._writer is not None}"
            )

            if self._writer is not None:
                _debug_logger.info(
                    "[save_to_lerobot] Calling writer.finalize() to generate meta files"
                )
                self._writer.finalize()
                _debug_logger.info(
                    "[save_to_lerobot] writer.finalize() completed successfully"
                )
            else:
                _debug_logger.warning(
                    f"[save_to_lerobot] Writer is None, cannot finalize! total_episodes_written={self._total_episodes_written}"
                )

            total_saved = self._total_episodes_written
            _debug_logger.info(
                f"[save_to_lerobot] Incremental mode: flushed {num_remaining} remaining episodes. "
                f"Total saved: {total_saved}"
            )
            return total_saved

        # Original batch write logic
        if not self._completed_episodes:
            _debug_logger.warning("[save_to_lerobot] No episodes to save")
            return 0

        from rlinf.data.lerobot_writer import LeRobotDatasetWriter

        # Determine dimensions from first episode
        first_ep = self._completed_episodes[0].to_export_dict()
        if first_ep is None:
            _debug_logger.warning("[save_to_lerobot] First episode has no data")
            return 0

        image_shape = first_ep["images"].shape[1:]  # [H, W, C]
        state_dim = first_ep["states"].shape[1]
        action_dim = first_ep["actions"].shape[1]

        _debug_logger.info(
            f"[save_to_lerobot] Data dimensions: image_shape={image_shape}, state_dim={state_dim}, action_dim={action_dim}"
        )
        _debug_logger.info(
            f"[save_to_lerobot] First episode keys: {list(first_ep.keys())}"
        )

        writer = LeRobotDatasetWriter(
            root_dir=save_dir,
            robot_type=robot_type,
            fps=fps,
            image_shape=image_shape,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        num_saved = 0
        num_skipped = 0
        for ep_idx, episode in enumerate(self._completed_episodes):
            ep_data = episode.to_export_dict()
            if ep_data is None:
                _debug_logger.warning(
                    f"[save_to_lerobot] Episode {ep_idx} has no data, skipping"
                )
                continue

            # Filter by success if requested
            if only_successful and not ep_data["is_success"]:
                num_skipped += 1
                continue

            # Log detailed info for each episode
            _debug_logger.info(
                f"[save_to_lerobot] Saving episode {ep_idx}: "
                f"length={len(ep_data['images'])}, "
                f"images_shape={ep_data['images'].shape}, "
                f"states_shape={ep_data['states'].shape}, "
                f"actions_shape={ep_data['actions'].shape}, "
                f"dones_shape={ep_data['dones'].shape}, "
                f"is_success={ep_data['is_success']}, "
                f"task='{ep_data['task']}'"
            )

            # Check dones values
            dones = ep_data["dones"]
            num_true_dones = np.sum(dones)
            _debug_logger.info(
                f"[save_to_lerobot] Episode {ep_idx} dones check: "
                f"total_steps={len(dones)}, num_true={num_true_dones}, "
                f"last_done={dones[-1]}, "
                f"first_5_dones={dones[:5].tolist()}, last_5_dones={dones[-5:].tolist()}"
            )

            writer.add_episode(
                images=ep_data["images"],
                wrist_images=ep_data["wrist_images"],
                states=ep_data["states"],
                actions=ep_data["actions"],
                task=ep_data["task"],
                is_success=ep_data["is_success"],
                dones=ep_data["dones"],  # Per-step done flags
            )
            num_saved += 1

        writer.finalize()
        _debug_logger.info(
            f"[save_to_lerobot] Saved {num_saved} episodes to {save_dir}, skipped {num_skipped} (only_successful={only_successful})"
        )
        return num_saved

    def clear(self) -> None:
        """Clear all collected data."""
        self._current_episodes = [None for _ in range(self.num_envs)]
        self._completed_episodes = []

    @property
    def num_completed_episodes(self) -> int:
        return len(self._completed_episodes)

    @property
    def num_successful_episodes(self) -> int:
        return sum(1 for ep in self._completed_episodes if ep.is_success)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about collected data."""
        # Calculate pending episodes in memory
        pending_episodes = len(self._completed_episodes)
        pending_successful = sum(1 for ep in self._completed_episodes if ep.is_success)
        pending_frames = sum(ep.length for ep in self._completed_episodes)

        # Incremental write mode: total = already written to disk + pending in memory
        if self._incremental_write_enabled:
            total_episodes = self._total_episodes_written + pending_episodes
            # Note: we don't separately track success count for episodes written to disk; only count pending
            total_successful = pending_successful  # Rough estimate
            total_frames = pending_frames  # Rough estimate
        else:
            total_episodes = pending_episodes
            total_successful = pending_successful
            total_frames = pending_frames

        stats = {
            "num_completed_episodes": total_episodes,  # Fix: includes episodes already written to disk
            "num_successful_episodes": total_successful,
            "success_rate": (total_successful / max(1, total_episodes)),
            "total_frames": total_frames,
        }

        # Additional statistics for incremental write mode
        if self._incremental_write_enabled:
            stats["incremental_write_enabled"] = True
            stats["flush_interval"] = self._flush_interval
            stats["total_episodes_written"] = self._total_episodes_written
            stats["episodes_pending_flush"] = pending_episodes
            # Flag whether finalize is needed (has written data or pending data)
            stats["needs_finalize"] = self._writer_initialized or pending_episodes > 0

        return stats


# Alias for backward compatibility
RolloutDataCollector = EnvDataCollector
