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

"""Bridge to load LeRobot-format datasets into RLinf's TrajectoryReplayBuffer.

This module converts data written by :class:`~rlinf.envs.wrappers.collect_episode.CollectEpisode`
(in LeRobot format via :class:`~rlinf.data.lerobot_writer.LeRobotDatasetWriter`) into
:class:`~rlinf.data.embodied_io_struct.Trajectory` objects compatible with
:class:`~rlinf.data.replay_buffer.TrajectoryReplayBuffer`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer

logger = logging.getLogger(__name__)


def load_lerobot_dataset_to_demo_buffer(
    dataset_path: str,
    demo_buffer: TrajectoryReplayBuffer,
    state_dim: int,
    action_dim: int,
    *,
    episodes: Optional[list[int]] = None,
    max_episode_length: int = 10000,
) -> int:
    """Load a LeRobot-format dataset into a demo :class:`TrajectoryReplayBuffer`.

    Reads data written by RLinf's ``CollectEpisode`` wrapper (LeRobot format)
    and converts each episode into a :class:`Trajectory` object, then adds
    them to *demo_buffer*.

    The LeRobot dataset must have been written with RLinf's
    :class:`~rlinf.data.lerobot_writer.LeRobotDatasetWriter`, which uses
    flat feature names: ``state``, ``actions``, ``image``, ``done``,
    ``is_success``, ``intervene_flag``, ``task``.

    Feature mapping to RLinf ``Trajectory`` format:

    =====================  ==========================================
    LeRobot feature         Trajectory field
    =====================  ==========================================
    ``state``              ``curr_obs["states"]`` / ``next_obs["states"]``
    ``actions``            ``forward_inputs["action"]``, ``actions``
    ``image``              ``curr_obs["main_images"]`` / ``next_obs["main_images"]``
    ``done``               ``dones``, ``terminations``
    ``intervene_flag``     ``intervene_flags``
    =====================  ==========================================

    Args:
        dataset_path: Path to the LeRobot dataset directory (must contain
            ``meta/info.json``).
        demo_buffer: Target buffer to add trajectories to.
        state_dim: Expected state vector dimension.
        action_dim: Expected action vector dimension.
        episodes: Optional list of episode indices to load.  If ``None``,
            all episodes are loaded.
        max_episode_length: Soft upper bound for episode truncation.

    Returns:
        Total number of frames loaded.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        dataset_path,
        episodes=episodes,
    )

    if dataset.num_episodes == 0:
        logger.warning("LeRobot dataset at %s has 0 episodes.", dataset_path)
        return 0

    total_frames = 0
    for ep_idx in range(dataset.num_episodes):
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()
        ep_len = ep_end - ep_start

        if ep_len == 0:
            continue

        # Gather all frames for this episode.
        ep_hf = dataset.hf_dataset.select(range(ep_start, ep_end))

        states_list = []
        actions_list = []
        images_list = []
        dones_list = []
        intervene_flags_list = []
        has_image = False

        for frame in ep_hf:
            # State.
            s = frame.get("state")
            if s is None:
                continue
            s = np.asarray(s, dtype=np.float32)
            if s.shape[-1] != state_dim:
                raise ValueError(
                    f"Expected state_dim={state_dim}, got {s.shape[-1]} "
                    f"in episode {ep_idx} of {dataset_path}"
                )
            states_list.append(s)

            # Action.
            a = frame.get("actions")
            if a is None:
                continue
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            if a.shape[-1] != action_dim:
                raise ValueError(
                    f"Expected action_dim={action_dim}, got {a.shape[-1]} "
                    f"in episode {ep_idx} of {dataset_path}"
                )
            actions_list.append(a)

            # Image (optional).
            img = frame.get("image")
            if img is not None:
                has_image = True
                img = np.asarray(img, dtype=np.uint8)
            images_list.append(img)

            # Done / termination.
            done = frame.get("done")
            if done is not None:
                dones_list.append(bool(np.asarray(done).reshape(-1)[0]))
            else:
                dones_list.append(False)

            # Intervene flag.
            iflag = frame.get("intervene_flag")
            if iflag is not None:
                intervene_flags_list.append(
                    bool(np.asarray(iflag).reshape(-1)[0])
                )
            else:
                intervene_flags_list.append(True)

        if not states_list or not actions_list:
            continue

        # Convert to tensors.
        T = len(states_list)
        states_t = torch.from_numpy(np.stack(states_list, axis=0))  # [T, Ds]
        actions_t = torch.from_numpy(np.stack(actions_list, axis=0))  # [T, Da]
        dones_t = torch.tensor(dones_list, dtype=torch.bool)  # [T]
        intervene_t = torch.tensor(intervene_flags_list, dtype=torch.bool)  # [T]

        # Mark the last frame as done if not already.
        dones_t[-1] = True

        # Build curr_obs / next_obs dicts.
        # Use the same observation for curr and next (offline data).
        curr_obs = {"states": states_t}
        next_obs = {"states": states_t.clone()}
        if has_image:
            valid_images = [img for img in images_list if img is not None]
            if valid_images:
                images_t = torch.from_numpy(
                    np.stack(valid_images, axis=0)
                )  # [T, H, W, C]
                curr_obs["main_images"] = images_t
                next_obs["main_images"] = images_t.clone()

        # Build Trajectory.
        trajectory = Trajectory(
            max_episode_length=max_episode_length,
            actions=actions_t,
            rewards=torch.zeros(T, 1, dtype=torch.float32),
            terminations=dones_t.clone(),
            truncations=torch.zeros(T, dtype=torch.bool),
            dones=dones_t,
            intervene_flags=intervene_t,
            prev_logprobs=torch.zeros(T, 1, action_dim, dtype=torch.float32),
            prev_values=torch.zeros(T, 1, 1, dtype=torch.float32),
            forward_inputs={"action": actions_t},
            curr_obs=curr_obs,
            next_obs=next_obs,
        )

        demo_buffer.add_trajectories([trajectory])
        total_frames += T
        logger.debug(
            "Loaded episode %d: %d frames (total: %d)",
            ep_idx, T, total_frames,
        )

    logger.info(
        "Loaded %d frames across %d episodes from %s",
        total_frames, dataset.num_episodes, dataset_path,
    )
    return total_frames
