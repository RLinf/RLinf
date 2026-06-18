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

Reads data written by :class:`~rlinf.envs.wrappers.collect_episode.CollectEpisode`
(in LeRobot format via :class:`~rlinf.data.lerobot_writer.LeRobotDatasetWriter`)
and converts each episode into a :class:`~rlinf.data.embodied_io_struct.Trajectory`
that is appended to a :class:`~rlinf.data.replay_buffer.TrajectoryReplayBuffer`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer

logger = logging.getLogger(__name__)


def _import_lerobot_dataset():
    """Import :class:`LeRobotDataset` from whichever path the installed lerobot exposes.

    LeRobot moved the package layout from ``lerobot.common.datasets`` (legacy)
    to ``lerobot.datasets`` (>= ccfd609e / SO-arms refactor). We support both
    so the bridge works regardless of which lerobot pin the user installed.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

        return LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

        return LeRobotDataset


def _open_lerobot_dataset(dataset_path: str, episodes: Optional[list[int]]):
    """Open a local LeRobot dataset by directory path.

    Picks a placeholder ``repo_id`` so the loader does not attempt a Hub
    lookup; the real metadata comes from ``{dataset_path}/meta/info.json``.
    """
    LeRobotDataset = _import_lerobot_dataset()
    root = Path(dataset_path)
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"{root}/meta/info.json is missing — pass a directory written by "
            "LeRobotDatasetWriter / CollectEpisode."
        )
    repo_id = f"local/{root.name}"
    try:
        return LeRobotDataset(repo_id, root=root, episodes=episodes)
    except TypeError:
        # Fallback for legacy lerobot where the constructor accepts a path
        # as the first positional argument.
        return LeRobotDataset(str(root), episodes=episodes)


def _episode_indices(dataset, ep_idx: int):
    """Return the row indices (inside ``hf_dataset``) that belong to ``ep_idx``.

    Handles both the legacy ``episode_data_index["from"/"to"]`` API and the
    v0.5.x API where episode boundaries are derived from the ``episode_index``
    column.
    """
    legacy = getattr(dataset, "episode_data_index", None)
    if legacy is not None and "from" in legacy and "to" in legacy:
        return range(int(legacy["from"][ep_idx]), int(legacy["to"][ep_idx]))
    column = np.asarray(dataset.hf_dataset["episode_index"])
    return np.flatnonzero(column == ep_idx).tolist()


def load_lerobot_dataset_to_demo_buffer(
    dataset_path: str,
    demo_buffer: TrajectoryReplayBuffer,
    state_dim: int,
    action_dim: int,
    *,
    episodes: Optional[list[int]] = None,
    max_episode_length: int = 10000,
) -> int:
    """Load a LeRobot-format dataset into ``demo_buffer``.

    The dataset must use the flat feature layout written by
    :class:`~rlinf.data.lerobot_writer.LeRobotDatasetWriter` (``state``,
    ``actions``, optional ``image``, ``done``, ``intervene_flag``).

    Returns:
        Total number of frames added across all loaded episodes.
    """
    dataset = _open_lerobot_dataset(dataset_path, episodes)
    if dataset.num_episodes == 0:
        logger.warning("LeRobot dataset at %s has 0 episodes.", dataset_path)
        return 0

    total_frames = 0
    for ep_idx in range(dataset.num_episodes):
        rows = _episode_indices(dataset, ep_idx)
        if not rows:
            continue
        ep_hf = dataset.hf_dataset.select(list(rows))

        states, actions, images, dones, interventions = [], [], [], [], []
        has_image = False

        for frame in ep_hf:
            s = frame.get("state")
            a = frame.get("actions")
            if s is None or a is None:
                continue
            s = np.asarray(s, dtype=np.float32)
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            if s.shape[-1] != state_dim:
                raise ValueError(
                    f"Expected state_dim={state_dim}, got {s.shape[-1]} "
                    f"in episode {ep_idx} of {dataset_path}"
                )
            if a.shape[-1] != action_dim:
                raise ValueError(
                    f"Expected action_dim={action_dim}, got {a.shape[-1]} "
                    f"in episode {ep_idx} of {dataset_path}"
                )
            states.append(s)
            actions.append(a)

            img = frame.get("image")
            if img is not None:
                has_image = True
                img = np.asarray(img, dtype=np.uint8)
            images.append(img)

            done = frame.get("done")
            dones.append(False if done is None else bool(np.asarray(done).reshape(-1)[0]))

            iflag = frame.get("intervene_flag")
            interventions.append(
                True if iflag is None else bool(np.asarray(iflag).reshape(-1)[0])
            )

        if not states:
            continue

        T = len(states)
        states_t = torch.from_numpy(np.stack(states, axis=0))
        actions_t = torch.from_numpy(np.stack(actions, axis=0))
        dones_t = torch.tensor(dones, dtype=torch.bool)
        dones_t[-1] = True
        intervene_t = torch.tensor(interventions, dtype=torch.bool)

        curr_obs: dict = {"states": states_t}
        next_obs: dict = {"states": states_t.clone()}
        if has_image:
            valid = [img for img in images if img is not None]
            if len(valid) == T:
                images_t = torch.from_numpy(np.stack(valid, axis=0))
                curr_obs["main_images"] = images_t
                next_obs["main_images"] = images_t.clone()

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

    logger.info(
        "Loaded %d frames across %d episodes from %s",
        total_frames,
        dataset.num_episodes,
        dataset_path,
    )
    return total_frames
