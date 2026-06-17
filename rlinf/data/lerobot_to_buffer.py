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

The default codepath reads parquet via pandas and does **not** require the
LeRobot package to be installed — it works on any machine with pandas and
pyarrow.  If LeRobot IS available, its :class:`~lerobot.datasets.lerobot_dataset.LeRobotDataset`
is used as a fallback for legacy dataset formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer

logger = logging.getLogger(__name__)


def _read_parquet_direct(root: Path):
    """Read a LeRobot dataset via plain pandas — no LeRobot import needed.

    Returns (df, info_dict) where *df* contains the full frame table with an
    ``episode_index`` column, and *info* is the parsed ``meta/info.json``.
    """
    pq_path = root / "data" / "chunk-000" / "file-000.parquet"
    info_path = root / "meta" / "info.json"

    if not pq_path.exists():
        raise FileNotFoundError(f"No parquet at {pq_path}")

    df = pd.read_parquet(pq_path)
    info = json.loads(info_path.read_text())
    return df, info


def _decode_image_column(col) -> np.ndarray | None:
    """Decode a single parquet cell from an ``dtype: image`` column.

    LeRobot stores images as ``{"bytes": <png bytes>, "path": <str>}`` dicts
    inside the parquet.  Returns the decoded (H, W, 3) uint8 array, or
    ``None`` if the column is empty / unreadable.
    """
    try:
        import cv2
    except ImportError:
        return None

    # Take the first non-None entry to infer whether the column stores images.
    item = None
    for v in col:
        if v is not None:
            item = v
            break
    if item is None:
        return None

    frames = []
    for v in col:
        if v is None:
            frames.append(None)
            continue
        if isinstance(v, dict) and "bytes" in v:
            buf = np.frombuffer(v["bytes"], np.uint8)
        elif isinstance(v, bytes):
            buf = np.frombuffer(v, np.uint8)
        else:
            frames.append(None)
            continue
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.stack(frames) if all(f is not None for f in frames) else None


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
    root = Path(dataset_path)

    # If the path is a parent directory containing multiple run_* subdirs
    # (each a standalone LeRobot dataset), load them all.
    if not (root / "meta" / "info.json").exists():
        run_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and d.name.startswith("run_") and (d / "meta" / "info.json").exists()
        )
        if not run_dirs:
            raise FileNotFoundError(
                f"{root}/meta/info.json is missing and no run_* subdirectories "
                "with meta/info.json were found — pass a directory written by "
                "LeRobotDatasetWriter / CollectEpisode."
            )
        total = 0
        for run_dir in run_dirs:
            total += load_lerobot_dataset_to_demo_buffer(
                str(run_dir), demo_buffer, state_dim, action_dim,
                episodes=episodes, max_episode_length=max_episode_length,
            )
        return total

    # Try the native LeRobot loader first; fall back to direct pandas.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(str(root), root=root, episodes=episodes)
        df = pd.DataFrame(dataset.hf_dataset[:])  # type: ignore[arg-type]
    except Exception:
        df, _ = _read_parquet_direct(root)

    if "episode_index" not in df.columns:
        logger.warning("No episode_index column in %s — treating as single episode.", root)
        df["episode_index"] = 0

    ep_ids = sorted(set(int(e) for e in df["episode_index"]))
    if episodes is not None:
        ep_ids = [e for e in ep_ids if e in episodes]

    total_frames = 0
    has_image_bar = "image" in df.columns

    for ep_id in ep_ids:
        mask = df["episode_index"] == ep_id
        ep_df = df[mask]
        if len(ep_df) == 0:
            continue

        states_list, actions_list, dones_list, interventions_list = [], [], [], []
        for _, row in ep_df.iterrows():
            s = row.get("state")
            a = row.get("actions")
            if s is None or a is None:
                continue
            states_list.append(np.asarray(s, dtype=np.float32).reshape(-1))
            actions_list.append(np.asarray(a, dtype=np.float32).reshape(-1))
            d = row.get("done")
            dones_list.append(False if d is None else bool(np.asarray(d).flat[0]))
            iflag = row.get("intervene_flag")
            interventions_list.append(
                True if iflag is None else bool(np.asarray(iflag).flat[0])
            )

        if not states_list:
            continue

        T = len(states_list)
        states_t = torch.from_numpy(np.stack(states_list))
        actions_t = torch.from_numpy(np.stack(actions_list))
        dones_t = torch.tensor(dones_list, dtype=torch.bool)
        dones_t[-1] = True
        intervene_t = torch.tensor(interventions_list, dtype=torch.bool)

        curr_obs: dict = {"states": states_t}
        next_obs: dict = {"states": states_t.clone()}

        if has_image_bar:
            images = _decode_image_column(ep_df["image"])
            if images is not None and len(images) == T:
                images_t = torch.from_numpy(images)
                curr_obs["main_images"] = images_t
                next_obs["main_images"] = images_t.clone()

        fwd_inputs = {
            "action": actions_t,
            "states": curr_obs["states"],
        }
        if "main_images" in curr_obs:
            fwd_inputs["main_images"] = curr_obs["main_images"]

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
            forward_inputs=fwd_inputs,
            curr_obs=curr_obs,
            next_obs=next_obs,
        )
        demo_buffer.add_trajectories([trajectory])
        total_frames += T

    logger.info(
        "Loaded %d frames across %d episodes from %s",
        total_frames, len(ep_ids), dataset_path,
    )
    return total_frames
