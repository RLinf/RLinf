"""HDF5 episode recorder for DOSW1 human-in-the-loop data collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rlinf.utils.logging import get_logger

try:
    import h5py
except ImportError:
    h5py = None


class DOSW1DataRecorder:
    """Buffers episode data in memory and flushes to HDF5 on episode end."""

    def __init__(self, data_dir: str) -> None:
        self._logger = get_logger()
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._episode_idx = self._next_episode_idx()
        self._buffer: list[dict[str, Any]] = []
        self._in_episode = False

        if h5py is None:
            self._logger.warning(
                "[DOSW1DataRecorder] h5py not installed. "
                "Data will NOT be persisted. Install with: pip install h5py"
            )

    def start_episode(self) -> None:
        """Begin a new episode and clear the internal buffer."""
        self._buffer.clear()
        self._in_episode = True

    def record_step(
        self,
        obs: dict,
        action: np.ndarray,
        reward: float,
        control_mode: int,
        timestamp: float,
    ) -> None:
        """Append one transition to the current episode buffer."""
        if not self._in_episode:
            return
        self._buffer.append(
            {
                "obs": obs,
                "action": np.asarray(action, dtype=np.float64).copy(),
                "reward": float(reward),
                "control_mode": int(control_mode),
                "timestamp": float(timestamp),
            }
        )

    def end_episode(self) -> None:
        """Flush the buffered episode to disk and increment the episode index."""
        if not self._in_episode:
            return
        self._in_episode = False
        if not self._buffer:
            return
        self._flush()
        self._buffer.clear()
        self._episode_idx += 1

    def _flush(self) -> None:
        if h5py is None:
            return

        filename = self._data_dir / f"episode_{self._episode_idx:06d}.hdf5"
        n_steps = len(self._buffer)

        with h5py.File(str(filename), "w") as file:
            file.attrs["num_steps"] = n_steps
            file.create_dataset(
                "action",
                data=np.stack([step["action"] for step in self._buffer]),
            )
            file.create_dataset(
                "reward",
                data=np.array([step["reward"] for step in self._buffer], dtype=np.float64),
            )
            file.create_dataset(
                "control_mode",
                data=np.array(
                    [step["control_mode"] for step in self._buffer], dtype=np.int32
                ),
            )
            file.create_dataset(
                "timestamp",
                data=np.array(
                    [step["timestamp"] for step in self._buffer], dtype=np.float64
                ),
            )

            state_group = file.create_group("obs/state")
            first_state = self._buffer[0]["obs"]["state"]
            for key in first_state:
                state_group.create_dataset(
                    key,
                    data=np.stack([step["obs"]["state"][key] for step in self._buffer]),
                )

            first_frames = self._buffer[0]["obs"].get("frames", {})
            if first_frames:
                frames_group = file.create_group("obs/frames")
                for camera_name in first_frames:
                    frames_group.create_dataset(
                        camera_name,
                        data=np.stack(
                            [step["obs"]["frames"][camera_name] for step in self._buffer]
                        ),
                        compression="gzip",
                        compression_opts=4,
                    )

        self._logger.info(
            "[DOSW1DataRecorder] Saved episode %s (%s steps) -> %s",
            self._episode_idx,
            n_steps,
            filename,
        )

    def _next_episode_idx(self) -> int:
        existing = list(self._data_dir.glob("episode_*.hdf5"))
        if not existing:
            return 0

        indices: list[int] = []
        for path in existing:
            try:
                indices.append(int(path.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
        return max(indices) + 1 if indices else 0
