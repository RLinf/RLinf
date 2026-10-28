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

"""Check metrics and exported trajectories from the RoboLab evaluation runner."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def check_results(
    log_dir: Path, num_envs: int, rollout_epochs: int, max_episode_steps: int
) -> None:
    """Validate completed episodes against the runner's aggregated metrics."""
    import h5py
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}
    for event_file in log_dir.rglob("events.out.tfevents.*"):
        events = EventAccumulator(str(event_file))
        events.Reload()
        for tag in events.Tags()["scalars"]:
            if tag.startswith("eval/"):
                if tag in metrics:
                    raise ValueError(
                        f"Expected one evaluation per log directory: {tag}"
                    )
                metrics[tag] = events.Scalars(tag)[-1].value

    expected_episodes = num_envs * rollout_epochs
    if metrics.get("eval/num_trajectories") != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episodes, found metrics {metrics}"
        )
    for tag in ("eval/success_once", "eval/score"):
        value = metrics.get(tag, float("nan"))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Expected {tag} in [0, 1], found {value}")

    recordings = sorted((log_dir / "native").glob("run_*.hdf5"))
    if len(recordings) != rollout_epochs:
        raise ValueError(f"Expected {rollout_epochs} recordings, found {recordings}")
    episodes = []
    for recording in recordings:
        with h5py.File(recording) as data:
            expected_demos = {f"demo_{i}" for i in range(num_envs)}
            if set(data["data"]) != expected_demos:
                raise ValueError(f"Expected demos {expected_demos} in {recording}")
            for name, demo in data["data"].items():
                length = int(demo.attrs["num_samples"])
                actions = demo["actions"][:]
                if not 0 < length <= max_episode_steps:
                    raise ValueError(
                        f"Expected 1..{max_episode_steps} steps, got {length}"
                    )
                if actions.shape != (length, 8) or not np.isfinite(actions).all():
                    raise ValueError(
                        f"Expected finite ({length}, 8) actions: {recording}/{name}"
                    )
                success = bool(demo.attrs["success"])
                score = 1.0 if success else float(demo["subtask/score"][-1])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(
                        f"Expected normalized score, got {score}: {recording}/{name}"
                    )
                episodes.append({"length": length, "success": success, "score": score})

    for tag, field in (
        ("eval/success_once", "success"),
        ("eval/score", "score"),
        ("eval/episode_len", "length"),
    ):
        expected = mean(episode[field] for episode in episodes)
        actual = metrics.get(tag, float("nan"))
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"Expected {tag}={expected} from recordings, found {actual}"
            )
    (log_dir / "verification.json").write_text(
        json.dumps({"metrics": metrics, "episodes": episodes}, indent=2) + "\n"
    )


def main() -> None:
    """Validate the artifacts produced by one CI evaluation configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--rollout-epochs", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    args = parser.parse_args()
    check_results(
        args.log_dir, args.num_envs, args.rollout_epochs, args.max_episode_steps
    )


if __name__ == "__main__":
    main()
