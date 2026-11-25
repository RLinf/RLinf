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

"""Replay timestamped Go2 sensor frames with an ABC policy on CPU."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from rlinf.utils.logging import get_logger

from .control import Go2Control, JointState, Perception


def main() -> None:
    """Replay timestamped JSONL sensor frames without opening an SDK2 channel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list("ABC"))
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument(
        "--checkpoint",
        type=Path,
        help="RLinf actor weights or an RSL-RL Go2 checkpoint",
    )
    weights.add_argument("--group", choices=["baseline", "ppo", "rlinf"])
    parser.add_argument(
        "--data",
        type=Path,
        default=os.environ.get(
            "GO2_ABC_DATA", Path(__file__).resolve().parents[3] / "data/go2_abc"
        ),
        help="Directory containing models/<task>/<group>.pt",
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames.resolve() == args.output.resolve():
        raise ValueError("Expected a separate output path for replay commands")
    controller = Go2Control(
        args.checkpoint or args.data / f"models/{args.task}/{args.group}.pt", args.task
    )
    count = 0
    if args.output.exists():
        raise FileExistsError(f"Replay output already exists: {args.output}")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / args.output.name
        with args.frames.open() as frames, staged.open("x") as output:
            for line in frames:
                frame = json.loads(line)
                result = controller.step(
                    JointState(**frame["state"]),
                    Perception(**frame["perception"]),
                    frame["command"],
                    frame["time"],
                )
                output.write(
                    json.dumps(
                        {
                            "time": frame["time"],
                            "target_q_sdk": result["target_q_sdk"].tolist(),
                            "torque_sdk": result["torque_sdk"].tolist(),
                        }
                    )
                    + "\n"
                )
                count += 1
        if not count:
            raise ValueError("Expected at least one sensor frame")
        args.output.hardlink_to(staged)
    get_logger().info("Replayed %s frames on CPU; SDK2 channels opened: 0", count)


if __name__ == "__main__":
    main()
