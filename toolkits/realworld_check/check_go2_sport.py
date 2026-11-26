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

"""Run one operator-confirmed direction, or read Go2 state without motion."""

import argparse
import json
import math
import os
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rlinf.utils.logging import get_logger

COMMANDS = {
    "forward": (0.3, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "left": (0.0, 0.3, 0.0),
    "right": (0.0, -0.3, 0.0),
    "turn_left": (0.0, 0.0, 0.5),
    "turn_right": (0.0, 0.0, -0.5),
}
logger = get_logger()


class SpeedLog:
    """Record SDK velocity estimates using host reception timestamps."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def callback(self, state: Any) -> None:
        """Ignore malformed or nonfinite SDK estimates."""
        try:
            velocity = [float(v) for v in state.velocity]
            yaw = float(state.yaw_speed)
            if len(velocity) != 3 or not all(map(math.isfinite, velocity + [yaw])):
                return
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        with self.lock:
            self.rows.append(
                {"t": time.monotonic(), "velocity": velocity, "yaw_speed": yaw}
            )

    def snapshot(self) -> list[dict[str, Any]]:
        """Return the received samples in time order."""
        with self.lock:
            return list(self.rows)

    def fresh(self) -> bool:
        """Require a sample received within 0.2 seconds."""
        rows = self.snapshot()
        return bool(rows) and time.monotonic() - rows[-1]["t"] <= 0.2

    def stopped(self, since: float) -> bool:
        """Check SDK estimates for 0.5 seconds; physical confirmation is separate."""
        rows = [r for r in self.snapshot() if r["t"] >= since]
        if len(rows) < 3 or time.monotonic() - rows[-1]["t"] > 0.2:
            return False
        start = previous = rows[-1]["t"]
        for row in reversed(rows):
            if previous - row["t"] > 0.2:
                return False
            previous = row["t"]
            if math.hypot(*row["velocity"][:2]) > 0.05 or abs(row["yaw_speed"]) > 0.1:
                return False
            if start - row["t"] >= 0.5:
                return True
        return False


def competing_processes() -> list[int]:
    """Find other local Go2 Python scripts; remote controllers are not visible."""
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [
                os.fsdecode(a) for a in (entry / "cmdline").read_bytes().split(b"\0")
            ]
            if not argv or "python" not in Path(argv[0]).name:
                continue
            scripts = [a for a in argv[1:] if a.endswith(".py")]
            modules = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "-m"]
            if argv[1:3] in (
                ["-m", "examples.embodiment.go2_abc.deploy"],
                ["-m", "toolkits.realworld_check.check_go2_lowstate"],
                ["-m", "toolkits.realworld_check.check_go2_camera"],
            ):
                continue
            if argv[1:3] == ["-m", "toolkits.realworld_check.check_go2_sport"] and (
                "--readonly" in argv or "--dry-run" in argv
            ):
                continue
            if any("go2" in Path(a).name.lower() for a in scripts + modules):
                matches.append(int(entry.name))
        except (OSError, ValueError):
            continue
    return matches


def run_check(interface: str, direction: str | None, seconds: int) -> dict[str, Any]:
    """Run a bounded check after the caller obtains operator confirmation.

    Args:
        interface: Network interface attached to the Go2.
        direction: One of COMMANDS, or None for state reception only.
        seconds: One or two seconds at ten steps per second.
    """
    if seconds not in (1, 2) or (direction is not None and direction not in COMMANDS):
        raise ValueError("Expected a named direction or None, and one or two seconds")

    import gymnasium as gym
    import numpy as np

    from rlinf.robotics import Go2Config, RobotDiscovery

    steps = seconds * 10
    report = {
        "interface": interface,
        "direction": direction,
        "command": COMMANDS[direction] if direction else None,
        "planned_seconds": seconds,
        "step_frequency_hz": 10,
        "uses_ppo": False,
        "operator_confirmed_motion_and_stop": None,
        "sdk_stop_estimate": "not_checked",
        "errors": [],
        "steps": [],
    }
    env = subscriber = None
    speeds = SpeedLog()
    try:
        cfg = Go2Config(
            node_rank=0, interface=interface, enable_motion=direction is not None
        )
        resource = RobotDiscovery.registry["UnitreeGo2"].discovery_cls.enumerate(
            0, [cfg]
        )
        env = gym.make(
            "rlinf.envs.real.go2:Go2DriveEnv-v1",
            override_cfg={"step_frequency": 10.0, "max_num_steps": steps},
            worker_info=None,
            robot_info=resource.infos[0],
            env_idx=0,
            env_cfg={},
        )
        if direction:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            subscriber.Init(speeds.callback, 10)
            deadline = time.monotonic() + 3
            while not speeds.fresh() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not speeds.fresh():
                raise TimeoutError("No fresh SportModeState; no Move will be sent")
        observation, _ = env.reset()
        for _ in range(steps):
            if direction:
                if not speeds.fresh():
                    raise TimeoutError("Speed feedback expired; requesting stop")
                observation, _, terminated, truncated, _ = env.step(
                    np.asarray(COMMANDS[direction], dtype=np.float32)
                )
            else:
                observation, _ = env.reset()
                time.sleep(0.1)
                terminated = truncated = False
            report["steps"].append(
                {
                    "t": time.monotonic(),
                    "observation_valid": env.observation_space.contains(observation),
                }
            )
            if not report["steps"][-1]["observation_valid"]:
                raise ValueError("Received state outside the Go2 observation space")
            if terminated or truncated:
                break
    except (Exception, KeyboardInterrupt) as exc:
        report["errors"].append(repr(exc))
    finally:
        try:
            if env is not None:
                env.close()
                report["close_completed"] = True
        except (Exception, KeyboardInterrupt) as exc:
            report["errors"].append("close: " + repr(exc))
        try:
            if direction and subscriber is not None:
                logger.warning(
                    "Stop requested. Use the remote if the robot is still moving."
                )
                since = time.monotonic()
                deadline = since + 2
                while not speeds.stopped(since) and time.monotonic() < deadline:
                    time.sleep(0.02)
                report["sdk_stop_estimate"] = (
                    "below_thresholds" if speeds.stopped(since) else "unconfirmed"
                )
        except (Exception, KeyboardInterrupt) as exc:
            report["errors"].append("stop observation: " + repr(exc))
        finally:
            if subscriber is not None:
                try:
                    subscriber.Close()
                except Exception as exc:
                    report["errors"].append("subscriber close: " + repr(exc))
            report["speeds"] = speeds.snapshot()
    return report


def main() -> None:
    """Preview by flag, or obtain confirmation before opening SDK channels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--direction", choices=COMMANDS)
    mode.add_argument("--readonly", action="store_true")
    parser.add_argument("--seconds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = {
        "interface": args.interface,
        "direction": args.direction,
        "command": COMMANDS.get(args.direction),
        "seconds": args.seconds,
        "hardware_access": False,
    }
    logger.info("Check plan: %s", json.dumps(plan))
    if args.dry_run:
        return
    if args.direction:
        conflicts = competing_processes()
        if conflicts:
            raise RuntimeError(f"Other local Go2 Python processes found: {conflicts}")
        logger.warning(
            "Keep the original controller active and the remote ready. Stop timing is not guaranteed."
        )
        try:
            if input("Standing, area clear, remote in hand: type GO: ").strip() != "GO":
                return
        except (EOFError, KeyboardInterrupt):
            return
    output = args.output or Path("outputs/go2_sport") / (
        (args.direction or "readonly")
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    with output.open("x") as stream:
        old_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            report = run_check(args.interface, args.direction, args.seconds)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        finally:
            signal.signal(signal.SIGTERM, old_handler)
    logger.info("Report: %s; physical result requires operator confirmation", output)
    if report["errors"] or (
        args.direction and report["sdk_stop_estimate"] != "below_thresholds"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
