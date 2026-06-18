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

"""Check whether RLinf can receive PICO / VR publisher data.

Usage::

    python toolkits/realworld_check/test_pico_data.py
    python toolkits/realworld_check/test_pico_data.py --zmq-addr tcp://192.168.1.10:5555
    python toolkits/realworld_check/test_pico_data.py --max-messages 20

The script subscribes to the same JSON ZeroMQ stream consumed by PicoExpert.
It does not require Franka hardware or Ray.
"""

import argparse
import json
import time
from typing import Any


def _format_vector(value: Any, precision: int = 3) -> str:
    if value is None:
        return "None"
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return str(value)
    return "[" + " ".join(f"{v:.{precision}f}" for v in values) + "]"


def _format_pose(pose: Any) -> str:
    if isinstance(pose, dict):
        pos = pose.get("position")
        quat = pose.get("orientation")
    elif isinstance(pose, (list, tuple)) and len(pose) >= 7:
        pos = pose[:3]
        quat = pose[3:7]
    else:
        return "pos=None quat=None"
    return f"pos={_format_vector(pos)} quat={_format_vector(quat)}"


def _controller_summary(data: dict[str, Any], hand: str) -> str:
    controller = data.get(f"{hand}_controller", {})
    pose = _format_pose(controller)
    grip = float(controller.get("grip", 0.0))
    trigger = float(controller.get("trigger", 0.0))
    return f"{hand}: {pose} grip={grip:.3f} trigger={trigger:.3f}"


def _buttons_summary(data: dict[str, Any]) -> str:
    buttons = data.get("buttons", {})
    if not isinstance(buttons, dict) or not buttons:
        return "buttons={}"
    active = [name for name, pressed in sorted(buttons.items()) if pressed]
    if active:
        return "buttons_active=" + ",".join(active)
    return "buttons_active=none"


def _frame_summary(data: dict[str, Any]) -> str:
    parts = [f"headset: {_format_pose(data.get('headset_pose'))}"]
    if "left_controller" in data:
        parts.append(_controller_summary(data, "left"))
    if "right_controller" in data:
        parts.append(_controller_summary(data, "right"))
    if "left_controller" not in data and "right_controller" not in data:
        parts.append(f"keys={','.join(sorted(data.keys()))}")
    parts.append(_buttons_summary(data))
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="PICO / VR ZeroMQ data check")
    parser.add_argument(
        "--zmq-addr",
        type=str,
        default="ipc:///tmp/vr_data.ipc",
        help="ZeroMQ address published by the VR data publisher.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="Receive timeout for each ZeroMQ poll.",
    )
    parser.add_argument(
        "--fail-after-s",
        type=float,
        default=10.0,
        help="Exit with an error if no valid JSON frame is received within this time.",
    )
    parser.add_argument(
        "--print-rate-hz",
        type=float,
        default=5.0,
        help="Maximum rate for printing received frames.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Stop after this many valid JSON frames. Use 0 to run until Ctrl-C.",
    )
    args = parser.parse_args()

    try:
        import zmq
    except ImportError:
        print("[ERROR] pyzmq is not installed. Install the franka env first.")
        return 1

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(args.zmq_addr)

    print(f"[INFO] Subscribed to PICO data at {args.zmq_addr}")
    print("[INFO] Move the PICO controller; press Ctrl-C to stop.")

    start_time = time.time()
    last_print_time = 0.0
    valid_messages = 0
    print_interval = 1.0 / max(args.print_rate_hz, 1e-6)

    try:
        while True:
            try:
                raw = socket.recv()
            except zmq.error.Again:
                elapsed = time.time() - start_time
                if valid_messages == 0 and elapsed >= args.fail_after_s:
                    print(
                        "[ERROR] No PICO JSON frames received. Check the VR "
                        "publisher process, ZeroMQ address, firewall, and route."
                    )
                    return 1
                continue

            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Received non-JSON PICO frame: {exc}")
                continue

            now = time.time()
            valid_messages += 1

            should_print = now - last_print_time >= print_interval
            reached_limit = (
                args.max_messages > 0 and valid_messages >= args.max_messages
            )
            if should_print or reached_limit:
                elapsed = max(now - start_time, 1e-6)
                recv_rate = valid_messages / elapsed
                print(
                    f"[{valid_messages:06d}] recv_rate={recv_rate:.1f}Hz | "
                    f"{_frame_summary(data)}",
                    flush=True,
                )
                last_print_time = now

            if reached_limit:
                print("[INFO] PICO data check completed.")
                return 0
    except KeyboardInterrupt:
        print("\n[INFO] Stopped PICO data check.")
        return 0
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
