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

"""Collect success / failure images for reward-classifier training.

The operator teleoperates the robot and presses a key to label frames.
Labelled transitions are saved as pickle files that can be fed directly
to ``toolkits.realworld_check.train_reward_classifier``.

Usage
-----
.. code-block:: bash

    export FRANKA_ROBOT_IP=172.16.0.2
    python -m toolkits.realworld_check.record_classifier_data \\
        --save_dir /path/to/classifier_data \\
        --successes_needed 200

Controls
--------
- **SpaceMouse** — move the arm.
- **Space bar** — mark current frame as *success*.
- All other frames are saved as *failure*.
- **Ctrl+C** — stop and save.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import pickle
import time

import numpy as np

from rlinf.envs.realworld.common.keyboard import KeyboardListener


def main():
    parser = argparse.ArgumentParser(
        description="Collect success/failure images for reward classifier training."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Directory to save pickle files.",
    )
    parser.add_argument(
        "--successes_needed", type=int, default=200,
        help="Number of success frames to collect before stopping.",
    )
    parser.add_argument(
        "--config_name", type=str, default=None,
        help="Hydra config name for creating the environment (optional).",
    )
    args = parser.parse_args()

    print(
        "收集分类器数据\n"
        "  按 空格键 标记当前帧为 *成功*\n"
        "  其余帧自动标记为 *失败*\n"
        "  按 Ctrl+C 结束并保存\n"
    )

    listener = KeyboardListener()

    successes: list[dict] = []
    failures: list[dict] = []

    print(
        f"目标: 收集 {args.successes_needed} 帧成功样本。\n"
        "请通过遥操作机器人，并在合适时刻按空格键。\n"
        "注意: 此脚本需要在有环境实例的情况下运行。\n"
        "如果不通过环境采集，也可以手动用摄像头采集图片，\n"
        "然后用 pickle 打包成 [{{'observations': {{'frames': {{'wrist_1': img}}}}}}, ...] 格式。"
    )

    # ── Manual camera-only collection mode ───────────────────────────
    # If no environment is specified, use the camera directly.
    try:
        from rlinf.envs.realworld.common.camera import Camera, CameraInfo

        cam_serial = os.environ.get("CAMERA_SERIAL", None)
        if cam_serial is None:
            print("\n未设置 CAMERA_SERIAL 环境变量，尝试自动检测 ...")
            try:
                import pyrealsense2 as rs
                ctx = rs.context()
                devices = ctx.query_devices()
                if len(devices) > 0:
                    cam_serial = devices[0].get_info(rs.camera_info.serial_number)
                    print(f"  检测到相机: {cam_serial}")
                else:
                    print("  未检测到相机。退出。")
                    return
            except ImportError:
                print("  pyrealsense2 未安装。退出。")
                return

        camera = Camera(CameraInfo(name="wrist_1", serial_number=cam_serial))
        camera.open()
        time.sleep(1.0)
        print("相机已就绪。开始采集 ...\n")

        try:
            while len(successes) < args.successes_needed:
                frame = camera.get_frame()
                key = listener.get_key()
                transition = {
                    "observations": {
                        "frames": {"wrist_1": frame.copy()},
                    },
                }
                if key == " ":
                    successes.append(copy.deepcopy(transition))
                    print(
                        f"\r  成功: {len(successes)}/{args.successes_needed}  "
                        f"失败: {len(failures)}",
                        end="", flush=True,
                    )
                else:
                    failures.append(copy.deepcopy(transition))
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            camera.close()

    except Exception as e:
        print(f"相机采集失败: {e}")
        return

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if successes:
        path = os.path.join(args.save_dir, f"success_{len(successes)}_{uuid}.pkl")
        with open(path, "wb") as f:
            pickle.dump(successes, f)
        print(f"\n保存 {len(successes)} 成功样本 → {path}")

    if failures:
        path = os.path.join(args.save_dir, f"failure_{len(failures)}_{uuid}.pkl")
        with open(path, "wb") as f:
            pickle.dump(failures, f)
        print(f"保存 {len(failures)} 失败样本 → {path}")

    print("完成。")


if __name__ == "__main__":
    main()
