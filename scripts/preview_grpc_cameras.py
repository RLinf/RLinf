# Copyright 2026 Shirui Chen
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

# !/usr/bin/env python3
"""Preview camera feeds received from RobotServer via gRPC.

Run this while robot_server is running to verify images are transmitted.

Usage:
    python scripts/preview_grpc_cameras.py [--url localhost:50051]

Press 'q' to quit.
"""

import argparse

import cv2
import numpy as np

from rlinf.envs.remote.proto import robot_env_pb2, robot_env_pb2_grpc

_MAX_MSG = 16 * 1024 * 1024


def decompress(data: bytes, h: int, w: int, compressed: bool) -> np.ndarray:
    if compressed:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        return img
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)[..., ::-1]


def main():
    import grpc

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:50051")
    args = parser.parse_args()

    channel = grpc.insecure_channel(
        args.url,
        options=[
            ("grpc.max_receive_message_length", _MAX_MSG),
        ],
    )
    stub = robot_env_pb2_grpc.RobotEnvServiceStub(channel)

    # Get image dimensions
    spaces = stub.GetSpaces(robot_env_pb2.Empty())
    h, w = spaces.img_height, spaces.img_width
    n_extra = spaces.num_extra_view_images
    print(f"Image size: {w}x{h}, extra views: {n_extra}")

    print("Calling Reset to get first observation...")
    obs = stub.Reset(robot_env_pb2.ResetRequest())

    print(f"Compressed: {obs.is_compressed}")
    print("Press 'q' to quit.\n")

    while True:
        # Decode main image
        main_img = decompress(obs.main_image, h, w, obs.is_compressed)
        cv2.putText(
            main_img, "cam_top", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        frames = [main_img]

        # Decode wrist images (cam_right, cam_left)
        wrist_names = ["cam_right", "cam_left"]
        for i, wrist_data in enumerate(obs.wrist_images):
            img = decompress(wrist_data, h, w, obs.is_compressed)
            label = wrist_names[i] if i < len(wrist_names) else f"wrist_{i}"
            cv2.putText(
                img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            frames.append(img)

        # Resize to same height and stack
        disp_h = 480
        resized = []
        for f in frames:
            scale = disp_h / f.shape[0]
            resized.append(cv2.resize(f, (int(f.shape[1] * scale), disp_h)))

        combined = np.hstack(resized)
        cv2.imshow("gRPC Camera Preview", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Get next observation via Reset (dummy mode, no action needed)
        try:
            obs = stub.Reset(robot_env_pb2.ResetRequest())
        except Exception as e:
            print(f"gRPC error: {e}")
            break

    cv2.destroyAllWindows()
    channel.close()


if __name__ == "__main__":
    main()
