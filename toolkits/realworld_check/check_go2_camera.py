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

"""Check the Go2 front camera through the read-only drive environment."""

import argparse
import time
from contextlib import closing
from hashlib import sha256

import gymnasium as gym

from rlinf.robotics import Go2Config, RobotDiscovery
from rlinf.utils.logging import get_logger


def main() -> None:
    """Read ten environment observations through the selected network interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    args = parser.parse_args()
    config = Go2Config(node_rank=0, interface=args.interface, enable_motion=False)
    resource = RobotDiscovery.registry["UnitreeGo2"].discovery_cls.enumerate(
        0, [config]
    )
    image_hashes = set()
    with closing(
        gym.make(
            "rlinf.envs.real.go2:Go2DriveEnv-v1",
            override_cfg={"enable_front_camera": True},
            worker_info=None,
            robot_info=resource.infos[0],
            env_idx=0,
            env_cfg={},
        )
    ) as env:
        for _ in range(10):
            observation, _ = env.reset()
            if not env.observation_space.contains(observation):
                raise ValueError("Go2 observation did not match its declared space.")
            frame = observation["frames"]["front"]
            image_hashes.add(sha256(frame).digest())
            time.sleep(0.1)
    get_logger().info(
        "Read 10 Go2 front-camera observations, %d distinct images, %s %s BGR.",
        len(image_hashes),
        frame.shape,
        frame.dtype,
    )


if __name__ == "__main__":
    main()
