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

import time

from rlinf.envs.realworld.xsquare.turtle2_smooth_controller import (
    Turtle2SmoothController,
)


def main():
    controller = Turtle2SmoothController.launch_controller(freq=50)
    # controller = Turtle2SmoothController()

    while True:
        try:
            cmd_str = input("Please input cmd:")
            if cmd_str == "q":
                break
            elif cmd_str == "getpos":
                print(controller.get_state().wait()[0].follow2_pos)
            elif cmd_str == "reset":
                controller.reset_arms().wait()
            elif cmd_str == "go":
                controller.move_arm(
                    [0, 0, 0, 0, 0, 0, 0],
                    [0.27, 0.09, 0.06, 0.0, 1.0, 0.5, 0.0],
                ).wait()
                time.sleep(3.0)
                controller.move_arm(
                    [0, 0, 0, 0, 0, 0, 0],
                    [0.27, 0.09, 0.02, 0.0, 1.0, 0.5, 0.0],
                ).wait()
            else:
                print(f"Unknown cmd: {cmd_str}")
        except KeyboardInterrupt:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
