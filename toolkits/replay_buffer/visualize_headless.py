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

"""Headless trajectory visualizer with command-line navigation.

For SSH environments without X11 display.
Navigate trajectories via command-line input and view saved images in VSCode.

Usage:
    python toolkits/replay_buffer/visualize_headless.py --replay_dir <path>
"""

import argparse

import matplotlib

matplotlib.use("Agg")

from visualize import MultiTrajectoryVisualizer


def main():
    """Interactive command-line navigation for headless environments."""
    parser = argparse.ArgumentParser(
        description="Headless trajectory visualizer with command-line navigation"
    )
    parser.add_argument(
        "--replay_dir",
        type=str,
        required=True,
        help="Path to replay buffer directory",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="main_images",
        choices=["main_images", "wrist_images", "extra_view_images"],
        help="Which camera view to visualize (default: main_images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="replay_buffer_viz.png",
        help="Output image path (default: replay_buffer_viz.png)",
    )

    args = parser.parse_args()

    viz = MultiTrajectoryVisualizer(
        args.replay_dir,
        camera_key=args.camera,
        save_image=True,
        output_image_path=args.output,
    )

    print("\n" + "=" * 60)
    print("HEADLESS MODE - Command-line Navigation")
    print("=" * 60)
    print(f"Image saved to: {args.output}")
    print("Open this file in VSCode to view the current frame.")
    print("\nCommands:")
    print("  n / next       : Next step (auto-switches to next trajectory at end)")
    print(
        "  p / prev       : Previous step (auto-switches to prev trajectory at start)"
    )
    print("  nt / nexttraj  : Next trajectory")
    print("  pt / prevtraj  : Previous trajectory")
    print("  nb / nextbatch : Next batch index")
    print("  pb / prevbatch : Previous batch index")
    print("  j <id>         : Jump to trajectory ID")
    print("  home           : First step of current trajectory")
    print("  end            : Last step of current trajectory")
    print("  info           : Show current position")
    print("  q / quit       : Exit")
    print("=" * 60 + "\n")

    while True:
        try:
            cmd = input("Command: ").strip().lower()

            if cmd in ["q", "quit", "exit"]:
                print("Exiting...")
                break
            elif cmd in ["n", "next"]:
                viz._next_step()
                viz.update_display()
            elif cmd in ["p", "prev"]:
                viz._prev_step()
                viz.update_display()
            elif cmd in ["nt", "nexttraj"]:
                viz._next_trajectory()
                viz.update_display()
            elif cmd in ["pt", "prevtraj"]:
                viz._prev_trajectory()
                viz.update_display()
            elif cmd in ["nb", "nextbatch"]:
                viz._next_batch()
                viz.update_display()
            elif cmd in ["pb", "prevbatch"]:
                viz._prev_batch()
                viz.update_display()
            elif cmd == "home":
                viz.step_idx = 0
                viz.update_display()
            elif cmd == "end":
                viz.step_idx = viz._get_max_step_idx()
                viz.update_display()
            elif cmd == "info":
                T, B = viz.current_traj_shape[:2]
                print("\nCurrent position:")
                print(
                    f"  Trajectory ID: {viz.current_traj_id} (#{viz.traj_idx}/{viz.buffer.size - 1})"
                )
                print(f"  Step: {viz.step_idx}/{T - 1}")
                print(f"  Batch: {viz.batch_idx}/{B - 1}")
                print(f"  Shape: {viz.current_traj_shape}\n")
            elif cmd.startswith("j "):
                try:
                    traj_id = int(cmd.split()[1])
                    viz.jump_to_trajectory(str(traj_id))
                except (ValueError, IndexError):
                    print("Usage: j <trajectory_id>")
            else:
                print(f"Unknown command: {cmd}. Type 'q' to quit.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
