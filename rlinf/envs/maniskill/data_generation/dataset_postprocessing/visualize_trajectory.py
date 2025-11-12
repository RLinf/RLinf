# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
import os

import cv2
import numpy as np


def create_video_from_trajectory(npz_path, output_path=None, fps=24, verbose=False):
    """
    Create an MP4 video from trajectory data stored in NPZ file.

    Args:
        npz_path (str): Path to the NPZ file containing trajectory data
        output_path (str): Path for output MP4 file. If None, uses same directory as NPZ file.
        fps (int): Frame rate for the output video
        verbose (bool): Enable verbose output
    """

    # Load the file
    if verbose:
        print(f"Loading trajectory data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # See what keys are available
    if verbose:
        print("Keys in the file:")
        print(data.files)

    # Extract the trajectory data
    trajectory_data = data["arr_0"].item()

    # Get images from the trajectory
    images = trajectory_data["image"]
    print(f"Found {len(images)} frames in trajectory")

    if len(images) == 0:
        print("No images found in trajectory data!")
        return

    # Set output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        output_dir = os.path.dirname(npz_path)
        output_path = os.path.join(output_dir, f"{base_name}_trajectory.mp4")

    # Get image dimensions from first image
    first_image = images[0]
    width, height = first_image.size

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating video with {len(images)} frames at {fps} FPS...")

    # Convert PIL images to OpenCV format and write to video
    for i, pil_image in enumerate(images):
        # Convert PIL to OpenCV format (RGB to BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        video_writer.write(opencv_image)

        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} frames")

    # Release video writer
    video_writer.release()
    print(f"Video saved to: {output_path}")

    # Print trajectory info
    if verbose:
        instruction = (
            trajectory_data["instruction"][0]
            if "instruction" in trajectory_data
            else "No instruction"
        )
        print(f"Instruction: {instruction}")
        print(f"Action shape: {trajectory_data['action'].shape}")
        print(f"State shape: {trajectory_data['state'].shape}")

        # Check if trajectory was successful
        if "info" in trajectory_data:
            last_info = trajectory_data["info"][-1]
            success = last_info["success"][0] if "success" in last_info else False
            print(f"Trajectory success: {success}")


def main():
    parser = argparse.ArgumentParser(
        description="Create MP4 video from trajectory data stored in NPZ file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python visualize_trajectory.py trajectory.npz
            python visualize_trajectory.py trajectory.npz --output video.mp4
            python visualize_trajectory.py trajectory.npz --output video.mp4 --fps 24
                    """,
    )

    parser.add_argument(
        "--npz_path", help="Path to the NPZ file containing trajectory data"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output MP4 file path. If not provided, auto-generates based on input filename",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,  # default is 24 FPS, which is the default frame rate in rlinf/envs/maniskill/data_generation/maniskill_custom_package/motionplanning/widowx/collect_simpler.py
        help="Frame rate for the output video (default: 24)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.npz_path):
        print(f"Error: NPZ file not found at {args.npz_path}")
        return 1

    try:
        create_video_from_trajectory(args.npz_path, args.output, args.fps, args.verbose)
        return 0
    except Exception as e:
        print(f"Error creating video: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
