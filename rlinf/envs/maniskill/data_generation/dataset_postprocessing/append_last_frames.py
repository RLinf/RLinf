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


"""
Appended data:
'arr_0'
        'is_image_encode': trajectory_data.get('is_image_encode', False),
        'image': new_images,
        'instruction': instruction,
        'action': new_actions,
        'state': new_states
        ('info' is intentionally excluded)
"""

import argparse
import os

import numpy as np


def append_last_frames(npz_path, output_path, n_last_frames, verbose=False):
    """
    Append the last n_last_frames to trajectory data and save to new NPZ file.

    Args:
        npz_path (str): Path to the input NPZ file containing trajectory data
        output_path (str): Path for output NPZ file
        n_last_frames (int): Number of last frames to append
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

    # Get the data arrays
    images = trajectory_data["image"]
    actions = trajectory_data["action"]
    states = trajectory_data["state"]
    instruction = trajectory_data["instruction"]

    if verbose:
        print(f"Original trajectory length: {len(images)} frames")
        print(f"Duplicating last frame {n_last_frames} times")

    # Check if we have at least one frame
    if len(images) == 0:
        print("Error: No frames available in trajectory")
        return

    # Get the last frame
    last_image = images[-1]
    last_action = actions[-1]
    last_state = states[-1]

    # Duplicate the last frame n_last_frames times
    duplicated_images = [last_image] * n_last_frames
    duplicated_actions = np.tile(last_action, (n_last_frames, 1))
    duplicated_states = np.tile(last_state, (n_last_frames, 1))

    # Append the duplicated frames to the original data
    new_images = images + duplicated_images
    new_actions = np.concatenate([actions, duplicated_actions], axis=0)
    new_states = np.concatenate([states, duplicated_states], axis=0)

    if verbose:
        print(f"New trajectory length: {len(new_images)} frames")
        print(f"Action shape: {new_actions.shape}")
        print(f"State shape: {new_states.shape}")

    # Create new trajectory data without info
    new_trajectory_data = {
        "is_image_encode": trajectory_data.get("is_image_encode", False),
        "image": new_images,
        "instruction": instruction,
        "action": new_actions,
        "state": new_states,
        # Note: 'info' is intentionally excluded
    }

    # Save to new NPZ file with compression (to match original file compression)
    if verbose:
        print(f"Saving modified trajectory to: {output_path}")

    np.savez_compressed(output_path, arr_0=new_trajectory_data)

    print(
        f"Successfully duplicated last frame {n_last_frames} times and saved to: {output_path}"
    )

    if verbose:
        instruction_text = instruction[0] if len(instruction) > 0 else "No instruction"
        print(f"Instruction: {instruction_text}")


def main():
    parser = argparse.ArgumentParser(
        description="Duplicate the last frame n times and append to trajectory data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python append_last_frames.py --npz_path trajectory.npz --output new_trajectory.npz --n_last_frames 5
            python append_last_frames.py --npz_path trajectory.npz --output new_trajectory.npz --n_last_frames 10 --verbose
                    """,
    )

    parser.add_argument(
        "--npz_path",
        required=True,
        help="Path to the input NPZ file containing trajectory data",
    )

    parser.add_argument("--output", "-o", required=True, help="Output NPZ file path")

    parser.add_argument(
        "--n_last_frames",
        type=int,
        required=True,
        help="Number of times to duplicate the last frame",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.npz_path):
        print(f"Error: NPZ file not found at {args.npz_path}")
        return 1

    # Validate n_last_frames
    if args.n_last_frames <= 0:
        print("Error: n_last_frames must be a positive integer")
        return 1

    try:
        append_last_frames(args.npz_path, args.output, args.n_last_frames, args.verbose)
        return 0
    except Exception as e:
        print(f"Error processing trajectory: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
