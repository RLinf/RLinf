# toolkits/ckpt_convertor/convert_pt_to_safetensors.py
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

import argparse
import os

import safetensors.torch
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pt checkpoint to safetensors format"
    )
    parser.add_argument(
        "--pt_path",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted safetensors file",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Optional key to extract from the checkpoint dict (e.g., 'model', 'state_dict')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load the .pt file
    print(f"Loading checkpoint from {args.pt_path}...")
    checkpoint = torch.load(args.pt_path, map_location="cpu")

    # Extract state_dict if needed
    if isinstance(checkpoint, dict):
        if args.key:
            if args.key in checkpoint:
                state_dict = checkpoint[args.key]
            else:
                print(
                    f"Warning: Key '{args.key}' not found in checkpoint. Available keys: {list(checkpoint.keys())}"
                )
                state_dict = checkpoint
        else:
            # Try common keys
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Use the entire dict if it looks like a state_dict
                state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save as safetensors
    print(f"Saving safetensors to {args.output_path}...")
    safetensors.torch.save_file(state_dict, args.output_path)

    print(f"Successfully converted {args.pt_path} to {args.output_path}")
