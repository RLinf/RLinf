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

import argparse
import os
import shutil

import torch
from safetensors.torch import save_file

DEFAULT_BASE_MODEL_PATH = "/workspace/RLinf/GR00T-N1.6-3B"
DEFAULT_SFT_PT_PATH = (
    "/workspace/test/RLinf/logs/20260430-16:26:37/gr00t_16_sft_libero/"
    "checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
)
DEFAULT_HF_OUTPUT_PATH = "/workspace/RLinf/GR00T-1.6-SFT-LIBERO-Spatial-RLinf-SFT-3000"
DEFAULT_PROCESSOR_PATH = "/workspace/Isaac-GR00T/results/libero_spatial_official_tb/processor"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an RLinf GR00T SFT full_weights.pt checkpoint to HF safetensors."
    )
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--sft-pt-path", default=DEFAULT_SFT_PT_PATH)
    parser.add_argument("--hf-output-path", default=DEFAULT_HF_OUTPUT_PATH)
    parser.add_argument(
        "--processor-path",
        default=DEFAULT_PROCESSOR_PATH,
        help="Optional processor directory to overlay into the HF output.",
    )
    return parser.parse_args()


def copy_non_weight_files(src_dir: str, dst_dir: str):
    for filename in os.listdir(src_dir):
        if filename.endswith((".safetensors", ".pt", ".bin", ".index.json")):
            continue
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    base_model_path = args.base_model_path
    sft_pt_path = args.sft_pt_path
    hf_output_path = args.hf_output_path

    os.makedirs(hf_output_path, exist_ok=True)

    # 1. Copy all non-weight files (config.json, experiment_cfg, and other metadata)
    print("[1/3] Copying model configuration files...")
    copy_non_weight_files(base_model_path, hf_output_path)
    if args.processor_path:
        copy_non_weight_files(args.processor_path, hf_output_path)

    # 2. Load the clean weights trained from SFT
    print("[2/3] Loading the SFT-trained PyTorch weights...")
    state_dict = torch.load(sft_pt_path, map_location="cpu")

    # Clean up redundant prefixes potentially left by FSDP
    clean_state_dict = {}
    for key, value in state_dict.items():
        # FSDP wrapping sometimes adds the '_fsdp_wrapped_module.' prefix, strip it off
        new_key = key.replace("_fsdp_wrapped_module.", "")
        clean_state_dict[new_key] = value

    # 3. Convert to safetensors (Loads faster and saves memory in the RL framework)
    print("[3/3] Packaging into Hugging Face Safetensors format...")
    save_file(clean_state_dict, os.path.join(hf_output_path, "model.safetensors"))

    print("\nConversion successfully completed!")
    print(
        f"You can now point 'actor.model_path' in your RL YAML config to: {hf_output_path}"
    )


if __name__ == "__main__":
    main()
