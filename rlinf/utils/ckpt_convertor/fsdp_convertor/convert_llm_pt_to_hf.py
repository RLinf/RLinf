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

"""Convert FSDP full_weights.pt to HuggingFace safetensors for standard LLMs.

Works with any HuggingFace AutoModelForCausalLM (Qwen3, Llama, etc.) without
requiring a custom model builder in ``get_model``.

Usage:
    python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_llm_pt_to_hf \
        --ckpt_path /mnt/public/xzxuan/repos/WideSeek-R1/logs/llm_sft/20260513-16:48:31/qwen3_sft_llm/checkpoints/global_step_238/actor/model_state_dict/full_weights.pt \
        --model_path /mnt/public/xzxuan/models/Qwen3-4B \
        --save_path /mnt/public/xzxuan/repos/WideSeek-R1/logs/llm_sft/20260513-16:48:31/hf
"""

import argparse
import os
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def convert(ckpt_path: str, model_path: str, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)

    # 1. Load model architecture from the base model config
    print(f"Loading model architecture from {model_path}...", flush=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    print(
        f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B",
        flush=True,
    )

    # 2. Load trained weights
    print(f"Loading checkpoint from {ckpt_path}...", flush=True)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    print(f"  Checkpoint keys: {len(state_dict)}", flush=True)

    # 3. Load state dict into model (strict=True to catch mismatches)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING - Missing keys ({len(missing)}):", flush=True)
        for k in missing[:10]:
            print(f"    {k}")
    if unexpected:
        print(f"  WARNING - Unexpected keys ({len(unexpected)}):", flush=True)
        for k in unexpected[:10]:
            print(f"    {k}")
    assert len(missing) == 0 and len(unexpected) == 0, (
        f"State dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys"
    )

    # 4. Save as HuggingFace safetensors format
    print(f"Saving to {save_path}...", flush=True)
    model.save_pretrained(save_path, safe_serialization=True)

    # 5. Copy non-model files (tokenizer, config overrides, etc.)
    for f in os.listdir(model_path):
        if "safetensors" in f or f.startswith("model-") or f == "model.safetensors.index.json":
            continue
        src = os.path.join(model_path, f)
        dst = os.path.join(save_path, f)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {f}", flush=True)

    print("Done!", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP full_weights.pt to HuggingFace safetensors"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to full_weights.pt (e.g. .../actor/model_state_dict/full_weights.pt)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base HuggingFace model (e.g. /path/to/Qwen3-4B)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output directory for the converted HuggingFace model",
    )
    args = parser.parse_args()
    convert(args.ckpt_path, args.model_path, args.save_path)


if __name__ == "__main__":
    main()
