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

"""Export a Qwen3.5 FSDP actor's full_weights.pt as a Hugging Face model."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    """Write model weights, model configuration, processor and tokenizer."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Expected an empty output directory, got {args.output_dir}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16
    )
    weights = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.save_pretrained(args.output_dir)
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
