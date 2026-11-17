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

"""Export a GR00T SFT full state dict and training processor as an HF checkpoint."""

import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = args.base_model / "config.json"
    required_processor_files = (
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    )
    for path in (
        args.weights,
        config,
        *(args.processor / name for name in required_processor_files),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    state = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(
        isinstance(v, torch.Tensor) for v in state.values()
    ):
        raise ValueError("Expected a tensor state dict from RLinf full_weights.pt")
    args.output.mkdir(parents=True)
    save_torch_state_dict(state, args.output, safe_serialization=True)
    shutil.copy2(config, args.output / "config.json")
    shutil.copytree(args.processor, args.output / "processor")
    metadata = args.base_model / "experiment_cfg" / "metadata.json"
    if metadata.is_file():
        (args.output / "experiment_cfg").mkdir()
        shutil.copy2(metadata, args.output / "experiment_cfg" / "metadata.json")


if __name__ == "__main__":
    main()
