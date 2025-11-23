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
import tempfile

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DCP checkpoint to state_dict checkpoint"
    )
    parser.add_argument(
        "--dcp_path",
        type=str,
        required=True,
        help="Path to the DCP checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted state_dict checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with tempfile.TemporaryDirectory() as temp_dir:
        dcp_to_torch_save(args.dcp_path, os.path.join(temp_dir, "temp_torch_save.pt"))
        temp_pt = torch.load(os.path.join(temp_dir, "temp_torch_save.pt"))
        model_state_dict = temp_pt["model"]
        torch.save(model_state_dict, args.output_path)

    print(
        f"Converted DCP checkpoint from {args.dcp_path} to state_dict at {args.output_path}"
    )
