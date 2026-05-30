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

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

"""
python rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py\
    --dcp_path /path/to/dcp_checkpoint/ \
    --output_path /path/to/save_path/model.pt
"""


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
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {}
    _load_state_dict(
        checkpoint,
        storage_reader=FileSystemReader(args.dcp_path),
        planner=_EmptyStateDictLoadPlanner(keys={"fsdp_checkpoint.model"}),
        no_dist=True,
    )

    try:
        model_state_dict = checkpoint["fsdp_checkpoint"]["model"]
    except KeyError as e:
        raise KeyError(
            "Could not find 'fsdp_checkpoint.model' in the DCP checkpoint. "
            f"Loaded top-level keys: {list(checkpoint.keys())}"
        ) from e

    torch.save(model_state_dict, args.output_path)

    print(
        f"Converted DCP checkpoint from {args.dcp_path} to state_dict at {args.output_path}"
    )
