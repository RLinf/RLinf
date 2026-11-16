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

"""Dual-ARX5 tcp_rot6d SFT data pipeline for openpi.

This pipeline expects 20D dual-ARX5 TCP rot6d actions,
gripper-first observations, and base/left-wrist/right-wrist images by default.

The dataset loader owns sampling and collation while the canonical RLinf
``DualARX5TcpRot6dDataConfig`` supplies model-facing transforms.
"""

from rlinf.data.datasets.openpi.dual_arx5_tcp_rot6d.dual_arx5_tcp_rot6d_sft_data_loader import (
    DualARX5TcpRot6dSftDataConfig,
    DualARX5TcpRot6dSftDataLoader,
    build_dual_arx5_tcp_rot6d_sft_dataloader,
    collate_dual_arx5_tcp_rot6d_sft_items,
    create_dual_arx5_tcp_rot6d_sft_data_loader,
)
from rlinf.data.datasets.openpi.dual_arx5_tcp_rot6d.dual_arx5_tcp_rot6d_sft_dataset import (
    DualARX5TcpRot6dSftDataset,
)

__all__ = [
    "DualARX5TcpRot6dSftDataConfig",
    "DualARX5TcpRot6dSftDataLoader",
    "DualARX5TcpRot6dSftDataset",
    "build_dual_arx5_tcp_rot6d_sft_dataloader",
    "collate_dual_arx5_tcp_rot6d_sft_items",
    "create_dual_arx5_tcp_rot6d_sft_data_loader",
]
