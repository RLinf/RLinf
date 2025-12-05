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

from .accelerator import Accelerator, AcceleratorType, AcceleratorUtil
from .amd_gpu import AMDGPUManager  # Must be imported to register the managers
from .ascend_npu import AscendNPUManager
from .intel_gpu import IntelGPUManager
from .nvidia_gpu import NvidiaGPUManager

__all__ = [
    "AcceleratorUtil",
    "Accelerator",
    "AcceleratorType",
    "AMDGPUManager",
    "AscendNPUManager",
    "IntelGPUManager",
    "NvidiaGPUManager",
]
