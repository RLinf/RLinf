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

import torch

from .tensor_operations import Operation


class DeviceInitializer:
    """A picklable device initializer class."""

    def __init__(self, use_gpu_num, use_gpu_index=None):
        self.use_gpu_num = use_gpu_num
        self.use_gpu_index = use_gpu_index

    def __call__(self):
        """Initialize device for the current process."""
        import multiprocessing as mp

        mp_name = mp.current_process().name
        mp_idx = int(mp_name.split("-")[-1]) - 1
        Operation.local_idx = mp_idx

        if self.use_gpu_num > 0:
            if self.use_gpu_index is not None:
                gpu_idx = self.use_gpu_index[mp_idx % self.use_gpu_num]
            else:
                gpu_idx = mp_idx % self.use_gpu_num
            torch.cuda.set_device(gpu_idx)
            Operation.global_device = f"cuda:{gpu_idx}"
        else:
            Operation.global_device = "cpu"

        torch.set_grad_enabled(False)


def get_device_initializer(args):
    """Returns a device initializer that can be pickled."""
    return DeviceInitializer(args.use_gpu_num, args.use_gpu_index)


def single_thread_init(args):
    if args.use_gpu_num > 0:
        if args.use_gpu_index is not None:
            Operation.global_device = f"cuda:{args.use_gpu_index[0]}"
        else:
            Operation.global_device = "cuda"
    else:
        Operation.global_device = "cpu"
    torch.set_grad_enabled(False)
