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

"""Fresh-process CUDA driver probe; deliberately imports neither Torch nor Ray."""

import ctypes
import json
import sys

CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


def probe() -> None:
    """Report visible SM limits and hold contexts until the parent closes stdin."""
    cuda = ctypes.CDLL("libcuda.so.1")

    def call(name, *args):
        status = getattr(cuda, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    def attribute(device, key):
        value = ctypes.c_int()
        call("cuDeviceGetAttribute", ctypes.byref(value), key, device)
        return value.value

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    devices, contexts = [], []
    try:
        for ordinal in range(count.value):
            device = ctypes.c_int()
            call("cuDeviceGet", ctypes.byref(device), ordinal)
            context = ctypes.c_void_p()
            call("cuCtxCreate_v2", ctypes.byref(context), 0, device)
            contexts.append(context)
            name = ctypes.create_string_buffer(256)
            call("cuDeviceGetName", name, len(name), device)
            devices.append(
                {
                    "name": name.value.decode(),
                    "sms": attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
                    "compute_capability": [
                        attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                        attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
                    ],
                }
            )
        if not devices:
            raise RuntimeError("No CUDA devices visible to MPS calibration")
        sys.stdout.write(json.dumps({"devices": devices}) + "\n")
        sys.stdout.flush()
        sys.stdin.read(1)
    finally:
        for context in reversed(contexts):
            call("cuCtxDestroy_v2", context)


if __name__ == "__main__":
    probe()
