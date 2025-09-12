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

import glob
import os
from typing import Literal

import safetensors
import safetensors.torch

from .tensor_operations import Load, Operation


class STLoaderLazy:
    def __init__(self, hf_model_dict: dict[str, str], prefix):
        super().__init__()
        self.hf_model_dict = hf_model_dict
        self.prefix = prefix

    @classmethod
    def from_file(cls, safetensor_files, prefix="", check_done=False):
        hf_model_dict = {}
        for safetensor_file in safetensor_files:
            if check_done:
                assert os.path.exists(safetensor_file) and os.path.exists(
                    f"{safetensor_file}.done"
                ), f"safetensor file {safetensor_file} or its done file does not exist."
            try:
                with safetensors.torch.safe_open(
                    safetensor_file, framework="pt", device="cpu"
                ) as file_handle:
                    for key in file_handle.keys():
                        assert key not in hf_model_dict, (
                            f"key in hf_model_dict is replicated, key is {key}"
                        )
                        hf_model_dict[key] = safetensor_file
            except Exception as e:
                print(f"loading file error: {safetensor_file}, exception: {e}")
                exit(1)
        return cls(hf_model_dict, prefix)

    @classmethod
    def from_path(cls, safetensor_path, prefix=""):
        safetensor_files = list(glob.glob(f"{safetensor_path}/*.safetensors"))
        safetensor_files.sort()
        return cls.from_file(safetensor_files, prefix)

    def sub_loader(self, suffix):
        return type(self)(self.hf_model_dict, self.prefix + suffix)

    def _read(self, src):
        if src not in self.hf_model_dict:
            raise KeyError(f"no {src}, full keys is {self.hf_model_dict.keys()}")
        safetensor_file = self.hf_model_dict[src]
        with safetensors.torch.safe_open(
            safetensor_file, framework="pt", device=Operation.global_device
        ) as file_handle:
            value = file_handle.get_tensor(src)
        return value

    def load(
        self,
        src: str,
        dtype_trans: Literal["auto", "fp8_bf16", "bf16_bf16", "fp8_fp8"] = "auto",
    ):
        return Load(self._read, f"{self.prefix}{src}", dtype_trans)

    def keys(self):
        return self.hf_model_dict.keys()


class STLoader:
    def __init__(self, hf_model_dict: dict, prefix):
        super().__init__()
        self.hf_model_dict = hf_model_dict
        self.prefix = prefix

    @classmethod
    def from_file(cls, safetensor_files, prefix="", check_done=False):
        hf_model_dict = {}
        for safetensor_file in safetensor_files:
            if check_done:
                assert os.path.exists(safetensor_file) and os.path.exists(
                    f"{safetensor_file}.done"
                )
            hf_model_dict.update(
                safetensors.torch.load_file(
                    safetensor_file, device=Operation.global_device
                )
            )
        return cls(hf_model_dict, prefix)

    @classmethod
    def from_path(cls, safetensor_path, prefix=""):
        safetensor_files = list(glob.glob(f"{safetensor_path}/*.safetensors"))
        safetensor_files.sort()
        return cls.from_file(safetensor_files, prefix)

    def sub_loader(self, suffix):
        return type(self)(self.hf_model_dict, f"{self.prefix}{suffix}")

    def load(self, src: str, dtype_trans: Literal["fp8_bf16", "bf16_bf16", "fp8_fp8"]):
        return Load(self.hf_model_dict.get, f"{self.prefix}{src}", dtype_trans)

    def keys(self):
        return self.hf_model_dict.keys()
