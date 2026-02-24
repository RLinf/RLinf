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

import itertools
import os
import pickle
from typing import Literal

import torch

from .tensor_operations import Load, Operation


class MyPickleModule:
    class MyNoneType:
        # types witch should not be loaded
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if f"{module}.{name}" in (
                "argparse.Namespace",
                "megatron.core.enums.ModelType",
                "megatron.core.transformer.enums.AttnBackend",
                "megatron.core.rerun_state_machine.RerunMode",
                "megatron.core.rerun_state_machine.RerunState",
                "megatron.core.rerun_state_machine.RerunDiagnostic",
            ):
                return MyPickleModule.MyNoneType
            else:
                try:
                    return super().find_class(module, name)
                except ModuleNotFoundError as e:
                    print(
                        f"find_class failed! module: {module}, name: {name}, error: {e}"
                    )
                    raise e


class MGLoaderGroupLazy:
    def __init__(self, model_dicts: list[dict[str, torch.Tensor]], prefix):
        super().__init__()
        self.model_dicts = model_dicts
        self.prefix = prefix

    @classmethod
    def from_file(cls, pt_files, prefix=""):
        model_dicts = []
        for pt_file in pt_files:
            assert os.path.exists(pt_file), f"pt_file not exist! {pt_file}"
            model_dicts.append(
                torch.load(
                    pt_file, map_location="cpu", pickle_module=MyPickleModule, mmap=True
                )
            )
        return cls(model_dicts, prefix)

    @classmethod
    def from_path(cls, pt_path, pp_rank, prefix=""):
        pt_files = []
        if Operation.global_tp >= Operation.global_tpe * Operation.global_ep:
            for tp_rank in range(Operation.global_tp):
                if Operation.global_pp == 1:
                    filename = f"{pt_path}/mp_rank_{tp_rank:02d}/model_optim_rng.pt"
                else:
                    filename = f"{pt_path}/mp_rank_{tp_rank:02d}_{pp_rank:03d}/model_optim_rng.pt"
                pt_files.append(filename)
        else:
            for tpe_rank, ep_rank in itertools.product(
                range(Operation.global_tpe), range(Operation.global_ep)
            ):
                if Operation.global_pp == 1:
                    filename = f"{pt_path}/mp_rank_for_expert{tpe_rank:02d}_{ep_rank:03d}/model_optim_rng.pt"
                else:
                    filename = f"{pt_path}/mp_rank_for_expert{tpe_rank:02d}_{pp_rank:03d}_{ep_rank:03d}/model_optim_rng.pt"
                pt_files.append(filename)
        return cls.from_file(pt_files, prefix)

    def sub_loader(self, suffix, need_ep_rank=None):
        if need_ep_rank is not None:
            all_size = max(
                Operation.global_tp, Operation.global_tpe * Operation.global_ep
            )
            assert len(self.model_dicts) == all_size
            dpe_size = all_size // (Operation.global_tpe * Operation.global_ep)
            parallel_product = itertools.product(
                range(Operation.global_tpe), range(Operation.global_ep), range(dpe_size)
            )
            model_dicts = []
            for model_dict, (tpe_rank, ep_rank, dpe_rank) in zip(
                self.model_dicts, parallel_product
            ):
                if need_ep_rank == ep_rank:
                    model_dicts.append(model_dict)
            return type(self)(model_dicts, self.prefix + suffix)
        else:
            return type(self)(self.model_dicts, self.prefix + suffix)

    @staticmethod
    def get_loader(model_dict, model_key_vpp, dtype_trans):
        def loader(name):
            tensor = model_dict[model_key_vpp].get(name)
            if dtype_trans == "not_tensor":
                return tensor
            else:
                assert torch.is_tensor(tensor), (
                    f"get tensor failed! (model_key_vpp, name) is ({model_key_vpp}, {name}); all keys are {model_dict[model_key_vpp].keys()}"
                )
                return tensor.to(Operation.global_device)

        return loader

    def load(
        self, model_key_vpp, src: str, dtype_trans: Literal["fp8", "bf16", "fp32"]
    ):
        name = f"{self.prefix}{src}"
        assert (
            model_key_vpp in self.model_dicts[0]
            and name in self.model_dicts[0][model_key_vpp]
        ), (
            f"get tensor failed! (model_key_vpp, name) is ({model_key_vpp}, {name}); all keys are {self.model_dicts[0][model_key_vpp].keys()}"
        )
        return [
            Load(
                self.get_loader(model_dict, model_key_vpp, dtype_trans),
                f"{self.prefix}{src}",
                dtype_trans,
            )
            for model_dict in self.model_dicts
        ]

    def keys(self, model_key_vpp, need_ep_rank=None):
        all_keys = []
        if need_ep_rank is not None:
            all_size = max(
                Operation.global_tp, Operation.global_tpe * Operation.global_ep
            )
            assert len(self.model_dicts) == all_size
            dpe_size = all_size // (Operation.global_tpe * Operation.global_ep)
            parallel_product = itertools.product(
                range(Operation.global_tpe), range(Operation.global_ep), range(dpe_size)
            )
            for model_dict, (tpe_rank, ep_rank, dpe_rank) in zip(
                self.model_dicts, parallel_product
            ):
                if need_ep_rank == ep_rank:
                    all_keys.extend(model_dict[model_key_vpp].keys())
        else:
            for model_dict in self.model_dicts:
                all_keys.extend(model_dict[model_key_vpp].keys())
        return all_keys

    def get_keys_value(self, key):
        assert key is not None
        if key in self.model_dicts[0]:
            return self.model_dicts[0][key]
        else:
            assert False, f"key {key} don't exist in model_dicts"
