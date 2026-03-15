# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py

import importlib
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Union

import torch.nn as nn

from ..utils import logging

logger = logging.get_logger(__name__)

MODELING_PATH = ["rlinf.models.embodimentlingbotvla.models.vla"]


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    modeling_path: list[str] = field(default_factory=list)
    model_arch_name_to_cls: dict[str, Union[type[nn.Module], str]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        for modeling_path in self.modeling_path:
            self._mapping_model_arch_name_to_cls(modeling_path)

    @property
    def supported_models(self) -> dict[str, type[nn.Module]]:
        return self.model_arch_name_to_cls.keys()

    def get_model_cls_from_model_arch(self, model_arch: str) -> type[nn.Module]:
        return self.model_arch_name_to_cls[model_arch]

    def _mapping_model_arch_name_to_cls(self, modeling_path: str):
        package = importlib.import_module(modeling_path)
        for _, name, ispkg in pkgutil.walk_packages(
            package.__path__, modeling_path + "."
        ):
            if not ispkg:
                try:
                    module = importlib.import_module(name)
                except Exception as e:
                    logger.warning(f"Ignore import error when loading {name}. {e}")
                    continue
                if hasattr(module, "ModelClass"):
                    entry = module.ModelClass
                    if isinstance(entry, list):
                        for tmp in entry:
                            assert tmp.__name__ not in self.model_arch_name_to_cls, (
                                f"Duplicated model implementation for {tmp.__name__}"
                            )
                            self.model_arch_name_to_cls[tmp.__name__] = tmp
                    else:
                        assert entry.__name__ not in self.model_arch_name_to_cls, (
                            f"Duplicated model implementation for {entry.__name__}"
                        )
                        self.model_arch_name_to_cls[entry.__name__] = entry


@lru_cache
def get_registry():
    return _ModelRegistry(modeling_path=MODELING_PATH)
