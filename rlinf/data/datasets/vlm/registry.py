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


from typing import TYPE_CHECKING, Callable, Optional, Union

from omegaconf import DictConfig
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from rlinf.data.datasets.vlm.base import VLMBaseDataset


class VLMDatasetRegistry:
    registry: dict[str, Callable[..., "VLMBaseDataset"]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Callable[..., "VLMBaseDataset"]], Callable[..., "VLMBaseDataset"]]:
        def decorator(klass: Callable[..., "VLMBaseDataset"]):
            cls.registry[name] = klass
            return klass

        return decorator

    @classmethod
    def create(
        cls,
        dataset_name: Optional[str],
        *,
        data_paths: Union[list[str], str],
        config: DictConfig,
        tokenizer: AutoTokenizer,
        **kwargs,
    ) -> "VLMBaseDataset":
        key = dataset_name.lower()
        dataset_class = cls.registry.get(key)
        return dataset_class(
            data_paths=data_paths, config=config, tokenizer=tokenizer, **kwargs
        )
