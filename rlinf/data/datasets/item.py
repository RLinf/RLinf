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

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch


@dataclass
class DatasetItem:
    """
    A single item in processed dataset.

    Attributes:
        prompt (torch.Tensor): Tokenized prompt input_ids tensor.
        length (int): Length of the prompt input_ids.
        answer (str | dict): The answer associated with the prompt.
        idx (int): Index of the item in the dataset.
        solution (Optional[str]): Optional solution text if exists.
        prompt_text (Optional[str]): Optional original prompt text before tokenization.
        meta (Optional[Dict[str, Any]]): Optional metadata dictionary.
        multi_modal_inputs (Optional[Dict[str, Any]]): Optional dictionary for additional multi-modal inputs.
    """

    prompt: torch.Tensor
    length: int
    answer: str | dict
    idx: int
    solution: Optional[str] = None
    image_data: Optional[list[Union[bytes, str]]] = None
    prompt_text: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    multi_modal_inputs: Optional[dict[str, Any]] = None
