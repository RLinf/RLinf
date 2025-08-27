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


import datetime
import uuid
from typing import Literal, Union

from pydantic import BaseModel, Field


class BaseCommand(BaseModel):
    command_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=datetime.datetime.now)
    command_type: str


class SyncHFWeightCommand(BaseCommand):
    command_type: Literal["sync_hf_weight"] = "sync_hf_weight"


class OffloadModelWeightCommand(BaseCommand):
    command_type: Literal["offload_model_weights"] = "offload_model_weights"


VLLMCommand = Union[SyncHFWeightCommand, OffloadModelWeightCommand]
