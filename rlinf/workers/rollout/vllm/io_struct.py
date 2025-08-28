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


import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class BaseCommand(BaseModel):
    command_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    command_type: str


class SyncHFWeightCommand(BaseCommand):
    command_type: Literal["sync_hf_weight"] = "sync_hf_weight"


class OffloadModelWeightCommand(BaseCommand):
    command_type: Literal["offload_model_weights"] = "offload_model_weights"


class CollectiveRpcCommand(BaseCommand):
    command_type: Literal["collective_rpc"] = "collective_rpc"
    method: Union[str, bytes]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = {}


VLLMCommand = Union[
    SyncHFWeightCommand, OffloadModelWeightCommand, CollectiveRpcCommand
]


class BaseResponse(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    response_type: str


class WorkerReadyResponse(BaseResponse):
    response_type: Literal["worker_ready"] = "worker_ready"
    rank: int


class SyncHFWeightResponse(BaseResponse):
    response_type: Literal["sync_hf_weight"] = "sync_hf_weight"
    command_id: uuid.UUID


class OffloadModelResponse(BaseResponse):
    response_type: Literal["offload_model_weights"] = "offload_model_weights"
    command_id: uuid.UUID


class CollectiveRpcResponse(BaseResponse):
    response_type: Literal["collective_rpc"] = "collective_rpc"
    command_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    rank: int
    data: Any
    success: bool
    error: Optional[str] = None


VLLMResponse = Union[
    WorkerReadyResponse,
    SyncHFWeightCommand,
    OffloadModelResponse,
    CollectiveRpcResponse,
]
