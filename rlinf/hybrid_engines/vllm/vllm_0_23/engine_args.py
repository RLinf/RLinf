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

from dataclasses import dataclass

from omegaconf import DictConfig

from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement


@dataclass
class RLinfEngineArgs:
    """RLinf arguments carried to the vllm workers via ``additional_config``.

    vllm resolves the worker class from a qualified name and then calls it with
    a fixed argument list, so extra constructor arguments cannot be passed
    directly. ``VllmConfig.additional_config`` is pickled into every worker
    process, and vllm calls ``compute_hash()`` on it instead of JSON-encoding it
    when the object provides one, which is what lets these non-JSON values ride
    along.
    """

    rlinf_config: DictConfig
    parent_address: WorkerAddress
    placement: ModelParallelComponentPlacement
    dp_rank: int

    def compute_hash(self) -> str:
        """Return a constant: these arguments must not affect vllm's compile cache."""
        return ""
