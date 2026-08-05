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

from omegaconf import DictConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from rlinf.scheduler.manager.worker_manager import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement

from .engine_args import RLinfEngineArgs

logger = init_logger(__name__)


class VLLMExecutor(MultiprocExecutor):
    """MultiprocExecutor that hands RLinf's arguments to the vllm workers.

    The vllm 0.8.5 variant had to re-implement ``_init_executor``,
    ``make_worker_process`` and ``worker_main`` to reach the worker constructor.
    Here the arguments travel inside ``additional_config`` instead, so vllm's
    own process handling is reused as is.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rlinf_config: DictConfig,
        dp_rank: int,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
    ):
        vllm_config.additional_config = RLinfEngineArgs(
            rlinf_config=rlinf_config,
            parent_address=parent_address,
            placement=placement,
            dp_rank=dp_rank,
        )
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        super()._init_executor()
        # Tag the loaded parameters, which the 0.8.5 executor did inside its
        # own worker process bootstrap.
        self.collective_rpc("use_sharded_weights")
