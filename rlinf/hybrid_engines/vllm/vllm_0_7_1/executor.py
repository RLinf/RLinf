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


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase


class VLLMExecutor(ExecutorBase):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

    def _init_executor(self):
        raise NotImplementedError("VLLMExecutor._init_executor is not implemented yet.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
    ) -> List[Any]:
        raise NotImplementedError("VLLMExecutor.collective_rpc is not implemented yet.")

    def check_health(self):
        raise NotImplementedError("VLLMExecutor.check_health is not implemented yet.")

    def offload_model_weights(self):
        raise NotImplementedError(
            "VLLMExecutor.offload_model_weights is not implemented yet."
        )

    def sync_hf_weight(self):
        raise NotImplementedError("VLLMExecutor.sync_hf_weight is not implemented yet.")
