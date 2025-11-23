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

from typing import Optional

import fastapi
from sglang.srt.managers.tokenizer_manager import TokenizerManager as _TokenizerManager
from sglang.srt.managers.tokenizer_manager import _Communicator
from sglang.srt.server_args import PortArgs, ServerArgs

from .io_struct import (
    AbortGenerationInput,
    AbortGenerationOutput,
    SyncHFWeightInput,
    SyncHFWeightOutput,
    TaskMethodInput,
    TaskMethodOutput,
)


# Add two methods and their communicators, input/output structs.
class TokenizerManager(_TokenizerManager):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        super().__init__(
            server_args=server_args,
            port_args=port_args,
        )

        self.run_task_method_communicator = _Communicator(
            self.send_to_scheduler,
            fan_out=server_args.dp_size,
        )
        self.sync_hf_weight_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.abort_generation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )

        self._result_dispatcher._mapping.extend(
            [
                (
                    TaskMethodOutput,
                    self.run_task_method_communicator.handle_recv,
                ),
                (
                    SyncHFWeightOutput,
                    self.sync_hf_weight_communicator.handle_recv,
                ),
                (
                    AbortGenerationOutput,
                    self.abort_generation_communicator.handle_recv,
                ),
            ]
        )

    async def run_task_method(
        self,
        obj: TaskMethodInput = None,
        request: Optional[fastapi.Request] = None,
    ):
        """
        Run a task method with the given name and arguments.
        """
        self.auto_create_handle_loop()
        if isinstance(obj, str):
            obj = TaskMethodInput(method_name=obj)
        res: list[TaskMethodOutput] = await self.run_task_method_communicator(obj)
        return res[0].result

    async def sync_hf_weight(
        self,
        obj: SyncHFWeightInput = None,
        request: Optional[fastapi.Request] = None,
    ):
        if obj is None:
            obj = SyncHFWeightInput()
        self.auto_create_handle_loop()
        await self.sync_hf_weight_communicator(obj)

    async def abort_generation(
        self,
        obj: AbortGenerationInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.abort_generation_communicator(obj)
