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

from sglang.srt.managers.detokenizer_manager import (
    BatchTokenIDOut,
    PortArgs,
    ServerArgs,
    configure_logger,
    get_exception_traceback,
    kill_itself_when_parent_died,
    logger,
    psutil,
    setproctitle,
    signal,
)
from sglang.srt.managers.detokenizer_manager import (
    DetokenizerManager as _DetokenizerManager,
)


class DetokenizerManager(_DetokenizerManager):
    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut):
        result = super().handle_batch_token_id_out(recv_obj)
        # for sglang 0.4.x, this will be None, then we can't get output_ids in tokenizer_manager, and then we can't get output_ids in result.
        # for sglang 0.5.x < 0.5.5, it has a bug in this, so we patched it. refer to https://github.com/sgl-project/sglang/pull/12628
        result.output_ids = recv_obj.output_ids
        return result


# It must be patched, otherwise sglang will use the original DetokenizerManager
def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = DetokenizerManager(server_args, port_args)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
