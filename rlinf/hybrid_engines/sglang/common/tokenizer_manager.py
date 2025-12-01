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

from importlib.metadata import version
from typing import Optional

import fastapi
from packaging.version import parse
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

        try:
            sglang_version = parse(version("sglang"))
        except Exception as e:
            raise ValueError(f"sglang version not supported: {e}")
        self.patch_return_output_ids = sglang_version < parse("0.5.0")

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

    # to return output_ids and response_text simaltaneously in sglang 0.4.x.
    # copied from srt/managers/tokenizer_manager.py (0.4.6) and only add "output_ids" in out_dict when isinstance(recv_obj, BatchStrOut) is True
    def _handle_batch_output(
        self,
        recv_obj,
    ):
        import json
        import time

        from sglang.srt.managers.tokenizer_manager import (
            BatchEmbeddingOut,
            BatchMultimodalOut,
            BatchStrOut,
            BatchTokenIDOut,
            logger,
        )

        # for sglang 0.5.0 and later, we use the original _handle_batch_output
        if not self.patch_return_output_ids:
            return super()._handle_batch_output(recv_obj)

        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
            }

            if getattr(state.obj, "return_logprob", False):
                self.convert_logprob_style(
                    meta_info,
                    state,
                    state.obj.top_logprobs_num,
                    state.obj.token_ids_logprob,
                    state.obj.return_text_in_logprobs
                    and not self.server_args.skip_tokenizer_init,
                    recv_obj,
                    i,
                )

            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if isinstance(recv_obj, BatchStrOut):
                state.text += recv_obj.output_strs[i]
                # ----- patched code start -----
                if state.obj.stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()
                # -----  patched code end  -----

                out_dict = {
                    "text": state.text,
                    # ----- patched code start -----
                    "output_ids": output_token_ids,
                    # -----  patched code end  -----
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchTokenIDOut):
                if self.server_args.stream_output and state.obj.stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids

                out_dict = {
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchMultimodalOut):
                if isinstance(recv_obj.outputs[i], str):
                    out_dict = {
                        "text": recv_obj.outputs[i],
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = {
                        "outputs": json.dumps(recv_obj.outputs[i]),
                        "meta_info": meta_info,
                    }
            else:
                assert isinstance(recv_obj, BatchEmbeddingOut)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                if self.server_args.speculative_algorithm:
                    meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time
                del self.rid_to_state[rid]

            state.out_list.append(out_dict)
            state.event.set()

            # Log metrics and dump
            if self.enable_metrics and state.obj.log_metrics:
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)
