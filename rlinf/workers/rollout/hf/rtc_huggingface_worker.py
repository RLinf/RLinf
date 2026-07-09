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

"""Real-Time Control (RTC) rollout worker for real-world OpenPI evaluation.

This subclass overrides :meth:`evaluate` to serve RTC bootstrap/replan
requests from the env worker, applying flow-matching guidance so each new
action chunk stays consistent with the unexecuted tail of the previous one.
"""

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig
from tqdm import tqdm

from rlinf.data.embodied_io_struct import RTCActionResponse, RTCRequest
from rlinf.scheduler import Channel, Worker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class RTCMultiStepRolloutWorker(MultiStepRolloutWorker):
    """Rollout worker that serves RTC requests during evaluation.

    Channel convention (same as the base class): ``input_channel`` carries
    data received from the env worker (RTC requests) and ``output_channel``
    carries data sent to the env worker (RTC action responses).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._rtc_eval_model_actions: torch.Tensor | None = None

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        """Serve RTC requests until the env worker sends an explicit stop."""
        if self.enable_offload:
            self.reload_model()

        self._rtc_eval_model_actions = None

        pbar = tqdm(desc="Evaluating RTC Rollout", disable=(self._rank != 0))
        try:
            while True:
                rtc_request: RTCRequest = await self.recv_rtc_request(input_channel)
                if rtc_request.request_type == "stop":
                    break

                rtc_response = self._predict_rtc(rtc_request)
                self.send_rtc_response(output_channel, rtc_response)
                pbar.update(1)
        finally:
            pbar.close()

        if self.enable_offload:
            self.offload_model()

    @Worker.timer("predict_rtc")
    def _predict_rtc(self, rtc_request: RTCRequest) -> RTCActionResponse:
        """Run one OpenPI inference for a bootstrap or replanning RTC request."""
        rtc_context = None
        guidance_applied = False
        if (
            rtc_request.request_type == "replan"
            and self._rtc_eval_model_actions is not None
        ):
            from rlinf.models.embodiment.openpi.rtc_guidance import RTCGuidanceContext

            # The previous model-space chunk is used to softly constrain the
            # overlap between already scheduled actions and the new plan.
            rtc_context = RTCGuidanceContext(
                prev_model_actions=self._rtc_eval_model_actions,
                executed_horizon=rtc_request.executed_horizon,
                delay_steps=rtc_request.predicted_delay_steps,
            )
            guidance_applied = True

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=rtc_request.obs,
                mode="eval",
                rtc_context=rtc_context,
            )

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        model_actions = result.get("model_actions")
        if isinstance(model_actions, np.ndarray):
            model_actions = torch.from_numpy(model_actions)
        if model_actions is not None:
            self._rtc_eval_model_actions = model_actions.detach().cpu().contiguous()

        return RTCActionResponse(
            actions=actions.detach().cpu().contiguous(),
            model_actions=self._rtc_eval_model_actions,
            chunk_id=rtc_request.chunk_id,
            guidance_applied=guidance_applied,
        )

    async def recv_rtc_request(self, input_channel: Channel) -> RTCRequest:
        """Receive the single real-world RTC request mapped to this rollout rank."""
        return await self.recv_from(
            group_name=self.cfg.env.group_name,
            channel=input_channel,
            tag="eval_rtc",
            route_key=0,
            async_op=True,
            batch_size=1,
            merge_fn=lambda items: items[0],
            infer_batch_size_fn=lambda data: 1,
        ).async_wait()

    def send_rtc_response(
        self, output_channel: Channel, rtc_response: RTCActionResponse
    ) -> None:
        """Send the RTC action response back to the matching env rank."""
        self.send_to(
            group_name=self.cfg.env.group_name,
            channel=output_channel,
            data=rtc_response,
            tag="eval_rtc",
            route_key=0,
            batch_size=1,
            split_fn=lambda data, sizes: [data],
        )
