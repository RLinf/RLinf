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

"""Multi-turn embodied VLM-as-Policy agent loop driving SessionServer."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.agents.embodied.env_tool_worker import EmbodiedEnvToolWorker
from rlinf.agents.tool_call.schema import ToolChannelRequest, ToolChannelResponse
from rlinf.models.embodiment.qwen_vlm.conversation import (
    render_incremental_user_delta,
)
from rlinf.models.embodiment.qwen_vlm.protocol import (
    DEFAULT_REASONING_EFFORT,
    LIBERO_SCHEMA,
    ROBODOJO_SCHEMA,
    InvalidAction,
    build_user_content,
    cameras_for_schema,
    chat_template_kwargs,
    parse_action,
    system_prompt,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    GroupDiscarded,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)
from rlinf.workers.agent.session_server_client import (
    SessionBinding,
    SessionDiscardedError,
)


class EmbodiedVLMAgentLoopWorker(MultiAgentLoopWorker):
    """Async multi-turn VLM-as-Policy agent loop on top of SessionServer.

    Each trajectory runs independently as an asyncio task:
    1. Opens a stateful simulator session on ``EmbodiedEnvToolWorker``.
    2. Loops turn-by-turn, sending only incremental delta text and the new
       turn's PNG camera images to ``SessionServerWorker`` (which keeps an
       engine-side SGLang streaming session open and writes incremental
       ``PayloadPart`` records to the payload channel).
    3. Parses the returned JSON macro action and steps the simulator.
    4. Always closes the simulator session in ``finally`` (including on
       ``GroupDiscarded`` or cancellation) and returns ``(None, close_fields,
       reward)`` so ``MultiAgentLoopWorker._run_session_server_group`` closes
       the session and reports the sparse 0/1 episode reward.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ) -> None:
        super().__init__(cfg, placement)
        model_cfg = cfg.rollout.model
        sglang_cfg = cfg.rollout.get("sglang", {})
        agentloop_cfg = cfg.agentloop

        self._model_path = str(model_cfg.model_path)
        self._trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        self._embodiment = str(model_cfg.get("embodiment", "libero")).lower()
        self._schema = (
            ROBODOJO_SCHEMA if self._embodiment == "robodojo" else LIBERO_SCHEMA
        )
        self._cameras = cameras_for_schema(self._schema)
        self._reasoning_effort = str(
            sglang_cfg.get(
                "reasoning_effort",
                model_cfg.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
            )
        )
        self._template_kwargs = chat_template_kwargs(self._reasoning_effort)
        self._response_budget = int(
            sglang_cfg.get(
                "response_budget",
                cfg.algorithm.sampling_params.get("max_new_tokens", 160_000),
            )
        )
        raw_max_turns = agentloop_cfg.get("max_turns", None)
        self._max_turns: Optional[int] = (
            int(raw_max_turns)
            if raw_max_turns is not None and int(raw_max_turns) > 0
            else None
        )
        self._tokens_per_image_estimate = int(
            agentloop_cfg.get("tokens_per_image_estimate", 256)
        )
        self._processor: Any = None

    def _processor_or_load(self) -> Any:
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self._model_path,
                trust_remote_code=self._trust_remote_code,
            )
        return self._processor

    def _env_tool_channel(self):
        worker_name = self.tool_name_map[EmbodiedEnvToolWorker.TOOL_NAME]
        return self.tool_channel_info_map[worker_name].input_channel

    async def _call_env_tool(
        self,
        env_session_id: str,
        request_type: str,
        tool_args: Optional[dict[str, Any]] = None,
    ) -> ToolChannelResponse:
        input_channel = self._env_tool_channel()
        await input_channel.put(
            ToolChannelRequest(
                session_id=env_session_id,
                request_type=request_type,
                tool_name=EmbodiedEnvToolWorker.TOOL_NAME
                if request_type == "execute"
                else None,
                tool_args=tool_args,
            ),
            async_op=True,
        ).async_wait()
        return await self.tool_worker_output_channel.get(
            env_session_id, async_op=True
        ).async_wait()

    def _estimate_obs_tokens(self, delta_text: str, num_images: int) -> int:
        token_ids = self.tokenizer.encode(delta_text, add_special_tokens=False)
        return len(token_ids) + num_images * self._tokens_per_image_estimate

    def _render_turn_delta(
        self,
        *,
        turn_index: int,
        instruction: str,
        state: dict[str, dict[str, float]],
        image_data: list[str],
        last_output_ended_with_eos: bool,
    ) -> str:
        processor = self._processor_or_load()
        user_msg = {
            "role": "user",
            "content": build_user_content(state, image_data, self._cameras),
        }
        if turn_index == 0:
            messages = [
                {"role": "system", "content": system_prompt(instruction, self._schema)},
                user_msg,
            ]
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self._template_kwargs,
            )
        delta_full, delta_no_eos = render_incremental_user_delta(
            processor, user_msg, self._template_kwargs
        )
        return delta_no_eos if last_output_ended_with_eos else delta_full

    async def run_one_query(self, prompt_ids: list[int], answer: str) -> Any:
        del prompt_ids, answer
        raise NotImplementedError(
            "EmbodiedVLMAgentLoopWorker requires session_server.enable=True "
            "and runs trajectories through run_session_trajectory()."
        )

    async def run_session_trajectory(
        self, binding: SessionBinding, input_ids: list[int], answer: str
    ) -> tuple[Optional[MultiAgentLoopOutput], dict[str, Any], float]:
        """Run one embodied VLM episode against SessionServer & EmbodiedEnvToolWorker."""
        del input_ids
        assert self.session_server_client is not None, (
            "EmbodiedVLMAgentLoopWorker requires an active session_server_client"
        )
        spec = json.loads(answer) if isinstance(answer, str) else dict(answer)
        env_session_id = f"{binding.session_id}-{uuid4().hex[:8]}"
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        env_started = False
        kept_lens: dict[int, int] = {}
        is_end: dict[int, bool] = {}
        reward = 0.0

        try:
            start_resp = await self._call_env_tool(
                env_session_id, "session_start", spec
            )
            if not start_resp.success:
                raise GroupDiscarded(f"env_session_start_failed: {start_resp.result}")
            env_started = True

            obs = start_resp.result
            instruction = str(obs.get("instruction") or spec.get("instruction", ""))
            state = obs["state"]
            image_data = list(obs["image_data"])
            remaining_budget = self._response_budget
            last_ended_with_eos = False
            turn_index = 0
            done = bool(obs.get("done", False))

            while not done and (
                self._max_turns is None or turn_index < self._max_turns
            ):
                delta_text = self._render_turn_delta(
                    turn_index=turn_index,
                    instruction=instruction,
                    state=state,
                    image_data=image_data,
                    last_output_ended_with_eos=last_ended_with_eos,
                )
                if turn_index > 0:
                    obs_tokens = self._estimate_obs_tokens(delta_text, len(image_data))
                    if obs_tokens >= remaining_budget:
                        break
                    remaining_budget -= obs_tokens

                if remaining_budget <= 0:
                    break

                try:
                    gen_resp = await self.session_server_client.generate_multimodal(
                        binding.group_id,
                        binding.session_id,
                        text=delta_text,
                        image_data=image_data,
                        sampling_params={
                            "max_new_tokens": remaining_budget,
                            "skip_special_tokens": False,
                        },
                    )
                except SessionDiscardedError as exc:
                    raise GroupDiscarded(exc.reason) from exc

                output_ids = list(gen_resp["output_ids"])
                finish_reason = gen_resp.get("finish_reason") or {}
                kept_lens[turn_index] = len(output_ids)
                remaining_budget -= len(output_ids)
                last_ended_with_eos = (
                    eos_id is not None and bool(output_ids) and output_ids[-1] == eos_id
                )

                if remaining_budget <= 0 or finish_reason.get("type") == "length":
                    is_end[turn_index] = False
                    reward = 0.0
                    break

                # Decode token IDs with skip_special_tokens=False so </think>
                # is preserved even if the engine strips special tokens in text.
                parsed_text = (
                    self.tokenizer.decode(output_ids, skip_special_tokens=False)
                    if output_ids
                    else str(gen_resp.get("text") or "")
                )
                try:
                    action = parse_action(parsed_text, self._schema)
                except InvalidAction:
                    is_end[turn_index] = True
                    reward = 0.0
                    break

                step_resp = await self._call_env_tool(
                    env_session_id,
                    "execute",
                    {"action": action},
                )
                if not step_resp.success:
                    raise GroupDiscarded(f"env_step_failed: {step_resp.result}")

                step_result = step_resp.result
                done = bool(step_result["done"])
                reward = float(step_result["reward"])
                is_end[turn_index] = done
                turn_index += 1
                if not done:
                    state = step_result["state"]
                    image_data = list(step_result["image_data"])
        finally:
            if env_started:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(
                            self._call_env_tool(env_session_id, "session_end")
                        ),
                        timeout=30.0,
                    )
                except Exception as exc:
                    self.log_warning(
                        f"Closing embodied_env session {env_session_id} failed: {exc!r}"
                    )

        if not kept_lens:
            raise GroupDiscarded("embodied_episode_zero_turns")

        return None, {"kept_lens": kept_lens, "is_end": is_end}, float(reward)
