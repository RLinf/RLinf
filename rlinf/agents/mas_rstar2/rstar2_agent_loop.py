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

import asyncio
import copy
import json
import re as _re
import time
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import AgentLoopOutput, MultiAgentLoopOutput, MultiAgentLoopWorker
from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import DynamicRolloutResult


class TimeTracker:
    def __init__(self, setter: callable):
        self.setter = setter

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        self.setter(elapsed_time)

class DownSamplingProcessor:
    def __init__(self, cfg: DictConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.do_down_sampling = self.cfg.algorithm.get("down_sampling", {}).get("do_down_sampling", False)

    def down_sample_batch(self, rollout_result: DynamicRolloutResult):
        if not self.do_down_sampling:
            return rollout_result

        down_sampling_config = self.cfg.algorithm.down_sampling.down_sampling_config

        def _build_group_uids_by_chunks(total_num: int, group_size: int):
            return [i // max(1, group_size) for i in range(total_num)]

        def _reject_equal_reward(uids, rewards):
            rewards_t = (
                rewards
                if isinstance(rewards, torch.Tensor)
                else torch.tensor(rewards, dtype=torch.float32)
            )
            uids_arr = np.array(uids)
            unique_uids = np.unique(uids_arr)
            valid_mask = torch.ones(len(uids), dtype=torch.bool)
            for uid in unique_uids:
                idxs = np.where(uids_arr == uid)[0]
                if len(idxs) == 0:
                    continue
                grp_rewards = rewards_t[idxs]
                if torch.allclose(grp_rewards[0], grp_rewards):
                    valid_mask[idxs] = False
            return valid_mask

        def _calc_penalty_weights(response_texts):
            def error_ratio(text, pattern=r"<tool_response>.*?</tool_response>"):
                matches = _re.findall(pattern, text, _re.DOTALL)
                error_count = len([m for m in matches if "error" in m.lower()])
                if len(matches) == 0:
                    return 0.5
                return error_count / len(matches)

            def answer_tag_penalty(
                text: str,
                answer_tags=None,
                answer_pattern=r"<answer>.*?</answer>",
                turn_pattern=r"<\|im_start\|>assistant.*?<\|im_end\|>",
            ):
                if answer_tags is None:
                    answer_tags = ["<answer>", "</answer>"]
                if any(tag not in text for tag in answer_tags):
                    return 1.0
                closed_cnt = len(_re.findall(answer_pattern, text, _re.DOTALL))
                tags_cnt = [text.count(tag) for tag in answer_tags]
                if any(c != closed_cnt for c in tags_cnt):
                    return 1.0
                turns = _re.findall(turn_pattern, text, _re.DOTALL)
                num_turns = len(turns)
                if num_turns == 0:
                    return 1.0
                return min((closed_cnt - 1) / num_turns, 1.0)

            err_w = np.array([error_ratio(t) for t in response_texts], dtype=float)
            fmt_w = np.array(
                [answer_tag_penalty(t) for t in response_texts], dtype=float
            )
            return err_w, fmt_w

        def _weighted_group_choice(uids, rewards, response_texts):
            cfg = down_sampling_config
            down_sample_to_n = int(cfg.get("down_sample_to_n", -1))
            if down_sample_to_n <= 0:
                return torch.ones(len(uids), dtype=torch.bool)

            roc_error_ratio = bool(cfg.get("roc_error_ratio", False))
            roc_answer_format = bool(cfg.get("roc_answer_format", False))
            min_zero = int(cfg.get("min_zero_reward_trace_num", 0))
            min_non_zero = int(cfg.get("min_non_zero_reward_trace_num", 0))

            err_w, fmt_w = _calc_penalty_weights(response_texts)

            uids_arr = np.array(uids)
            unique_uids = np.unique(uids_arr)
            rewards_t = (
                rewards
                if isinstance(rewards, torch.Tensor)
                else torch.tensor(rewards, dtype=torch.float32)
            )

            valid_mask = torch.zeros(len(uids), dtype=torch.bool)
            for uid in unique_uids:
                idxs = np.where(uids_arr == uid)[0]
                if len(idxs) < down_sample_to_n:
                    continue
                if len(idxs) == down_sample_to_n:
                    valid_mask[idxs] = True
                    continue
                grp_rewards = rewards_t[idxs]
                grp_err_w = err_w[idxs]
                grp_fmt_w = fmt_w[idxs]
                penalty = (grp_err_w if roc_error_ratio else 0) + (
                    grp_fmt_w if roc_answer_format else 0
                )

                zero_pairs = [
                    (i, p)
                    for i, r, p in zip(idxs, grp_rewards, penalty, strict=False)
                    if r <= 0
                ]
                non_zero_pairs = [
                    (i, p)
                    for i, r, p in zip(idxs, grp_rewards, penalty, strict=False)
                    if r > 0
                ]

                non_zero_pairs.sort(key=lambda x: x[1])

                z_quota = round(len(zero_pairs) * down_sample_to_n / len(idxs))
                nz_quota = round(len(non_zero_pairs) * down_sample_to_n / len(idxs))

                if z_quota <= min(min_zero, len(zero_pairs)):
                    z_quota = min(min_zero, len(zero_pairs))
                    nz_quota = down_sample_to_n - z_quota
                if nz_quota <= min(min_non_zero, len(non_zero_pairs)):
                    nz_quota = min(min_non_zero, len(non_zero_pairs))
                    z_quota = down_sample_to_n - nz_quota

                chosen = [i for i, _ in non_zero_pairs[:nz_quota]] + [
                    i for i, _ in zero_pairs[:z_quota]
                ]
                if len(chosen) != down_sample_to_n:
                    all_sorted = [
                        i
                        for i, _ in sorted(
                            non_zero_pairs + zero_pairs, key=lambda x: x[1]
                        )
                    ]
                    chosen = all_sorted[:down_sample_to_n]
                valid_mask[torch.tensor(chosen, dtype=torch.long)] = True

            return valid_mask

        reject_equal = bool(down_sampling_config.get("reject_equal_reward", False))

        # Dynamic rollout keeps turn-level sequences; down-sample by trajectory.
        traj_to_turn_indices: dict[int, list[int]] = {}
        for turn_idx, traj_idx in enumerate(rollout_result.idx_to_traj):
            traj_to_turn_indices.setdefault(traj_idx, []).append(turn_idx)
        traj_ids = sorted(traj_to_turn_indices.keys())
        if len(traj_ids) == 0:
            return rollout_result
        rep_turn_indices = [traj_to_turn_indices[traj_idx][-1] for traj_idx in traj_ids]
        uids = _build_group_uids_by_chunks(len(rep_turn_indices), max(1, len(rep_turn_indices)))

        rep_rewards = None
        if rollout_result.rewards is not None:
            rewards_t = (
                rollout_result.rewards
                if isinstance(rollout_result.rewards, torch.Tensor)
                else torch.tensor(rollout_result.rewards, dtype=torch.float32)
            )
            rep_rewards = rewards_t[torch.tensor(rep_turn_indices, dtype=torch.long)]

        if reject_equal and rep_rewards is not None:
            mask1 = _reject_equal_reward(uids, rep_rewards)
        else:
            mask1 = torch.ones(len(rep_turn_indices), dtype=torch.bool)

        response_texts = []
        for turn_idx in rep_turn_indices:
            prompt_len = int(rollout_result.prompt_lengths[turn_idx])
            resp_len = int(rollout_result.response_lengths[turn_idx])
            resp_ids = rollout_result.input_ids[turn_idx][
                prompt_len : prompt_len + resp_len
            ]
            response_texts.append(self.tokenizer.decode(resp_ids, skip_special_tokens=True))

        if rep_rewards is not None:
            mask2 = _weighted_group_choice(uids, rep_rewards, response_texts)
        else:
            mask2 = torch.ones(len(rep_turn_indices), dtype=torch.bool)

        final_mask = mask1 & mask2
        if not torch.any(final_mask):
            final_mask = torch.ones(len(rep_turn_indices), dtype=torch.bool)
        selected_traj_ids = {
            traj_id
            for traj_id, keep in zip(traj_ids, final_mask.tolist(), strict=False)
            if keep
        }
        seq_mask = torch.tensor(
            [traj_idx in selected_traj_ids for traj_idx in rollout_result.idx_to_traj],
            dtype=torch.bool,
        )

        def _apply_mask_to_list(lst, mask):
            return [x for i, x in enumerate(lst) if mask[i].item()]

        def _apply_mask_to_tensor(t, mask):
            return t[mask]

        rr = rollout_result
        rr.prompt_lengths = _apply_mask_to_list(rr.prompt_lengths, seq_mask)
        rr.response_lengths = _apply_mask_to_list(rr.response_lengths, seq_mask)
        rr.input_ids = _apply_mask_to_list(rr.input_ids, seq_mask)
        rr.is_end = _apply_mask_to_list(rr.is_end, seq_mask)
        rr.idx_to_traj = _apply_mask_to_list(rr.idx_to_traj, seq_mask)
        if rr.rewards is not None:
            rr.rewards = (
                rr.rewards
                if isinstance(rr.rewards, torch.Tensor)
                else torch.tensor(rr.rewards)
            )
            rr.rewards = _apply_mask_to_tensor(rr.rewards, seq_mask)
        if rr.rollout_logprobs is not None:
            rr.rollout_logprobs = _apply_mask_to_list(rr.rollout_logprobs, seq_mask)
        if rr.ref_logprobs is not None:
            rr.ref_logprobs = _apply_mask_to_tensor(rr.ref_logprobs, seq_mask)
        if rr.prev_logprobs is not None:
            rr.prev_logprobs = _apply_mask_to_tensor(rr.prev_logprobs, seq_mask)
        if rr.recompute_prev_logprobs is not None:
            rr.recompute_prev_logprobs = _apply_mask_to_tensor(
                rr.recompute_prev_logprobs, seq_mask
            )

        if rr.extra_fields_turn is not None:
            for k, v in rr.extra_fields_turn.items():
                if isinstance(v, list) and len(v) == len(seq_mask):
                    rr.extra_fields_turn[k] = _apply_mask_to_list(v, seq_mask)
                elif isinstance(v, torch.Tensor) and v.size(0) == len(seq_mask):
                    rr.extra_fields_turn[k] = _apply_mask_to_tensor(v, seq_mask)
        assert rr.extra_fields_train is not None
        for k, v in rr.extra_fields_train.items():
            if isinstance(v, list) and len(v) == len(seq_mask):
                rr.extra_fields_train[k] = _apply_mask_to_list(v, seq_mask)
            elif isinstance(v, torch.Tensor) and v.size(0) == len(seq_mask):
                rr.extra_fields_train[k] = _apply_mask_to_tensor(v, seq_mask)

        kept_traj_ids = sorted(set(rr.idx_to_traj))
        traj_remap = {old_id: new_id for new_id, old_id in enumerate(kept_traj_ids)}
        rr.idx_to_traj = [traj_remap[traj_id] for traj_id in rr.idx_to_traj]
        rr.num_sequence = len(rr.idx_to_traj)
        rr.group_size = len(kept_traj_ids)

        _dsn = int(down_sampling_config.get("down_sample_to_n", -1))
        if _dsn > 0 and rr.group_size > _dsn:
            rr.group_size = _dsn
        return rr

class MasRstar2AgentLoopWorker(MultiAgentLoopWorker):
    """Simple tool agent loop that can interact with tools."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        if self.cfg.rollout.get("custom_chat_template", None) is not None:
            self.tokenizer.chat_template = cfg.rollout.custom_chat_template

        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.actor.model.encoder_seq_length)
        self.max_resp_len = max(1, max_total_len - self.max_prompt_len)

        self.max_user_turns = cfg.agentloop.get("max_user_turns", 5)
        self.max_assistant_turns = cfg.agentloop.get("max_assistant_turns", 5)
        self.max_parallel_calls = cfg.agentloop.get("max_parallel_calls", 3)
        self.max_tool_response_length = cfg.agentloop.get(
            "max_tool_response_length", 500
        )
        self.tool_response_truncate_side = cfg.agentloop.get(
            "tool_response_truncate_side", "right"
        )
        self.apply_chat_template_kwargs = cfg.data.get("apply_chat_template_kwargs", {})
        self.system_prompt = self.tokenizer.apply_chat_template(
            [{}],
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        assert self.toolcall_parser is not None, "toolcall_parser must be set in rstar2"
        self.reward = get_reward_class(self.cfg.reward.reward_type)(cfg.reward)
        self.down_sampling_processor = DownSamplingProcessor(cfg, self.tokenizer)
        self.extra_keys_traj = [
            "user_turns",
            "assistant_turns",
            "num_turns",
            "reward_time",
            "llm_time",
            "toolcall_time",
        ]

    async def pre_process_query(self, prompt_ids: list[int], answer: str):
        generate_context = {
            "answer": answer,
            "tool_session_ids": {},
            "user_turns": 0,
            "assistant_turns": 0,
            "num_turns": 0,
            "reward_time": 0.0,
            "llm_time": 0.0,
            "toolcall_time": 0.0,
        }
        return prompt_ids[: self.max_prompt_len], generate_context

    async def post_process_query(
        self, generate_context: dict[str, Any], output: MultiAgentLoopOutput
    ) -> dict[str, Any]:
        # Release tool sessions
        for tool_worker_name, session_id in generate_context[
            "tool_session_ids"
        ].items():
            if self.tool_channel_info_map[tool_worker_name].has_session:
                # tool need session
                await self.tool_session_release(tool_worker_name, session_id)

        # Calculate reward score
        final_response_ids = generate_context["final_response_ids"]
        final_response_text = self.tokenizer.decode(final_response_ids)
        answer = generate_context["answer"]
        def time_setter(x):
            generate_context["reward_time"] += x
        with TimeTracker(time_setter):
            reward_score = (await self.reward.async_get_reward([final_response_text], [answer]))[0]
        for single_turn_output in output.single_turn_outputs:
            single_turn_output.reward_score = reward_score

        # Set extra fields
        user_turns = generate_context["user_turns"]
        assistant_turns = generate_context["assistant_turns"]
        output.extra_fields = {
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "num_turns": user_turns + assistant_turns + 1,
            "reward_time": generate_context["reward_time"],
            "llm_time": generate_context["llm_time"],
            "toolcall_time": generate_context["toolcall_time"],
        }
        if self.print_outputs:
            output.trace_prints.append(
                {
                    # "prompt": output.prompt_text,
                    # "generate": output.response_text,
                    "prompt": self.tokenizer.decode(output.single_turn_outputs[0].prompt_ids),
                    "generate": self.tokenizer.decode(output.single_turn_outputs[-1].response_ids),
                    "num_turns": output.extra_fields["num_turns"],
                }
            )
        return output

    async def tool_session_get(
        self, generate_context: dict[str, Any], tool_name: str
    ) -> Any:
        tool_worker_name = self.tool_name_map[tool_name]
        tool_channel_info = self.tool_channel_info_map[tool_worker_name]
        if tool_worker_name in generate_context["tool_session_ids"]:
            return generate_context["tool_session_ids"][tool_worker_name]
        session_id = uuid4().hex
        generate_context["tool_session_ids"][tool_worker_name] = session_id
        if tool_channel_info.has_session:
            # tool need session
            await tool_channel_info.input_channel.put(
                ToolChannelRequest(session_id=session_id, request_type="session_start"),
                async_op=True,
            ).async_wait()
            self.log_debug("session_start put")
            response: ToolChannelResponse = await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
            assert response.success
            self.log_debug("session_start get")
        return session_id

    async def tool_session_release(self, tool_worker_name, session_id) -> str | dict:
        tool_channel_info = self.tool_channel_info_map[tool_worker_name]
        await tool_channel_info.input_channel.put(
            ToolChannelRequest(session_id=session_id, request_type="session_end"),
            async_op=True,
        ).async_wait()
        self.log_debug("session_end put")
        response: ToolChannelResponse = await self.tool_worker_output_channel.get(
            session_id, async_op=True
        ).async_wait()
        assert response.success
        self.log_debug("session_end get")

    async def tool_call(
        self,
        generate_context: dict[str, Any],
        tool_request: ToolRequest | ToolChannelResponse,
    ) -> ToolResponse:
        if isinstance(tool_request, ToolChannelResponse):
            return ToolResponse(text=tool_request.result)
        elif tool_request.name not in self.tool_name_map:
            return ToolResponse(
                text=f"Error when executing tool: '{tool_request.name}'"
            )
        else:
            tool_name, tool_args = tool_request.name, tool_request.arguments
            tool_channel_info = self.tool_channel_info_map[
                self.tool_name_map[tool_name]
            ]
            tool_input_channel = tool_channel_info.input_channel
            session_id = await self.tool_session_get(generate_context, tool_name)
            await tool_input_channel.put(
                ToolChannelRequest(
                    session_id=session_id,
                    request_type="execute",
                    tool_name=tool_name,
                    tool_args=tool_args,
                ),
                async_op=True,
            ).async_wait()
            self.log_debug("tool execute put")
            response: ToolChannelResponse = await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
            self.log_debug("tool execute get")
            self.log_info(f"{response}")
            if isinstance(response.result, (list, dict)):
                result_text = json.dumps(response.result)
            else:
                result_text = str(response.result)

            if result_text and len(result_text) > self.max_tool_response_length:
                if self.tool_response_truncate_side == "left":
                    result_text = (
                        result_text[: self.max_tool_response_length] + "...(truncated)"
                    )
                elif self.tool_response_truncate_side == "right":
                    result_text = (
                        "(truncated)..." + result_text[-self.max_tool_response_length :]
                    )
                else:
                    length = self.max_tool_response_length // 2
                    result_text = (
                        result_text[:length]
                        + "...(truncated)..."
                        + result_text[-length:]
                    )
            return ToolResponse(text=result_text)

    async def generate_llm_response(
        self,
        generate_context: dict[str, Any],
        trace_prints,
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
    ):
        # Generate response from LLM
        max_resp_len = self.max_resp_len - (
            len(turn_prompt_ids) - len(problem_prompt_ids)
        )

        def time_setter(x):
            generate_context["llm_time"] += x
        with TimeTracker(time_setter):
            generate_result = await self.generate(
                turn_prompt_ids, sampling_params={"max_new_tokens": max_resp_len}
            )
        llm_response_ids = generate_result["output_ids"]
        if self.return_logprobs:
            generate_logprobs = generate_result["logprobs"]
        if len(llm_response_ids) > max_resp_len:
            llm_response_ids = llm_response_ids[:max_resp_len]
            if self.return_logprobs:
                generate_logprobs = generate_logprobs[:max_resp_len]

        llm_response_logprobs = None
        if self.return_logprobs:
            llm_response_logprobs = generate_logprobs
        llm_response_text = self.tokenizer.decode(llm_response_ids)

        llm_output = AgentLoopOutput(
            prompt_ids=copy.deepcopy(turn_prompt_ids),
            response_ids=llm_response_ids,
            response_logprobs=llm_response_logprobs,
        )
        next_turn_prompt_ids = turn_prompt_ids + llm_response_ids
        all_response_ids = next_turn_prompt_ids[len(problem_prompt_ids):]
        generate_context["final_response_ids"] = all_response_ids
        generate_context["assistant_turns"] += 1

        is_continue = True
        if is_continue and len(llm_response_ids) == max_resp_len:
            is_continue = False
        if is_continue and (
            self.max_assistant_turns
            and generate_context["assistant_turns"] >= self.max_assistant_turns
        ):
            is_continue = False
        if is_continue and (
            self.max_user_turns
            and generate_context["user_turns"] >= self.max_user_turns
        ):
            is_continue = False
        if not is_continue:
            return False, None, None, llm_output
        return True, llm_response_ids, llm_response_text, llm_output

    async def generate_tool_response(
        self,
        generate_context: dict[str, Any],
        trace_prints,
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
        llm_response_ids: list[int],
        llm_response_text: str,
    ):
        # Extract tool calls from response
        _, tool_requests = await self.toolcall_parser(llm_response_text)
        if len(tool_requests) == 0:
            return False, None

        # Execute tools in parallel with history propagation
        tool_responses: list[ToolResponse | int] = []
        run_tool_requests = []
        for tool_request in tool_requests[: self.max_parallel_calls]:
            if isinstance(tool_request, ToolResponse):
                tool_responses.append(tool_request)
            else:
                tool_responses.append(len(run_tool_requests))
                run_tool_requests.append(tool_request)
        run_tool_tasks = [
            self.tool_call(generate_context, tool_request)
            for tool_request in run_tool_requests
        ]
        def time_setter(x):
            generate_context["toolcall_time"] += x
        with TimeTracker(time_setter):
            run_tool_responses: list[ToolResponse] = await asyncio.gather(*run_tool_tasks)
        tool_responses: list[ToolResponse] = [
            item if isinstance(item, ToolResponse) else run_tool_responses[item]
            for item in tool_responses
        ]
        if any(not isinstance(item, ToolResponse) for item in tool_responses):
            return False, None

        # Convert tool responses to messages and tokenize
        tool_messages = []
        for tool_response in tool_responses:
            message = {"role": "tool", "content": tool_response.text}
            tool_messages.append(message)

        # Tokenize tool responses
        tool_response_ids = self.tokenizer.apply_chat_template(
            tool_messages,
            add_generation_prompt=True,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        tool_response_ids = tool_response_ids[len(self.system_prompt) :]
        max_tool_resp_len = self.max_resp_len - (
            len(turn_prompt_ids) + len(llm_response_ids) - len(problem_prompt_ids)
        )
        if len(tool_response_ids) > max_tool_resp_len:
            return False, None

        generate_context["user_turns"] += 1
        next_turn_prompt_ids = turn_prompt_ids + llm_response_ids + tool_response_ids
        return True, next_turn_prompt_ids

    def get_rollout_result(
        self,
        task_results: list[MultiAgentLoopOutput],
        extra_fields_turn: Optional[dict],
        extra_fields_traj: Optional[dict],
        extra_fields_group: Optional[dict],
        extra_fields_train: dict,
    ) -> DynamicRolloutResult:
        result = super().get_rollout_result(
            task_results,
            extra_fields_turn,
            extra_fields_traj,
            extra_fields_group,
            extra_fields_train,
        )
        return self.down_sampling_processor.down_sample_batch(result)

    def get_rollout_metrics(
        self,
        rollout_result: DynamicRolloutResult,
    ) -> dict:
        return {
            "__mean__/agentloop/mean/user_turns":           (sum(rollout_result.extra_fields_traj["user_turns"]), rollout_result.group_size),
            "__mean__/agentloop/mean/assistant_turns":      (sum(rollout_result.extra_fields_traj["assistant_turns"]), rollout_result.group_size),
            "__mean__/agentloop/mean/num_turns":            (sum(rollout_result.extra_fields_traj["num_turns"]), rollout_result.group_size),
            "__mean__/agentloop/mean/reward_time":          (sum(rollout_result.extra_fields_traj["reward_time"]), rollout_result.group_size),
            "__mean__/agentloop/mean/llm_time":             (sum(rollout_result.extra_fields_traj["llm_time"]), rollout_result.group_size),
            "__mean__/agentloop/mean/toolcall_time":        (sum(rollout_result.extra_fields_traj["toolcall_time"]), rollout_result.group_size),
            "__max__/agentloop/max/user_turns":             max(rollout_result.extra_fields_traj["user_turns"]),
            "__max__/agentloop/max/assistant_turns":        max(rollout_result.extra_fields_traj["assistant_turns"]),
            "__max__/agentloop/max/num_turns":              max(rollout_result.extra_fields_traj["num_turns"]),
            "__max__/agentloop/max/reward_time":            max(rollout_result.extra_fields_traj["reward_time"]),
            "__max__/agentloop/max/llm_time":               max(rollout_result.extra_fields_traj["llm_time"]),
            "__max__/agentloop/max/toolcall_time":          max(rollout_result.extra_fields_traj["toolcall_time"]),
            "__sum__/agentloop/sum/filtered_batch_size":    len(set(rollout_result.idx_to_traj)),
        }
