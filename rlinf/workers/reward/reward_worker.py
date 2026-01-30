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


import re as _re

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement


class RewardWorker(Worker):
    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        self.cfg = cfg
        self.component_placement = placement
        self.tokenizer = hf_tokenizer(cfg.reward.tokenizer.tokenizer_model)
        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )
        self.do_down_sampling = hasattr(
            self.cfg.algorithm, "down_sampling"
        ) and getattr(self.cfg.algorithm.down_sampling, "do_down_sampling", False)
        if self.do_down_sampling:
            self.down_sampling_config = getattr(
                self.cfg.algorithm.down_sampling, "down_sampling_config", {}
            )
        else:
            self.down_sampling_config = None

    def init_worker(self):
        if self.cfg.reward.use_reward_model:
            raise NotImplementedError("Reward model is not implemented yet.")
        else:
            self.reward = get_reward_class(self.cfg.reward.reward_type)(self.cfg.reward)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()
        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence
            with self.worker_timer():
                if rollout_result.rewards is None:
                    if self.cfg.reward.use_reward_model:
                        with self.device_lock:
                            batch = rollout_result.to_actor_batch(
                                self.cfg.data.max_prompt_length,
                                self.cfg.actor.model.encoder_seq_length,
                                self.tokenizer.eos_token_id,
                            )
                            rollout_result.rewards = (
                                self.compute_batch_rewards_with_model(batch)
                            )
                    else:
                        rollout_result.rewards = self._compute_rule_based_rewards(
                            rollout_result
                        )
            rollout_result = self.down_sample_batch(rollout_result)
            output_channel.put(rollout_result, async_op=True)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def _compute_rule_based_rewards(self, rollout_result: RolloutResult):
        # Decode only the generated tokens; response_ids are already the post-prompt tokens
        texts = rollout_result.response_texts
        if texts is None:
            texts = self.tokenizer.batch_decode(
                rollout_result.response_ids, skip_special_tokens=True
            )

        kwargs = {}
        if getattr(self.cfg.reward, "use_prompt", False):
            prompts = rollout_result.prompt_texts
            if prompts is None:
                prompts = self.tokenizer.batch_decode(
                    rollout_result.prompt_ids, skip_special_tokens=True
                )
            kwargs["prompts"] = prompts
        scores = self.reward.get_reward(texts, rollout_result.answers, **kwargs)
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )

    def compute_batch_rewards_with_model(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError("Reward model is not implemented yet.")

    def down_sample_batch(self, rollout_result: RolloutResult):
        if not self.do_down_sampling:
            return rollout_result

        down_sampling_config = self.down_sampling_config

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

        original_group_size = int(self.cfg.algorithm.group_size)
        uids = _build_group_uids_by_chunks(
            rollout_result.num_sequence, original_group_size
        )

        if reject_equal and rollout_result.rewards is not None:
            mask1 = _reject_equal_reward(uids, rollout_result.rewards)
        else:
            mask1 = torch.ones(rollout_result.num_sequence, dtype=torch.bool)

        if rollout_result.response_texts is not None:
            response_texts = rollout_result.response_texts
        else:
            response_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in rollout_result.response_ids
            ]

        mask2 = _weighted_group_choice(uids, rollout_result.rewards, response_texts)

        final_mask = mask1 & mask2

        def _apply_mask_to_list(lst, mask):
            return [x for i, x in enumerate(lst) if mask[i].item()]

        def _apply_mask_to_tensor(t, mask):
            return t[mask]

        idx_mask = final_mask
        rr = rollout_result
        rr.prompt_lengths = _apply_mask_to_list(rr.prompt_lengths, idx_mask)
        rr.prompt_ids = _apply_mask_to_list(rr.prompt_ids, idx_mask)
        rr.response_lengths = _apply_mask_to_list(rr.response_lengths, idx_mask)
        rr.response_ids = _apply_mask_to_list(rr.response_ids, idx_mask)
        rr.is_end = _apply_mask_to_list(rr.is_end, idx_mask)
        if rr.rewards is not None:
            rr.rewards = (
                rr.rewards
                if isinstance(rr.rewards, torch.Tensor)
                else torch.tensor(rr.rewards)
            )
            rr.rewards = _apply_mask_to_tensor(rr.rewards, idx_mask)
        if rr.prompt_texts is not None:
            rr.prompt_texts = _apply_mask_to_list(rr.prompt_texts, idx_mask)
        if rr.response_texts is not None:
            rr.response_texts = _apply_mask_to_list(rr.response_texts, idx_mask)
        if rr.answers is not None:
            rr.answers = _apply_mask_to_list(rr.answers, idx_mask)
        if rr.response_mask is not None:
            rr.response_mask = _apply_mask_to_list(rr.response_mask, idx_mask)
        if rr.rollout_logprobs is not None:
            rr.rollout_logprobs = _apply_mask_to_list(rr.rollout_logprobs, idx_mask)
        if rr.ref_logprobs is not None:
            rr.ref_logprobs = _apply_mask_to_tensor(rr.ref_logprobs, idx_mask)
        if rr.prev_logprobs is not None:
            rr.prev_logprobs = _apply_mask_to_tensor(rr.prev_logprobs, idx_mask)
        if rr.recompute_prev_logprobs is not None:
            rr.recompute_prev_logprobs = _apply_mask_to_tensor(
                rr.recompute_prev_logprobs, idx_mask
            )

        _dsn = int(down_sampling_config.get("down_sample_to_n", -1))
        if _dsn > 0:
            rr.group_size = _dsn
        return rr
