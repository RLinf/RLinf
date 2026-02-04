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
