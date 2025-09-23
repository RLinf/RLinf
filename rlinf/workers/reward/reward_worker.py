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

from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement


class RewardWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        super().__init__(cfg.reward)
        self.cfg = cfg
        self.component_placement = placement

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )

    def init_worker(self):
        if self.cfg.reward.use_reward_model:
            self.setup_model_and_optimizer()
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
        else:
            self.reward = get_reward_class(self.cfg.reward.reward_type)(self.cfg.reward)

    def get_batch(
        self, channel: Channel
    ) -> Tuple[Dict[str, torch.Tensor], RolloutResult]:
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

        with self.worker_timer():
            recv_batch_size = 0
            while recv_batch_size < self.total_batch_size_per_dp:
                batch, rollout_result = self.get_batch(input_channel)
                recv_batch_size += rollout_result.num_sequence

                # Compute rule-based reward
                if rollout_result.rewards is None:
                    rollout_result.rewards = self._compute_batch_rewards(
                        batch, rollout_result.answers
                    )
                output_channel.put(rollout_result)

            assert recv_batch_size == self.total_batch_size_per_dp, (
                f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
            )

    def _compute_batch_rewards(
        self, batch: Dict[str, torch.Tensor], answers: List[str | dict]
    ):
        """Reward computation using non-model based reward."""

        if self.cfg.reward.use_reward_model:
            return self.compute_batch_rewards_with_model(batch)

        texts = []
        for response, response_len in zip(
            batch["input_ids"],
            batch["response_lengths"],
        ):
            response = response[
                self.cfg.data.max_prompt_length : self.cfg.data.max_prompt_length
                + response_len
            ]
            texts.append(
                self.tokenizer.decode(response.tolist(), skip_special_tokens=True)
            )
        reward_scores = self.reward.get_reward(texts, answers)

        all_reward_scores = torch.as_tensor(
            reward_scores,
            dtype=torch.float,
            device=torch.device("cpu"),
        ).view(-1, 1)
        return all_reward_scores.flatten()

    def compute_batch_rewards_with_model(self, batch: Dict[str, torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            # TODO: fix this
            rewards = self.model(batch["input_ids"], batch["attention_mask"])
        return rewards
