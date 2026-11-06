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

"""Synchronous PPO scheduling for multi-turn agents."""

from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle


class AgentPPORunner(AgentRunner):
    """Complete both model updates and weight installation before each rollout."""

    def init_rollout_workers(self) -> None:
        """Reuse agent initialization and convert the independent critic checkpoint."""
        super().init_rollout_workers()
        if self.cfg.runner.resume_dir is None and self.cfg.critic.megatron.use_hf_ckpt:
            from rlinf.utils.ckpt_convertor.megatron_convertor.convert_hf_to_mg import (
                convert_hf_to_mg,
            )

            convert_hf_to_mg(
                self.cfg.critic.megatron.ckpt_convertor.hf_model_path,
                self.cfg.critic.megatron.ckpt_convertor,
            )

    def _run_training(
        self, rollout_handle: Handle
    ) -> tuple[dict | None, list[dict], dict]:
        """Compute fixed critic targets, update the critic, then update the actor."""
        critic_infer_handle = self.critic.run_inference(
            input_channel=self.rollout_channel,
            output_channel=self.value_channel,
            do_offload=False,
        )
        critic_infer_handle.wait()
        critic_handle = self.critic.run_training(
            input_channel=self.value_channel,
            output_channel=self.critic_output_channel,
            compute_rollout_metrics=False,
        )
        critic_training_metrics = critic_handle.wait()[0][1]

        actor_input_channel = self.critic_output_channel
        actor_infer_handle = None
        if self.recompute_logprobs:
            actor_infer_handle = self.actor.run_inference(
                input_channel=actor_input_channel,
                output_channel=self.actor_inference_channel,
                compute_ref_logprobs=False,
            )
            actor_infer_handle.wait()
            actor_input_channel = self.actor_inference_channel
        actor_handle = self.actor.run_training(input_channel=actor_input_channel)
        rollout_metrics, training_metrics = actor_handle.wait()[0]
        for actor_metrics, critic_metrics in zip(
            training_metrics, critic_training_metrics, strict=True
        ):
            actor_metrics.update(critic_metrics)
        durations = {
            "training": actor_handle.consume_duration(),
            "critic/training": critic_handle.consume_duration(),
            "critic/inference": critic_infer_handle.consume_duration(
                reduction_type="min"
            ),
        }
        if actor_infer_handle is not None:
            durations["inference"] = actor_infer_handle.consume_duration(
                reduction_type="min"
            )
        return rollout_metrics, training_metrics, durations
