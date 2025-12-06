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

import copy
import gc

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
import logging

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult, EnvOutput
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.scheduler import Cluster, Worker
from rlinf.scheduler.channel.channel import Channel
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement


class MultiStepRolloutWorker(Worker):
    """Rollout worker for huggingface models with multi-step environment interaction."""

    def __init__(self, cfg: DictConfig):
        """Initialize the rollout worker.

        Args:
            cfg (DictConfig): Configuration for the rollout worker.
        """
        Worker.__init__(self)

        self.cfg = cfg
        self.device = torch.cuda.current_device()
        self.actor_group_name = cfg.actor.group_name
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)
        self.get_batch_cnt = 0

        self.placement = HybridComponentPlacement(cfg, Cluster())

        # Batching parameters
        only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or only_eval
        self.train_num_envs_per_stage = (
            self.cfg.env.train.total_num_envs
            // self._world_size
            // self.num_pipeline_stages
        )
        self.train_num_group_envs_per_stage = (
            self.train_num_envs_per_stage // self.cfg.env.train.group_size
        )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs
                // self._world_size
                // self.num_pipeline_stages
            )
            self.eval_num_group_envs_per_stage = (
                self.eval_num_envs_per_stage // self.cfg.env.eval.group_size
            )

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            model_config, input_processor = get_vla_model_config_and_processor(
                self.cfg.actor
            )
            self.hf_model.setup_config_and_processor(
                model_config, self.cfg, input_processor
            )

        self.hf_model.eval()

        self.setup_sample_params()
        if self.enable_offload:
            self._offload_model()

    def load_checkpoint(self, load_path):
        try:
            model_dict = torch.load(load_path)
            self.hf_model.load_state_dict(model_dict)
        except Exception as e:
            logging.warning("Failed to load checkpoint from %s: %s", load_path, e)
            pass

    def setup_sample_params(self):
        """Setup sampling parameters for rollout."""
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, do_sample=True, mode="train"):
        """Predict actions using the Huggingface model."""
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )
        kwargs["do_sample"] = do_sample

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
        ]:
            kwargs = {"mode": mode}

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def get_dones_and_rewards(
        self, env_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_batch: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards) tensors. Both can be None if this is the first step.
        """
        # First step: no rewards yet, only dones
        if env_batch["rewards"] is None:
            return env_batch["dones"].bool().cpu().contiguous(), None

        dones = env_batch["dones"].bool().cpu().contiguous()
        rewards = env_batch["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head"):
                final_obs_list = env_batch.get("final_obs", None)
                batch_env_sizes = env_batch.get("_batch_env_sizes", None)

                if (
                    final_obs_list is not None
                    and isinstance(final_obs_list, list)
                    and batch_env_sizes is not None
                ):
                    final_values_all = self._compute_bootstrap_values(
                        final_obs_list, batch_env_sizes, dones
                    )
                    rewards[:, -1] += self.cfg.algorithm.gamma * final_values_all

        return dones, rewards

    def _compute_bootstrap_values(
        self,
        final_obs_list: list[dict | None],
        batch_env_sizes: list[int],
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bootstrap values for done environments from final_obs list.

        Args:
            final_obs_list: List of final_obs, one per batch (may contain None)
            batch_env_sizes: List of environment counts per batch
            dones: Dones tensor of shape [total_envs, num_chunks]

        Returns:
            Bootstrap values tensor of shape [total_envs]
        """
        last_step_dones = dones[:, -1]  # [bsz, ] on CPU
        final_values_all = torch.zeros(dones.shape[0], dtype=torch.float32)

        env_idx = 0
        for i, final_obs in enumerate(final_obs_list):
            batch_size = batch_env_sizes[i]
            if final_obs is not None:
                batch_env_range = slice(env_idx, env_idx + batch_size)
                batch_dones = last_step_dones[batch_env_range]

                # Only process if there are done environments in this batch
                if batch_dones.any():
                    with torch.no_grad():
                        actions, result = self.predict(final_obs)
                        if "prev_values" in result:
                            _final_values = result["prev_values"]
                        else:
                            _final_values = torch.zeros_like(actions[:, 0])

                    if _final_values.ndim > 1:
                        batch_final_values = _final_values[:, 0].cpu()
                    else:
                        batch_final_values = _final_values.cpu()

                    final_values_all[batch_env_range][batch_dones] = batch_final_values[
                        batch_dones
                    ]

            env_idx += batch_size

        return final_values_all

    def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = self.recv(self.actor_group_name, src_rank=self._rank)

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)

    def get_batch(self, input_channel: Channel, num_group_envs: int):
        """Get a batch of group environment outputs from the input channel."""
        env_outputs: list[EnvOutput] = []
        for _ in range(num_group_envs):
            env_output: EnvOutput = input_channel.get(key=self.get_batch_cnt)
            env_outputs.append(env_output)
        env_batch = EnvOutput.merge_to_batch(env_outputs)
        self.get_batch_cnt += 1
        return env_batch, env_outputs

    def put_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | list | dict,
        env_outputs: list[EnvOutput],
        num_group_envs: int,
    ):
        """Put actions into the output channel to send to the environment.

        It first splits the actions according to the number of environment groups,
        then sends each split action to the corresponding environment based on the env_outputs' worker_rank and stage_id.
        An env ID is also sent along with the action to identify which env the action belongs to.

        Args:
            output_channel (Channel): Channel to send actions to the environment.
            chunk_actions (torch.Tensor | list | dict): Actions to be sent to the environment.
            env_outputs (list[EnvOutput]): List of EnvOutput corresponding to each environment.
            num_group_envs (int): Number of group environment.
        """
        split_actions = EnvOutput.split_value(chunk_actions, split_size=num_group_envs)
        assert len(env_outputs) == num_group_envs, (
            f"Number of env outputs {len(env_outputs)} does not match the num_group_envs per stage in rollout {num_group_envs}"
        )
        for action, env_output in zip(split_actions, env_outputs):
            assert env_output.num_group_envs == 1, (
                "The put_actions should only put the actions for one group env number."
            )
            assert len(env_output.group_env_ids) == 1, (
                "The put_actions should only put the actions for one group env number."
            )
            # The key is (worker_rank, stage_id) to ensure the action is sent to the correct env worker and stage
            # The group env ID is not added to the key but sent as part of the item to avoid creating too many queues in the channel (each key creates a separate queue), because num_group_envs can be large while num_pipeline_stages and env worker size are relatively small.
            output_channel.put(
                item=(env_output.group_env_ids[0], action),
                key=(env_output.worker_rank, env_output.stage_id),
            )

    def put_train_batch(self, actor_channel: Channel, stage_id: int):
        """Put the rollout batch into the actor channel for training."""
        # send rollout_batch to actor
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        # Split only along batch dimension, keep time dimension as-is.
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i])

    def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        """Generate rollout by interacting with the environment.

        Args:
            input_channel (Channel): Channel to receive environment outputs.
            output_channel (Channel): Channel to send actions to the environment.
            actor_channel (Channel): Channel to send rollout data to the actor for training.
        """
        if self.enable_offload:
            self._load_model()
        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_episode_steps
            // self.cfg.actor.model.num_action_chunks
        )

        self.device_lock.acquire()
        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    # Get env output
                    self.device_lock.release()  # Release lock to allow EnvWorker to run
                    env_batch, env_outputs = self.get_batch(
                        input_channel, self.train_num_group_envs_per_stage
                    )
                    self.device_lock.acquire()  # Re-acquire lock for prediction

                    dones, rewards = self.get_dones_and_rewards(env_batch)

                    # Predict actions
                    with self.worker_timer():
                        actions, result = self.predict(env_batch["obs"])

                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=result["forward_inputs"],
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)

                    # Send actions to env
                    self.put_actions(
                        output_channel,
                        actions,
                        env_outputs,
                        self.train_num_group_envs_per_stage,
                    )

            for stage_id in range(self.num_pipeline_stages):
                self.device_lock.release()  # Release lock to allow EnvWorker to run
                env_batch, _ = self.get_batch(
                    input_channel, self.train_num_group_envs_per_stage
                )
                self.device_lock.acquire()  # Re-acquire lock for prediction

                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards = self.get_dones_and_rewards(env_batch)
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].rewards.append(rewards)

                with self.worker_timer():
                    actions, result = self.predict(env_batch["obs"])

                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )

        for stage_id in range(self.num_pipeline_stages):
            self.put_train_batch(actor_channel, stage_id)

        if self.enable_offload:
            self._offload_model()

        self.device_lock.release()

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        """Evaluate the model by interacting with the environment."""
        if self.enable_offload:
            self._load_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_episode_steps
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_batch, env_outputs = self.get_batch(
                        input_channel, self.eval_num_envs_per_stage
                    )
                    actions, _ = self.predict(env_batch["obs"], mode="eval")
                    self.put_actions(
                        output_channel,
                        actions,
                        env_outputs,
                        self.eval_num_envs_per_stage,
                    )

        if self.enable_offload:
            self._offload_model()

    def _offload_model(self):
        """Offload the model to CPU to save GPU memory."""
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def _load_model(self):
        self.hf_model = self.hf_model.to(self.device)
