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

import gc

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.data.io_struct import EmbodiedRolloutResult, EnvOutput
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
        env_world_size = self.placement.get_world_size("env")

        # Batching parameters
        # NOTE: The num_envs is already divided by num_pipeline_stages and env_world_size
        self.train_batch_size_per_stage = (
            self.cfg.env.train.num_envs * env_world_size // self._world_size
        )
        self.train_group_size = self.cfg.env.train.group_size
        self.train_num_groups_per_stage = (
            self.train_batch_size_per_stage // self.train_group_size
        )

        self.eval_batch_size_per_stage = (
            self.cfg.env.eval.num_envs * env_world_size // self._world_size
        )
        self.eval_group_size = self.cfg.env.eval.group_size
        self.eval_num_groups_per_stage = (
            self.eval_batch_size_per_stage // self.eval_group_size
        )

    def init_worker(self):
        """Initialize the rollout worker model and sampling parameters."""
        # NOTE: because pi series have some different dtype params, we can not call `to`
        # after get_model, here we simply change actor.model.precision to rollout.precision
        # and after get_model we change it back. THIS CODE SHOULD BE REFACTORED SOON.
        with open_dict(self.cfg):
            original_precision = self.cfg.actor.model.precision
            self.cfg.actor.model.precision = self.cfg.rollout.precision
        self.hf_model = get_model(self.cfg.rollout.model_dir, self.cfg.actor.model)
        with open_dict(self.cfg):
            self.cfg.actor.model.precision = original_precision

        if self.cfg.actor.model.model_name in ["openvla", "openvla_oft"]:
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

        if self.cfg.actor.model.model_name in ["openpi", "mlp_policy", "gr00t"]:
            kwargs = {"mode": mode}

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def update_env_output(self, stage_id: int, env_batch: dict[str, torch.Tensor]):
        """Update the environment output buffer with new data."""
        # first step for env_batch
        if env_batch["rewards"] is None:
            self.buffer_list[stage_id].dones.append(
                env_batch["dones"].contiguous().cpu()
            )
            return

        self.buffer_list[stage_id].rewards.append(
            env_batch["rewards"].cpu().contiguous()
        )
        self.buffer_list[stage_id].dones.append(
            env_batch["dones"].bool().cpu().contiguous()
        )

        # Note: currently this is not correct for chunk-size>1 with partial reset
        if env_batch["dones"].any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head"):
                dones = env_batch["dones"]

                final_obs = env_batch["final_obs"]
                with torch.no_grad():
                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                self.buffer_list[stage_id].rewards[-1][:, -1] += (
                    self.cfg.algorithm.gamma * final_values.cpu()
                )

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

    def get_batch(self, input_channel: Channel, num_groups: int):
        """Get a batch of environment outputs from the input channel."""
        env_outputs: list[EnvOutput] = []
        env_batches: list[dict[str, torch.Tensor]] = []
        for _ in range(num_groups):
            env_output: EnvOutput = input_channel.get(key=self.get_batch_cnt)
            env_outputs.append(env_output)
            env_batches.append(env_output.to_batch())
        env_batch = EnvOutput.merge_batches(env_batches)
        self.get_batch_cnt += 1
        return env_batch, env_outputs

    def put_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | list | dict,
        env_outputs: list[EnvOutput],
        num_groups: int,
    ):
        """Put actions into the output channel to send to the environment.

        It first splits the actions according to the number of environment groups,
        then sends each split action to the corresponding environment group based on the env_outputs' worker_rank and stage_id.
        A group ID is also sent along with the action to identify which group the action belongs to.

        Args:
            output_channel (Channel): Channel to send actions to the environment.
            chunk_actions (torch.Tensor | list | dict): Actions to be sent to the environment.
            env_outputs (list[EnvOutput]): List of EnvOutput corresponding to each environment group.
            num_groups (int): Number of environment groups.
        """
        split_actions = EnvOutput.split_value(chunk_actions, num_groups)
        assert len(env_outputs) == num_groups, (
            f"Number of env outputs {len(env_outputs)} does not match the num_groups per stage in rollout {num_groups}"
        )
        for action, env_output in zip(split_actions, env_outputs):
            assert env_output.num_groups == 1, (
                "The put_actions should only put the actions for one env group."
            )
            assert len(env_output.group_ids) == 1, (
                "The put_actions should only put the actions for one env group."
            )
            # The key is (worker_rank, stage_id) to ensure the action is sent to the correct env worker and stage
            # The group ID is not added to the key but sent as part of the item to avoid creating too many queues in the channel (each key creates a separate queue), because num_groups can be large while num_pipeline_stages and env worker size are relatively small.
            output_channel.put(
                item=(env_output.group_ids[0], action),
                key=(env_output.worker_rank, env_output.stage_id),
            )

    def put_train_batch(self, actor_channel: Channel, stage_id: int):
        """Put the rollout batch into the actor channel for training."""
        # send rollout_batch to actor
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
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
            EmbodiedRolloutResult() for _ in range(self.num_pipeline_stages)
        ]

        self.device_lock.acquire()
        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.cfg.algorithm.n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    # Get env output
                    self.device_lock.release()  # Release lock to allow EnvWorker to run
                    env_batch, env_outputs = self.get_batch(
                        input_channel, self.train_num_groups_per_stage
                    )
                    self.device_lock.acquire()  # Re-acquire lock for prediction

                    # Update buffer and predict actions
                    self.update_env_output(stage_id, env_batch)
                    with self.worker_timer():
                        actions, result = self.predict(env_batch["obs"])
                    self.buffer_list[stage_id].append_result(result)

                    # Send actions to env
                    self.put_actions(
                        output_channel,
                        actions,
                        env_outputs,
                        self.train_num_groups_per_stage,
                    )

            for stage_id in range(self.num_pipeline_stages):
                self.device_lock.release()  # Release lock to allow EnvWorker to run
                env_batch, _ = self.get_batch(
                    input_channel, self.train_num_groups_per_stage
                )
                self.device_lock.acquire()  # Re-acquire lock for prediction

                self.update_env_output(stage_id, env_batch)
                with self.worker_timer():
                    actions, result = self.predict(env_batch["obs"])
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

        for _ in range(self.cfg.algorithm.n_eval_chunk_steps):
            for _ in range(self.num_pipeline_stages):
                env_batch, env_outputs = self.get_batch(
                    input_channel, self.eval_num_groups_per_stage
                )
                actions, _ = self.predict(env_batch["obs"], mode="eval")
                self.put_actions(
                    output_channel, actions, env_outputs, self.eval_num_groups_per_stage
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
