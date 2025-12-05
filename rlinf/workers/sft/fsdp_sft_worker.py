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

import os

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.data.io_struct import RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from openpi.training import config as _config
import openpi.training.data_loader as _data
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
    seq_mean_token_mean,
    seq_mean_token_sum,
)
from rlinf.workers.rollout.utils import RankMapper

class EmbodiedSFTActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(self._world_size,), mesh_dim_names=["fsdp"]
        )
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self._replay_buffer_name = cfg.actor.channel.queue_name

        self.channel = self.connect_channel(cfg.actor.channel.name)
        training_config_name = cfg.actor.get("config_name", "pi0_maniskill")
        data_loader_config = _config.get_config(training_config_name)
        self.data_loader = _data.create_data_loader(data_loader_config, framework="pytorch", shuffle=True)
        self.sft_loss_weight = cfg.actor.get("sft_loss_weight", 0.1)

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.runner.get("resume_dir", None) is not None:
            actor_checkpoint_path = os.path.join(self.cfg.runner.resume_dir, "actor")
            self.load_checkpoint(actor_checkpoint_path)

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def run_training(self):
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        self.model.train()
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)

        with torch.no_grad():
            for key, value in self.rollout_batch.items():
                if key in ["dones", "prev_values"]:
                    value = value[:-1]
                if "env_info" in key:
                    continue
                if value is None:
                    continue
                value = value.reshape(rollout_size, *value.shape[2:])

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            for train_global_batch in self.data_loader:
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                ), "BS not supported."
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )
                train_micro_batch = get_iterator_k_split(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )
                self.optimizer.zero_grad()
                for idx, data in enumerate(train_micro_batch):
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    real_observation, real_actions = data #self.get_next_real_data_batch()
                    observation = jax.tree.map(lambda x: x.to(self.device), real_observation)  # noqa: PLW2901
                    actions = real_actions.to(torch.float32)  # noqa: PLW2901
                    actions = actions.to(self.device)  # noqa: PLW2
                    sft_losses = self.model(data=dict(observation=observation, actions=actions), mode="sft")
                    # Ensure losses is a tensor and handle different return types
                    if isinstance(sft_losses, list | tuple):
                        sft_losses = torch.stack(sft_losses)
                    elif not isinstance(sft_losses, torch.Tensor):
                        sft_losses = torch.tensor(sft_losses, device=self.device, dtype=torch.float32)

                    sft_loss = sft_losses.mean()
                    metrics_data["sft_loss"] = sft_loss.clone().detach().item()
                    loss = self.sft_loss_weight * sft_loss # self.cfg.actor.get("sft_loss_weight", 0
                    
                    loss /= self.gradient_accumulation
                    loss.backward()
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()
                    print(f"""Loss: {metrics_data["loss"]}""")

                    metrics_data["loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)
                torch.cuda.empty_cache()
                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step):
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
