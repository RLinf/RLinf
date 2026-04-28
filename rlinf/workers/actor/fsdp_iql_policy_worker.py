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

import asyncio
import os
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.iql_mlp_policy import IQLMLPPolicy
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def iql_expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Expectile loss: weight = |expectile - 1(diff < 0)|, return weight * diff^2."""
    neg_mask = (diff < 0).to(dtype=diff.dtype)
    return torch.abs(float(expectile) - neg_mask) * diff.square()


class EmbodiedIQLFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.offline_torch_compile = True
        self.offline_compile_mode = "reduce-overhead"
        self.offline_strict_mode = True
        self.offline_metric_sync_interval = 1000

        self._obs_dim = None
        self._action_dim = None
        self._offline_dataset = None
        self.offline_data_loader = None
        self.offline_data_iter = None
        self._data_epoch = 0
        self._data_iter_offset = 0
        self._dataset_size = 0

        self.critic_model = None
        self.target_model = None
        self.value_model = None
        self.target_model_initialized = False
        self.eval_returns: list[tuple[int, float]] = []
        self._global_step = 0

        self.model = None
        self.optimizer = None
        self.qf_optimizer = None
        self.vf_optimizer = None
        self.lr_scheduler = None
        self.qf_lr_scheduler = None
        self.vf_lr_scheduler = None
        self._compiled_update_step = None
        self._critic_params: list[torch.Tensor] = []
        self._target_params: list[torch.Tensor] = []

        self._use_fsdp_wrap = False

    def build_offline_dataloader(self) -> None:
        """Build per-rank ``DataLoader`` + ``DistributedSampler`` (SFT worker pattern)."""
        from rlinf.data.datasets.d4rl import build_d4rl_dataset_from_cfg

        dataset_cfg = self.cfg.get("data", {})
        dataset_type = dataset_cfg.get("dataset_type", None)
        if dataset_type is None:
            raise ValueError(
                "data.dataset_type is required for offline IQL (e.g. data.dataset_type: d4rl)."
            )
        if str(dataset_type).lower() != "d4rl":
            raise AssertionError(
                f"offline IQL only supports dataset_type='d4rl', got {dataset_type!r}."
            )

        self._offline_dataset = build_d4rl_dataset_from_cfg(self.cfg)
        per_rank_bs = int(self.batch_size)

        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                self._offline_dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=bool(dataset_cfg.get("shuffle", True)),
                seed=int(dataset_cfg.get("seed", 42)),
                drop_last=True,
            )
        else:
            sampler = None

        nw = int(dataset_cfg.get("num_workers", 0))
        self.offline_data_loader = DataLoader(
            self._offline_dataset,
            batch_size=per_rank_bs,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=nw,
            drop_last=True,
            pin_memory=bool(dataset_cfg.get("pin_memory", True)),
            persistent_workers=(
                nw > 0 and bool(dataset_cfg.get("persistent_workers", True))
            ),
        )
        if len(self._offline_dataset) < per_rank_bs:
            raise ValueError(
                f"Dataset size ({len(self._offline_dataset)}) must be >= batch_size ({per_rank_bs})."
            )

        self._data_epoch = 0
        self._data_iter_offset = 0
        if sampler is not None:
            sampler.set_epoch(self._data_epoch)
        self.offline_data_iter = iter(self.offline_data_loader)

        self._obs_dim, self._action_dim = self._offline_dataset.get_obs_action_dims()
        self._dataset_size = self._offline_dataset.get_dataset_size()
        self.log_info(
            "IQL offline: DataLoader with "
            f"{self._dataset_size} samples, per_rank_batch_size={per_rank_bs}."
        )
        if self._obs_dim <= 0 or self._action_dim <= 0:
            raise ValueError("Failed to infer obs_dim/action_dim from offline dataset.")

    def setup_iql_components(self):
        """Initialize IQL-specific offline components (SFT-style: data lives on the actor).

        Batches come from the actor's ``offline_data_loader`` (see
        ``build_offline_dataloader``).
        """
        actor_cfg = self.cfg.actor
        self.offline_torch_compile = bool(actor_cfg.get("offline_torch_compile", True))
        self.offline_compile_mode = str(
            actor_cfg.get("offline_compile_mode", "reduce-overhead")
        )
        self.offline_strict_mode = bool(actor_cfg.get("offline_strict_mode", True))
        self.offline_metric_sync_interval = int(
            actor_cfg.get("offline_metric_sync_interval", 1000)
        )

        self._seed = int(actor_cfg.seed)
        self._save_dir = self.cfg.runner.get("save_dir", None)
        if self._save_dir is None:
            runner_logger = self.cfg.runner.get("logger", None)
            if runner_logger is not None:
                log_path = runner_logger.get("log_path", ".")
                exp_name = runner_logger.get("experiment_name", "offline")
                self._save_dir = os.path.join(log_path, exp_name)
            else:
                self._save_dir = "."
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
        np.random.seed(self._seed)
        raw_device = self.device
        if isinstance(raw_device, torch.device):
            self.device = raw_device
        elif isinstance(raw_device, int):
            self.device = (
                torch.device(f"cuda:{raw_device}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        elif raw_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(raw_device)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        self.discount = float(self.cfg.algorithm.gamma)
        self.tau = float(self.cfg.algorithm.tau)
        self.expectile = float(self.cfg.algorithm.expectile)
        self.temperature = float(self.cfg.algorithm.temperature)
        global_bs = int(actor_cfg.global_batch_size)
        assert global_bs % self._world_size == 0, (
            f"actor.global_batch_size ({global_bs}) must be divisible by world_size ({self._world_size})"
        )
        self.batch_size = global_bs // self._world_size
        if int(actor_cfg.micro_batch_size) != self.batch_size:
            raise ValueError(
                f"actor.micro_batch_size must equal per-rank batch size (actor.global_batch_size // world_size); got micro_batch_size={actor_cfg.micro_batch_size}, expected {self.batch_size}."
            )
        self.build_offline_dataloader()

    def init_worker(self):
        """Initialize IQL worker"""
        self.setup_iql_components()
        os.makedirs(self._save_dir or ".", exist_ok=True)
        self.setup_model_and_optimizer(initialize_target=True)
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def setup_model_and_optimizer(self, initialize_target: bool = True) -> None:
        """Setup models, optimizer and scheduler.

        When offline_torch_compile is true, ensure fsdp_config has use_orig_params: true.
        """
        if self._obs_dim is None or self._action_dim is None:
            raise ValueError(
                "actor.model.obs_dim/action_dim must be set before actor init."
            )
        obs_dim = int(self._obs_dim)
        action_dim = int(self._action_dim)

        module = self.model_provider_func(obs_dim, action_dim, type_name="actor")
        critic_module = self.model_provider_func(
            obs_dim, action_dim, type_name="critic"
        )
        value_module = self.model_provider_func(obs_dim, action_dim, type_name="value")

        if initialize_target:
            target_module = self.model_provider_func(
                obs_dim, action_dim, type_name="critic"
            )

        use_fsdp_wrap = self.cfg.actor.get("use_fsdp_wrap", False)
        if use_fsdp_wrap:
            self.model = self._strategy.wrap_model(
                model=module, device_mesh=self._device_mesh
            )
            self.critic_model = self._strategy.wrap_model(
                model=critic_module, device_mesh=self._device_mesh
            )
            self.value_model = self._strategy.wrap_model(
                model=value_module, device_mesh=self._device_mesh
            )
            if initialize_target:
                self.target_model = self._strategy.wrap_model(
                    model=target_module, device_mesh=self._device_mesh
                )
        else:
            self.model = module
            self.critic_model = critic_module
            self.value_model = value_module
            if initialize_target:
                self.target_model = target_module

        self._use_fsdp_wrap = use_fsdp_wrap
        self.log_info(f"IQL offline: use_fsdp_wrap={use_fsdp_wrap}.")

        # Initialize target model
        if initialize_target:
            self.target_model.load_state_dict(self.critic_model.state_dict())
            self.target_model.eval()
            self.target_model_initialized = True
            for p in self.target_model.parameters():
                p.requires_grad_(False)
            self._critic_params = list(self.critic_model.parameters())
            self._target_params = list(self.target_model.parameters())

        actor_optim_cfg = self.cfg.actor.optim
        critic_optim_cfg = self.cfg.actor.critic_optim
        value_optim_cfg = self.cfg.actor.value_optim
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(actor_optim_cfg.lr),
            betas=(
                float(actor_optim_cfg.adam_beta1),
                float(actor_optim_cfg.adam_beta2),
            ),
            eps=float(actor_optim_cfg.adam_eps),
        )
        self.vf_optimizer = torch.optim.Adam(
            self.value_model.parameters(),
            lr=float(value_optim_cfg.lr),
            betas=(
                float(value_optim_cfg.adam_beta1),
                float(value_optim_cfg.adam_beta2),
            ),
            eps=float(value_optim_cfg.adam_eps),
        )
        self.qf_optimizer = torch.optim.Adam(
            self.critic_model.parameters(),
            lr=float(critic_optim_cfg.lr),
            betas=(
                float(critic_optim_cfg.adam_beta1),
                float(critic_optim_cfg.adam_beta2),
            ),
            eps=float(critic_optim_cfg.adam_eps),
        )
        self.build_lr_schedulers()
        self.log_info("IQL offline: using separated Adam optimizers.")
        if self.offline_torch_compile and self._use_fsdp_wrap:
            self.log_warning(
                "IQL offline: disable torch.compile when use_fsdp_wrap=True "
            )
            self.offline_torch_compile = False
        if self.offline_torch_compile:
            self._compiled_update_step = torch.compile(
                self.update_step_forward,
                mode=self.offline_compile_mode,
                dynamic=False,
                fullgraph=False,
            )
            self.log_info("IQL offline: torch.compile enabled (fullgraph=False).")

    def build_lr_schedulers(self) -> None:
        """Build IQL schedulers (actor uses cosine annealing)."""
        assert self.optimizer is not None, (
            "setup_model_and_optimizer must be called first."
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(self.cfg.runner.max_steps)),
            eta_min=0.0,
        )
        self.qf_lr_scheduler = None
        self.vf_lr_scheduler = None

    def build_iql_module(
        self, obs_dim: int, action_dim: int, type_name: str
    ) -> torch.nn.Module:
        model_cfg = self.cfg.actor.model
        if model_cfg.get("num_action_chunks", None) is None:
            raise ValueError(
                "actor.model.num_action_chunks is required for IQL offline training."
            )
        num_chunks = int(model_cfg.num_action_chunks)
        if self.cfg.algorithm.get("hidden_dims", None) is None:
            raise ValueError(
                "algorithm.hidden_dims is required for IQL offline training."
            )
        base_hidden: Sequence[int] = tuple(self.cfg.algorithm.hidden_dims)
        iql_config: dict[str, Any] = {"type": type_name, "hidden_dims": base_hidden}
        if type_name == "actor":
            if model_cfg.get("iql_config", None) is None:
                raise ValueError("actor.model.iql_config is required for IQL actor.")
            ic = OmegaConf.to_container(model_cfg.iql_config, resolve=True)
            if not isinstance(ic, dict):
                raise ValueError("actor.model.iql_config must be a mapping.")
            required_ic = (
                "hidden_dims",
                "dropout_rate",
                "state_dependent_std",
                "log_std_min",
                "log_std_max",
            )
            missing = [k for k in required_ic if k not in ic]
            if missing:
                raise ValueError(
                    f"actor.model.iql_config missing required keys: {missing}"
                )
            iql_config = {
                "type": type_name,
                "hidden_dims": tuple(ic["hidden_dims"]),
                "dropout_rate": ic["dropout_rate"],
                "state_dependent_std": bool(ic["state_dependent_std"]),
                "log_std_min": float(ic["log_std_min"]),
                "log_std_max": float(ic["log_std_max"]),
            }
        elif (
            "dropout_rate" in self.cfg.algorithm
            and self.cfg.algorithm.dropout_rate is not None
        ):
            iql_config["dropout_rate"] = self.cfg.algorithm.dropout_rate
        model = IQLMLPPolicy(
            obs_dim,
            action_dim,
            num_action_chunks=num_chunks,
            add_value_head=(type_name == "actor"),
            add_q_head=False,
        )
        model.configure_iql(iql_config)
        return model.to(self.device)

    def model_provider_func(self, obs_dim: int, action_dim: int, type_name: str):
        """SAC-style provider entry for IQL modules."""
        if type_name in {"actor", "value"}:
            return self.build_iql_module(obs_dim, action_dim, type_name=type_name)
        if type_name == "critic":
            return self.build_critic_module(obs_dim, action_dim)
        raise ValueError(f"Unsupported provider type: {type_name}")

    def build_critic_module(self, obs_dim: int, action_dim: int) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {
                "q1": self.build_iql_module(obs_dim, action_dim, type_name="critic"),
                "q2": self.build_iql_module(obs_dim, action_dim, type_name="critic"),
            }
        ).to(self.device)

    def forward_critic_module(
        self,
        critic_module: torch.nn.ModuleDict,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = critic_module["q1"](
            forward_type=ForwardType.IQL_CRITIC,
            observations=observations,
            actions=actions,
        )
        q2 = critic_module["q2"](
            forward_type=ForwardType.IQL_CRITIC,
            observations=observations,
            actions=actions,
        )
        return q1, q2

    def forward_value(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.vf_optimizer is not None, (
            "setup_model_and_optimizer must be called first."
        )
        with torch.no_grad():
            q1_t, q2_t = self.forward_critic_module(self.target_model, obs, actions)
            q_t = torch.min(q1_t, q2_t)
        v = self.value_model(forward_type=ForwardType.IQL_VALUE, observations=obs)
        value_loss = iql_expectile_loss(q_t - v, self.expectile).mean()
        self.vf_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.vf_optimizer.step()
        return v, value_loss

    def forward_actor(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.optimizer is not None, (
            "setup_model_and_optimizer must be called first."
        )
        with torch.no_grad():
            new_v = self.value_model(
                forward_type=ForwardType.IQL_VALUE, observations=obs
            )
            q1_t, q2_t = self.forward_critic_module(self.target_model, obs, actions)
            q_t = torch.min(q1_t, q2_t)
            adv = q_t - new_v
            exp_a = torch.exp(adv * self.temperature).clamp(max=100.0)
        log_probs = self.model(
            forward_type=ForwardType.IQL_ACTOR, observations=obs, actions=actions
        )
        actor_loss = -(exp_a * log_probs).mean()
        self.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return adv, actor_loss

    def forward_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.qf_optimizer is not None, (
            "setup_model_and_optimizer must be called first."
        )
        with torch.no_grad():
            next_v = self.value_model(
                forward_type=ForwardType.IQL_VALUE, observations=next_obs
            )
            target_q = rewards + self.discount * masks * next_v
        q1, q2 = self.forward_critic_module(self.critic_model, obs, actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        self.qf_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.qf_optimizer.step()
        return q1, q2, critic_loss

    def update_step_forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled path: one IQL step (value + actor + critic + target)."""
        v, value_loss = self.forward_value(obs, actions)
        adv, actor_loss = self.forward_actor(obs, actions)
        q1, q2, critic_loss = self.forward_critic(
            obs, actions, rewards, masks, next_obs
        )
        self.soft_update_target_model()
        return torch.stack(
            [
                critic_loss.detach(),
                q1.detach().mean(),
                q2.detach().mean(),
                value_loss.detach(),
                v.detach().mean(),
                actor_loss.detach(),
                adv.detach().mean(),
                adv.detach().std(),
            ]
        )

    def _next_train_batch(self) -> dict[str, torch.Tensor]:
        if self.offline_data_iter is None:
            raise RuntimeError("Offline DataLoader is not initialized.")
        try:
            batch = next(self.offline_data_iter)
            self._data_iter_offset += 1
        except StopIteration:
            self._data_epoch += 1
            if hasattr(self.offline_data_loader, "sampler") and hasattr(
                self.offline_data_loader.sampler, "set_epoch"
            ):
                self.offline_data_loader.sampler.set_epoch(self._data_epoch)
            self.offline_data_iter = iter(self.offline_data_loader)
            batch = next(self.offline_data_iter)
            self._data_iter_offset = 1
        return batch

    def _load_offline_data_state(self, state_payload: dict[str, Any]) -> None:
        """Restore dataloader position after checkpoint (aligned with SFT VLM worker)."""
        if "data_epoch" not in state_payload or "data_iter_offset" not in state_payload:
            return
        self._data_epoch = int(state_payload.get("data_epoch", 0))
        self._data_iter_offset = int(state_payload.get("data_iter_offset", 0))

        if self.offline_data_loader is None:
            return

        if hasattr(self.offline_data_loader, "sampler") and hasattr(
            self.offline_data_loader.sampler, "set_epoch"
        ):
            self.offline_data_loader.sampler.set_epoch(self._data_epoch)

        self.offline_data_iter = iter(self.offline_data_loader)
        for _ in range(self._data_iter_offset):
            try:
                next(self.offline_data_iter)
            except StopIteration:
                self._data_epoch += 1
                if hasattr(self.offline_data_loader, "sampler") and hasattr(
                    self.offline_data_loader.sampler, "set_epoch"
                ):
                    self.offline_data_loader.sampler.set_epoch(self._data_epoch)
                self.offline_data_iter = iter(self.offline_data_loader)

    def prepare_batch(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        required = ["observations", "actions", "rewards", "masks", "next_observations"]
        prepared_batch: dict[str, torch.Tensor] = {}
        for key in required:
            if key not in batch:
                raise KeyError(f"prepare_batch: missing key '{key}' in batch.")
            value = batch[key]
            if isinstance(value, torch.Tensor):
                tensor = value
                if tensor.dtype != torch.float32:
                    tensor = tensor.float()
                if tensor.device != self.device:
                    tensor = tensor.to(self.device, non_blocking=True)
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            prepared_batch[key] = tensor
        return prepared_batch

    def _pack_train_batch(
        self, prepared: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            prepared["observations"],
            prepared["actions"],
            prepared["rewards"],
            prepared["masks"],
            prepared["next_observations"],
        )

    # Training
    @Worker.timer("update_one_epoch")
    def update_one_epoch(
        self,
        batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> dict[str, Any]:
        self.model.train()
        self.critic_model.train()
        self.value_model.train()

        obs, actions, rewards, masks, next_obs = batch
        assert self.lr_scheduler is not None, (
            "setup_model_and_optimizer must be called first."
        )
        if int(obs.shape[0]) != self.batch_size:
            raise ValueError(
                f"Offline IQL requires static batch size. Got {int(obs.shape[0])}, expected {self.batch_size}. Use a fixed-size sampler (e.g., drop_last=True)."
            )
        use_compiled_update = bool(self.offline_torch_compile)
        if use_compiled_update:
            if hasattr(torch, "compiler") and hasattr(
                torch.compiler, "cudagraph_mark_step_begin"
            ):
                torch.compiler.cudagraph_mark_step_begin()

            try:
                flat = self._compiled_update_step(
                    obs,
                    actions,
                    rewards,
                    masks,
                    next_obs,
                )
            except Exception as e:
                raise RuntimeError(
                    "torch.compile IQL update step failed; set actor.offline_torch_compile to false or fix the underlying error."
                ) from e
            if use_compiled_update:
                metric_device = obs.device
                result = {
                    "critic_loss": flat[0].detach(),
                    "q1": flat[1].detach(),
                    "q2": flat[2].detach(),
                    "value_loss": flat[3].detach(),
                    "v": flat[4].detach(),
                    "actor_loss": flat[5].detach(),
                    "adv_mean": flat[6].detach(),
                    "adv_std": flat[7].detach(),
                    "lr_actor": torch.tensor(
                        float(self.optimizer.param_groups[0]["lr"]),
                        device=metric_device,
                    ),
                    "lr_value": torch.tensor(
                        float(self.vf_optimizer.param_groups[0]["lr"]),
                        device=metric_device,
                    ),
                    "lr_critic": torch.tensor(
                        float(self.qf_optimizer.param_groups[0]["lr"]),
                        device=metric_device,
                    ),
                    "use_fsdp_wrap": self._use_fsdp_wrap,
                }
                return result

        flat = self.update_step_forward(
            obs,
            actions,
            rewards,
            masks,
            next_obs,
        )
        metric_device = obs.device
        result = {
            "critic_loss": flat[0].detach(),
            "q1": flat[1].detach(),
            "q2": flat[2].detach(),
            "value_loss": flat[3].detach(),
            "v": flat[4].detach(),
            "actor_loss": flat[5].detach(),
            "adv_mean": flat[6].detach(),
            "adv_std": flat[7].detach(),
            "lr_actor": torch.tensor(
                float(self.optimizer.param_groups[0]["lr"]),
                device=metric_device,
            ),
            "lr_value": torch.tensor(
                float(self.vf_optimizer.param_groups[0]["lr"]),
                device=metric_device,
            ),
            "lr_critic": torch.tensor(
                float(self.qf_optimizer.param_groups[0]["lr"]),
                device=metric_device,
            ),
            "use_fsdp_wrap": self._use_fsdp_wrap,
        }
        return result

    def aggregate_update_info(
        self, summed: Optional[dict[str, torch.Tensor]], update_info: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Merge update_info into summed; skip non-tensor keys like use_fsdp_wrap."""
        if summed is None:
            summed = {}
            for k, v in update_info.items():
                if k == "use_fsdp_wrap" or not isinstance(v, torch.Tensor):
                    continue
                summed[k] = v.detach().clone()
        else:
            for k, v in update_info.items():
                if k == "use_fsdp_wrap" or not isinstance(v, torch.Tensor):
                    continue
                summed[k].add_(v.detach())
        return summed

    @Worker.timer("run_training")
    def run_training(self):
        """IQL training with actor-local offline dataloader."""
        assert self.model is not None, (
            "init_worker() must be called before run_training()."
        )
        local_update_steps = max(1, int(self.cfg.runner.local_update_steps))
        if self._global_step < 0:
            self._global_step = 0
        max_steps = int(self.cfg.runner.max_steps)
        remaining_steps = max_steps - int(self._global_step)
        local_update_steps = min(local_update_steps, max(1, remaining_steps))

        # Ensure train mode before update loop
        self.model.train()
        self.critic_model.train()
        self.value_model.train()

        summed_metrics: Optional[dict[str, torch.Tensor]] = None

        # Main update loop
        for _ in range(local_update_steps):
            prepared = self.prepare_batch(self._next_train_batch())
            packed_batch = self._pack_train_batch(prepared)

            update_info = self.update_one_epoch(packed_batch)
            summed_metrics = self.aggregate_update_info(summed_metrics, update_info)

            self._global_step += 1

        # No eval here; runner may call env worker for evaluation.
        mean_metric_dict: dict[str, Any] = {}
        if summed_metrics is not None:
            for k, v in summed_metrics.items():
                mean_metric_dict[k] = float((v / local_update_steps).item())
        mean_metric_dict["__global_step"] = int(self._global_step)
        return mean_metric_dict

    def compute_advantages_and_returns(self):
        """
        IQL doesn't compute rollout advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def set_global_step(self, step: int):
        self._global_step = int(step)
        return None

    def get_policy_state_dict(self) -> dict[str, torch.Tensor]:
        """Return actor policy state dict on its native device for weight syncing."""
        assert self.model is not None, (
            "init_worker() must be called before get_policy_state_dict()."
        )
        if self._use_fsdp_wrap:
            state_dict = self._strategy.get_model_state_dict(
                self.model,
                cpu_offload=False,
                full_state_dict=True,
            )
        else:
            state_dict = self.model.state_dict()
        return state_dict

    async def sync_model_to_rollout(self) -> None:
        """Sync policy weights to rollout workers using the configured weight syncer."""

        if not self._weight_dst_rank_in_rollout:
            self.log_debug(
                f"Actor rank {self._rank} has no rollout weight-sync destination."
            )
            if self.enable_offload:
                if not self.is_optimizer_offloaded:
                    self.offload_optimizer()
                if not self.is_weight_offloaded:
                    self.offload_param_and_grad(True)
            return

        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        if self._use_fsdp_wrap:
            state_dict = self.get_model_state_dict(
                cpu_offload=False, full_state_dict=False
            )
        else:
            state_dict = self.get_policy_state_dict()

        async def send_func(data):
            handles = []
            for rank in self._weight_dst_rank_in_rollout:
                handles.append(
                    self.send(
                        data,
                        dst_group_name=self._rollout_group_name,
                        dst_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            await asyncio.gather(*handles)

        async def recv_func():
            handles = []
            for rank in self._weight_dst_rank_in_rollout:
                handles.append(
                    self.recv(
                        src_group_name=self._rollout_group_name,
                        src_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            metadata_list = await asyncio.gather(*handles)
            metadata = metadata_list[0]
            for other_metadata in metadata_list[1:]:
                if other_metadata != metadata:
                    raise ValueError("Patch metadata differs across rollout ranks")
            return metadata

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
            )

        await self.weight_syncer.sync(state_dict, send_func, version=self.version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    def soft_update_target_model(self):
        assert self.target_model_initialized
        with torch.no_grad():
            if self._use_fsdp_wrap:
                # FSDP may expose params as views; avoid in-place math on these
                # views by blending full state dict tensors out-of-place.
                target_state = self._strategy.get_model_state_dict(
                    self.target_model,
                    cpu_offload=False,
                    full_state_dict=True,
                )
                critic_state = self._strategy.get_model_state_dict(
                    self.critic_model,
                    cpu_offload=False,
                    full_state_dict=True,
                )
                one_minus_tau = 1.0 - float(self.tau)
                mixed_state: dict[str, Any] = {}
                for name, target_value in target_state.items():
                    critic_value = critic_state.get(name, None)
                    if isinstance(target_value, torch.Tensor) and isinstance(
                        critic_value, torch.Tensor
                    ):
                        mixed_state[name] = (
                            target_value * one_minus_tau
                            + critic_value * float(self.tau)
                        )
                    elif critic_value is not None:
                        mixed_state[name] = critic_value
                    else:
                        mixed_state[name] = target_value
                self._strategy.load_model_with_state_dict(
                    self.target_model,
                    mixed_state,
                    cpu_offload=False,
                    full_state_dict=True,
                )
                return

            if self._critic_params and self._target_params:
                torch._foreach_mul_(self._target_params, 1.0 - self.tau)
                torch._foreach_add_(
                    self._target_params, self._critic_params, alpha=self.tau
                )
            else:
                for p, tp in zip(
                    self.critic_model.parameters(), self.target_model.parameters()
                ):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def save_checkpoint(self, save_base_path, step):
        assert self.model is not None, "init_worker() must initialize self.model first."
        os.makedirs(save_base_path, exist_ok=True)

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer] if self.optimizer is not None else [],
            lr_schedulers=[self.lr_scheduler] if self.lr_scheduler is not None else [],
            save_path=os.path.join(save_base_path, "actor_policy"),
            checkpoint_format="local_shard",
        )
        self._strategy.save_checkpoint(
            model=self.critic_model,
            optimizers=[self.qf_optimizer] if self.qf_optimizer is not None else [],
            lr_schedulers=[self.qf_lr_scheduler]
            if self.qf_lr_scheduler is not None
            else [],
            save_path=os.path.join(save_base_path, "critic"),
            checkpoint_format="local_shard",
        )
        self._strategy.save_checkpoint(
            model=self.value_model,
            optimizers=[self.vf_optimizer] if self.vf_optimizer is not None else [],
            lr_schedulers=[self.vf_lr_scheduler]
            if self.vf_lr_scheduler is not None
            else [],
            save_path=os.path.join(save_base_path, "value"),
            checkpoint_format="local_shard",
        )

        # Save iql components
        components_path = os.path.join(save_base_path, "iql_components")
        os.makedirs(components_path, exist_ok=True)
        # save target model
        if self._use_fsdp_wrap:
            target_state = self._strategy.get_model_state_dict(
                self.target_model, cpu_offload=False, full_state_dict=True
            )
        else:
            target_state = self.target_model.state_dict()
            target_state = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in target_state.items()
            }
        torch.save(
            target_state,
            os.path.join(components_path, "target_critic.pt"),
        )
        state_payload = {
            "step": int(step),
            "global_step": int(self._global_step),
            "data_epoch": int(self._data_epoch),
            "data_iter_offset": int(self._data_iter_offset),
        }
        torch.save(state_payload, os.path.join(components_path, "state.pt"))

    def load_checkpoint(self, load_base_path: str):
        assert self.model is not None, "init_worker() must initialize self.model first."

        # Load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer] if self.optimizer is not None else [],
            lr_schedulers=[self.lr_scheduler] if self.lr_scheduler is not None else [],
            load_path=os.path.join(load_base_path, "actor_policy"),
            checkpoint_format="local_shard",
        )
        self._strategy.load_checkpoint(
            model=self.critic_model,
            optimizers=[self.qf_optimizer] if self.qf_optimizer is not None else [],
            lr_schedulers=[self.qf_lr_scheduler]
            if self.qf_lr_scheduler is not None
            else [],
            load_path=os.path.join(load_base_path, "critic"),
            checkpoint_format="local_shard",
        )
        self._strategy.load_checkpoint(
            model=self.value_model,
            optimizers=[self.vf_optimizer] if self.vf_optimizer is not None else [],
            lr_schedulers=[self.vf_lr_scheduler]
            if self.vf_lr_scheduler is not None
            else [],
            load_path=os.path.join(load_base_path, "value"),
            checkpoint_format="local_shard",
        )

        # Load iql components
        components_path = os.path.join(load_base_path, "iql_components")
        # load target model
        target_path = os.path.join(components_path, "target_critic.pt")
        target_legacy_path = os.path.join(components_path, "target_critic_q1q2.pt")
        if os.path.exists(target_path):
            target_state = torch.load(
                target_path, map_location=self.device, weights_only=True
            )
            if self._use_fsdp_wrap:
                self._strategy.load_model_with_state_dict(
                    self.target_model,
                    target_state,
                    cpu_offload=False,
                    full_state_dict=True,
                )
            else:
                self.target_model.load_state_dict(target_state, strict=True)
        elif os.path.exists(target_legacy_path):
            # Backward compatibility: old checkpoint with q1/q2 keys
            legacy = torch.load(
                target_legacy_path, map_location=self.device, weights_only=True
            )
            for key in ("q1", "q2"):
                if key not in legacy:
                    continue
                if self._use_fsdp_wrap:
                    self._strategy.load_model_with_state_dict(
                        self.target_model[key],
                        legacy[key],
                        cpu_offload=False,
                        full_state_dict=True,
                    )
                else:
                    self.target_model[key].load_state_dict(legacy[key], strict=True)
        # load runner state
        state_path = os.path.join(components_path, "state.pt")
        if os.path.exists(state_path):
            state_payload = torch.load(state_path, map_location=self.device)
            if "global_step" not in state_payload:
                raise KeyError(
                    "Checkpoint state.pt must contain key 'global_step' for offline IQL resume."
                )
            self._global_step = int(state_payload["global_step"])
            self._load_offline_data_state(state_payload)
