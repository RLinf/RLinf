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
from typing import Any, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig

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
        # Hardcoded fast path: compile, minimal sync; async prefetch on by default.
        self.offline_torch_compile = True
        self.offline_compile_mode = "reduce-overhead"
        self.offline_strict_mode = True
        self.offline_metric_sync_interval = 1000

        self.env = None
        self._prefetched_batches = None
        self._obs_dim = None
        self._action_dim = None
        self.device = None

        # IQL State
        self.critic_model = None
        self.target_model = None
        self.value_model = None
        self.target_model_initialized = False
        self.eval_returns: list[tuple[int, float]] = []
        self._global_step = 0

        # Build model / optimizer (JAX-style functional schedule, no lr_scheduler)
        self.model = None
        self.optimizer = None
        self._compiled_update_step = None
        self._critic_params: list[torch.Tensor] = []
        self._target_params: list[torch.Tensor] = []

        self._use_fsdp_wrap = True

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

    def init_worker(self):
        """Initialize IQL worker."""
        dataset_cfg = getattr(self.cfg, "dataset", None)
        self._dataset_group_name = str(
            getattr(dataset_cfg, "group_name", "DatasetGroup")
        )
        self.setup_iql_components()
        os.makedirs(self._save_dir or ".", exist_ok=True)
        self.setup_model_and_optimizer(initialize_target=True)
        # Setup actor->rollout rank mapping for weight sync.
        self._setup_rollout_weight_dst_ranks()

    def build_iql_module(
        self, obs_dim: int, action_dim: int, type_name: str
    ) -> torch.nn.Module:
        hidden_dims: Sequence[int] = tuple(self.cfg.algorithm.hidden_dims)
        iql_config = {"type": type_name, "hidden_dims": hidden_dims}
        if type_name == "actor":
            iql_config.update(
                {
                    "dropout_rate": self.cfg.algorithm.dropout_rate,
                    "state_dependent_std": False,
                    "log_std_min": -5.0,
                    "log_std_max": 2.0,
                }
            )
        model = IQLMLPPolicy(
            obs_dim,
            action_dim,
            num_action_chunks=1,
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
            forward_type=ForwardType.IQL, observations=observations, actions=actions
        )
        q2 = critic_module["q2"](
            forward_type=ForwardType.IQL, observations=observations, actions=actions
        )
        return q1, q2

    def compute_iql_step_outputs(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> tuple[
        torch.Tensor, ...
    ]:  # value_loss, actor_loss, critic_loss, v, q1, q2, adv
        """Single source of truth for IQL loss computations (value, actor, critic).
        Returns all losses and intermediates; caller performs backward and optimizer step.
        """
        # Target Q from critic (no grad)
        with torch.no_grad():
            q1_t, q2_t = self.forward_critic_module(self.target_model, obs, actions)
            q_t = torch.min(q1_t, q2_t)

        # Value loss
        v = self.value_model(forward_type=ForwardType.IQL, observations=obs)
        value_loss = iql_expectile_loss(q_t - v, self.expectile).mean()

        # Actor loss
        with torch.no_grad():
            new_v = self.value_model(forward_type=ForwardType.IQL, observations=obs)
            adv = q_t - new_v
            exp_a = torch.exp(adv * self.temperature).clamp(max=100.0)
        log_probs = self.model(
            forward_type=ForwardType.IQL, observations=obs, actions=actions
        )
        actor_loss = -(exp_a * log_probs).mean()

        # Critic loss
        with torch.no_grad():
            next_v = self.value_model(
                forward_type=ForwardType.IQL, observations=next_obs
            )
            target_q = rewards + self.discount * masks * next_v
        q1, q2 = self.forward_critic_module(self.critic_model, obs, actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        return value_loss, actor_loss, critic_loss, v, q1, q2, adv

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

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        obs_np = np.asarray(observations, dtype=np.float32, copy=True)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        prev_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            actions = (
                self.model(
                    forward_type=ForwardType.IQL,
                    observations=obs,
                    temperature=temperature,
                )
                .cpu()
                .numpy()
            )
        self.model.train(prev_mode)
        if actions.shape[0] == 1:
            actions = actions[0]
        return np.clip(actions, -1, 1)

    def get_policy_state_dict(self) -> dict[str, torch.Tensor]:
        """Return actor policy state_dict on CPU for external rollout/eval workers."""
        assert self.model is not None, (
            "init_worker() must be called before get_policy_state_dict()."
        )
        if self._use_fsdp_wrap:
            state = self._strategy.get_model_state_dict(
                self.model,
                cpu_offload=False,
                full_state_dict=True,
            )
        else:
            state = self.model.state_dict()
        return {k: v.detach().cpu() for k, v in state.items()}

    def sync_model_to_rollout(self, dst_rank: Optional[int] = None) -> None:
        """Sync current policy weights to rollout group via Worker.send()."""
        # Keep the FSDP path aligned with the base actor implementation.
        if self._use_fsdp_wrap and dst_rank is None:
            super().sync_model_to_rollout()
            return

        state = self.get_policy_state_dict()
        if dst_rank is not None:
            self.send(
                state,
                dst_group_name=self._rollout_group_name,
                dst_rank=dst_rank,
                async_op=False,
                options=self._sync_weight_comm_options,
            )
            return

        dst_ranks = self._weight_dst_rank_in_rollout
        if dst_ranks is None:
            return
        if not isinstance(dst_ranks, list):
            dst_ranks = [dst_ranks]
        for rank in dst_ranks:
            self.send(
                state,
                dst_group_name=self._rollout_group_name,
                dst_rank=rank,
                async_op=False,
                options=self._sync_weight_comm_options,
            )

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

    def update_step_forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        next_obs: torch.Tensor,
        step: torch.Tensor,
        max_steps: torch.Tensor,
        use_cosine: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled path: one IQL step (forward + backward + schedule + step + target) then return stacked metrics."""
        value_loss, actor_loss, critic_loss, v, q1, q2, adv = (
            self.compute_iql_step_outputs(obs, actions, rewards, masks, next_obs)
        )
        assert self.optimizer is not None, (
            "setup_model_and_optimizer must be called first."
        )
        total_loss = value_loss + actor_loss + critic_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        decay = use_cosine * (
            0.5 * (1.0 + torch.cos(torch.pi * step.float() / max_steps.float()))
        ) + (1.0 - use_cosine)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.mul_(decay)
        self.optimizer.step()
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
        step_tensor = torch.tensor(
            self._global_step, dtype=torch.float32, device=obs.device
        )
        max_steps_int = max(1, int(self.cfg.runner.get("max_steps", 1)))
        max_steps_tensor = torch.tensor(
            max_steps_int, dtype=torch.float32, device=obs.device
        )
        use_cosine = (
            self.cfg.algorithm.get("opt_decay_schedule", "cosine") == "cosine"
            and max_steps_int > 0
        )
        use_cosine_tensor = torch.tensor(
            float(use_cosine), dtype=torch.float32, device=obs.device
        )
        if int(obs.shape[0]) != self.batch_size:
            raise ValueError(
                "Offline IQL requires static batch size. "
                f"Got {int(obs.shape[0])}, expected {self.batch_size}. "
                "Use a fixed-size sampler (e.g., drop_last=True)."
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
                    step_tensor,
                    max_steps_tensor,
                    use_cosine_tensor,
                )
            except Exception as e:
                self.log_warning(
                    f"torch.compile update step failed ({e}); fallback to eager update."
                )
                self.offline_torch_compile = False
                self._compiled_update_step = None
                use_compiled_update = False
            if use_compiled_update:
                return {
                    "critic_loss": flat[0].detach(),
                    "q1": flat[1].detach(),
                    "q2": flat[2].detach(),
                    "value_loss": flat[3].detach(),
                    "v": flat[4].detach(),
                    "actor_loss": flat[5].detach(),
                    "adv_mean": flat[6].detach(),
                    "adv_std": flat[7].detach(),
                    "use_fsdp_wrap": self._use_fsdp_wrap,
                }

        flat = self.update_step_forward(
            obs,
            actions,
            rewards,
            masks,
            next_obs,
            step_tensor,
            max_steps_tensor,
            use_cosine_tensor,
        )
        return {
            "critic_loss": flat[0].detach(),
            "q1": flat[1].detach(),
            "q2": flat[2].detach(),
            "value_loss": flat[3].detach(),
            "v": flat[4].detach(),
            "actor_loss": flat[5].detach(),
            "adv_mean": flat[6].detach(),
            "adv_std": flat[7].detach(),
            "use_fsdp_wrap": self._use_fsdp_wrap,
        }

    def setup_model_and_optimizer(self, initialize_target: bool = True) -> None:
        """Setup model, lr_scheduler and optimizers.

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

        use_fsdp_wrap = self.cfg.actor.get("use_fsdp_wrap", True)
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

        # Single unified optimizer (actor + value + critic), one backward + one step
        actor_lr = self.cfg.algorithm.actor_lr
        value_lr = self.cfg.algorithm.value_lr
        critic_lr = self.cfg.algorithm.critic_lr
        unified_params = [
            {"params": list(self.model.parameters()), "lr": actor_lr},
            {"params": list(self.value_model.parameters()), "lr": value_lr},
            {"params": list(self.critic_model.parameters()), "lr": critic_lr},
        ]
        self.optimizer = torch.optim.Adam(unified_params)
        self.log_info("IQL offline: using unified optimizer (1 step).")
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

    def setup_iql_components(self):
        """Initialize IQL-specific offline components."""
        # Read actor.offline_* from config (same style as fsdp_sac_policy_worker)
        actor_cfg = self.cfg.actor
        self.offline_torch_compile = bool(actor_cfg.get("offline_torch_compile", True))
        self.offline_compile_mode = str(
            actor_cfg.get("offline_compile_mode", "reduce-overhead")
        )
        self.offline_strict_mode = bool(actor_cfg.get("offline_strict_mode", True))
        self.offline_metric_sync_interval = int(
            actor_cfg.get("offline_metric_sync_interval", 1000)
        )

        # Read runner.*, env.*, algorithm.*
        self._seed = int(self.cfg.actor.get("seed", 42))
        self._save_dir = self.cfg.runner.get("save_dir", None)
        if self._save_dir is None:
            runner_logger = self.cfg.runner.get("logger", None)
            if runner_logger is not None:
                log_path = runner_logger.get("log_path", ".")
                exp_name = runner_logger.get("experiment_name", "offline")
                self._save_dir = os.path.join(log_path, exp_name)
            else:
                self._save_dir = "."
        # NOTE: Offline eval is handled by env workers via runner; actor doesn't own env.
        self.env = None

        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        self.device = self.cfg.actor.get("device", None)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        self.discount = self.cfg.algorithm.discount
        self.tau = self.cfg.algorithm.tau
        self.expectile = self.cfg.algorithm.expectile
        self.temperature = self.cfg.algorithm.temperature
        self.batch_size = int(self.cfg.algorithm.get("batch_size", 256))
        # Actor only consumes batches forwarded from dataset worker by runner.
        model_cfg = self.cfg.actor.model
        self._obs_dim = int(model_cfg.get("obs_dim", 0))
        self._action_dim = int(model_cfg.get("action_dim", 0))
        if self._obs_dim <= 0 or self._action_dim <= 0:
            raise ValueError(
                "actor.model.obs_dim/action_dim must be set before actor init when "
                "using runner-forwarded dataset batches."
            )

    @Worker.timer("run_training")
    def run_training(
        self, batches: Optional[list[dict[str, np.ndarray | torch.Tensor]]] = None
    ):
        """IQL training using batches forwarded from dataset worker by runner."""
        if batches is None and self._prefetched_batches is not None:
            batches = self._prefetched_batches
            self._prefetched_batches = None
        assert self.model is not None, (
            "init_worker() must be called before run_training()."
        )
        if batches is None:
            raise RuntimeError(
                "No training batches provided. Runner must prefetch and send batches "
                "from dataset worker before actor.run_training()."
            )
        local_update_steps = max(1, len(batches))
        if self._global_step < 0:
            self._global_step = 0
        max_steps = int(self.cfg.runner.get("max_steps", 1))
        remaining_steps = max_steps - int(self._global_step)
        local_update_steps = min(local_update_steps, max(1, remaining_steps))

        # Ensure train mode before update loop
        self.model.train()
        self.critic_model.train()
        self.value_model.train()

        summed_metrics: Optional[dict[str, torch.Tensor]] = None
        eval_metrics: dict[str, Any] = {}
        # Evaluation is performed outside actor (env worker + runner).

        # Main update loop
        for i in range(local_update_steps):
            prepared = self.prepare_batch(batches[i])
            packed_batch = (
                prepared["observations"],
                prepared["actions"],
                prepared["rewards"],
                prepared["masks"],
                prepared["next_observations"],
            )

            update_info = self.update_one_epoch(packed_batch)
            summed_metrics = self.aggregate_update_info(summed_metrics, update_info)

            self._global_step += 1

            # No eval here; runner may call env worker for evaluation.

        mean_metric_dict: dict[str, Any] = {}
        if summed_metrics is not None:
            for k, v in summed_metrics.items():
                mean_metric_dict[k] = float((v / local_update_steps).item())
        mean_metric_dict.update(eval_metrics)
        mean_metric_dict["__global_step"] = int(self._global_step)
        return mean_metric_dict

    def set_prefetched_batches(
        self, batches: list[dict[str, np.ndarray | torch.Tensor]]
    ) -> None:
        """Cache prefetched batches from runner for next run_training call."""
        self._prefetched_batches = batches

    def recv_dataset_batches(self) -> None:
        """Receive prefetched batches from dataset workers via Worker.recv()."""
        dataset_world_size = int(self._component_placement.get_world_size("dataset"))
        src_rank = self._rank % max(1, dataset_world_size)
        batches = self.recv(
            src_group_name=self._dataset_group_name,
            src_rank=src_rank,
        )
        assert isinstance(batches, list), (
            f"Expected list of batches, got {type(batches)}"
        )
        self._prefetched_batches = batches

    def compute_advantages_and_returns(self):
        """
        IQL doesn't compute rollout advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def set_global_step(self, step: int):
        self._global_step = int(step)
        return None

    def save_checkpoint(self, save_base_path, step):
        assert self.model is not None, "init_worker() must initialize self.model first."
        os.makedirs(save_base_path, exist_ok=True)

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer] if self.optimizer is not None else [],
            lr_schedulers=[],
            save_path=os.path.join(save_base_path, "actor_policy"),
            checkpoint_format="local_shard",
        )
        self._strategy.save_checkpoint(
            model=self.critic_model,
            optimizers=[],
            lr_schedulers=[],
            save_path=os.path.join(save_base_path, "critic"),
            checkpoint_format="local_shard",
        )
        self._strategy.save_checkpoint(
            model=self.value_model,
            optimizers=[],
            lr_schedulers=[],
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
        # save runner state
        state_payload = {
            "step": int(step),
            "global_step": int(self._global_step),
        }
        torch.save(state_payload, os.path.join(components_path, "state.pt"))

    def load_checkpoint(self, load_base_path: str):
        assert self.model is not None, "init_worker() must initialize self.model first."

        # Load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer] if self.optimizer is not None else [],
            lr_schedulers=[],
            load_path=os.path.join(load_base_path, "actor_policy"),
            checkpoint_format="local_shard",
        )
        self._strategy.load_checkpoint(
            model=self.critic_model,
            optimizers=[],
            lr_schedulers=[],
            load_path=os.path.join(load_base_path, "critic"),
            checkpoint_format="local_shard",
        )
        self._strategy.load_checkpoint(
            model=self.value_model,
            optimizers=[],
            lr_schedulers=[],
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
            self._global_step = int(
                state_payload.get("global_step", state_payload.get("step", 0))
            )
