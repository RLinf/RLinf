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
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.datasets.d4rl_offline import (
    evaluate_policy,
    make_d4rl_env_and_dataset,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy import get_model
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

InfoDict = Dict[str, Any]


def iql_expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(
        diff > 0,
        torch.full_like(diff, expectile),
        torch.full_like(diff, 1 - expectile),
    )
    return weight * (diff**2)


class EmbodiedIQLFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.env = None
        self.dataset = None
        self.device = None

        # IQL State
        self.critic_model = None
        self.target_model = None
        self.value_model = None
        self.target_model_initialized = False
        self.eval_returns = []
        self._global_step = 0

        # Build model / optimizer / scheduler
        self.model = None
        self.optimizer = None
        self.qf_optimizer = None
        self.vf_optimizer = None
        self.lr_scheduler = None
        self.qf_lr_scheduler = None
        self.vf_lr_scheduler = None
        self.optimizers = []
        self.lr_schedulers = []

        # IQL Hyperparameters
        self.discount = 0.99
        self.tau = 0.005
        self.expectile = 0.8
        self.temperature = 0.1

    def init_worker(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.setup_iql_components()
        self.setup_model_and_optimizer(initialize_target=True)
        self.soft_update_target_model(tau=1.0)

    def build_iql_module(self, obs_dim: int, action_dim: int, kind: str) -> torch.nn.Module:
        hidden_dims: Sequence[int] = tuple(self.cfg.algorithm.hidden_dims)
        iql_config = {"kind": kind, "hidden_dims": hidden_dims}
        if kind == "actor":
            iql_config.update(
                {
                    "dropout_rate": self.cfg.algorithm.dropout_rate,
                    "state_dependent_std": False,
                    "log_std_min": -5.0,
                    "log_std_max": 2.0,
                }
            )
        model_cfg = OmegaConf.create(
            {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "num_action_chunks": 1,
                "add_value_head": False,
                "add_q_head": False,
                "iql_config": iql_config,
            }
        )
        return get_model(model_cfg).to(self.device)

    def model_provider_func(self, obs_dim: int, action_dim: int, kind: str):
        """SAC-style provider entry for IQL modules."""
        if kind in {"actor", "value"}:
            return self.build_iql_module(obs_dim, action_dim, kind=kind)
        if kind == "critic":
            return self.build_critic_module(obs_dim, action_dim)
        raise ValueError(f"Unsupported provider kind: {kind}")

    def build_critic_module(
        self, obs_dim: int, action_dim: int
    ) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {
                "q1": self.build_iql_module(obs_dim, action_dim, kind="critic"),
                "q2": self.build_iql_module(obs_dim, action_dim, kind="critic"),
            }
        ).to(self.device)

    def forward_critic_module(
        self,
        critic_module: torch.nn.ModuleDict,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = critic_module["q1"](
            forward_type=ForwardType.IQL,
            observations=observations,
            actions=actions,
        )
        q2 = critic_module["q2"](
            forward_type=ForwardType.IQL,
            observations=observations,
            actions=actions,
        )
        return q1, q2

    def _prepare_batch(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            "observations": torch.as_tensor(
                batch["observations"], dtype=torch.float32, device=self.device
            ),
            "actions": torch.as_tensor(
                batch["actions"], dtype=torch.float32, device=self.device
            ),
            "rewards": torch.as_tensor(
                batch["rewards"], dtype=torch.float32, device=self.device
            ),
            "masks": torch.as_tensor(
                batch["masks"], dtype=torch.float32, device=self.device
            ),
            "next_observations": torch.as_tensor(
                batch["next_observations"], dtype=torch.float32, device=self.device
            ),
        }

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        prev_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            actions = self.model(
                forward_type=ForwardType.IQL,
                observations=obs,
                temperature=temperature,
            ).cpu().numpy()
        self.model.train(prev_mode)
        if actions.shape[0] == 1:
            actions = actions[0]
        return np.clip(actions, -1, 1)

    def soft_update_target_model(self, tau: Optional[float] = None):
        if tau is None:
            tau = self.tau
        assert self.target_model_initialized
        with torch.no_grad():
            for p, tp in zip(self.critic_model.parameters(), self.target_model.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)


    def forward_value(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q1_t, q2_t = self.forward_critic_module(self.target_model, obs, actions)
            q_t = torch.min(q1_t, q2_t)
        v = self.value_model(forward_type=ForwardType.IQL, observations=obs)
        value_loss = iql_expectile_loss(q_t - v, self.expectile).mean()
        self.vf_optimizer.zero_grad()
        value_loss.backward()
        self.vf_optimizer.step()
        return v, value_loss

    def forward_actor(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            new_v = self.value_model(forward_type=ForwardType.IQL, observations=obs)
            q1_a, q2_a = self.forward_critic_module(self.target_model, obs, actions)
            q_a = torch.min(q1_a, q2_a)
            adv = q_a - new_v
            exp_a = torch.exp(adv * self.temperature).clamp(max=100.0)
        log_probs = self.model(
            forward_type=ForwardType.IQL,
            observations=obs,
            actions=actions,
        )
        actor_loss = -(exp_a * log_probs).mean()
        self.optimizer.zero_grad()
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
        with torch.no_grad():
            next_v = self.value_model(
                forward_type=ForwardType.IQL, observations=next_obs
            )
            target_q = rewards + self.discount * masks * next_v
        q1, q2 = self.forward_critic_module(self.critic_model, obs, actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        self.qf_optimizer.zero_grad()
        critic_loss.backward()
        self.qf_optimizer.step()
        return q1, q2, critic_loss

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, batch: dict[str, np.ndarray]) -> InfoDict:
        self.model.train()
        self.critic_model.train()
        self.value_model.train()

        b = self._prepare_batch(batch)
        obs = b["observations"]
        actions = b["actions"]
        rewards = b["rewards"]
        masks = b["masks"]
        next_obs = b["next_observations"]

        # Value update with target critic
        v, value_loss = self.forward_value(obs, actions)

        # Actor update with exponential advantage weights
        adv, actor_loss = self.forward_actor(obs, actions)

        # Critic update with bootstrap target from value network
        q1, q2, critic_loss = self.forward_critic(
            obs, actions, rewards, masks, next_obs
        )

        # Soft target update
        self.soft_update_target_model()

        adv_np = adv.detach().cpu().numpy()
        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "q1": float(q1.detach().mean().cpu().item()),
            "q2": float(q2.detach().mean().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "v": float(v.detach().mean().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "adv_mean": float(np.mean(adv_np)),
            "adv_std": float(np.std(adv_np)),
        }

    def build_lr_schedulers(self) -> None:
        self.lr_scheduler = None
        if (
            self.cfg.algorithm.get("opt_decay_schedule", "cosine") == "cosine"
            and self.cfg.max_steps is not None
        ):
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(self.cfg.max_steps)),
            )
        self.qf_lr_scheduler = None
        self.vf_lr_scheduler = None

    def setup_model_and_optimizer(self, initialize_target: bool = True) -> None:
        """Setup model, lr_scheduler and optimizers."""
        obs_dim = int(self.env.observation_space.sample().shape[-1])
        action_dim = int(self.env.action_space.sample().shape[-1])

        module = self.model_provider_func(obs_dim, action_dim, kind="actor")
        critic_module = self.model_provider_func(obs_dim, action_dim, kind="critic")
        value_module = self.model_provider_func(obs_dim, action_dim, kind="value")
        if initialize_target:
            target_module = self.model_provider_func(obs_dim, action_dim, kind="critic")

        # Build model (prefer strategy wrapping when available)
        if hasattr(self, "_strategy") and hasattr(self, "_device_mesh"):
            self.model = self._strategy.wrap_model(
                model=module, device_mesh=self._device_mesh
            )
            self.critic_model = torch.nn.ModuleDict(
                {
                    "q1": self._strategy.wrap_model(
                        model=critic_module["q1"], device_mesh=self._device_mesh
                    ),
                    "q2": self._strategy.wrap_model(
                        model=critic_module["q2"], device_mesh=self._device_mesh
                    ),
                }
            )
            self.value_model = self._strategy.wrap_model(
                model=value_module, device_mesh=self._device_mesh
            )
            if initialize_target:
                self.target_model = torch.nn.ModuleDict(
                    {
                        "q1": self._strategy.wrap_model(
                            model=target_module["q1"], device_mesh=self._device_mesh
                        ),
                        "q2": self._strategy.wrap_model(
                            model=target_module["q2"], device_mesh=self._device_mesh
                        ),
                    }
                )
        else:
            self.model = module
            self.critic_model = critic_module
            self.value_model = value_module
            if initialize_target:
                self.target_model = target_module

        # Initialize target model
        if initialize_target:
            self.target_model.load_state_dict(self.critic_model.state_dict())
            self.target_model.eval()
            self.target_model_initialized = True
            for p in self.target_model.parameters():
                p.requires_grad_(False)

        # Build optimizers and schedulers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.algorithm.actor_lr
        )
        self.vf_optimizer = torch.optim.Adam(
            self.value_model.parameters(), lr=self.cfg.algorithm.value_lr
        )
        self.qf_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.cfg.algorithm.critic_lr
        )
        self.build_lr_schedulers()

        self.optimizers = [self.optimizer, self.qf_optimizer, self.vf_optimizer]
        self.lr_schedulers = [
            sch
            for sch in [self.lr_scheduler, self.qf_lr_scheduler, self.vf_lr_scheduler]
            if sch is not None
        ]

    def setup_iql_components(self):
        """Initialize IQL-specific offline components."""
        # Initialize offline environment and dataset
        self.env, self.dataset = make_d4rl_env_and_dataset(
            self.cfg.env_name, self.cfg.seed
        )

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = self.cfg.get("device", None)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        self.discount = self.cfg.algorithm.discount
        self.tau = self.cfg.algorithm.tau
        self.expectile = self.cfg.algorithm.expectile
        self.temperature = self.cfg.algorithm.temperature


    @Worker.timer("run_training")
    def run_training(self):
        """IQL training using offline dataset."""
        assert self.model is not None and self.dataset is not None, (
            "init_worker() must be called before run_training()."
        )
        if self._global_step <= 0:
            self._global_step = 1

        # Sample one batch from offline dataset
        batch = self.dataset.sample(self.cfg.batch_size)
        update_info = self.update_one_epoch(batch)

        # Evaluate policy at configured interval
        eval_metrics = {}
        should_eval = self._global_step % self.cfg.eval_interval == 0
        if should_eval:
            eval_stats = evaluate_policy(self, self.env, self.cfg.eval_episodes)
            self.eval_returns.append((self._global_step, eval_stats["return"]))
            np.savetxt(
                os.path.join(self.cfg.save_dir, f"{self.cfg.seed}.txt"),
                self.eval_returns,
                fmt=["%d", "%.1f"],
            )
            eval_metrics = {
                f"evaluation/average_{k}s": v for k, v in eval_stats.items()
            }

        # Aggregate train/eval metrics
        metrics = {
            "critic_loss": update_info["critic_loss"],
            "value_loss": update_info["value_loss"],
            "actor_loss": update_info["actor_loss"],
            "q1": update_info["q1"],
            "q2": update_info["q2"],
            "v": update_info["v"],
            "adv_mean": update_info["adv_mean"],
            "adv_std": update_info["adv_std"],
        }
        metrics.update(eval_metrics)
        return metrics

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

        checkpoint_format = (
            "local_shard"
            if self.cfg.actor.get("fsdp_config", {}).get("use_orig_params", False)
            else "dcp"
        )

        # Save actor
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer] if self.optimizer is not None else [],
            lr_schedulers=[self.lr_scheduler] if self.lr_scheduler is not None else [],
            save_path=os.path.join(save_base_path, "actor_policy"),
            checkpoint_format=checkpoint_format,
        )

        # Save critic
        self._strategy.save_checkpoint(
            model=self.critic_model,
            optimizers=[self.qf_optimizer] if self.qf_optimizer is not None else [],
            lr_schedulers=[self.qf_lr_scheduler]
            if self.qf_lr_scheduler is not None
            else [],
            save_path=os.path.join(save_base_path, "critic"),
            checkpoint_format=checkpoint_format,
        )

        # Save value
        self._strategy.save_checkpoint(
            model=self.value_model,
            optimizers=[self.vf_optimizer] if self.vf_optimizer is not None else [],
            lr_schedulers=[self.vf_lr_scheduler]
            if self.vf_lr_scheduler is not None
            else [],
            save_path=os.path.join(save_base_path, "value"),
            checkpoint_format=checkpoint_format,
        )

        # Save IQL components
        components_path = os.path.join(save_base_path, "iql_components")
        os.makedirs(components_path, exist_ok=True)

        # Save target model
        target_q1_state_dict = self._strategy.get_model_state_dict(
            self.target_model["q1"], cpu_offload=False, full_state_dict=True
        )
        target_q2_state_dict = self._strategy.get_model_state_dict(
            self.target_model["q2"], cpu_offload=False, full_state_dict=True
        )
        torch.save(
            {"q1": target_q1_state_dict, "q2": target_q2_state_dict},
            os.path.join(components_path, "target_critic_q1q2.pt"),
        )

        # Save IQL state
        state_payload = {
            "step": int(step),
            "global_step": int(self._global_step),
            "eval_returns": self.eval_returns,
        }
        torch.save(state_payload, os.path.join(components_path, "state.pt"))

    def load_checkpoint(self, load_base_path: str):
        assert self.model is not None, "init_worker() must initialize self.model first."

        checkpoint_format = (
            "local_shard"
            if self.cfg.actor.get("fsdp_config", {}).get("use_orig_params", False)
            else "dcp"
        )

        # Load actor
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer] if self.optimizer is not None else [],
            lr_schedulers=[self.lr_scheduler] if self.lr_scheduler is not None else [],
            load_path=os.path.join(load_base_path, "actor_policy"),
            checkpoint_format=checkpoint_format,
        )

        # Load critic
        self._strategy.load_checkpoint(
            model=self.critic_model,
            optimizers=[self.qf_optimizer] if self.qf_optimizer is not None else [],
            lr_schedulers=[self.qf_lr_scheduler]
            if self.qf_lr_scheduler is not None
            else [],
            load_path=os.path.join(load_base_path, "critic"),
            checkpoint_format=checkpoint_format,
        )

        # Load value
        self._strategy.load_checkpoint(
            model=self.value_model,
            optimizers=[self.vf_optimizer] if self.vf_optimizer is not None else [],
            lr_schedulers=[self.vf_lr_scheduler]
            if self.vf_lr_scheduler is not None
            else [],
            load_path=os.path.join(load_base_path, "value"),
            checkpoint_format=checkpoint_format,
        )

        # Load IQL components
        components_path = os.path.join(load_base_path, "iql_components")
        target_critic_path = os.path.join(components_path, "target_critic_q1q2.pt")
        if os.path.exists(target_critic_path):
            target_model_state_dict = torch.load(
                target_critic_path, map_location=self.device
            )
            self._strategy.load_model_with_state_dict(
                self.target_model["q1"],
                target_model_state_dict["q1"],
                cpu_offload=False,
                full_state_dict=True,
            )
            self._strategy.load_model_with_state_dict(
                self.target_model["q2"],
                target_model_state_dict["q2"],
                cpu_offload=False,
                full_state_dict=True,
            )

        state_path = os.path.join(components_path, "state.pt")
        if os.path.exists(state_path):
            state_payload = torch.load(state_path, map_location=self.device)
            self.eval_returns = state_payload.get("eval_returns", [])
            self._global_step = int(
                state_payload.get("global_step", state_payload.get("step", 0))
            )
