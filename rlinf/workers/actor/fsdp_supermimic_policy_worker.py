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

"""FSDP actor of SuperMimic: replay, world model, imagination and critic.

The env worker streams one transition per control step into this worker's
replay. Each global update fits the world model on replay windows, trains the
motion policy through H-step imagination and fits the critic to the
lambda-returns of the same imagined trajectory.

Every phase runs through the FSDP root forward with only that phase's
parameters trainable, so gradients still flow through the frozen world
model and critic into the actions and states. Every rank trains on the same
number of rows, so FSDP's gradient mean is the global mean.
"""

from __future__ import annotations

import contextlib
import json
import math
import os

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from rlinf.algorithms.supermimic.objectives import (
    lambda_returns,
    reward_per_second,
    value_ceiling,
)
from rlinf.algorithms.supermimic.replay import TransitionRing
from rlinf.envs.sim.g1_mimic.g1_mimic_env import build_motion_library
from rlinf.hybrid_engines.fsdp import FSDPModule
from rlinf.hybrid_engines.fsdp.utils import (
    clip_grad_by_total_norm_,
    get_grad_norm,
    to_local_if_dtensor,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.supermimic.networks import hidden_matrix_names
from rlinf.scheduler import Worker
from rlinf.utils.precision import torch_dtype_from_precision
from rlinf.utils.profiler import PyTorchProfiler
from rlinf.utils.utils import (
    collect_param_names_need_sync,
    seed_everything,
    warmup_optimizer_state,
)
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

WORLD_MODEL = ForwardType.SUPERMIMIC_WORLD_MODEL
ACTOR = ForwardType.SUPERMIMIC_ACTOR
CRITIC = ForwardType.SUPERMIMIC_CRITIC

# The rollout worker only acts; it never reads these networks.
_ACTOR_ONLY_PREFIXES = ("dynamics.", "critic.")
_COUNTERS_FILE = "supermimic_counters.json"
# Phase -> (module prefix, Adam config under ``actor``). The motion policy's
# Adam is the main optimizer of ``build_optimizers``.
_PHASES = {
    "world_model": ("dynamics.", "world_model_optim"),
    "actor": ("actor.", "optim"),
    "critic": ("critic.", "critic_optim"),
}
_PHASE_TYPES = {"world_model": WORLD_MODEL, "actor": ACTOR, "critic": CRITIC}


def clip_step(
    optimizers: list[torch.optim.Optimizer],
    params: list,
    norm: float | torch.Tensor,
    max_norm: float,
    value_clip: float | None,
) -> None:
    """Clip by the global gradient ``norm``, clamp each element, step every optimizer."""
    clip_grad_by_total_norm_(params, max_norm, norm)
    if value_clip is not None:
        for param in params:
            if param.grad is not None:
                to_local_if_dtensor(param.grad).clamp_(-value_clip, value_clip)
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


class EmbodiedSuperMimicFSDPPolicy(EmbodiedFSDPActor):
    """FSDP actor that owns one env rank's replay and runs every training phase."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        alg = cfg.algorithm
        world = self._world_size
        self.k = int(alg.k)
        self.horizon = int(alg.H)
        self.gamma = float(alg.gamma)
        self.lam = float(alg.lam)
        self.value_max = value_ceiling(self.gamma)
        self.wm_batches = int(alg.wm_batches)
        self.wm_window_batch = int(alg.wm_window_batch) // world
        self.imagine_batch = int(alg.imagine_batch) // world
        self.critic_epochs = int(alg.critic_epochs)
        self.critic_minibatches = int(alg.critic_minibatches)
        self.num_envs = int(cfg.env.train.total_num_envs) // world

        self.optimizer_steps = dict.fromkeys(_PHASES, 0)
        self.profiler = None
        if cfg.actor.get("use_profiler", False):
            # One profiler step per global update, counted from this launch.
            self.profiler = PyTorchProfiler.from_config(cfg.actor.profiler)
            self.profiler.init_fwd_bwd_schedule(
                self.profiler.schedule_warmup + self.profiler.schedule_active
            )

    # ------------------------------------------------------------------ setup

    def init_worker(self) -> None:
        """Build identical networks on every rank, then seed each rank's streams."""
        seed = int(self.cfg.actor.seed)
        seed_everything(seed)
        self.setup_model_and_optimizer()
        seed_everything(seed + self._rank)
        device = torch.device(self.device)
        self.motion = build_motion_library(self.cfg.env.train, device)
        # An episode ends with its clip, so only a clip of k control steps can
        # fill a world-model window; with none, no window is ever legal.
        longest = int(self.motion.lengths.max())
        if longest < self.k:
            raise ValueError(
                f"the longest clip in env.train.motion_path has {longest} control "
                f"steps; world-model windows need at least algorithm.k={self.k}"
            )
        self.replay = TransitionRing(
            self.num_envs,
            int(self.cfg.algorithm.replay_length),
            self.model.representation.state_dim,
            device,
        )

    def setup_model_and_optimizer(self) -> None:
        """Wrap the policy with FSDP and build two optimizers per phase.

        Muon updates the hidden matrices of every MLP and Adam the remaining
        parameters. ``fsdp_config.amp_autocast`` applies inside the network
        forwards only; kinematics, rewards and losses stay float32.
        """
        module = self.model_provider_func()
        module.configure_training(self.cfg.algorithm)
        amp = self.cfg.actor.fsdp_config.amp_autocast
        module.set_autocast(
            torch_dtype_from_precision(amp.precision) if amp.enabled else None
        )
        self.param_names_need_sync = [
            name
            for name in collect_param_names_need_sync(module)
            if not name.startswith(_ACTOR_ONLY_PREFIXES)
        ]
        hidden = hidden_matrix_names(module)
        self.model: FSDPModule = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )

        # Substring filters, matched in order: the exact hidden-matrix names
        # come before the module prefixes. The motion policy's remaining
        # parameters fall through to the main Adam.
        actor_cfg = self.cfg.actor
        param_filters, optim_cfgs = {}, {}
        for phase, (prefix, _) in _PHASES.items():
            param_filters[f"{phase}_muon"] = [
                name for name in hidden if name.startswith(prefix)
            ]
            optim_cfgs[f"{phase}_muon"] = actor_cfg.muon_optim
        for phase, (prefix, cfg_name) in _PHASES.items():
            if phase != "actor":
                param_filters[phase] = [prefix]
                optim_cfgs[phase] = actor_cfg[cfg_name]
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=actor_cfg.optim,
            param_filters=param_filters,
            filtered_optim_config=optim_cfgs,
        )
        by_name = dict(zip(["actor", *param_filters], optimizers, strict=True))
        optim_cfgs["actor"] = actor_cfg.optim
        self.phase_optimizers = {
            phase: [by_name[phase], by_name[f"{phase}_muon"]] for phase in _PHASES
        }
        self.phase_optim_cfgs = {
            phase: actor_cfg[cfg_name] for phase, (_, cfg_name) in _PHASES.items()
        }
        self.all_optimizers = list(by_name.values())
        self.all_schedulers = []
        for name, optimizer in by_name.items():
            warmup_optimizer_state(optimizer)
            self.all_schedulers.append(
                self.build_lr_scheduler(optimizer, optim_cfgs[name])
            )
        self.optimizer = by_name["actor"]
        self.lr_scheduler = self.all_schedulers[0]

    # ------------------------------------------------------------- collection

    @Worker.timer("ingest")
    def ingest(self, num_steps: int) -> None:
        """Store ``num_steps`` vector steps from this rank's env worker.

        Each transition also carries the merged moments the rollout worker
        folded into its input norms; folding the same values keeps both
        copies equal. The world-model norms fold the step's encodings and
        executed PD targets.
        """
        model = self.model
        fields = set(self.replay.storage)
        for _ in range(num_steps):
            step = self.recv(self._env_group_name, src_rank=self._rank)
            model.fold_policy_stats(step["enc_stats"], step["goal_stats"])
            model.fold_world_model_stats(step["enc_stats"], step["target_stats"])
            self.replay.add({name: step[name] for name in fields})

    def replay_ready(self) -> bool:
        """Whether this rank's replay holds a legal world-model window."""
        return self.replay.has_window(self.k)

    # --------------------------------------------------------------- training

    @Worker.timer("run_updates")
    def run_updates(self, num_updates: int) -> dict[str, float]:
        """Run ``num_updates`` global updates; returns the last update's metrics."""
        metrics: dict[str, float | torch.Tensor] = {}
        for _ in range(num_updates):
            if self.profiler is not None:
                self.profiler.start()
            self._run_update(metrics)
            if self.profiler is not None:
                self.profiler.stop()
        self.model.reshard()
        # Checkpoint loading drops the optimizer state of frozen parameters.
        self.model.requires_grad_(True)
        names = [name for name, value in metrics.items() if torch.is_tensor(value)]
        hosted = []
        if names:
            hosted = torch.stack(
                [metrics[name].detach().float().reshape(()) for name in names]
            ).tolist()
        out = {
            name: float(value)
            for name, value in metrics.items()
            if not torch.is_tensor(value)
        }
        out.update(zip(names, hosted, strict=True))
        # Non-finite gradients are stepped, not skipped, so the weights
        # are already poisoned; stop before they are synced or saved.
        poisoned = sorted(
            name for name, value in out.items() if not math.isfinite(value)
        )
        if poisoned:
            raise RuntimeError(
                f"non-finite training metrics {poisoned} after global update "
                f"{self.optimizer_steps['actor']}; stopping instead of training on NaN"
            )
        out.update(
            {
                f"{phase}_optimizer_steps": float(n)
                for phase, n in self.optimizer_steps.items()
            }
        )
        return out

    def _run_update(self, metrics: dict) -> None:
        with self._phase("world_model"):
            metrics.update(self._fit_world_model())
        with self._phase("actor"):
            actor_metrics, rollout = self._train_actor()
        with self._phase("critic"):
            metrics.update(self._fit_critic(*rollout))
        metrics.update(actor_metrics)
        for scheduler in self.all_schedulers:
            scheduler.step()

    @contextlib.contextmanager
    def _phase(self, name: str):
        """Time one training phase and label it in ``torch.profiler`` traces."""
        with self.worker_timer(name), torch.profiler.record_function(name):
            yield

    def _enter_phase(self, forward_type: ForwardType) -> None:
        """Freeze every parameter outside one phase and clear stale gradients.

        FSDP copies ``requires_grad`` from the sharded parameters when it
        gathers them, so the model is resharded before the switch.
        """
        self.model.reshard()
        self.model.zero_grad(set_to_none=True)
        self.model.set_trainable(forward_type)

    def _optimizer_step(self, phase: str) -> torch.Tensor:
        """Clip and step one phase's optimizers; returns the pre-clip norm.

        The norm stays on the device, so the step never waits for the host.
        """
        optimizers = self.phase_optimizers[phase]
        params = self.model.phase_parameters(_PHASE_TYPES[phase])
        norm = get_grad_norm(params, self._dp_group, as_tensor=True)
        optim_cfg = self.phase_optim_cfgs[phase]
        value_clip = optim_cfg.get("clip_grad_value", None)
        clip_step(
            optimizers,
            params,
            norm,
            float(optim_cfg.clip_grad),
            None if value_clip is None else float(value_clip),
        )
        self.optimizer_steps[phase] += 1
        return norm

    def _fit_world_model(self) -> dict[str, float | torch.Tensor]:
        """One optimizer step on each of ``wm_batches`` replay window batches."""
        self._enter_phase(WORLD_MODEL)
        losses, norms = [], []
        for _ in range(self.wm_batches):
            batch = self.replay.sample(self.k, self.wm_window_batch)
            out = self.model(
                forward_type=WORLD_MODEL,
                states=batch["state"],
                actions=batch["action"],
                context=batch["sim_context"][:, 0],
            )
            out["loss"].backward()
            norms.append(self._optimizer_step("world_model"))
            losses.append(
                torch.cat((out["loss"].detach().reshape(1), out["group_mse"]))
            )
        means = torch.stack(losses).mean(0)
        names = ["state", *self.model.representation.block_widths]
        metrics = {
            f"world_model/{name}_loss": value
            for name, value in zip(names, means.unbind(0), strict=True)
        }
        metrics["world_model/grad_norm"] = sum(norms) / len(norms)
        return metrics

    def _imagination_inputs(self, starts: dict[str, torch.Tensor]) -> dict:
        """Goal windows at ``t_0 .. t_H`` and reference bodies at ``t_0 .. t_{H-1}``.

        ``t_k`` is ``k`` frames after the start's frame, held at its clip's
        last frame.
        """
        frame = starts["motion_time"][:, 0]
        offsets = torch.arange(self.horizon + 1, device=frame.device)
        times = torch.minimum(
            frame.unsqueeze(1) + offsets, (self.motion.clip_end(frame) - 1).unsqueeze(1)
        )
        return {
            "start": starts["state"][:, 0],
            "prev_action": starts["prev_action"][:, 0],
            "context": starts["sim_context"][:, 0],
            "windows": self.motion.goal_window(times),
            "refs": self.motion.body_state(times[:, : self.horizon]),
        }

    def _train_actor(self) -> tuple[dict[str, float | torch.Tensor], tuple]:
        """Backpropagate the H-step imagined return into the motion policy.

        Returns the metrics and the detached critic features, rewards and
        termination probabilities of the same trajectory.
        """
        self._enter_phase(ACTOR)
        starts = self.replay.sample(1, self.imagine_batch)
        out = self.model(forward_type=ACTOR, **self._imagination_inputs(starts))
        out["loss"].backward()
        grad_norm = self._optimizer_step("actor")
        term_prob = out["term_prob"]
        metrics = {
            "actor/reward": reward_per_second(out["return"], self.horizon),
            "actor/grad_norm": grad_norm,
            "imagination/survival": (1.0 - term_prob).prod(dim=1).mean(),
            "imagination/termination_prob": term_prob.mean(),
            "imagination/step_reward": out["rewards"].mean(),
        }
        return metrics, (out["steps"], out["terminal"], out["rewards"], term_prob)

    def _fit_critic(
        self,
        steps: torch.Tensor,
        terminal: torch.Tensor,
        rewards: torch.Tensor,
        term_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Regress the critic onto the lambda-returns of the imagined trajectory.

        ``steps`` holds the critic features of the ``H`` states the policy
        acted in, ``[B, H, F]``, and ``terminal`` those of the last state.
        The rows are fitted time-major in ``critic_minibatches`` fixed chunks,
        ``critic_epochs`` times.
        """
        self._enter_phase(CRITIC)
        batch, horizon, width = steps.shape
        features = steps.transpose(0, 1).reshape(-1, width)
        with torch.no_grad():
            values = self.model(
                forward_type=CRITIC, features=torch.cat((features, terminal))
            )["values"]
        self.model.reshard()
        values, last_value = values[:-batch].view(horizon, batch), values[-batch:]
        targets = lambda_returns(
            values,
            last_value,
            rewards.transpose(0, 1),
            term_prob.transpose(0, 1),
            self.gamma,
            self.lam,
        ).clamp(max=self.value_max)
        targets = targets.reshape(-1)
        chunks = self.critic_minibatches
        order = torch.randperm(
            features.shape[0] // chunks * chunks, device=features.device
        ).view(chunks, -1)
        losses = []
        for _ in range(self.critic_epochs):
            for rows in order:
                out = self.model(
                    forward_type=CRITIC, features=features[rows], targets=targets[rows]
                )
                out["loss"].backward()
                self._optimizer_step("critic")
                losses.append(out["loss"].detach())
        return {
            "critic/loss": torch.stack(losses).mean(),
            "critic/value_mean": values.mean(),
            "critic/target_mean": targets.mean(),
        }

    # ------------------------------------------------------- weights and state

    def get_rollout_state_dict(self) -> dict:
        self.model.reshard()
        return super().get_rollout_state_dict()

    def save_checkpoint(self, save_base_path: str, step: int = 0) -> None:
        """Networks, optimizers, schedulers and RNG of every rank, plus counters.

        The replay is not saved; a resumed run refills it before training.
        """
        self.model.reshard()
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=self.all_optimizers,
            lr_schedulers=self.all_schedulers,
            save_path=save_base_path,
            save_full_model_weights=True,
        )
        if self._rank == 0:
            counters = {"optimizer_steps": self.optimizer_steps}
            with open(os.path.join(save_base_path, _COUNTERS_FILE), "w") as file:
                json.dump(counters, file, indent=2)
        dist.barrier()

    def load_checkpoint(self, load_base_path: str) -> None:
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=self.all_optimizers,
            lr_schedulers=self.all_schedulers,
            load_path=load_base_path,
        )
        with open(os.path.join(load_base_path, _COUNTERS_FILE)) as file:
            counters = json.load(file)
        self.optimizer_steps = {
            phase: int(counters["optimizer_steps"][phase]) for phase in _PHASES
        }
