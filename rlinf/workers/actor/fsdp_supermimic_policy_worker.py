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

"""FSDP actor of SuperMimic: replay, world model, imagination and critics.

The env worker streams one transition per control step into this worker's
replay. Each global update fits the world model on replay windows and then,
after the world-model-only prefix, trains the motion policy through H-step
imagination, the motion critic on the same imagined trajectory and, with
velocity tracking, the velocity policy and its critic.

Every phase runs through the FSDP root forward with only that phase's
parameters trainable, so gradients still flow through the frozen world
model and critic into the actions and states. Ranks average gradients
weighted by their own row counts; a rank without rows joins every forward
and backward with a zero loss.
"""

from __future__ import annotations

import contextlib
import json
import math
import os

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from rlinf.algorithms.supermimic.objectives import gae_targets
from rlinf.algorithms.supermimic.replay import SuperMimicReplay
from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits
from rlinf.envs.sim.g1_mimic.g1_mimic_env import build_motion_library, make_goal
from rlinf.envs.sim.g1_mimic.velocity import heading_velocity
from rlinf.hybrid_engines.fsdp import FSDPModule
from rlinf.hybrid_engines.fsdp.utils import (
    clip_grad_by_total_norm_,
    get_grad_norm,
    to_local_if_dtensor,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.supermimic.interface import (
    check_interface,
    find_interface,
    interface_record,
    write_interface,
)
from rlinf.models.embodiment.supermimic.supermimic_policy import goal_width
from rlinf.scheduler import Worker
from rlinf.utils.precision import torch_dtype_from_precision
from rlinf.utils.utils import (
    collect_param_names_need_sync,
    seed_everything,
    warmup_optimizer_state,
)
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

WORLD_MODEL = ForwardType.SUPERMIMIC_WORLD_MODEL
ACTOR = ForwardType.SUPERMIMIC_ACTOR
CRITIC = ForwardType.SUPERMIMIC_CRITIC
VELOCITY_ACTOR = ForwardType.SUPERMIMIC_VELOCITY_ACTOR
VELOCITY_CRITIC = ForwardType.SUPERMIMIC_VELOCITY_CRITIC

# The rollout worker acts and bootstraps with the policy, the motion critic
# and the input norms; it never reads these networks.
_ACTOR_ONLY_PREFIXES = ("dynamics.", "termination.", "velocity_critic.")
_COUNTERS_FILE = "supermimic_counters.json"
# Optimizer-step counter of each phase; the velocity actor has none.
_STEP_COUNTERS = {
    "world_model": "wm_optimizer_steps",
    "actor": "actor_optimizer_steps",
    "critic": "critic_optimizer_steps",
    "velocity_critic": "critic_optimizer_steps",
}


class EmbodiedSuperMimicFSDPPolicy(EmbodiedFSDPActor):
    """FSDP actor that owns one env rank's replay and runs every training phase."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        alg = cfg.algorithm
        model_cfg = cfg.actor.model
        world = self._world_size
        self.k = int(alg.k)
        self.horizon = int(alg.H)
        self.gamma = float(alg.gamma)
        self.lam = float(alg.lam)
        self.residual_frame = str(alg.residual_frame)
        self.learned_termination = str(alg.termination_head) == "learned"
        self.wm_batches = int(alg.wm_batches)
        self.wm_window_batch = int(alg.wm_window_batch) // world
        self.imagine_batch = int(alg.imagine_batch) // world
        self.critic_epochs = int(alg.critic_epochs)
        self.critic_minibatches = int(alg.critic_minibatches)
        self.wm_only_updates = int(alg.wm_only_updates)
        self.max_nonfinite_skips = int(alg.max_nonfinite_skips)
        self.velocity = bool(model_cfg.velocity_tracking)
        self.goal_encoding = str(model_cfg.goal_encoding)
        self.goal_anchor = bool(model_cfg.goal_anchor)
        self.num_envs = int(cfg.env.train.total_num_envs) // world

        self.wm_optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.critic_optimizer_steps = 0
        self.nonfinite_skips = 0
        self.consecutive_nonfinite = 0
        self._pending_skips: list[tuple[str, torch.Tensor]] = []
        self.torch_profile_update = cfg.actor.get("torch_profile_update", None)

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
        # fill a world-model window; with none, collection never yields one.
        longest = int(self.motion.lengths.max())
        if longest < self.k:
            raise ValueError(
                f"the longest clip in env.train.motion_path has {longest} control "
                f"steps; world-model windows need at least algorithm.k={self.k}"
            )
        self.q0 = self.motion.q0.to(device)
        joint_low, joint_high = asset_joint_limits()
        self.joint_low = joint_low.to(device)
        self.joint_high = joint_high.to(device)
        self.replay = SuperMimicReplay(
            self.num_envs,
            int(self.cfg.algorithm.replay_length),
            device,
            obs_dim=int(self.cfg.actor.model.obs_dim),
        )
        self._env_ids = torch.arange(self.num_envs, device=device)

    def setup_model_and_optimizer(self) -> None:
        """Wrap the policy with FSDP and build one optimizer per phase.

        ``fsdp_config.amp_autocast`` applies inside the network forwards only;
        kinematics, rewards and losses stay float32.
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
        self.model: FSDPModule = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )

        # Substring filters; the velocity critic must match before the motion
        # critic's "critic." does. The encoder and decoder stay in the main group.
        actor_cfg = self.cfg.actor
        param_filters = {"world_model": ["dynamics.", "termination."]}
        optim_cfgs = {"world_model": actor_cfg.world_model_optim}
        if self.velocity:
            param_filters["velocity_actor"] = ["velocity_encoder."]
            param_filters["velocity_critic"] = ["velocity_critic."]
            optim_cfgs["velocity_actor"] = actor_cfg.velocity_actor_optim
            optim_cfgs["velocity_critic"] = actor_cfg.velocity_critic_optim
        param_filters["critic"] = ["critic."]
        optim_cfgs["critic"] = actor_cfg.critic_optim
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=actor_cfg.optim,
            param_filters=param_filters,
            filtered_optim_config=optim_cfgs,
        )
        names = ["actor", *param_filters]
        optim_cfgs["actor"] = actor_cfg.optim
        self.phase_optimizers = dict(zip(names, optimizers, strict=True))
        self.phase_optim_cfgs = optim_cfgs
        self.phase_schedulers = {}
        for name, optimizer in self.phase_optimizers.items():
            warmup_optimizer_state(optimizer)
            self.phase_schedulers[name] = self.build_lr_scheduler(
                optimizer, optim_cfgs[name]
            )
        self.optimizer = self.phase_optimizers["actor"]
        self.lr_scheduler = self.phase_schedulers["actor"]

    # ------------------------------------------------------------- collection

    @Worker.timer("ingest")
    def ingest(self, num_steps: int) -> dict[str, float]:
        """Store ``num_steps`` vector steps from this rank's env worker.

        Each transition also carries the statistics the rollout worker folded
        into its input norms; folding the same values keeps both copies equal.
        The world-model norms fold every transition, merged over the actor ranks.
        """
        dynamics = self.model.dynamics
        for _ in range(num_steps):
            step, counts = self.recv(self._env_group_name, src_rank=self._rank)
            if "obs_stats" in step:
                self.model.fold_policy_stats(
                    step["obs_stats"],
                    counts["obs_count"],
                    step["goal_stats"],
                    counts["goal_count"],
                )
            with torch.no_grad():
                feat, target = dynamics.norm_rows(
                    step["state"],
                    step["executed"],
                    step["xi"],
                    step["next_state"],
                    self.residual_frame,
                )
            dynamics.fold_norm_rows(feat, target, group=self._dp_group)
            self.replay.add(
                self._env_ids,
                step["state"],
                step["next_state"],
                step["action"],
                step["executed"],
                step["reward"],
                step["end_kind"],
                step["xi"],
                step["motion_id"],
                step["motion_time"],
                step["time_in_episode"],
                prev_action=step["prev_action"],
                obs_noisy=step["obs"],
                task_id=step["task"],
                command=step["command"],
                command_from_ref=step["command_from_ref"],
            )
        return {"buffer/valid_windows": float(self.replay.num_valid_windows(self.k))}

    # --------------------------------------------------------------- training

    @Worker.timer("run_updates")
    def run_updates(self, num_updates: int, global_step: int) -> dict[str, float]:
        """Run ``num_updates`` global updates; ``global_step`` counts finished ones.

        Metrics stay on the device until the last update; one read then brings
        them to the host and settles the steps fused Adam skipped.
        """
        metrics: dict[str, float | torch.Tensor] = {}
        for step in range(global_step, global_step + num_updates):
            traced = step == self.torch_profile_update
            with self._torch_profile(step) if traced else contextlib.nullcontext():
                self._run_update(step, metrics)
        self.model.reshard()
        metrics = self._read_metrics(metrics)
        metrics.update(
            {
                "wm_only": float(global_step + num_updates <= self.wm_only_updates),
                "wm_optimizer_steps": float(self.wm_optimizer_steps),
                "actor_optimizer_steps": float(self.actor_optimizer_steps),
                "critic_optimizer_steps": float(self.critic_optimizer_steps),
                "nonfinite_skips": float(self.nonfinite_skips),
            }
        )
        return metrics

    def _run_update(self, step: int, metrics: dict) -> None:
        """Run global update ``step`` and add its metrics to ``metrics``."""
        with self._phase("world_model"):
            metrics.update(self._fit_world_model())
        actor_updated = step >= self.wm_only_updates
        if actor_updated:
            with self._phase("motion_actor"):
                actor_metrics, critic_batch = self._train_motion_actor()
            with self._phase("motion_critic"):
                metrics.update(self._train_motion_critic(*critic_batch))
            metrics.update(actor_metrics)
            if self.velocity:
                with self._phase("velocity_actor"):
                    velocity_metrics, critic_batch = self._train_velocity_actor()
                with self._phase("velocity_critic"):
                    self._train_velocity_critic(*critic_batch)
                metrics.update(velocity_metrics)
        self._finish_update(actor_updated)

    @contextlib.contextmanager
    def _phase(self, name: str):
        """Time one training phase and label it in ``torch.profiler`` traces."""
        with self.worker_timer(name), torch.profiler.record_function(name):
            yield

    @contextlib.contextmanager
    def _torch_profile(self, update: int):
        """Trace one update with ``torch.profiler``.

        Operator tables (``.txt``) and a Chrome trace (``.json``) go to
        ``<log_path>/<experiment_name>/torch_profile``.
        """
        on_cuda = Worker.torch_device_type == "cuda"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if on_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profiler:
            yield
        logger_cfg = self.cfg.runner.logger
        out_dir = os.path.join(
            logger_cfg.log_path, logger_cfg.experiment_name, "torch_profile"
        )
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"update_{update}_rank_{self._rank}")
        averages = profiler.key_averages()
        sort_keys = ["cpu_time_total", "self_cpu_time_total"]
        if on_cuda:
            sort_keys.append("self_device_time_total")
        with open(stem + ".txt", "w") as file:
            for key in sort_keys:
                table = averages.table(sort_by=key, row_limit=50)
                file.write(f"sorted by {key}\n{table}\n\n")
        profiler.export_chrome_trace(stem + ".json")
        self.log_info(f"torch.profiler output of update {update}: {stem}.txt, .json")

    def _read_metrics(
        self, metrics: dict[str, float | torch.Tensor]
    ) -> dict[str, float]:
        """Read the device metrics and skip flags with one host synchronization.

        A flagged step was left undone by fused Adam; it leaves the phase's
        step counter and counts as a non-finite skip, in step order.
        """
        names = [name for name, value in metrics.items() if torch.is_tensor(value)]
        skips, self._pending_skips = self._pending_skips, []
        device_values = [metrics[name] for name in names]
        device_values += [flag for _, flag in skips]
        hosted = []
        if device_values:
            hosted = torch.stack(
                [value.detach().float().reshape(()) for value in device_values]
            ).tolist()
        out = {
            name: float(value)
            for name, value in metrics.items()
            if not torch.is_tensor(value)
        }
        out.update(zip(names, hosted[: len(names)], strict=True))
        self._replay_skips(skips, hosted[len(names) :])
        return out

    def _settle_skips(self) -> None:
        """Count the pending fused steps now, ahead of a host-checked step.

        The consecutive-skip count follows step order, so the fused flags are
        replayed before a host-checked step resets or extends it.
        """
        skips, self._pending_skips = self._pending_skips, []
        if skips:
            self._replay_skips(skips, torch.stack([flag for _, flag in skips]).tolist())

    def _replay_skips(
        self, skips: list[tuple[str, torch.Tensor]], flags: list[float]
    ) -> None:
        """Apply fused Adam's skip flags in step order."""
        for (phase, _), flag in zip(skips, flags, strict=True):
            if not flag:
                self.consecutive_nonfinite = 0
                continue
            counter = _STEP_COUNTERS.get(phase)
            if counter is not None:
                setattr(self, counter, getattr(self, counter) - 1)
            self._note_nonfinite(phase)

    def _enter_phase(self, forward_type: ForwardType) -> None:
        """Freeze every parameter outside one phase and clear stale gradients.

        FSDP copies ``requires_grad`` from the sharded parameters when it
        gathers them, so the model is resharded before the switch.
        """
        self.model.reshard()
        self.model.zero_grad(set_to_none=True)
        self.model.set_trainable(forward_type)

    def _global_sum(self, values: list[float]) -> list[float]:
        """Sum per-rank host values over the actor ranks."""
        if self._world_size == 1:
            return [float(value) for value in values]
        packed = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(packed, group=self._dp_group)
        return packed.tolist()

    def _loss_scale(self, weight: float, total: float) -> float:
        """Scale that turns FSDP's mean over ranks into a row-weighted mean."""
        return self._world_size * weight / total

    def _note_nonfinite(self, phase: str) -> None:
        self.nonfinite_skips += 1
        self.consecutive_nonfinite += 1
        if self.consecutive_nonfinite > self.max_nonfinite_skips:
            raise RuntimeError(
                f"non-finite gradients in {self.consecutive_nonfinite} consecutive "
                f"optimizer steps (last: {phase}); stopping instead of training on NaN"
            )

    def _optimizer_step(
        self, phase: str, forward_type: ForwardType, total: float
    ) -> tuple[bool, float | torch.Tensor]:
        """Clip and step one phase's optimizer on every rank alike.

        ``total`` is the global row count behind the gradient; with no rows
        the step is skipped. A non-finite gradient norm skips the step too,
        right here or, with fused Adam, inside the kernel (:meth:`_fused_step`).
        """
        optimizer = self.phase_optimizers[phase]
        if total <= 0:
            optimizer.zero_grad(set_to_none=True)
            return False, 0.0
        params = self.model.phase_parameters(forward_type)
        if optimizer.param_groups[0].get("fused"):
            return True, self._fused_step(phase, optimizer, params)
        self._settle_skips()
        norm = get_grad_norm(params, self._dp_group)
        if not math.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            self._note_nonfinite(phase)
            return False, float("nan")
        self.consecutive_nonfinite = 0
        optim_cfg = self.phase_optim_cfgs[phase]
        clip_grad_by_total_norm_(params, float(optim_cfg.clip_grad), norm)
        value_clip = optim_cfg.get("clip_grad_value", None)
        if value_clip is not None:
            for param in params:
                if param.grad is not None:
                    to_local_if_dtensor(param.grad).clamp_(
                        -float(value_clip), float(value_clip)
                    )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return True, norm

    def _fused_step(
        self, phase: str, optimizer: torch.optim.Optimizer, params: list
    ) -> torch.Tensor:
        """Clip and step fused Adam without waiting for the device.

        A non-finite norm becomes the kernel's ``found_inf``, which leaves the
        parameters, moments and step count unchanged; :meth:`_replay_skips`
        counts such steps later, in step order. Returns the pre-clip norm.
        """
        grads = self._local_grads(params)
        norm = self._grad_norm(grads)
        optim_cfg = self.phase_optim_cfgs[phase]
        scale = (float(optim_cfg.clip_grad) / (norm + 1.0e-6)).clamp(max=1.0)
        torch._foreach_mul_(grads, scale)
        value_clip = optim_cfg.get("clip_grad_value", None)
        if value_clip is not None:
            torch._foreach_clamp_min_(grads, -float(value_clip))
            torch._foreach_clamp_max_(grads, float(value_clip))
        found_inf = (~torch.isfinite(norm)).float()
        optimizer.found_inf = found_inf
        try:
            optimizer.step()
        finally:
            optimizer.found_inf = None
        optimizer.zero_grad(set_to_none=True)
        self._pending_skips.append((phase, found_inf))
        return norm

    @staticmethod
    def _local_grads(params) -> list[torch.Tensor]:
        return [to_local_if_dtensor(p.grad) for p in params if p.grad is not None]

    def _grad_norm(self, grads: list[torch.Tensor]) -> torch.Tensor:
        """L2 norm of the local gradient shards over the actor ranks, on the device."""
        squared = (
            torch.nn.utils.get_total_norm(grads).square()
            if grads
            else torch.zeros((), device=self.device)
        )
        if self._world_size > 1:
            dist.all_reduce(squared, group=self._dp_group)
        return squared.sqrt()

    def _critic_chunk_totals(self, batch: int) -> list[float]:
        """Global row count of each critic minibatch slot.

        Chunk sizes depend only on the local batch, so one reduction serves
        every epoch.
        """
        sizes = [
            float(chunk.numel())
            for chunk in torch.tensor_split(
                torch.arange(batch), self.critic_minibatches
            )
        ]
        return self._global_sum(sizes)

    def _fit_world_model(self) -> dict[str, float | torch.Tensor]:
        self._enter_phase(WORLD_MODEL)
        held: torch.Tensor | None = None
        norm: float | torch.Tensor = 0.0
        for _ in range(self.wm_batches):
            batch = self.replay.sample_world_model_windows(self.k, self.wm_window_batch)
            weight = 0 if batch is None else int(batch["state"].shape[0])
            motion = 0
            if batch is not None and self.velocity:
                motion = int((batch["task_id"] == 0).sum())
            total, motion_total = self._global_sum([weight, motion])
            if total <= 0:
                continue
            if batch is None:
                out = self.model(forward_type=WORLD_MODEL, empty=True)
            else:
                # Each rank's BCE averages its own motion rows; this weight
                # makes the row-weighted gradient mean the mean over all of them.
                bce_scale = 1.0
                if self.velocity:
                    bce_scale = (
                        motion * total / (weight * motion_total)
                        if motion_total > 0
                        else 0.0
                    )
                out = self.model(
                    forward_type=WORLD_MODEL,
                    **self._world_model_inputs(batch),
                    bce_scale=bce_scale,
                )
                held = torch.stack(
                    (out["loss"].detach(), out["one_step_loss"], out["k_step_loss"])
                )
            (out["loss"] * self._loss_scale(weight, total)).backward()
            stepped, step_norm = self._optimizer_step("world_model", WORLD_MODEL, total)
            if stepped:
                self.wm_optimizer_steps += 1
                norm = step_norm
        if held is None:
            return {"wm/grad_norm": norm}
        return {
            "wm/state_loss": held[0],
            "wm/grad_norm": norm,
            "wm/one_step_loss": held[1],
            "wm/k_step_loss": held[2],
        }

    def _world_model_inputs(self, batch: dict[str, torch.Tensor]) -> dict:
        state = batch["state"]
        if self.learned_termination:
            refs = torch.stack(
                [
                    self.motion.state_at(
                        batch["motion_id"], batch["motion_time"] + (t + 1)
                    )
                    for t in range(self.k)
                ],
                dim=1,
            )
        else:
            refs = state[:, 1:]
        return {
            "state0": state[:, 0],
            "actions": batch["action_executed"],
            "xi": batch["xi"],
            "targets": state[:, 1:],
            "refs": refs,
            "end_kind": batch["end_kind"],
            "task": batch["task_id"] if self.velocity else None,
        }

    def _motion_inputs(self, starts: dict[str, torch.Tensor]) -> dict:
        """Reference states, goals and body poses for times ``t .. t+H``.

        Goals that ignore the imagined state are precomputed in one batched
        query. The anchor goal reads the imagined torso, so its 680-d part is
        passed instead and composed step by step.
        """
        motion_id, time0 = starts["motion_id"], starts["motion_time"]
        batch, steps = motion_id.shape[0], self.horizon + 1
        offsets = torch.arange(steps, device=time0.device, dtype=time0.dtype)
        flat_id = motion_id.unsqueeze(1).expand(-1, steps).reshape(-1)
        flat_time = (time0.unsqueeze(1) + offsets).reshape(-1)
        refs = self.motion.state_at(flat_id, flat_time)
        ref_pos, ref_rot, ref_lin, ref_ang = (
            part.reshape(batch, steps, *part.shape[1:]).contiguous()
            for part in self.motion.kinematics_at(flat_id, flat_time)
        )
        inputs = {
            "state": starts["state"],
            "prev_action": starts["prev_action"],
            "xi": starts["xi"],
            "history": starts["history"],
            "refs": refs.view(batch, steps, -1),
            "ref_pos": ref_pos,
            "ref_rot": ref_rot,
            "ref_lin": ref_lin,
            "ref_ang": ref_ang,
            "q0": self.q0,
            "joint_low": self.joint_low,
            "joint_high": self.joint_high,
        }
        if self.goal_anchor:
            goal_refs = self.motion.goal_ref(flat_id, flat_time)
            inputs["goal_refs"] = goal_refs.view(batch, steps, -1)
        else:
            goals = make_goal(
                self.motion, self.goal_encoding, False, flat_id, flat_time, None
            )
            inputs["goals"] = goals.view(batch, steps, -1)
        return inputs

    def _train_motion_actor(self) -> tuple[dict[str, float | torch.Tensor], tuple]:
        """Backpropagate the H-step imagination into the encoder and decoder.

        Returns the metrics and the detached critic features, rewards and
        termination probabilities of the same trajectory.
        """
        self._enter_phase(ACTOR)
        rows = self.imagine_batch // 2 if self.velocity else self.imagine_batch
        starts = self.replay.sample_imagination_starts(
            rows, task=0 if self.velocity else None
        )
        weight = 0 if starts is None else int(starts["state"].shape[0])
        (total,) = self._global_sum([weight])
        out = None
        if total > 0:
            if starts is None:
                loss = self.model(forward_type=ACTOR, empty=True)["loss"]
            else:
                out = self.model(forward_type=ACTOR, **self._motion_inputs(starts))
                loss = out["loss"]
            (loss * self._loss_scale(weight, total)).backward()
        # The decoder's last layer has gain 0.01, so the encoder sees a much
        # smaller gradient than the decoder itself; both are logged.
        encoder_norm = self._grad_norm(
            self._local_grads(self.model.encoder.parameters())
        )
        decoder_norm = self._grad_norm(
            self._local_grads(self.model.decoder.parameters())
        )
        stepped, grad_norm = self._optimizer_step("actor", ACTOR, total)
        if stepped:
            self.actor_optimizer_steps += 1
        if out is None:
            width = self.model.critic.net[0].in_features
            empty = torch.zeros(0, self.horizon, device=self.device)
            return {}, (
                torch.zeros(0, self.horizon + 1, width, device=self.device),
                empty,
                empty,
            )
        d_hat = out["d"]
        metrics = {
            "actor/objective": out["objective"],
            "actor/grad_norm": grad_norm,
            "actor/encoder_grad_norm": encoder_norm,
            "actor/decoder_grad_norm": decoder_norm,
            "imagination/survival": (1.0 - d_hat).prod(dim=1).mean(),
            "imagination/termination_prob": d_hat.mean(),
            "imagination/reward": out["rewards"].mean(),
        }
        return metrics, (out["features"], out["rewards"], d_hat)

    def _train_motion_critic(
        self, features: torch.Tensor, rewards: torch.Tensor, d_hat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Regress V onto the lambda-returns of the actor phase's trajectory.

        ``features`` is ``[B, H+1, F]``. The running norm folds them once, and
        the normalized batch serves the targets and every minibatch.
        """
        self._enter_phase(CRITIC)
        batch, width = features.shape[0], features.shape[-1]
        norm = self.model.critic.norm
        if norm is not None:
            norm.fold(features.reshape(-1, width), self._dp_group)
            features = norm(features).detach()
        with torch.no_grad():
            values = self.model(
                forward_type=CRITIC, features=features.reshape(-1, width)
            )["values"].reshape(batch, self.horizon + 1)
        self.model.reshard()
        targets = gae_targets(rewards, d_hat, values, self.gamma, self.lam)
        fit_features = features[:, : self.horizon]
        totals = self._critic_chunk_totals(batch)
        held = None
        for _ in range(self.critic_epochs):
            chunks = torch.tensor_split(
                torch.randperm(batch, device=self.device), self.critic_minibatches
            )
            for chunk, total in zip(chunks, totals, strict=True):
                if total <= 0:
                    continue
                rows = int(chunk.numel())
                if rows:
                    out = self.model(
                        forward_type=CRITIC,
                        features=fit_features[chunk].reshape(-1, width),
                        targets=targets[chunk].reshape(-1),
                    )
                    held = out["loss"].detach()
                else:
                    out = self.model(forward_type=CRITIC, empty=True)
                (out["loss"] * self._loss_scale(rows, total)).backward()
                stepped, _ = self._optimizer_step("critic", CRITIC, total)
                if stepped:
                    self.critic_optimizer_steps += 1
        if batch == 0:
            return {}
        metrics = {
            "critic/value_mean": values[:, : self.horizon].mean(),
            "critic/target_mean": targets.mean(),
        }
        if held is not None:
            metrics["critic/loss"] = held
        return metrics

    def _velocity_inputs(self, starts: dict[str, torch.Tensor]) -> dict:
        """Reference commands for every start and motion goals for ``from_ref`` rows."""
        motion_id, time0 = starts["motion_id"], starts["motion_time"]
        from_ref = starts["command_from_ref"]
        ref_commands = torch.stack(
            [
                self.motion.reference_velocity_command(motion_id, time0 + t)
                for t in range(self.horizon + 1)
            ],
            dim=1,
        )
        rows = from_ref.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            width = goal_width(self.goal_encoding, False)
            ref_goals = torch.zeros(0, self.horizon, width, device=self.device)
        else:
            offsets = torch.arange(self.horizon, device=time0.device, dtype=time0.dtype)
            flat_id = motion_id[rows].unsqueeze(1).expand(-1, self.horizon).reshape(-1)
            flat_time = (time0[rows].unsqueeze(1) + offsets).reshape(-1)
            ref_goals = make_goal(
                self.motion, self.goal_encoding, False, flat_id, flat_time, None
            ).view(rows.numel(), self.horizon, -1)
        return {
            "state": starts["state"],
            "prev_action": starts["prev_action"],
            "xi": starts["xi"],
            "history": starts["history"],
            "command": starts["command"],
            "from_ref": from_ref,
            "ref_commands": ref_commands,
            "ref_goals": ref_goals,
            "q0": self.q0,
            "joint_low": self.joint_low,
            "joint_high": self.joint_high,
        }

    def _train_velocity_actor(
        self,
    ) -> tuple[dict[str, float | torch.Tensor], tuple]:
        """Update the velocity encoder through imagination plus latent alignment.

        The RL term averages over every start and the alignment term over the
        ``from_ref`` starts, so each rank scales its terms by its share of
        those global counts and the summed gradient is the global mean.
        """
        self._enter_phase(VELOCITY_ACTOR)
        starts = self.replay.sample_imagination_starts(self.imagine_batch // 2, task=1)
        local = [0.0, 0.0]
        if starts is not None:
            local = [
                float(starts["state"].shape[0]),
                float(starts["command_from_ref"].sum()),
            ]
        rows_total, ref_total = self._global_sum(local)
        out = None
        if rows_total > 0:
            if starts is None:
                loss = self.model(forward_type=VELOCITY_ACTOR, empty=True)["loss"]
            else:
                out = self.model(
                    forward_type=VELOCITY_ACTOR,
                    **self._velocity_inputs(starts),
                    rl_share=local[0] / rows_total,
                    align_share=local[1] / ref_total if ref_total > 0 else 0.0,
                )
                loss = out["loss"]
            # The shares already sum to one over the ranks; undo FSDP's mean.
            (loss * self._world_size).backward()
        _, grad_norm = self._optimizer_step(
            "velocity_actor", VELOCITY_ACTOR, rows_total
        )
        if out is None:
            state_dim = self.replay.state.shape[-1]
            xi_dim = self.replay.xi.shape[-1]
            empty = torch.zeros(0, self.horizon, device=self.device)
            return {}, (
                torch.zeros(0, self.horizon + 1, state_dim, device=self.device),
                torch.zeros(0, self.horizon + 1, 3, device=self.device),
                torch.zeros(0, xi_dim, device=self.device),
                empty,
                empty,
            )
        states, commands = out["states"], out["commands"]
        rewards, d_hat = out["rewards"], out["d"]
        planar, yaw_rate = heading_velocity(states[:, 1:])
        commands_next = commands[:, 1:]
        metrics = {
            "velocity/objective": out["objective"],
            "velocity/grad_norm": grad_norm,
            "velocity/align": out["align"],
            "velocity/survival": (1.0 - d_hat).prod(dim=1).mean(),
            "velocity/fall_prob": d_hat.mean(),
            "velocity/reward": rewards.mean(),
            "velocity/imagined_lin_err": (planar - commands_next[..., :2])
            .norm(dim=-1)
            .mean(),
            "velocity/imagined_yaw_err": (yaw_rate - commands_next[..., 2])
            .abs()
            .mean(),
        }
        return metrics, (states, commands, starts["xi"], rewards, d_hat)

    def _train_velocity_critic(
        self,
        states: torch.Tensor,
        commands: torch.Tensor,
        xi: torch.Tensor,
        rewards: torch.Tensor,
        d_hat: torch.Tensor,
    ) -> None:
        """Regress the velocity critic onto the velocity trajectory's lambda-returns."""
        self._enter_phase(VELOCITY_CRITIC)
        batch, steps = states.shape[0], self.horizon + 1

        def flat(values: torch.Tensor, length: int) -> torch.Tensor:
            return values[:, :length].reshape(-1, values.shape[-1])

        def flat_xi(rows: torch.Tensor, length: int) -> torch.Tensor:
            return rows.unsqueeze(1).expand(-1, length, -1).reshape(-1, rows.shape[-1])

        critic = self.model.velocity_critic
        critic.fold_norm(
            flat(states, steps),
            flat(commands, steps),
            flat_xi(xi, steps),
            self._dp_group,
        )
        with torch.no_grad():
            values = self.model(
                forward_type=VELOCITY_CRITIC,
                state=flat(states, steps),
                command=flat(commands, steps),
                xi=flat_xi(xi, steps),
            )["values"].reshape(batch, steps)
        self.model.reshard()
        targets = gae_targets(rewards, d_hat, values, self.gamma, self.lam)
        totals = self._critic_chunk_totals(batch)
        for _ in range(self.critic_epochs):
            chunks = torch.tensor_split(
                torch.randperm(batch, device=self.device), self.critic_minibatches
            )
            for chunk, total in zip(chunks, totals, strict=True):
                if total <= 0:
                    continue
                rows = int(chunk.numel())
                if rows:
                    out = self.model(
                        forward_type=VELOCITY_CRITIC,
                        state=flat(states[chunk], self.horizon),
                        command=flat(commands[chunk], self.horizon),
                        xi=flat_xi(xi[chunk], self.horizon),
                        targets=targets[chunk].reshape(-1),
                    )
                else:
                    out = self.model(forward_type=VELOCITY_CRITIC, empty=True)
                (out["loss"] * self._loss_scale(rows, total)).backward()
                stepped, _ = self._optimizer_step(
                    "velocity_critic", VELOCITY_CRITIC, total
                )
                if stepped:
                    self.critic_optimizer_steps += 1

    def _finish_update(self, actor_updated: bool) -> None:
        """Advance the schedulers of the phases this update ran."""
        self.phase_schedulers["world_model"].step()
        if actor_updated:
            for name, scheduler in self.phase_schedulers.items():
                if name != "world_model":
                    scheduler.step()

    # ------------------------------------------------------- weights and state

    def get_rollout_state_dict(self) -> dict:
        self.model.reshard()
        return super().get_rollout_state_dict()

    def _interface(self) -> dict:
        return interface_record(self.cfg.actor.model, self.cfg.algorithm)

    def save_checkpoint(self, save_base_path: str, step: int = 0) -> None:
        """Networks, optimizers, schedulers and RNG of every rank, plus counters.

        The replay is not saved; a resumed run refills it before training.
        """
        self.model.reshard()
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=list(self.phase_optimizers.values()),
            lr_schedulers=list(self.phase_schedulers.values()),
            save_path=save_base_path,
            save_full_model_weights=True,
        )
        if self._rank == 0:
            write_interface(save_base_path, self._interface())
            counters = {
                "wm_optimizer_steps": self.wm_optimizer_steps,
                "actor_optimizer_steps": self.actor_optimizer_steps,
                "critic_optimizer_steps": self.critic_optimizer_steps,
                "nonfinite_skips": self.nonfinite_skips,
            }
            with open(os.path.join(save_base_path, _COUNTERS_FILE), "w") as file:
                json.dump(counters, file, indent=2)
        dist.barrier()

    def load_checkpoint(self, load_base_path: str) -> None:
        check_interface(
            find_interface(load_base_path, levels=0), self._interface(), load_base_path
        )
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=list(self.phase_optimizers.values()),
            lr_schedulers=list(self.phase_schedulers.values()),
            load_path=load_base_path,
        )
        with open(os.path.join(load_base_path, _COUNTERS_FILE)) as file:
            counters = json.load(file)
        self.wm_optimizer_steps = int(counters["wm_optimizer_steps"])
        self.actor_optimizer_steps = int(counters["actor_optimizer_steps"])
        self.critic_optimizer_steps = int(counters["critic_optimizer_steps"])
        self.nonfinite_skips = int(counters["nonfinite_skips"])
