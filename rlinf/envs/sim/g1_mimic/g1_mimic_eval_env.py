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

"""G1 evaluation vector for SuperMimic, driven by the generic eval loop.

Evaluation is a table of jobs: one per reference clip (motion tracking) or
per push trial. The jobs are dealt round-robin to the env ranks, and every
rollout epoch starts the next ``num_envs`` jobs of this rank together; after
``rollout_epoch`` epochs the next evaluation starts from the first jobs
again. Rows run without randomization, reset noise or action noise until
their job ends. The vector reports done only when every row has ended, and
then returns one result row per env in ``infos["episode"]``. Rows past this
rank's last job are padding with ``valid = 0``.

``tracking`` starts every clip at its first frame, as training resets do:
the state is written at a frame and the reference clock points at the next
one. ``push_recovery`` holds a static standing reference.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.supermimic.representations import make_representation
from rlinf.algorithms.supermimic.scan import root_goals
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    DT,
    MAX_TRACKING_REWARD,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
)
from rlinf.envs.sim.g1_mimic.eval_metrics import EVAL_TASKS, PUSH_RECOVERY, TRACKING
from rlinf.envs.sim.g1_mimic.features import quat_to_rotmat
from rlinf.envs.sim.g1_mimic.g1_fk import (
    PELVIS_BODY,
    TORSO_BODY,
    asset_joint_limits,
    set_kinematics_compile,
)
from rlinf.envs.sim.g1_mimic.g1_mimic_env import build_motion_library, env_device
from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env
from rlinf.envs.sim.g1_mimic.push_recovery import (
    OUTCOMES,
    REASONS,
    RecoveryCriteria,
    RecoveryTracker,
    check_push_protocol,
    control_steps,
    criteria_dict,
    default_standing_state,
    hold_pose_library,
    initialization_reason,
    is_stable,
    reference_steps,
    root_tilt,
    torso_orientation_error,
    trial_plan,
    world_push_xy,
)
from rlinf.envs.sim.g1_mimic.reward import (
    failure_signals,
    rotation_angle,
    tracking_reward,
    xy_yaw_drift,
)

_PADDING = -1


def checked_policy_action(action: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Reject non-finite actions on active rows and zero the other rows."""
    invalid = ~torch.isfinite(action).all(dim=-1) & active
    if bool(invalid.any()):
        rows = invalid.nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(
            f"non-finite policy actions in active evaluation rows {rows}"
        )
    return torch.where(active.unsqueeze(-1), action, torch.zeros_like(action))


def check_eval_states(state: torch.Tensor, active: torch.Tensor) -> None:
    """Reject non-finite simulator states on active rows."""
    invalid = ~torch.isfinite(state).all(dim=-1) & active
    if bool(invalid.any()):
        rows = invalid.nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"non-finite evaluation states in active rows {rows}")


def _quat_geodesic(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    dot = (q * ref).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


class G1MimicEvalEnv:
    """One rank's evaluation vector. See the module docstring for the protocol."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed = int(cfg.seed)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.device = env_device()
        set_kinematics_compile(bool(cfg.torch_compile))
        self.tasks = [str(task) for task in cfg.eval_tasks]
        if self.tasks not in ([TRACKING], [PUSH_RECOVERY]):
            raise ValueError(
                "env.eval.eval_tasks must be [tracking] or [push_recovery], "
                f"got {self.tasks}"
            )
        self.representation = make_representation(str(cfg.representation)).to(
            self.device
        )
        self._chunk = 0
        self._t = 0
        self._build_sim()
        joint_low, joint_high = asset_joint_limits()
        self.joint_low = joint_low.to(self.device)
        self.joint_high = joint_high.to(self.device)
        self._build_jobs()
        self._allocate_rows()

    # ----------------------------------------------------------------- setup

    def _build_sim(self) -> None:
        cfg = self.cfg
        if PUSH_RECOVERY in self.tasks:
            push = cfg.push_recovery
            self.criteria = RecoveryCriteria(
                speed_threshold=float(push.speed_threshold),
                tilt_threshold=float(push.tilt_threshold),
                torso_ori_threshold=float(push.torso_ori_threshold),
                stable_seconds=float(push.stable_seconds),
                max_xy_drift=None
                if push.get("max_xy_drift", None) is None
                else float(push.max_xy_drift),
            )
            self.push_speeds = check_push_protocol(
                list(push.speeds),
                int(push.trials),
                float(push.settle_seconds),
                float(push.max_seconds),
                self.criteria,
                str(push.frame),
                str(push.push_mode),
                push.direction,
            )
            self.settle_steps = control_steps(float(push.settle_seconds))
            self.post_steps = control_steps(float(push.max_seconds))
            self.motion = hold_pose_library(
                default_standing_state(),
                reference_steps(self.settle_steps, self.post_steps),
            )
        else:
            self.motion = build_motion_library(cfg, self.device)
        self.sim = MjlabG1Env(
            num_envs=self.num_envs,
            motion=self.motion,
            device=self.device,
            max_episode_seconds=None,
            randomize=False,
        )

    def _build_jobs(self) -> None:
        """Global job table, then this rank's share of it."""
        task_ids, args, lengths = [], [], []

        def add(task, arg, length):
            task_ids.append(EVAL_TASKS.index(task))
            args.append(int(arg))
            lengths.append(int(length))

        if TRACKING in self.tasks:
            clip_lengths = self.motion.clip_length(
                torch.arange(self.motion.num_clips, device=self.device)
            ).tolist()
            for clip, length in enumerate(clip_lengths):
                # The first frame is the start state; every later frame is
                # one control step.
                add(TRACKING, clip, max(length - 1, 1))
        if PUSH_RECOVERY in self.tasks:
            push = self.cfg.push_recovery
            self.push_plan = trial_plan(
                self.push_speeds, int(push.trials), push.direction, self.seed
            )
            for item in self.push_plan:
                add(PUSH_RECOVERY, item["trial"], self.settle_steps + self.post_steps)
            self.push_directions = torch.tensor(
                [[item["direction_x"], item["direction_y"]] for item in self.push_plan],
                device=self.device,
            )
            self.push_speed_of = torch.tensor(
                [item["speed_m_s"] for item in self.push_plan], device=self.device
            )
            self.save_trajectory = bool(push.get("save_trajectory", False))

        self.num_jobs = len(task_ids)
        local = slice(self.seed_offset, None, self.total_num_processes)
        self.job_task = torch.tensor(task_ids[local], dtype=torch.long)
        self.job_id = torch.arange(len(task_ids))[local]
        self.job_arg = torch.tensor(args[local], dtype=torch.long)
        self.job_length = torch.tensor(lengths[local], dtype=torch.long)
        num_jobs = int(self.job_task.numel())
        rollout_epoch = int(self.cfg.rollout_epoch)
        if rollout_epoch * self.num_envs < num_jobs:
            raise ValueError(
                f"env rank {self.seed_offset} holds {num_jobs} evaluation jobs but "
                f"rollout_epoch * envs per rank = {rollout_epoch} * {self.num_envs}; "
                "raise env.eval.rollout_epoch or env.eval.total_num_envs"
            )
        longest = max(lengths) if lengths else 0
        if longest > int(self.cfg.max_steps_per_rollout_epoch):
            raise ValueError(
                f"the longest evaluation job runs {longest} control steps, more "
                "than env.eval.max_steps_per_rollout_epoch="
                f"{self.cfg.max_steps_per_rollout_epoch}"
            )

    def _allocate_rows(self) -> None:
        n, device = self.num_envs, self.device

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(n, *shape, dtype=dtype, device=device)

        self.row_task = torch.full((n,), _PADDING, dtype=torch.long, device=device)
        self.row_job = torch.full_like(self.row_task, _PADDING)
        self.row_arg = torch.zeros_like(self.row_task)
        self.row_length = torch.zeros_like(self.row_task)
        self.alive = zeros(dtype=torch.bool)
        self.prev = zeros(ACTION_DIM)
        self.acc = {
            name: zeros()
            for name in (
                "return_sum",
                "body_pos_err",
                "torso_pos_err",
                "torso_ori_err",
                "free_root_pos_err",
                "free_root_ori_err",
                "free_root_xy_drift",
                "free_root_yaw_drift",
                "steps",
                "fail",
                "clip_end",
                "fail_step",
            )
        }
        if PUSH_RECOVERY in self.tasks:
            self.outcome = zeros(dtype=torch.long)
            self.reason = zeros(dtype=torch.long)
            self.decided = zeros(dtype=torch.bool)
            self.push_values = zeros(4)
            self.origin = zeros(2)
            self.recovery_time = zeros()
            self.onset_time = zeros()
            self.tracker = RecoveryTracker(n, self.criteria, device)
            if self.save_trajectory:
                steps = self.settle_steps + 1 + self.post_steps
                self.trajectory = zeros(steps, 8)

    def _is(self, task: str) -> torch.Tensor:
        return self.row_task == EVAL_TASKS.index(task)

    # ------------------------------------------------------------- interface

    def reset(self, *args, **kwargs):
        """Start the ``num_envs`` jobs of the current chunk of this rank."""
        self._start_jobs()
        return self._observe(), {}

    def update_reset_state_ids(self) -> None:
        """Move to the next chunk of jobs, and back to the first after the last.

        The eval loop calls this once after every rollout epoch, so every
        evaluation in the process covers all of this rank's jobs.
        """
        self._chunk = (self._chunk + 1) % int(self.cfg.rollout_epoch)

    def chunk_step(self, chunk_actions):
        """One control step of every row. Chunks hold a single action."""
        actions = torch.as_tensor(chunk_actions, dtype=torch.float32)
        actions = actions.reshape(self.num_envs, -1, ACTION_DIM)
        if actions.shape[1] != 1:
            raise ValueError(
                f"G1 evaluation steps one action per call, got chunks of "
                f"{actions.shape[1]}"
            )
        if bool(self.alive.any()):
            self._step(actions[:, 0].to(self.device))
        done = bool((~self.alive).all())
        infos = {"episode": self._result_rows()} if done else {}
        terminations = torch.full((self.num_envs, 1), done, dtype=torch.bool)
        truncations = torch.zeros_like(terminations)
        rewards = torch.zeros(self.num_envs, 1)
        return [self._observe()], rewards, terminations, truncations, [infos]

    def metadata(self) -> dict:
        """Names and settings the summaries and exported tables need."""
        meta = {
            "tasks": list(self.tasks),
            "num_jobs": self.num_jobs,
            "num_clips": int(self.motion.num_clips),
            "clip_names": list(self.motion.clip_names),
        }
        if PUSH_RECOVERY in self.tasks:
            push = self.cfg.push_recovery
            meta.update(
                push_speeds=list(self.push_speeds),
                push_trials=int(push.trials),
                push_frame=str(push.frame),
                push_mode=str(push.push_mode),
                push_direction=OmegaConf.to_container(push.direction)
                if OmegaConf.is_config(push.direction)
                else push.direction,
                push_settle_seconds=float(push.settle_seconds),
                push_max_seconds=float(push.max_seconds),
                push_criteria=criteria_dict(self.criteria),
                push_seed=self.seed,
            )
        return meta

    # ----------------------------------------------------------------- jobs

    def _start_jobs(self) -> None:
        n = self.num_envs
        first = self._chunk * n
        self._t = 0
        count = max(0, min(n, int(self.job_task.numel()) - first))
        take = slice(first, first + count)
        self.row_task.fill_(_PADDING)
        self.row_job.fill_(_PADDING)
        self.row_arg.zero_()
        self.row_length.zero_()
        self.row_task[:count] = self.job_task[take].to(self.device)
        self.row_job[:count] = self.job_id[take].to(self.device)
        self.row_arg[:count] = self.job_arg[take].to(self.device)
        self.row_length[:count] = self.job_length[take].to(self.device)
        self.alive = self.row_task != _PADDING
        self.prev.zero_()
        for value in self.acc.values():
            value.zero_()

        ids = torch.arange(n, device=self.device)
        clip = torch.where(self._is(TRACKING), self.row_arg, torch.zeros_like(ids))
        self.sim.reset(ids, clip, torch.zeros_like(ids))
        self.sim.motion_time += 1
        check_eval_states(self.sim.state, self.alive)
        if PUSH_RECOVERY in self.tasks:
            self._start_push_trials()

    def _start_push_trials(self) -> None:
        self.outcome.fill_(OUTCOMES.index("initialization_failed"))
        self.reason.zero_()
        self.decided.zero_()
        self.push_values.fill_(float("nan"))
        self.origin.zero_()
        self.recovery_time.fill_(float("nan"))
        self.onset_time.fill_(float("nan"))
        if self.save_trajectory:
            self.trajectory.fill_(float("nan"))
        sim = self.sim
        _, _, failed = failure_signals(sim.state, sim.reference_state())
        self._stop_before_push(
            self.alive,
            initialization_reason(sim.state, self.joint_low, self.joint_high, failed),
        )
        if self.settle_steps == 0:
            self._apply_push(self.alive & self._is(PUSH_RECOVERY))

    def _stop_before_push(self, rows: torch.Tensor, code: torch.Tensor) -> None:
        stop = rows & (code >= 0)
        self.reason = torch.where(stop, code, self.reason)
        self.alive = self.alive & ~stop

    # ----------------------------------------------------------------- step

    def _step(self, action: torch.Tensor) -> None:
        t = self._t
        self._t += 1
        running = self.alive.clone()
        action = checked_policy_action(action, running)
        result = self.sim.step(action)
        is_push = self._is(PUSH_RECOVERY)
        check_eval_states(self.sim.state, running & ~is_push)
        if TRACKING in self.tasks:
            rows = running & self._is(TRACKING)
            self._track(rows, action, result)
            self._count_ends(t, rows, result)
            ended = result["failure"] | result["clip_end"]
            self.alive = torch.where(rows, ~ended, self.alive)
        self.prev = torch.where(running.unsqueeze(-1), action, self.prev)
        if PUSH_RECOVERY in self.tasks:
            self._push_step(t, running & is_push, result)
        self.alive = self.alive & (t + 1 < self.row_length)

    def _track(self, rows, action, result) -> None:
        sim = self.sim
        state, ref_state = sim.state, result["ref_state"]
        pos, quat, lin, ang = sim.body_state()
        body = (pos, quat_to_rotmat(quat), lin, ang)
        ref = self.motion.body_state(result["frame"])
        reward = tracking_reward(
            state[:, ROOT_POS],
            quat_to_rotmat(state[:, ROOT_QUAT]),
            body,
            ref,
            state[:, 7:36],
            action,
            self.prev,
            sim.soft_limits,
        )
        relative = pos - pos[:, PELVIS_BODY : PELVIS_BODY + 1]
        ref_relative = ref[0] - ref[0][:, PELVIS_BODY : PELVIS_BODY + 1]
        xy, yaw = xy_yaw_drift(state, ref_state)
        acc = self.acc
        for name, value in (
            ("return_sum", reward),
            ("body_pos_err", (relative - ref_relative).norm(dim=-1).mean(-1)),
            (
                "torso_pos_err",
                (pos[:, TORSO_BODY] - ref[0][:, TORSO_BODY]).norm(dim=-1),
            ),
            (
                "torso_ori_err",
                rotation_angle(body[1][:, TORSO_BODY], ref[1][:, TORSO_BODY]),
            ),
            (
                "free_root_pos_err",
                (state[:, ROOT_POS] - ref_state[:, ROOT_POS]).norm(dim=-1),
            ),
            (
                "free_root_ori_err",
                _quat_geodesic(state[:, ROOT_QUAT], ref_state[:, ROOT_QUAT]),
            ),
        ):
            acc[name] = torch.where(rows, acc[name] + value, acc[name])
        for name, value in (("free_root_xy_drift", xy), ("free_root_yaw_drift", yaw)):
            acc[name] = torch.where(rows, torch.maximum(acc[name], value), acc[name])
        acc["steps"] = torch.where(rows, acc["steps"] + 1, acc["steps"])

    def _count_ends(self, t, rows, result) -> None:
        """A fall on the clip's last frame counts as reaching the clip end."""
        acc = self.acc
        clip_end = rows & result["clip_end"]
        new_fail = rows & result["failure"] & ~result["clip_end"]
        acc["fail"] = acc["fail"] + new_fail.float()
        acc["clip_end"] = acc["clip_end"] + clip_end.float()
        acc["fail_step"] = torch.where(
            new_fail, torch.full_like(acc["fail_step"], float(t)), acc["fail_step"]
        )

    def _push_step(self, t: int, rows: torch.Tensor, result: dict) -> None:
        if not bool(rows.any()):
            return
        if bool((rows & result["clip_end"]).any()):
            raise RuntimeError("the static reference clip ended during push recovery")
        sim = self.sim
        state, ref = sim.state, result["ref_state"]
        if t < self.settle_steps:
            self._record(rows, t, 0, state, sim.reference_state())
            self._stop_before_push(
                rows,
                initialization_reason(
                    state, self.joint_low, self.joint_high, result["failure"]
                ),
            )
            if t == self.settle_steps - 1:
                self._apply_push(rows & self.alive)
            return
        index = t - self.settle_steps
        self._record(rows, self.settle_steps + 1 + index, 2, state, ref)
        nonfinite = ~torch.isfinite(state).all(dim=-1)
        failed = nonfinite | result["failure"]
        stable = is_stable(
            state,
            ref,
            self.origin,
            self.criteria,
            torso_ori_err=torso_orientation_error(state, ref),
        )
        undecided = rows & ~self.decided
        fallen, recovered = self.tracker.step(index, undecided, stable, failed)
        timeout = undecided & ~fallen & ~recovered & (index == self.post_steps - 1)
        for mask, outcome, reason in (
            (fallen & ~nonfinite, "fallen", "fallen"),
            (fallen & nonfinite, "fallen", "nonfinite"),
            (recovered, "recovered", "recovered"),
            (timeout, "timeout", "timeout"),
        ):
            self.outcome = torch.where(mask, OUTCOMES.index(outcome), self.outcome)
            self.reason = torch.where(mask, REASONS.index(reason), self.reason)
        self.recovery_time = torch.where(
            recovered,
            torch.full_like(self.recovery_time, (index + 1) * self.criteria.dt),
            self.recovery_time,
        )
        self.onset_time = torch.where(
            recovered, self.tracker.onset_time(), self.onset_time
        )
        self.decided = self.decided | fallen | recovered | timeout
        # A recorded trajectory runs on after recovery until a fall or the
        # end of the budget; the verdict stays the first one.
        stop = (rows & failed) | timeout
        if not self.save_trajectory:
            stop = stop | recovered
        self.alive = self.alive & ~stop

    def _apply_push(self, rows: torch.Tensor) -> None:
        """Push every listed row once, after its settle steps."""
        ids = rows.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return
        sim = self.sim
        push = self.cfg.push_recovery
        before = sim.state[ids].clone()
        plan = self.row_arg[ids]
        delta = world_push_xy(
            self.push_directions[plan], 1.0, str(push.frame), before[:, ROOT_QUAT]
        ) * self.push_speed_of[plan].unsqueeze(-1)
        sim.apply_horizontal_velocity_push(ids, delta, mode=str(push.push_mode))
        after = sim.state[ids]
        measured = after[:, ROOT_LIN_VEL] - before[:, ROOT_LIN_VEL]
        self.push_values[ids] = torch.cat((delta, measured[:, :2]), dim=-1)
        self._record(rows, self.settle_steps, 1, sim.state, sim.reference_state())
        bad = torch.zeros_like(rows)
        bad[ids] = ~torch.isfinite(after).all(dim=-1)
        self._stop_before_push(
            bad, torch.where(bad, REASONS.index("nonfinite"), self.reason)
        )
        self.origin[ids] = before[:, ROOT_POS][:, :2]
        self.tracker.reset(ids)
        self.outcome = torch.where(rows & ~bad, OUTCOMES.index("timeout"), self.outcome)

    def _record(self, rows, index, phase, state, ref) -> None:
        if not self.save_trajectory:
            return
        frame = torch.cat(
            (
                torch.full_like(state[:, :1], float(phase)),
                state[:, ROOT_POS],
                state[:, ROOT_LIN_VEL],
                root_tilt(state, ref).unsqueeze(-1),
            ),
            dim=-1,
        )
        self.trajectory[:, index] = torch.where(
            rows.unsqueeze(-1), frame, self.trajectory[:, index]
        )

    # ---------------------------------------------------------------- output

    def _observe(self) -> dict[str, torch.Tensor]:
        """``states``: the representation's encoding and the actor goal."""
        sim = self.sim
        full = self.representation.stored_state(sim.state, sim.body_state())
        _, goal = root_goals(
            self.representation.root(full), *self.motion.goal_window(sim.frame())
        )
        return {"states": torch.cat((self.representation.encode(full), goal), dim=-1)}

    def _result_rows(self) -> dict[str, torch.Tensor]:
        acc = self.acc
        rows = {
            "task": self.row_task,
            "job": self.row_job,
            "valid": (self.row_task != _PADDING).long(),
        }
        if TRACKING in self.tasks:
            scheduled = self.row_length.float().clamp_min(1.0)
            steps = acc["steps"].clamp_min(1.0)
            rows.update(
                {
                    "tracking.clip_id": self.row_arg,
                    "tracking.normalized_return": acc["return_sum"]
                    / (MAX_TRACKING_REWARD * scheduled),
                    **{
                        f"tracking.{name}": acc[name] / steps
                        for name in (
                            "body_pos_err",
                            "torso_pos_err",
                            "torso_ori_err",
                            "free_root_pos_err",
                            "free_root_ori_err",
                        )
                    },
                    "tracking.free_root_xy_drift": acc["free_root_xy_drift"],
                    "tracking.free_root_yaw_drift": acc["free_root_yaw_drift"],
                    "tracking.fail": acc["fail"].long(),
                    "tracking.clip_end": acc["clip_end"].long(),
                    "tracking.fail_step": acc["fail_step"].long(),
                    "tracking.fail_time_s": (acc["fail_step"] + 1.0) * DT * acc["fail"],
                    "tracking.episode_length_s": acc["steps"] * DT,
                    "tracking.steps": acc["steps"].long(),
                    "tracking.return_sum": acc["return_sum"],
                    "tracking.scheduled_steps": scheduled.long(),
                }
            )
        if PUSH_RECOVERY in self.tasks:
            plan = self.row_arg
            rows.update(
                {
                    "push_recovery.trial": plan,
                    "push_recovery.speed_m_s": self.push_speed_of[plan],
                    "push_recovery.direction_x": self.push_directions[plan, 0],
                    "push_recovery.direction_y": self.push_directions[plan, 1],
                    "push_recovery.outcome": self.outcome,
                    "push_recovery.reason": self.reason,
                    "push_recovery.recovery_time_s": self.recovery_time,
                    "push_recovery.stable_onset_time_s": self.onset_time,
                    "push_recovery.world_push_x": self.push_values[:, 0],
                    "push_recovery.world_push_y": self.push_values[:, 1],
                    "push_recovery.measured_dvx": self.push_values[:, 2],
                    "push_recovery.measured_dvy": self.push_values[:, 3],
                }
            )
            if self.save_trajectory:
                rows["push_recovery.trajectory"] = self.trajectory
        return {key: value.detach().clone() for key, value in rows.items()}
