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

Evaluation is a table of jobs: one per reference clip (motion tracking), per
smoke-test command, per Appendix G.2 trial and command, or per push trial.
The jobs are dealt round-robin to the env ranks, and every rollout epoch
starts the next ``num_envs`` jobs of this rank together; after
``rollout_epoch`` epochs the next evaluation starts from the first jobs
again. Rows run without
sensor or action noise until their job ends. The vector reports done only
when every row has ended, and then returns one result row per env in
``infos["episode"]``. Rows past this rank's last job are padding with
``valid = 0``.

Supported task sets: ``tracking`` and ``velocity_smoke`` (alone or
together), ``velocity_g2`` alone, ``push_recovery`` alone. The G.2 test
draws the sim context from the training distribution; push recovery holds a
static standing reference.
"""

from __future__ import annotations

import contextlib

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    DT,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    MAX_TRACKING_REWARD,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
)
from rlinf.envs.sim.g1_mimic.eval_metrics import (
    DEFAULT_SMOKE_COMMANDS,
    EVAL_TASKS,
    PUSH_RECOVERY,
    TRACKING,
    VELOCITY_G2,
    VELOCITY_SMOKE,
)
from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits, set_kinematics_compile
from rlinf.envs.sim.g1_mimic.g1_mimic_env import (
    build_motion_library,
    env_device,
    make_goal,
)
from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env
from rlinf.envs.sim.g1_mimic.obs import RollingHistory, analytic_obs
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
    TORSO,
    analytic_reward,
    failure_signals,
    forward_kinematics_pair,
    xy_yaw_drift,
)
from rlinf.envs.sim.g1_mimic.velocity import (
    heading_velocity,
    normalize_command,
    step_test_commands,
    velocity_fall,
)
from rlinf.scheduler import Worker

_TASK_SETS = (
    {TRACKING},
    {VELOCITY_SMOKE},
    {TRACKING, VELOCITY_SMOKE},
    {VELOCITY_G2},
    {PUSH_RECOVERY},
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


def checked_eval_states(
    next_state: torch.Tensor, ref_state: torch.Tensor, running: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject invalid active states and keep the other rows finite."""
    invalid = running & (
        ~torch.isfinite(next_state).all(dim=-1) | ~torch.isfinite(ref_state).all(dim=-1)
    )
    if bool(invalid.any()):
        rows = invalid.nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"non-finite evaluation states in active rows {rows}")
    next_state = next_state.clone()
    ref_state = ref_state.clone()
    inactive = ~running
    if bool(inactive.any()):
        replacement = ref_state[inactive]
        bad = ~torch.isfinite(replacement).all(dim=-1)
        if bool(bad.any()):
            replacement = replacement.clone()
            replacement[bad] = 0
            replacement[bad, 3] = 1
        finite_next = next_state[inactive]
        use_ref = ~torch.isfinite(finite_next).all(dim=-1)
        if bool(use_ref.any()):
            finite_next = finite_next.clone()
            finite_next[use_ref] = replacement[use_ref]
        next_state[inactive] = finite_next
        ref_state[inactive] = replacement
    return next_state, ref_state


def _rot_angle(rot: torch.Tensor, ref_rot: torch.Tensor) -> torch.Tensor:
    trace = (ref_rot * rot).sum(dim=(-2, -1))
    return torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))


def _quat_geodesic(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    dot = (q * ref).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _seconds_to_steps(seconds: float) -> int:
    return int(round(float(seconds) / DT))


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
        if set(self.tasks) not in _TASK_SETS or len(set(self.tasks)) != len(self.tasks):
            raise ValueError(
                "env.eval.eval_tasks must be tracking and/or velocity_smoke, "
                f"velocity_g2 alone, or push_recovery alone; got {self.tasks}"
            )
        self.velocity = bool(cfg.velocity_tracking)
        if not self.velocity and (
            VELOCITY_SMOKE in self.tasks or VELOCITY_G2 in self.tasks
        ):
            raise ValueError(
                "velocity evaluation needs a policy trained with velocity_tracking=true"
            )
        self.obs_dim = int(cfg.obs_dim)
        self.include_ref_ori = self.obs_dim == 99
        self.goal_encoding = str(cfg.goal_encoding)
        self.goal_anchor = bool(cfg.goal_anchor)
        self.reward_anchor = str(cfg.reward_anchor)
        self.termination_body = str(cfg.termination_body)
        self._chunk = 0
        self._t = 0

        on_accelerator = self.device.type != "cpu"
        self._rng_devices = [self.device] if on_accelerator else []
        self._rng_device_type = self.device.type if on_accelerator else None
        with self._fork_rng():
            torch.manual_seed(self.seed + self.seed_offset)
            self._save_rng()
        with self._own_rng():
            self._build_sim()
        joint_low, joint_high = asset_joint_limits()
        self.joint_low = joint_low.to(self.device)
        self.joint_high = joint_high.to(self.device)
        self.q0 = self.motion.q0.to(self.device)
        self._build_jobs()
        self._allocate_rows()

    # ----------------------------------------------------------------- setup

    def _fork_rng(self):
        return torch.random.fork_rng(
            devices=self._rng_devices, device_type=self._rng_device_type
        )

    def _save_rng(self) -> None:
        self._cpu_rng = torch.get_rng_state()
        self._device_rng = (
            Worker.torch_platform.get_rng_state(self.device)
            if self._rng_devices
            else None
        )

    @contextlib.contextmanager
    def _own_rng(self):
        """Draw from this env's generator, leaving the process generator as it was.

        The training env shares the process; evaluation must not shift its
        random stream.
        """
        with self._fork_rng():
            torch.set_rng_state(self._cpu_rng)
            if self._device_rng is not None:
                Worker.torch_platform.set_rng_state(self._device_rng, self.device)
            yield
            self._save_rng()

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
            eval_mode = True
        else:
            self.motion = build_motion_library(cfg, self.device)
            eval_mode = VELOCITY_G2 not in self.tasks
        self.sim = MjlabG1Env(
            num_envs=self.num_envs,
            motion=self.motion,
            device=self.device,
            max_episode_seconds=1.0e6,
            termination_body=self.termination_body,
            eval_mode=eval_mode,
            com_sampling="per_reset",
        )

    def _build_jobs(self) -> None:
        """Global job table, then this rank's share of it."""
        task_ids, args, trials, lengths = [], [], [], []

        def add(task, arg, trial, length):
            task_ids.append(EVAL_TASKS.index(task))
            args.append(int(arg))
            trials.append(int(trial))
            lengths.append(int(length))

        cfg = self.cfg
        if TRACKING in self.tasks:
            clip_lengths = self.motion.clip_length(
                torch.arange(self.motion.num_clips, device=self.device)
            ).tolist()
            for clip, length in enumerate(clip_lengths):
                add(TRACKING, clip, 0, length)
        if VELOCITY_SMOKE in self.tasks:
            smoke = cfg.velocity_smoke
            commands = smoke.get("commands", None)
            commands = (
                OmegaConf.to_container(commands, resolve=True)
                if commands
                else DEFAULT_SMOKE_COMMANDS
            )
            self.smoke_names = list(commands)
            self.smoke_commands = torch.tensor(
                [[float(x) for x in commands[name]] for name in self.smoke_names],
                device=self.device,
            )
            self.smoke_start_clip = int(smoke.start_clip)
            steps = int(round(float(smoke.seconds) / 0.02))
            for index in range(len(self.smoke_names)):
                add(VELOCITY_SMOKE, index, 0, steps)
        if VELOCITY_G2 in self.tasks:
            g2 = cfg.velocity_g2
            table, self.g2_names, self.g2_planes = step_test_commands()
            self.g2_commands = table.to(self.device)
            self.g2_trials = int(g2.trials)
            steps = _seconds_to_steps(g2.seconds)
            self.g2_steady_steps = _seconds_to_steps(g2.steady_seconds)
            if not 0 < self.g2_steady_steps <= steps:
                raise ValueError(
                    "velocity_g2.steady_seconds must be positive and at most "
                    f"velocity_g2.seconds, got {g2.steady_seconds} and {g2.seconds}"
                )
            for trial in range(self.g2_trials):
                for index in range(len(self.g2_names)):
                    add(VELOCITY_G2, index, trial, steps)
        if PUSH_RECOVERY in self.tasks:
            push = cfg.push_recovery
            self.push_plan = trial_plan(
                self.push_speeds, int(push.trials), push.direction, self.seed
            )
            for item in self.push_plan:
                add(
                    PUSH_RECOVERY,
                    item["trial"],
                    item["trial"],
                    self.settle_steps + self.post_steps,
                )
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
        self.job_trial = torch.tensor(trials[local], dtype=torch.long)
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
        self.row_trial = torch.zeros_like(self.row_task)
        self.row_length = torch.zeros_like(self.row_task)
        self.alive = zeros(dtype=torch.bool)
        self.prev = zeros(ACTION_DIM)
        self.command = zeros(3)
        self.acc = {
            name: zeros()
            for name in (
                "return_sum",
                "body_tracking",
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
                "lin_vel_err",
                "yaw_rate_err",
                "count",
                "fall",
                "fall_step",
                "fell",
            )
        }
        self.error_sum = zeros(3)
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
        with self._own_rng():
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
        with self._own_rng():
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
        if VELOCITY_SMOKE in self.tasks:
            meta["smoke_names"] = list(self.smoke_names)
        if VELOCITY_G2 in self.tasks:
            meta.update(
                g2_names=list(self.g2_names),
                g2_planes=list(self.g2_planes),
                g2_trials=self.g2_trials,
                g2_commands=self.g2_commands.cpu().tolist(),
            )
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
        self.row_trial.zero_()
        self.row_length.zero_()
        self.row_task[:count] = self.job_task[take].to(self.device)
        self.row_job[:count] = self.job_id[take].to(self.device)
        self.row_arg[:count] = self.job_arg[take].to(self.device)
        self.row_trial[:count] = self.job_trial[take].to(self.device)
        self.row_length[:count] = self.job_length[take].to(self.device)
        self.alive = self.row_task != _PADDING
        self.prev.zero_()
        self.command.zero_()
        for value in self.acc.values():
            value.zero_()
        self.error_sum.zero_()

        ids = torch.arange(n, device=self.device)
        if VELOCITY_G2 in self.tasks:
            # The step test reseeds once per vector of trials; with one rank
            # the seeds follow the job chunks.
            torch.manual_seed(
                self.seed + self._chunk * self.total_num_processes + self.seed_offset
            )
            self.sim.reset_standing(ids)
            g2 = self._is(VELOCITY_G2)
            self.command[g2] = self.g2_commands[self.row_arg[g2]]
        else:
            clip = torch.zeros_like(ids)
            tracking = self._is(TRACKING)
            clip[tracking] = self.row_arg[tracking]
            if VELOCITY_SMOKE in self.tasks:
                smoke = self._is(VELOCITY_SMOKE)
                clip[smoke] = self.smoke_start_clip
                self.command[smoke] = self.smoke_commands[self.row_arg[smoke]]
            self.sim.reset(ids, clip, torch.zeros_like(ids))

        sim = self.sim
        state, ref = checked_eval_states(sim.state, sim.reference_state(), self.alive)
        self.history = RollingHistory(
            analytic_obs(state, self.prev, sim.xi, self.q0, ref_quat=self._quat(ref))
        )
        if PUSH_RECOVERY in self.tasks:
            self._start_push_trials()

    def _quat(self, ref: torch.Tensor) -> torch.Tensor | None:
        return ref[..., ROOT_QUAT] if self.include_ref_ori else None

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
        _, _, failed = failure_signals(
            sim.state, sim.reference_state(), termination_body=self.termination_body
        )
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
        end_kind = result["end_kind"]
        next_raw, ref_raw = result["next_state"], result["ref_state"]
        is_push = self._is(PUSH_RECOVERY)
        next_state, ref_state = checked_eval_states(
            next_raw, ref_raw, running & ~is_push
        )
        if TRACKING in self.tasks:
            rows = running & self._is(TRACKING)
            self._track(rows, action, next_state, ref_state)
            self._count_ends(t, rows, end_kind)
            self.alive = torch.where(rows, end_kind == EPISODE_CONT, self.alive)
        if VELOCITY_SMOKE in self.tasks or VELOCITY_G2 in self.tasks:
            self._follow_command(t, running, next_state)
        self.prev = torch.where(running.unsqueeze(-1), action, self.prev)

        sim = self.sim
        fresh = analytic_obs(
            sim.state, self.prev, sim.xi, self.q0, ref_quat=self._quat(ref_raw)
        )
        # Tracking and push rows keep the frame of the step they ran; velocity
        # rows stop updating once they fall.
        velocity_rows = self._is(VELOCITY_SMOKE) | self._is(VELOCITY_G2)
        rows = torch.where(velocity_rows, self.alive, running)
        use = rows.unsqueeze(-1) & torch.isfinite(fresh)
        self.history.push(torch.where(use, fresh, self.history.last()))

        if PUSH_RECOVERY in self.tasks:
            self._push_step(t, running & is_push, result)
        self.alive = self.alive & (t + 1 < self.row_length)

    def _track(self, rows, action, next_state, ref_state) -> None:
        state_fk, ref_fk = forward_kinematics_pair(next_state, ref_state)
        pack = analytic_reward(
            next_state,
            ref_state,
            action,
            self.prev,
            self.joint_low,
            self.joint_high,
            state_fk=state_fk,
            ref_fk=ref_fk,
            reward_anchor=self.reward_anchor,
        )
        xy, yaw = xy_yaw_drift(next_state, ref_state)
        body_pos, body_rot, _, _ = state_fk
        ref_body_pos, ref_body_rot, _, _ = ref_fk
        acc = self.acc
        for name, value in (
            ("return_sum", pack["reward"]),
            ("body_tracking", pack["body_tracking"]),
            (
                "torso_pos_err",
                (body_pos[:, TORSO] - ref_body_pos[:, TORSO]).norm(dim=-1),
            ),
            ("torso_ori_err", _rot_angle(body_rot[:, TORSO], ref_body_rot[:, TORSO])),
            (
                "free_root_pos_err",
                (next_state[..., :3] - ref_state[..., :3]).norm(dim=-1),
            ),
            (
                "free_root_ori_err",
                _quat_geodesic(next_state[..., 3:7], ref_state[..., 3:7]),
            ),
        ):
            acc[name] = torch.where(rows, acc[name] + value, acc[name])
        for name, value in (("free_root_xy_drift", xy), ("free_root_yaw_drift", yaw)):
            acc[name] = torch.where(rows, torch.maximum(acc[name], value), acc[name])
        acc["steps"] = torch.where(rows, acc["steps"] + 1, acc["steps"])

    def _count_ends(self, t, rows, end_kind) -> None:
        acc = self.acc
        new_fail = rows & (end_kind == EPISODE_FAILURE)
        acc["fail"] = acc["fail"] + new_fail.float()
        acc["clip_end"] = acc["clip_end"] + (rows & (end_kind == EPISODE_CLIP_END))
        acc["fail_step"] = torch.where(
            new_fail, torch.full_like(acc["fail_step"], float(t)), acc["fail_step"]
        )

    def _follow_command(self, t, running, next_state) -> None:
        acc = self.acc
        planar, yaw_rate = heading_velocity(next_state)
        error = torch.cat(
            (
                planar - self.command[:, :2],
                (yaw_rate - self.command[:, 2]).unsqueeze(-1),
            ),
            dim=-1,
        )
        fallen = velocity_fall(next_state) | ~torch.isfinite(next_state).all(dim=-1)

        smoke = running & self._is(VELOCITY_SMOKE)
        acc["lin_vel_err"] = torch.where(
            smoke, acc["lin_vel_err"] + error[:, :2].norm(dim=-1), acc["lin_vel_err"]
        )
        acc["yaw_rate_err"] = torch.where(
            smoke, acc["yaw_rate_err"] + error[:, 2].abs(), acc["yaw_rate_err"]
        )
        acc["count"] = torch.where(smoke, acc["count"] + 1, acc["count"])
        new_fall = smoke & fallen
        acc["fall"] = acc["fall"] + new_fall.float()
        acc["fall_step"] = torch.where(
            new_fall, torch.full_like(acc["fall_step"], float(t)), acc["fall_step"]
        )

        g2 = running & self._is(VELOCITY_G2)
        if VELOCITY_G2 in self.tasks:
            steady = g2 & (t >= self.row_length - self.g2_steady_steps)
            self.error_sum = torch.where(
                steady.unsqueeze(-1), self.error_sum + error, self.error_sum
            )
            acc["fell"] = torch.where(g2 & fallen, 1.0, acc["fell"])
        self.alive = self.alive & ~((smoke | g2) & fallen)

    def _push_step(self, t: int, rows: torch.Tensor, result: dict) -> None:
        if not bool(rows.any()):
            return
        if bool((rows & (result["end_kind"] == EPISODE_CLIP_END)).any()):
            raise RuntimeError("the static reference clip ended during push recovery")
        sim = self.sim
        state, ref = result["next_state"], result["ref_state"]
        if t < self.settle_steps:
            self._record(rows, t, 0, sim.state, sim.reference_state())
            failed = result["motion_failure"] | (result["end_kind"] == EPISODE_FAILURE)
            self._stop_before_push(
                rows,
                initialization_reason(
                    sim.state, self.joint_low, self.joint_high, failed
                ),
            )
            if t == self.settle_steps - 1:
                self._apply_push(rows & self.alive)
            return
        index = t - self.settle_steps
        self._record(rows, self.settle_steps + 1 + index, 2, state, ref)
        nonfinite = ~torch.isfinite(state).all(dim=-1)
        failed = nonfinite | result["motion_failure"]
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
        """``states``: history, goal and, with velocity tracking, command and task.

        Velocity rows carry task 1 and read the normalized command; the other
        rows read the goal.
        """
        sim = self.sim
        goal = make_goal(
            self.motion,
            self.goal_encoding,
            self.goal_anchor,
            sim.motion_id,
            sim.motion_time,
            sim.state,
        )
        parts = [self.history.flat(), goal]
        if self.velocity:
            velocity_rows = self._is(VELOCITY_SMOKE) | self._is(VELOCITY_G2)
            parts += [
                normalize_command(self.command),
                velocity_rows.float().unsqueeze(-1),
            ]
        return {"states": torch.cat(parts, dim=-1)}

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
                    "tracking.body_tracking": acc["body_tracking"],
                    **{
                        f"tracking.{name}": acc[name] / steps
                        for name in (
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
        if VELOCITY_SMOKE in self.tasks:
            count = acc["count"].clamp_min(1.0)
            rows.update(
                {
                    "velocity_smoke.command_index": self.row_arg,
                    "velocity_smoke.command": self.command,
                    "velocity_smoke.lin_vel_err": acc["lin_vel_err"] / count,
                    "velocity_smoke.yaw_rate_err": acc["yaw_rate_err"] / count,
                    "velocity_smoke.fall": acc["fall"].long(),
                    "velocity_smoke.fall_time_s": (acc["fall_step"] + 1.0)
                    * DT
                    * acc["fall"],
                    "velocity_smoke.survival_s": acc["count"] * DT,
                    "velocity_smoke.scheduled_s": self.row_length.float() * DT,
                }
            )
        if VELOCITY_G2 in self.tasks:
            relative = (self.error_sum / float(self.g2_steady_steps)).norm(
                dim=-1
            ) / self.command.norm(dim=-1)
            fell = acc["fell"] > 0
            rows.update(
                {
                    "velocity_g2.trial": self.row_trial,
                    "velocity_g2.command_index": self.row_arg,
                    "velocity_g2.relative_error": torch.where(
                        fell, torch.full_like(relative, float("nan")), relative
                    ),
                    "velocity_g2.fell": fell.long(),
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
