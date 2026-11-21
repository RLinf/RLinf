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
import concurrent.futures
import copy
import os
import queue
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.schema.embodied_types import (
    EnvOutput,
    EnvPart,
    EnvTransition,
    PolicyInput,
    TrajectoryKey,
    TrajectorySource,
)
from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.layout import EnvAssignment, EnvLayout
from rlinf.envs.policy_obs import PolicyObsSpec
from rlinf.envs.utils import get_env_attr, to_tensor
from rlinf.envs.wrappers import InsertDelay, RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.env_helpers import HistoryManager, SmoothInterveneController
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    update_nested_cfg,
)
from rlinf.utils.obs_compression import compress_obs, is_compression_enabled
from rlinf.utils.placement import HybridComponentPlacement


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    getter = getattr(obj, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(obj, key, default)


class EnvWorker(Worker):
    # Class-level default so the observation send path is safe even when the
    # instance is built without running ``__init__`` (e.g. ``object.__new__`` in
    # unit tests). ``None`` means "use the scheduler's default split"; when
    # ``env.obs_compression`` is enabled, ``__init__`` installs the compressing
    # ``split_fn`` (see ``_split_and_compress_obs``).
    _obs_split_fn = None
    _policy_input_split_fn = None
    # Same reason; ``__init__`` installs the specs ``env.<mode>.policy_obs`` gives.
    _train_policy_obs_spec = PolicyObsSpec()
    _eval_policy_obs_spec = PolicyObsSpec()
    collect_final_values = True
    enable_rlt = False
    discard_episodes = False
    _release_at_episode_end = False

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self._prefetched_train_bootstrap: list[EnvOutput] | None = None
        self._trajectory_step = 0
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.collect_final_values = self.cfg.rollout.get("collect_final_values", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        self.enable_rlt = OmegaConf.select(
            self.cfg, "algorithm.loss_type", default=""
        ) in {"rlt_ac", "rlt_td3"}
        self.use_training_pipeline = self.cfg.runner.get("use_training_pipeline", False)
        # Optional lossless compression of image observations before they are
        # sent to the rollout workers. Disabled unless `env.obs_compression`
        # is present and `enable: true`. Compression runs inside a custom
        # `split_fn` (see `_split_and_compress_obs`) so it happens *after* the
        # channel splits the batch across ranks, keeping the scheduler's
        # batch-size inference and splitting operating on plain tensors.
        self.obs_compression_cfg = OmegaConf.select(
            self.cfg, "env.obs_compression", default=None
        )
        if is_compression_enabled(self.obs_compression_cfg):
            self._obs_split_fn = self._split_and_compress_obs
            self._policy_input_split_fn = self._split_and_compress_policy_input
        else:
            self._obs_split_fn = None
            self._policy_input_split_fn = PolicyInput.split

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self.history_reward_assign = self.cfg.get("reward", {}).get(
            "history_reward_assign", False
        )
        self.use_reward_model = self.cfg.get("reward", {}).get(
            "use_reward_model", False
        )
        self.use_realworld_reward = self.cfg.get("reward", {}).get(
            "standalone_realworld", False
        )
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info", "_final_info")
        if self.use_external_reward_model:
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)

        # Env configurations
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.model_cfg = (
            self.cfg.rollout.model if self.only_eval else self.cfg.actor.model
        )
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            self.cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        # The assignment holds the env this rank runs, including its loop.
        self.train_assignment: EnvAssignment | None = (
            EnvLayout(train_env_cfg, self._world_size, "env.train").assign(self._rank)
            if self.enable_train
            else None
        )
        self.eval_assignment: EnvAssignment | None = (
            EnvLayout(eval_env_cfg, self._world_size, "env.eval").assign(self._rank)
            if self.enable_eval
            else None
        )
        self.rollout_epoch = (
            self.train_assignment.cfg.rollout_epoch if self.enable_train else 1
        )
        self.eval_rollout_epoch = (
            self.eval_assignment.cfg.rollout_epoch if self.enable_eval else 1
        )
        self._train_policy_obs_spec = PolicyObsSpec.from_cfg(train_env_cfg)
        self._eval_policy_obs_spec = PolicyObsSpec.from_cfg(eval_env_cfg)
        self.train_enable_offload = (
            self.train_assignment.cfg.get("enable_offload", False)
            if self.train_assignment is not None
            else False
        )
        self.eval_enable_offload = (
            self.eval_assignment.cfg.get("enable_offload", False)
            if self.eval_assignment is not None
            else False
        )
        if self.enable_train:
            self.enable_online_lerobot = bool(
                OmegaConf.select(
                    self.cfg,
                    "algorithm.dagger.online_lerobot.enabled",
                    default=False,
                )
            )
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
            self.train_batch_size = self.cfg.env.train.total_num_envs // self.stage_num
        else:
            self.enable_online_lerobot = False
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
            self.eval_batch_size = self.cfg.env.eval.total_num_envs // self.stage_num
        self.n_train_chunk_steps = 0
        if self.enable_train:
            self.n_train_chunk_steps = (
                self.train_assignment.cfg.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                self.eval_assignment.cfg.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        if self.enable_train:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)

        # env.train.discard_episodes (Tianji HIL "丢弃本条"): an episode's
        # chunks are held until it finishes so discarded episodes never enter
        # the actor replay buffer or updates.
        self.discard_episodes = bool(
            self.enable_train and _cfg_get(train_env_cfg, "discard_episodes", False)
        )
        if self.discard_episodes:
            assert self.train_num_envs_per_stage == 1 and not self.use_training_pipeline, (
                "env.train.discard_episodes needs one env per stage and no training pipeline"
            )
            self._episode_serial = [0] * self.stage_num
            self._discarded_serials = [set() for _ in range(self.stage_num)]

        self.smooth_intervene = SmoothInterveneController.from_cfg(
            self.cfg,
            stage_num=self.stage_num,
            enable_train=self.enable_train,
            train_num_envs_per_stage=(
                self.train_num_envs_per_stage if self.enable_train else 0
            ),
        )
        if self.smooth_intervene.enabled and self.env_decoupled_mode:
            raise ValueError("smooth_intervene does not support decoupled env mode")
        # Actions a stage sent with its request, run without waiting for a reply.
        self._supplied_actions: dict[int, torch.Tensor] = {}

        if self.env_decoupled_mode:
            # Init the batch_router for env decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {}
            assert self._component_placement.get_world_size(
                "env"
            ) >= self._component_placement.get_world_size("rollout"), (
                "the world size of env must be greater than the world size of rollout in env_decoupled_mode"
            )

    def set_global_step(self, global_step: int) -> None:
        """Set the trajectory step used to correlate distributed events."""
        self._trajectory_step = global_step

    def init_worker(self):
        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        if self.enable_train:
            self.env_list = self._setup_env_and_wrappers(
                self.train_assignment,
                num_envs_per_stage=self.train_num_envs_per_stage,
                insert_delay=True,
            )
            if self.train_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.env_list
                ), "train envs must have an offload method to enable offload!"
            self._release_at_episode_end = self._defer_episode_resets()

        if self.enable_eval:
            self.eval_env_list = self._setup_env_and_wrappers(
                self.eval_assignment,
                num_envs_per_stage=self.eval_num_envs_per_stage,
                insert_delay=False,
            )
            if self.eval_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.eval_env_list
                ), "eval envs must have an offload method to enable offload!"

        if self.enable_train:
            if self.reward_mode == "history_buffer":
                self.train_history_managers = [
                    HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                    for _ in range(self.stage_num)
                ]
                self.history_lengths = [{} for _ in range(self.stage_num)]

        self._init_env()

    def update_env_cfg(self):
        for assignment in (self.train_assignment, self.eval_assignment):
            if assignment is None:
                continue
            self._apply_rank_override_cfg(assignment)
            self._inject_realworld_reward_cfg(assignment.cfg)

    @staticmethod
    def _apply_rank_override_cfg(assignment: EnvAssignment) -> None:
        """Merge this rank's ``override_cfgs`` entry into the env's ``override_cfg``.

        ``override_cfgs`` lists one entry per worker hosting the env, indexed by
        the worker's rank among those hosts.
        """
        env_cfg = assignment.cfg
        override_cfgs = env_cfg.get("override_cfgs", None)
        if override_cfgs is None:
            return
        assert len(override_cfgs) > assignment.rank, (
            f"override_cfgs lists {len(override_cfgs)} entries, but the env is "
            f"hosted by {assignment.world_size} workers and this is worker "
            f"{assignment.rank}."
        )
        shared_override_cfg = env_cfg.get("override_cfg", None)
        base_cfg = update_nested_cfg(
            {},
            OmegaConf.to_container(shared_override_cfg, resolve=True)
            if shared_override_cfg is not None
            else {},
        )
        base_cfg = update_nested_cfg(
            base_cfg,
            OmegaConf.to_container(override_cfgs[assignment.rank], resolve=True).copy(),
        )
        setattr(env_cfg, "override_cfg", OmegaConf.create(base_cfg))

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if SupportedEnvType(env_cfg.env_type) is not SupportedEnvType.REAL:
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(
        self,
        assignment: EnvAssignment,
        num_envs_per_stage: int,
        insert_delay: bool,
    ):
        env_cfg = assignment.cfg
        env_cls = get_env_cls(env_cfg.env_type, env_cfg)
        env_list = []

        for stage_id in range(self.stage_num):
            # Seeds and reset-state shards are split among the workers hosting
            # this env, so it behaves as it would running alone on them.
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=assignment.rank * self.stage_num + stage_id,
                total_num_processes=assignment.world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if insert_delay and self.cfg.env.get("delay_sampler", None):
                env = InsertDelay(env, self.cfg.env.delay_sampler)
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle_episode"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                    resume=bool(getattr(env_cfg.data_collection, "resume", False)),
                    streaming=getattr(env_cfg.data_collection, "streaming", False),
                )
            env_list.append(env)
        return env_list

    def _init_env(self):
        for i in range(self.stage_num):
            if self.enable_train:
                if self.train_assignment.cfg.auto_reset:
                    extracted_obs, _ = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    self.last_intervened_info_list.append((None, None))
                if self.train_enable_offload and self.train_assignment.cfg.get(
                    "enable_init_offload", True
                ):
                    get_env_attr(self.env_list[i], "offload")()
            if self.enable_eval:
                if self.eval_enable_offload:
                    get_env_attr(self.eval_env_list[i], "offload")()

    async def _maybe_wait_env_delay(self, stage_id: int) -> None:
        """Wait out the delay ``InsertDelay`` sampled for this stage, if it is on.

        The wrapper only samples the delay; waiting here keeps the emulated sensor
        latency off the event loop so co-scheduled coroutines keep running.
        """
        env = self.env_list[stage_id]
        if get_env_attr(env, "wait_delay") is None:
            return
        await env.wait_delay()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self,
        chunk_actions: torch.Tensor,
        stage_id: int,
        on_step=None,
        on_step_at=None,
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
        """Execute a chunk and build environment output and LeRobot payload.

        ``obs_list`` in the payload stays post-action. The caller snapshots the
        observation from before this chunk for LeRobot action pairing.
        """
        t0 = time.time()
        prev_end = getattr(self, "_chunk_end_t", None)
        if prev_end is not None:
            self._chunk_gaps = getattr(self, "_chunk_gaps", []) + [t0 - prev_end]

        exec_actions = prepare_actions(
            raw_chunk_actions=chunk_actions["raw_actions"]
            if isinstance(chunk_actions, dict)
            else chunk_actions,
            env_type=self.train_assignment.cfg.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.train_assignment.cfg.get("wm_env_type", None),
            env_cfg=self.train_assignment.cfg,
        )
        if isinstance(chunk_actions, dict):
            chunk_actions["actions"] = exec_actions
        else:
            chunk_actions = exec_actions

        env = self.env_list[stage_id]
        if on_step is not None or on_step_at is not None:
            obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
                env.chunk_step(chunk_actions, on_step=on_step, on_step_at=on_step_at)
            )
        else:
            obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
                env.chunk_step(chunk_actions)
            )

        env_output, env_info, chunk_step_payload = self._chunk_env_output(
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
            exec_actions=exec_actions,
        )
        if getattr(self, "_release_at_episode_end", False):
            pending_fn = get_env_attr(env, "auto_reset_pending")
            if callable(pending_fn) and pending_fn():
                # Episode ended in this chunk; _complete_episode_reset rebuilds
                # this output once the finished episode is handed off and reset finishes.
                if getattr(self, "_pending_chunks", None) is None:
                    self._pending_chunks = {}
                self._pending_chunks[stage_id] = (
                    obs_list,
                    chunk_rewards,
                    chunk_terminations,
                    chunk_truncations,
                    infos_list,
                    exec_actions,
                )

        if getattr(self, "enable_rlt", False) or on_step is not None or on_step_at is not None:
            self._chunk_end_t = time.time()
            self._chunk_exec = getattr(self, "_chunk_exec", []) + [self._chunk_end_t - t0]
            if len(self._chunk_exec) >= 30:
                gaps = np.asarray(getattr(self, "_chunk_gaps", []) or [0.0])
                exec_arr = np.asarray(self._chunk_exec)
                freq = float(
                    OmegaConf.select(
                        getattr(self, "cfg", None),
                        "env.train.override_cfg.step_frequency",
                        default=None,
                    )
                    or _cfg_get(self.train_assignment.cfg, "step_frequency", 30)
                )
                period = 1.0 / freq
                n_steps = (
                    int(exec_actions.shape[1])
                    if isinstance(exec_actions, torch.Tensor) and exec_actions.ndim >= 2
                    else int(self.model_cfg.num_action_chunks)
                )
                self.log_info(
                    "[ChunkTiming] 执行 %.3fs/块(理论 %.3f) 块间空档 均值 %.3fs 最大 %.3fs 超过一周期 %d/%d 块 → 步频 %.1f Hz"
                    % (
                        exec_arr.mean(),
                        n_steps * period,
                        gaps.mean(),
                        gaps.max(),
                        int((gaps > period).sum()),
                        len(gaps),
                        n_steps / (exec_arr.mean() + gaps.mean()),
                    )
                )
                self._chunk_exec = []
                self._chunk_gaps = []

        return env_output, env_info, chunk_step_payload

    def _chunk_env_output(
        self,
        obs_list,
        chunk_rewards,
        chunk_terminations,
        chunk_truncations,
        infos_list,
        *,
        exec_actions: torch.Tensor | None = None,
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
        """Build EnvOutput, episode metrics, and LeRobot payload from one chunk_step result."""
        env_info: dict[str, Any] = {}
        extracted_obs = obs_list
        infos = infos_list
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        infos = infos or {}
        valid_action_mask = None
        if self.enable_online_lerobot and isinstance(infos, dict):
            # The train env publishes this after auto-reset. Removing it keeps
            # the mask out of the stored step metadata.
            valid_action_mask = infos.pop("valid_action_mask", None)
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.train_assignment.cfg.auto_reset:
            if self.train_assignment.cfg.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info.get("episode", {}):
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )
        if self.train_assignment.cfg.auto_reset and chunk_dones.any():
            final_info = infos.get("final_info", infos)
            if isinstance(final_info, dict) and "intervene_action" in final_info:
                intervene_actions = final_info["intervene_action"]
                intervene_flags = final_info["intervene_flag"]

        iv_kind = infos["iv_kind"] if "iv_kind" in infos else None
        recorded_exec_actions = infos["exec_action"] if "exec_action" in infos else None
        if (
            self.train_assignment.cfg.auto_reset
            and chunk_dones.any()
            and "final_info" in infos
            and isinstance(infos["final_info"], dict)
        ):
            if (
                "iv_kind" in infos["final_info"]
                and "exec_action" in infos["final_info"]
            ):
                iv_kind = infos["final_info"]["iv_kind"]
                recorded_exec_actions = infos["final_info"]["exec_action"]
        if isinstance(iv_kind, torch.Tensor) and iv_kind.dim() != 2:
            iv_kind, recorded_exec_actions = None, None
        action_info = (
            infos.get("final_info", infos)
            if self.train_assignment.cfg.auto_reset and chunk_dones.any()
            else infos
        )
        if not isinstance(action_info, dict):
            action_info = {}

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            env_infos=infos if isinstance(infos, dict) else None,
            transition=EnvTransition(
                rewards=chunk_rewards,
                dones=chunk_dones,
                terminations=chunk_terminations,
                truncations=chunk_truncations,
                intervene_actions=intervene_actions,
                intervene_flags=intervene_flags,
                rlt_switch_flags=rlt_switch_flags,
                executed_mask=infos.get("executed_mask"),
                iv_kind=iv_kind,
                exec_actions=recorded_exec_actions,
                exec_action_available=action_info.get("exec_action_available"),
                intervene_action_available=action_info.get("intervene_action_available"),
                transition_padding=action_info.get("transition_padding"),
                exec_actions_joint=(
                    action_info.get("exec_action_joint")
                    if recorded_exec_actions is not None
                    else None
                ),
                intervene_actions_tcp=(
                    action_info.get("intervene_action_tcp")
                    if recorded_exec_actions is not None
                    else None
                ),
            ),
        )
        chunk_step_payload = {
            "chunk_actions": exec_actions,
            "obs_list": obs_list,
            "terminations": chunk_terminations,
            "truncations": chunk_truncations,
            "infos_list": infos_list,
            "valid_action_mask": valid_action_mask,
        }
        return env_output, env_info, chunk_step_payload

    def env_evaluate_step(
        self,
        raw_actions: torch.Tensor,
        stage_id: int,
        on_step=None,
        on_step_at=None,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.eval_assignment.cfg.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.eval_assignment.cfg.get("wm_env_type", None),
            env_cfg=self.eval_assignment.cfg,
        )
        env_info = {}

        env = self.eval_env_list[stage_id]
        if on_step is not None or on_step_at is not None:
            obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
                env.chunk_step(chunk_actions, on_step=on_step, on_step_at=on_step_at)
            )
        else:
            obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
                env.chunk_step(chunk_actions)
            )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones.any(dim=1)  # [num_envs] bool
        if self.eval_assignment.cfg.auto_reset:
            newly_done = current_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if newly_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            env_infos=infos if isinstance(infos, dict) else None,
            transition=EnvTransition(
                dones=chunk_dones,
                terminations=chunk_terminations,
                truncations=chunk_truncations,
                rlt_switch_flags=rlt_switch_flags,
            ),
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Copy the last observation and restore completed environments' terminal rows."""
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        def replace_terminal_rows(
            observation: dict[str, Any], terminal: dict[str, Any], done_mask: np.ndarray
        ) -> None:
            for key, value in observation.items():
                if key not in terminal:
                    continue
                final_value = terminal[key]
                if isinstance(value, dict) and isinstance(final_value, dict):
                    replace_terminal_rows(value, final_value, done_mask)
                elif isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    value[dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    value[done_mask] = final_value[done_mask]
                elif isinstance(value, list) and isinstance(final_value, list):
                    for index in np.flatnonzero(done_mask):
                        value[index] = copy.deepcopy(final_value[index])

        merged_final_obs = copy.deepcopy(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            replace_terminal_rows(merged_final_obs, final_obs, done_mask)

        return merged_final_obs

    @staticmethod
    def _infer_rollout_batch_size(data: Any) -> int:
        """Infer the leading batch dimension of rollout or reward payloads."""

        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return int(data.shape[0])
        if hasattr(data, "actions") and isinstance(
            data.actions, (torch.Tensor, np.ndarray)
        ):
            return int(data.actions.shape[0])
        from rlinf.scheduler import infer_batch_size

        return infer_batch_size(data)

    @staticmethod
    def _merge_rollout_outputs(outputs: list[Any]) -> Any:
        if len(outputs) == 1:
            return outputs[0]
        first = outputs[0]
        if hasattr(first, "merge") and callable(first.merge):
            return type(first).merge(outputs)
        if isinstance(first, torch.Tensor):
            return torch.cat(outputs, dim=0)
        return outputs

    def finish_rollout(self, mode="train"):
        # Seeds restart at zero for every env of a heterogeneous mode, so each env
        # writes its videos under its own name.
        if mode == "train":
            for i in range(self.stage_num):
                if self.train_assignment.cfg.video_cfg.save_video:
                    flush_video = get_env_attr(self.env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video(self.train_assignment.name)
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.eval_assignment.cfg.video_cfg.save_video:
                    flush_video = get_env_attr(self.eval_env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video(self.eval_assignment.name)
                eval_env_cfg = self.eval_assignment.cfg
                if not eval_env_cfg.auto_reset and not eval_env_cfg.get(
                    "use_fixed_reset_state_ids", False
                ):
                    self.eval_env_list[i].update_reset_state_ids()

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        stage_id: int | None = None,
        last_run: bool = False,
    ):
        if self.reward_mode in {"per_step", "history_buffer"}:
            observations = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            observations = env_output.final_obs
        else:
            return None
        reward_input = dict(observations)
        if env_output.env_infos is not None:
            reward_input["env_infos"] = self._select_reward_env_infos(
                env_output.env_infos
            )

        dones = env_output.dones
        if dones is not None:
            if getattr(dones, "ndim", 0) > 1:
                dones = dones.any(dim=-1)
            reward_input["dones"] = dones

        if self.reward_mode == "history_buffer":
            if stage_id is None:
                raise ValueError("stage_id is required for history-buffer reward.")
            history_manager = self.train_history_managers[stage_id]
            history_manager.append_to_history_entries(observations)
            history_input, history_lengths = history_manager.build_history_input(
                dones=dones
            )
            reward_input["history_input"] = history_input
            self.history_lengths[stage_id] = dict(history_lengths)

        if last_run:
            reward_input.update(
                {
                    "last_run": torch.ones(
                        (self.train_num_envs_per_stage, 1), dtype=torch.bool
                    )
                }
            )
        self.send_to(
            group_name=self.cfg.reward.group_name,
            channel=send_channel,
            data=reward_input,
            tag="train_reward_obs",
            async_op=True,
            decoupled_mode=self.env_decoupled_mode,
        )
        reward_output = self.recv_from(
            group_name=self.cfg.reward.group_name,
            channel=recv_channel,
            tag="train_reward_obs",
            batch_size=self.train_batch_size,
            decoupled_mode=self.env_decoupled_mode,
        )
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _select_reward_env_infos(self, env_infos: dict[str, Any]) -> dict[str, Any]:
        reward_env_infos = {}
        for key in self.env_infos_reward_keys:
            if key not in env_infos:
                continue
            reward_env_infos[key] = clone_nested_to_cpu(env_infos[key])
        return reward_env_infos

    def _scatter_terminal_reward_output(
        self,
        env_output: EnvOutput,
        reward_output: torch.Tensor,
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    @Worker.timer("env/bootstrap_step")
    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.model_cfg.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.train_assignment.cfg.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    transition=EnvTransition(
                        dones=dones,
                        terminations=terminations,
                        truncations=truncations,
                    ),
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    env_infos=infos if isinstance(infos, dict) else None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                if _cfg_get(_cfg_get(getattr(self, "cfg", None), "rollout", None), "prefetch_at", None) is None:
                    base_env = getattr(
                        self.env_list[stage_id], "unwrapped", self.env_list[stage_id]
                    )
                    refresh = getattr(base_env, "refresh_observation", None)
                    fresh = refresh() if callable(refresh) else None
                    if fresh is not None:
                        self.last_obs_list[stage_id] = fresh
                        self.last_intervened_info_list[stage_id] = (None, None)
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    transition=EnvTransition(
                        dones=dones,
                        terminations=terminations,
                        truncations=truncations,
                        intervene_actions=self.last_intervened_info_list[stage_id][0],
                        intervene_flags=self.last_intervened_info_list[stage_id][1],
                    ),
                )
                env_outputs.append(env_output)

        return env_outputs

    def _build_rollout_input_data(self, env_batch: dict[str, Any]) -> dict[str, Any]:
        final_obs = env_batch["final_obs"]
        data = {
            "obs": self._eval_policy_obs_spec.apply(env_batch["obs"]),
            "final_obs": (
                self._eval_policy_obs_spec.apply(final_obs)
                if final_obs is not None
                else None
            ),
        }
        if self.enable_rlt:
            data["rlt_switch_flags"] = env_batch.get("rlt_switch_flags", None)
            data["intervene_flags"] = env_batch.get("intervene_flags", None)
        return data

    def _build_env_transition(
        self,
        env_output: EnvOutput,
        *,
        stage_id: int | None = None,
        reward_model_output: torch.Tensor | None = None,
        chunk_step_data: dict[str, Any] | None = None,
    ) -> EnvTransition:
        """Attach collector metadata to one environment transition."""
        reward_assign_lengths = None
        if (
            stage_id is not None
            and reward_model_output is not None
            and self.reward_mode == "history_buffer"
            and self.history_reward_assign
        ):
            reward_assign_lengths = [
                min(
                    lengths[env_id]
                    for lengths in self.history_lengths[stage_id].values()
                )
                for env_id in range(self.train_num_envs_per_stage)
            ]
        return env_output.transition.with_trajectory_data(
            reward_model_output=reward_model_output,
            reward_assign_lengths=reward_assign_lengths,
            episode_data=chunk_step_data if self.enable_online_lerobot else None,
        )

    def _train_policy_obs(
        self, env_output: EnvOutput, obs: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the copy of a train observation that goes to the policy."""
        return self._train_policy_obs_spec.apply(env_output.prepare_observations(obs))

    def _build_policy_input(
        self,
        env_output: EnvOutput,
    ) -> PolicyInput:
        """Build the model input without trajectory-owned environment data."""
        return PolicyInput(
            obs=self._train_policy_obs(env_output, env_output.obs),
            rlt_switch_flags=env_output.rlt_switch_flags,
            intervene_flags=env_output.intervene_flags,
        )

    def _send_policy_input(
        self,
        rollout_channel: Channel,
        policy_input: PolicyInput,
        stage_id: int,
        key: TrajectoryKey | None,
    ) -> None:
        if key is not None:
            policy_input.sources = [
                TrajectorySource(key=key, size=self.train_num_envs_per_stage)
            ]
        # Rollout does not reply to a request that carries its own actions, so
        # the stage runs them at once while the request is still in transit.
        supplies_actions = not policy_input.requires_inference
        if supplies_actions:
            self._supplied_actions[stage_id] = policy_input.external_actions
        route_key = (
            (key.step_id if key is not None else self._trajectory_step)
            if self.env_decoupled_mode
            else stage_id
        )
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=rollout_channel,
            data=policy_input,
            mode="train",
            tag="policy",
            route_key=route_key,
            async_op=supplies_actions,
            batch_size=self.train_batch_size,
            split_fn=self._policy_input_split_fn or PolicyInput.split,
            decoupled_mode=self.env_decoupled_mode,
        )

    def _publish_step(
        self,
        rollout_channel: Channel,
        env_output: EnvOutput,
        initial_transition: EnvTransition,
        reward_model_output: torch.Tensor | None,
        chunk_step_data: dict[str, Any] | None,
        epoch_id: int,
        chunk_id: int,
        stage_id: int,
        *,
        extra_next_rlt_updates: dict[str, Any] | None = None,
        send_completion_only: bool = False,
        is_last_override: bool | None = None,
        has_cached_next_rlt: bool = False,
    ) -> EnvPart | None:
        """Publish one environment outcome and schedule its next model input."""
        # The policy and environment halves of a chunk share this identity.
        key = TrajectoryKey(
            self._trajectory_step, epoch_id, self._rank, stage_id, chunk_id
        )
        source = TrajectorySource(key=key, size=self.train_num_envs_per_stage)
        is_last = (
            bool(is_last_override)
            if is_last_override is not None
            else (chunk_id == self.n_train_chunk_steps - 1)
        )
        has_terminal_state = env_output.dones is not None and bool(
            env_output.dones.any()
        )
        # A terminal forward pass supplies value/transition data, not actions.
        needs_terminal = (
            self.collect_final_values
            and not self.enable_online_lerobot
            and (is_last or has_terminal_state)
            and not has_cached_next_rlt
        )

        next_obs_override = None
        if has_terminal_state and env_output.final_obs is not None:
            next_obs_override = env_output.final_obs
        elif needs_terminal and send_completion_only:
            next_obs_override = env_output.obs

        smooth_intervene = getattr(self, "smooth_intervene", None)
        continue_smooth = (
            smooth_intervene.on_chunk_done(
                stage_id, env_output.intervene_flags, env_output.dones
            )
            if smooth_intervene is not None
            else False
        )
        ivpf = bool(
            _cfg_get(
                _cfg_get(_cfg_get(getattr(self, "cfg", None), "env", None), "train", None),
                "intervene_prefetch",
                False,
            )
        )

        if send_completion_only:
            next_key = None
            policy_input = PolicyInput(
                obs={},
                request_sizes=[self.train_num_envs_per_stage],
            )
        else:
            next_key = None
            if not is_last:
                next_key = TrajectoryKey(
                    self._trajectory_step,
                    epoch_id,
                    self._rank,
                    stage_id,
                    chunk_id + 1,
                )
            if continue_smooth and not is_last and not ivpf and smooth_intervene is not None:
                policy_input = smooth_intervene.build_external_policy_input(
                    stage_id,
                    env=self.env_list[stage_id],
                    obs=self._train_policy_obs(env_output, env_output.obs),
                )
            else:
                policy_input = self._build_policy_input(env_output)

        # Piggyback the completed transition on the next policy request so
        # Rollout remains the Actor channel's only producer.
        env_part = EnvPart(
            sources=[source],
            transition=self._build_env_transition(
                env_output,
                stage_id=stage_id,
                reward_model_output=reward_model_output,
                chunk_step_data=chunk_step_data,
            ),
            next_obs=(
                self._train_policy_obs(env_output, next_obs_override)
                if next_obs_override is not None
                else None
            ),
            next_rlt_obs=(
                {"__tianji_updates__": extra_next_rlt_updates}
                if extra_next_rlt_updates is not None
                else None
            ),
            requires_inference=needs_terminal,
            # Only chunk zero owns the boundary before the first action.
            initial_transition=initial_transition if chunk_id == 0 else None,
        )
        policy_input.env_parts = [env_part]
        policy_input.is_last = is_last
        self._send_policy_input(
            rollout_channel,
            policy_input,
            stage_id,
            next_key,
        )
        return None

    def _split_and_compress_obs(
        self, data: dict[str, Any], split_sizes: list[int]
    ) -> list[dict[str, Any]]:
        """Split a rollout-input payload by batch, then compress each shard.

        Used as the ``split_fn`` for observation ``send_to`` calls when
        ``env.obs_compression`` is enabled. Splitting first keeps the
        scheduler's ``infer_batch_size`` / ``split_batch`` operating on plain
        tensors; compression is applied per shard so the rollout worker can
        reconstruct it after receiving. When compression is disabled this
        function is not installed and the default ``split_batch`` is used.
        """
        from rlinf.scheduler.worker.routing import split_batch

        return [
            compress_obs(shard, self.obs_compression_cfg)
            for shard in split_batch(data, split_sizes)
        ]

    def _split_and_compress_policy_input(
        self, policy_input: PolicyInput, split_sizes: list[int]
    ) -> list[PolicyInput]:
        """Split a policy request, then compress its observation payloads."""
        shards = policy_input.split(split_sizes)
        for shard in shards:
            shard.obs = compress_obs(shard.obs, self.obs_compression_cfg)
            for env_part in shard.env_parts:
                if env_part is not None and env_part.next_obs is not None:
                    env_part.next_obs = compress_obs(
                        env_part.next_obs, self.obs_compression_cfg
                    )
        return shards

    def _send_train_bootstrap(
        self,
        rollout_channel: Channel,
        env_outputs: list[EnvOutput],
        step_id: int,
        epoch_id: int,
        skip_stage_ids: set[int] | None = None,
    ) -> None:
        ivpf = bool(
            _cfg_get(
                _cfg_get(_cfg_get(getattr(self, "cfg", None), "env", None), "train", None),
                "intervene_prefetch",
                False,
            )
        )
        smooth_intervene = getattr(self, "smooth_intervene", None)
        for stage_id in range(self.stage_num):
            if skip_stage_ids and stage_id in skip_stage_ids:
                continue
            # Bootstrap starts chunk zero; no earlier transition exists yet.
            key = TrajectoryKey(step_id, epoch_id, self._rank, stage_id, 0)
            if smooth_intervene is not None and smooth_intervene.is_active(stage_id) and not ivpf:
                policy_input = smooth_intervene.build_external_policy_input(
                    stage_id,
                    env=self.env_list[stage_id],
                    obs=self._train_policy_obs(
                        env_outputs[stage_id], env_outputs[stage_id].obs
                    ),
                )
            else:
                policy_input = self._build_policy_input(env_outputs[stage_id])
            policy_input.env_parts = [None]
            self._send_policy_input(
                rollout_channel,
                policy_input,
                stage_id,
                key,
            )

    def _bootstrap_and_send_train(
        self,
        rollout_channel: Channel,
        step_id: int,
        epoch_id: int,
        skip_stage_ids: set[int] | None = None,
    ) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(
            rollout_channel,
            env_outputs,
            step_id,
            epoch_id,
            skip_stage_ids=skip_stage_ids,
        )
        return env_outputs

    def prefetch_train_bootstrap(
        self,
        rollout_channel: Channel,
    ) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel,
            self._trajectory_step,
            0,
        )

    def record_env_metrics(
        self,
        env_metrics: dict[str, list],
        env_info: dict[str, Any],
    ):
        for key, value in env_info.items():
            env_metrics.setdefault(key, []).append(value)

    def _expects_train_policy_payload(self) -> bool:
        cfg = getattr(self, "cfg", None)
        rollout_cfg = _cfg_get(cfg, "rollout", None)
        return bool(
            getattr(self, "enable_rlt", False)
            or _cfg_get(rollout_cfg, "rlt_feature_model", None) is not None
            or _cfg_get(rollout_cfg, "prefetch_at", None) is not None
        )

    def _rlt_stage_now(self, stage_id: int, evaluation: bool = False):
        """Current stage index from the wrapper's ``rlt_switch_flags`` per sub-env, or None."""
        try:
            envs = self.eval_env_list if evaluation else self.env_list
            vals = envs[stage_id].env.call("get_wrapper_attr", "rlt_switch_flags")
            return [int(v) for v in vals]
        except Exception:
            return None

    @staticmethod
    def _stage_head_switches_of(fi, by_stage, stages):
        """Yield ``(env_idx, infer_stage, exec_stage, new_head_output)`` for stage changes."""
        n = int(os.environ.get("RLT_NUM_STAGES", "8"))
        pro = fi.get("proprio", None)
        if (
            n <= 0
            or not isinstance(pro, torch.Tensor)
            or pro.shape[-1] < n
            or pro.shape[0] != len(stages)
        ):
            return []
        oh = pro.reshape(pro.shape[0], -1)[:, -n:]
        infer_stage = torch.where(
            oh.sum(-1) > 0, oh.argmax(-1) + 1, torch.zeros_like(oh.argmax(-1))
        )
        S = int(by_stage.shape[1])
        switches = []
        for b_i, st in enumerate(stages):
            inf = int(infer_stage[b_i])
            if inf < 1 or st < 1:
                continue
            k_new, k_old = min(int(st), S), min(inf, S)
            if k_new != k_old:
                switches.append((b_i, inf, st, by_stage[b_i, k_new - 1]))
        return switches

    def _select_stage_head(self, policy_output, stages) -> None:
        """Select updated stage head output if stage switched between prefetch and execution."""
        fi = getattr(policy_output, "forward_inputs", None)
        if not fi:
            return
        by_stage = fi.pop("actions_by_stage", None)
        target_by_stage = fi.pop("target_actions_by_stage", None)
        if (
            by_stage is None
            or stages is None
            or getattr(policy_output, "actions", None) is None
        ):
            return
        acts = policy_output.actions
        for b_i, inf, st, new in self._stage_head_switches_of(fi, by_stage, stages):
            new = new.to(device=acts.device, dtype=acts.dtype)
            acts[b_i] = new.reshape(acts[b_i].shape)
            if target_by_stage is not None and "target_policy_action" in fi:
                target = target_by_stage[b_i, min(int(st), target_by_stage.shape[1]) - 1]
                fi["target_policy_action"][b_i] = target.to(fi["target_policy_action"])
            if isinstance(fi.get("policy_action_tcp"), torch.Tensor):
                from rlinf.algorithms.rlt.rollout import split_actor_critic_actions

                plan = fi["policy_action"].clone()
                plan[b_i] = new.reshape(plan[b_i].shape).to(plan.dtype)
                fi["action"] = plan
                split_actor_critic_actions(fi, int(self.model_cfg.num_action_chunks))
            elif isinstance(fi.get("action"), torch.Tensor):
                fi["action"][b_i] = new.reshape(fi["action"][b_i].shape).to(
                    fi["action"].dtype
                )
            self._stage_head_switches = getattr(self, "_stage_head_switches", 0) + 1
            self.log_info(
                "[StageHead] 切换块改用新阶段头:推理时阶段 %d → 执行起点阶段 %d(累计 %d 次)"
                % (inf, int(st), self._stage_head_switches)
            )

    def _offer_early_chunk(self, stage_id: int, evaluation: bool = False) -> None:
        """Offer the received next chunk early to the realworld env."""
        cache = "_early_eval_recv" if evaluation else "_early_policy_recv"
        state = (getattr(self, cache, None) or {}).get(stage_id)
        if state is None or state["output"] is None or state.get("offered"):
            return
        state["offered"] = True
        envs = self.eval_env_list if evaluation else self.env_list
        env_cfg = self.cfg.env.eval if evaluation else self.cfg.env.train
        offer = getattr(envs[stage_id], "offer_next_chunk", None)
        output = state["output"]
        if offer is None or getattr(output, "actions", None) is None:
            return
        try:
            acts = output.actions.clone()
            fi = getattr(output, "forward_inputs", None) or {}
            by_stage = fi.get("actions_by_stage")
            stages = self._rlt_stage_now(stage_id, evaluation=evaluation)
            if by_stage is not None and stages is not None:
                for b_i, _, _, new in self._stage_head_switches_of(fi, by_stage, stages):
                    acts[b_i] = new.to(device=acts.device, dtype=acts.dtype).reshape(
                        acts[b_i].shape
                    )
            if "target_actions_by_stage" in fi and by_stage is not None and stages is not None:
                for b_i, _, st, _ in self._stage_head_switches_of(fi, by_stage, stages):
                    if "target_policy_action" in fi:
                        fi["target_policy_action"][b_i] = fi["target_actions_by_stage"][
                            b_i, min(int(st), fi["target_actions_by_stage"].shape[1]) - 1
                        ]
            acts = self._row_scaled_actions(
                output, acts, stage_id=stage_id, evaluation=evaluation
            )
            offer(
                chunk_actions=prepare_actions(
                    raw_chunk_actions=acts,
                    env_type=env_cfg.env_type,
                    model_type=self.model_cfg.model_type,
                    num_action_chunks=self.model_cfg.num_action_chunks,
                    action_dim=self.model_cfg.action_dim,
                    policy=self.model_cfg.get("policy_setup", None),
                    wm_env_type=self.cfg.env.train.get("wm_env_type", None)
                    if hasattr(self.cfg.env, "train")
                    else None,
                    env_cfg=env_cfg,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- early offer is optimization only
            self.log_warning(f"[rt] next chunk not offered early: {exc}")

    @staticmethod
    def _relabel_stage_onehot(policy_output, stages) -> None:
        """Relabel proprio/critic_proprio one-hot stage suffix to execution-start stage."""
        if stages is None or policy_output is None:
            return
        fi = getattr(policy_output, "forward_inputs", None)
        if not fi:
            return
        n = int(os.environ.get("RLT_NUM_STAGES", "8"))
        if n <= 0:
            return
        for key in (
            "proprio",
            "rlt_transition_proprio",
            "critic_proprio",
            "rlt_transition_critic_proprio",
        ):
            t = fi.get(key, None)
            if not isinstance(t, torch.Tensor) or t.shape[-1] < n or t.shape[0] != len(stages):
                continue
            t = t.clone()
            oh = torch.zeros(t.shape[0], n, dtype=t.dtype, device=t.device)
            for b_i, st in enumerate(stages):
                if 1 <= st <= n:
                    oh[b_i, st - 1] = 1
            t[..., -n:] = oh
            fi[key] = t

    def _row_scaled_actions(
        self, policy_output, actions, stage_id=None, evaluation=False
    ):
        """Apply ActionGrid / TIANJI_ROW_SCALE resampling to chunk actions."""
        from rlinf.envs.realworld.tianji.tianji_contract import row_scale_from_env

        scale = row_scale_from_env()
        fi = getattr(policy_output, "forward_inputs", None) or {}
        if actions is not None and "target_action_grid" in fi:
            from rlinf.algorithms.rlt.action_grid import ActionGrid
            from rlinf.envs.realworld.tianji.tianji_kinematics import (
                joint_chunk_rl_to_tcp_rl,
            )

            grid = ActionGrid.from_metadata(fi["target_action_grid"])
            ref = fi["ref_chunk"].detach().to(device="cpu", dtype=torch.float32)
            deploy = bool(
                _cfg_get(_cfg_get(getattr(self, "cfg", None), "runner", None), "deploy_actor_updates", True)
            )
            if evaluation or deploy:
                target = (
                    fi["target_policy_action"]
                    .detach()
                    .to(ref)
                    .reshape(ref.shape[0], grid.target_steps, ref.shape[-1])
                )
            else:
                target = grid.reference_targets(ref)
            fi["target_policy_action"] = target.reshape(target.shape[0], -1).to(
                fi["ref_chunk"]
            )
            fi["target_policy_action_tcp"] = joint_chunk_rl_to_tcp_rl(
                fi["target_policy_action"].double(), grid.target_steps
            ).to(fi["ref_chunk"])
            executed = grid.to_env(target)
            fi["policy_action"] = executed.reshape(executed.shape[0], -1).to(
                fi["ref_chunk"]
            )
            fi["policy_action_tcp"] = joint_chunk_rl_to_tcp_rl(
                fi["policy_action"].double(), grid.env_steps
            ).to(fi["ref_chunk"])
            fi["action"] = fi["policy_action_tcp"].clone()
            if stage_id is not None:
                self._offer_dense_rows(
                    stage_id,
                    ref,
                    grid.offset,
                    grid.env_steps,
                    executed,
                    targets=target,
                    evaluation=evaluation,
                )
            return executed.reshape(actions.shape).to(actions)
        if scale == 1.0 or actions is None:
            return actions
        ref = fi.get("ref_chunk", None)
        if not isinstance(ref, torch.Tensor) or ref.dim() != 3:
            raise RuntimeError(
                "TIANJI_ROW_SCALE needs forward_inputs['ref_chunk'] (batch, horizon, dim)"
            )
        ref = ref.detach().to(device="cpu", dtype=torch.float32)
        batch, dim = ref.shape[0], ref.shape[2]
        rows = actions.numel() // (batch * dim)
        offset = float(int(os.environ.get("RLT_REF_OFFSET", "0")))
        resampled = self._resample_rows(ref, offset, scale, rows)
        executed = resampled.reshape(actions.shape).to(
            device=actions.device, dtype=actions.dtype
        )
        if stage_id is not None:
            self._offer_dense_rows(
                stage_id,
                ref,
                offset,
                rows,
                executed.reshape(batch, rows, dim),
                evaluation=evaluation,
            )
        return executed

    def _offer_dense_rows(
        self, stage_id, ref, offset, rows, executed, targets=None, evaluation=False
    ):
        from rlinf.envs.realworld.tianji.tianji_contract import dense_rows_from_env

        k = dense_rows_from_env()
        if not k:
            return
        envs = self.eval_env_list if evaluation else self.env_list
        offer = getattr(envs[stage_id], "offer_dense_rows", None)
        if offer is None:
            return
        dense = (
            targets
            if targets is not None
            else self._resample_rows(ref, offset, 1.0, int(-(-k * rows // 1)))
        ).to(dtype=executed.dtype)
        offer(executed.detach().cpu(), dense.detach().cpu())

    @staticmethod
    def _resample_rows(ref, start, step, count):
        horizon = ref.shape[1]
        pos = start + step * torch.arange(count, dtype=torch.float64)
        if count and float(pos[-1]) > horizon - 1 + 1e-6:
            raise ValueError(
                f"resampled rows reach model row {float(pos[-1]):.2f} of a {horizon}-row reference"
            )
        lo = pos.floor().long().clamp(0, horizon - 1)
        hi = (lo + 1).clamp(max=horizon - 1)
        w = (pos - lo.double()).to(ref.dtype).view(1, -1, 1)
        return ref[:, lo] * (1 - w) + ref[:, hi] * w

    def _resample_policy_output(self, policy_output, stage_id, evaluation=False) -> None:
        if policy_output is None or getattr(policy_output, "actions", None) is None:
            return
        resampled = self._row_scaled_actions(
            policy_output, policy_output.actions, stage_id=stage_id, evaluation=evaluation
        )
        if resampled is policy_output.actions:
            return
        policy_output.actions = resampled.detach().clone()
        ref = (
            policy_output.forward_inputs["ref_chunk"]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        if "target_action_grid" in policy_output.forward_inputs:
            from rlinf.algorithms.rlt.action_grid import ActionGrid

            grid = ActionGrid.from_metadata(
                policy_output.forward_inputs["target_action_grid"]
            )
            if grid.target_stride == 1:
                target = (
                    policy_output.forward_inputs["target_policy_action"]
                    .detach()
                    .cpu()
                    .reshape(ref.shape[0], grid.target_steps, ref.shape[-1])
                )
                ref = grid.execution_reference(ref, target)
        cache = "_eval_row_scale_ref" if evaluation else "_row_scale_ref"
        self.__dict__.setdefault(cache, {})[stage_id] = ref.clone()

    def _row_scale_prefix(self, obs, stage_id, prefetch_at, evaluation=False) -> None:
        from rlinf.envs.realworld.tianji.tianji_contract import row_scale_from_env

        scale = row_scale_from_env()
        if scale == 1.0 or prefetch_at is None or not isinstance(obs, dict):
            return
        prev = obs.get("rtc_prev_actions", None)
        if not isinstance(prev, torch.Tensor) or prev.dim() != 3:
            return
        offset = int(os.environ.get("RLT_REF_OFFSET", "0"))
        cache = "_eval_row_scale_ref" if evaluation else "_row_scale_ref"
        ref = (self.__dict__.get(cache) or {}).get(stage_id)
        if ref is None:
            if prev.shape[1] == 0:
                return
            idx = (
                (torch.arange(offset, dtype=torch.float64) / scale)
                .floor()
                .long()
                .clamp(max=prev.shape[1] - 1)
            )
            obs["rtc_prev_actions"] = prev[:, idx]
            return
        start = offset + (int(prefetch_at) + 1) * scale
        obs["rtc_prev_actions"] = self._resample_rows(ref, start, 1.0, offset).to(
            dtype=prev.dtype
        )

    @staticmethod
    def _mask_time_scaled(policy_output, stages) -> None:
        if stages is None or policy_output is None:
            return
        from rlinf.envs.realworld.tianji.tianji_contract import (
            row_scale_from_env,
            stage_time_scale_from_env,
        )

        scales = stage_time_scale_from_env()
        resampled = row_scale_from_env() != 1.0
        fi = getattr(policy_output, "forward_inputs", None)
        record = (
            fi.get("record_transition", None)
            if (scales or resampled) and fi
            else None
        )
        if not isinstance(record, torch.Tensor) or record.shape[0] != len(stages):
            return
        keep = torch.tensor(
            [not resampled and scales.get(int(s), 1.0) == 1.0 for s in stages],
            dtype=torch.bool,
            device=record.device,
        ).reshape(-1, *([1] * (record.dim() - 1)))
        fi["record_transition"] = record.to(torch.bool) & keep

    def _episode_archive(self, stage_id: int):
        """Return the training env's episode archive if available, else None."""
        archives = self.__dict__.setdefault("_episode_archives", {})
        if stage_id not in archives:
            archive = None
            if getattr(self, "enable_train", False) and getattr(
                self, "enable_rlt", False
            ):
                try:
                    getter = get_env_attr(self.env_list[stage_id], "episode_archive")
                    archive = getter() if callable(getter) else None
                    if archive is not None:
                        archive.set_run_info(self._archive_run_info())
                except Exception as exc:  # noqa: BLE001
                    self.log_warning(f"[EpisodeArchive] unavailable: {exc}")
                    archive = None
            archives[stage_id] = archive
        return archives[stage_id]

    def _archive_run_info(self) -> dict:
        def plain(path):
            try:
                value = OmegaConf.select(self.cfg, path, default=None)
                return (
                    OmegaConf.to_container(value, resolve=True)
                    if OmegaConf.is_config(value)
                    else value
                )
            except Exception:  # noqa: BLE001
                return None

        return {
            "algorithm": plain("algorithm"),
            "num_action_chunks": plain("actor.model.num_action_chunks"),
            "executor_action_grid": plain("actor.model.executor_action_grid"),
            "action_grid_offset": plain("actor.model.action_grid_offset"),
            "action_grid_scale": plain("actor.model.action_grid_scale"),
            "action_grid_dense": plain("actor.model.action_grid_dense"),
            "prefetch_at": plain("rollout.prefetch_at"),
            "ref_offset": os.environ.get("RLT_REF_OFFSET"),
            "vla_model_path": plain("rollout.rlt_feature_model.model_path"),
            "resume_dir": plain("runner.resume_dir"),
            "log_path": plain("runner.logger.log_path"),
        }

    def _archive_snapshot(self, policy_output, keep=False):
        kept = self.__dict__.setdefault("_archive_snapshots", {})
        if not keep:
            hit = kept.pop(id(policy_output), None)
            if hit is not None and hit[0] is policy_output:
                return hit
        try:
            from rlinf.envs.realworld.tianji.tianji_episode_archive import (
                snapshot_inference,
            )

            snapshot = (policy_output, snapshot_inference(policy_output))
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"[EpisodeArchive] inference not copied: {exc}")
            snapshot = (policy_output, None)
        if keep:
            kept[id(policy_output)] = snapshot
        return snapshot

    def _archive_inference(
        self, archive, serial, received, policy_output, hold_first, stages
    ):
        try:
            fi = getattr(policy_output, "forward_inputs", None) or {}
            if received is None:
                use = "no_inference"
            elif received[0] is not policy_output:
                use = "replaced_by_hold"
            elif hold_first:
                use = "during_takeover"
            else:
                use = "policy_chunk"
            actions_val = getattr(policy_output, "actions", policy_output)
            archive.record_inference(
                serial,
                received[1] if received is not None else None,
                {
                    "actions": actions_val,
                    "action": fi.get("action"),
                    "proprio": fi.get("proprio"),
                    "critic_proprio": fi.get("critic_proprio"),
                    "target_policy_action": fi.get("target_policy_action"),
                    "target_policy_action_tcp": fi.get("target_policy_action_tcp"),
                    "target_action_grid": fi.get("target_action_grid"),
                },
                {"inference_use": use, "stage_at_start": stages},
            )
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"[EpisodeArchive] inference not recorded: {exc}")

    def _defer_episode_resets(self) -> bool:
        """Enable deferred auto-reset for single-env single-stage single-epoch Tianji runs."""
        self._pending_chunks = {}
        if not (
            getattr(self, "discard_episodes", False)
            and self.stage_num == 1
            and self.rollout_epoch == 1
            and self.train_assignment.cfg.auto_reset
            and _cfg_get(_cfg_get(getattr(self, "cfg", None), "rollout", None), "prefetch_at", None)
            is not None
            and not getattr(self, "enable_online_lerobot", False)
        ):
            return False
        for env in self.env_list:
            if not callable(get_env_attr(env, "defer_auto_resets")) or (
                getattr(env, "unwrapped", env) is not env
            ):
                return False
        for env in self.env_list:
            get_env_attr(env, "defer_auto_resets")(True)
        return True

    def _complete_episode_reset(self, stage_id: int) -> tuple[EnvOutput, dict[str, Any]]:
        """Run deferred auto-reset after the finished episode has been emitted."""
        (
            obs_list,
            rewards,
            terminations,
            truncations,
            infos_list,
            exec_actions,
        ) = self._pending_chunks.pop(stage_id)
        complete = get_env_attr(self.env_list[stage_id], "complete_auto_reset")
        pool = getattr(self, "_ivpf_pool", None)
        obs_list[-1], infos_list[-1] = (
            pool.submit(complete).result() if pool is not None else complete()
        )
        env_output, env_info, _ = self._chunk_env_output(
            obs_list,
            rewards,
            terminations,
            truncations,
            infos_list,
            exec_actions=exec_actions,
        )
        self._chunk_end_t = time.time()
        return env_output, env_info

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    @staticmethod
    def _infer_action_batch_size(actions: Any) -> int:
        """Infer the batch dimension of a policy response."""
        if hasattr(actions, "actions"):
            actions = actions.actions
        return int(actions.shape[0])

    async def _recv_actions(
        self, input_channel: Channel, stage_id: int
    ) -> torch.Tensor:
        """Return the actions the stage supplied, else receive the policy's reply."""
        supplied = self._supplied_actions.pop(stage_id, None)
        if supplied is not None:
            return supplied
        return await self.recv_from(
            group_name=self.cfg.rollout.group_name,
            channel=input_channel,
            tag="train_policy" if self.env_decoupled_mode else "policy",
            route_key=stage_id if not self.env_decoupled_mode else None,
            batch_size=self.train_batch_size,
            infer_batch_size_fn=self._infer_action_batch_size,
            async_op=True,
            decoupled_mode=self.env_decoupled_mode,
        ).async_wait()

    def _recv_train_policy_output(
        self, input_channel: Channel, stage_id: int, **kwargs
    ):
        recv_kwargs: dict[str, Any] = {
            "group_name": self.cfg.rollout.group_name,
            "channel": input_channel,
            "tag": "train_policy" if self.env_decoupled_mode else "policy",
            "route_key": stage_id if not self.env_decoupled_mode else None,
            "batch_size": self.train_batch_size,
            "infer_batch_size_fn": self._infer_action_batch_size,
            "decoupled_mode": self.env_decoupled_mode,
        }
        if self.env_decoupled_mode or self._expects_train_policy_payload():
            recv_kwargs["merge_fn"] = self._merge_rollout_outputs
        recv_kwargs.update(kwargs)
        return self.recv_from(**recv_kwargs)

    def _poll_early_policy_recv(self, input_channel: Channel, stage_id: int) -> None:
        pending = getattr(self, "_early_policy_recv", None)
        if pending is None:
            pending = self._early_policy_recv = {}
        state = pending.get(stage_id)
        if state is None:
            pending[stage_id] = {
                "work": self._recv_train_policy_output(
                    input_channel, stage_id, async_op=True
                ),
                "output": None,
            }
        elif state["output"] is None and state["work"].done():
            state["output"] = state["work"].wait()
            if os.environ.get("RLT_LAT_TRACE") == "1":
                with open("/tmp/rlt_lat_env_recv.log", "a") as _f:
                    _f.write("%.4f %d\n" % (time.time(), time.monotonic_ns()))

    async def _take_train_policy_output(
        self, input_channel: Channel, stage_id: int
    ) -> Any:
        state = (getattr(self, "_early_policy_recv", None) or {}).pop(stage_id, None)
        if state is None:
            return await self._recv_train_policy_output(
                input_channel, stage_id, async_op=True
            ).async_wait()
        if state["output"] is not None:
            return state["output"]
        return await state["work"].async_wait()

    @staticmethod
    def _next_prefetch_key(
        step_id: int,
        epoch_id: int,
        env_rank: int,
        stage_id: int,
        chunk_id: int,
        n_train_chunk_steps: int,
        rollout_epoch: int,
    ) -> TrajectoryKey:
        if chunk_id + 1 < n_train_chunk_steps:
            return TrajectoryKey(step_id, epoch_id, env_rank, stage_id, chunk_id + 1)
        if epoch_id + 1 < rollout_epoch:
            return TrajectoryKey(step_id, epoch_id + 1, env_rank, stage_id, 0)
        return TrajectoryKey(step_id + 1, 0, env_rank, stage_id, 0)

    def _build_tianji_post_exec_updates(
        self,
        stage_id: int,
        policy_output: Any,
        env_output: EnvOutput,
        *,
        is_stale: bool,
        episode_end: bool,
    ) -> dict[str, Any] | None:
        expects_payload = self._expects_train_policy_payload()
        discard = bool(getattr(self, "discard_episodes", False))
        if not expects_payload and not discard:
            return None
        updates: dict[str, Any] = {}
        if is_stale:
            updates["_is_stale"] = True
        fi = getattr(policy_output, "forward_inputs", None)
        if isinstance(fi, dict):
            for key in (
                "proprio",
                "critic_proprio",
                "rlt_transition_proprio",
                "rlt_transition_critic_proprio",
                "action",
                "policy_action",
                "policy_action_tcp",
                "target_policy_action",
                "target_policy_action_tcp",
                "target_grid_observed",
                "record_transition",
            ):
                if key in fi and fi[key] is not None:
                    updates[key] = fi[key]
            actor_model = _cfg_get(
                _cfg_get(getattr(self, "cfg", None), "actor", None), "model", None
            )
            if (
                actor_model is not None
                and _cfg_get(actor_model, "executor_action_grid", None) is not None
            ):
                from rlinf.algorithms.rlt.action_grid import ActionGrid

                grid = ActionGrid.from_config(actor_model)
                if grid.aligned:
                    info = env_output.env_infos or {}
                    matches = (
                        info.get("target_grid_matches_plan")
                        if isinstance(info, dict)
                        else None
                    )
                    if matches is None and isinstance(info, dict):
                        final = info.get("final_info")
                        matches = (
                            final.get("target_grid_matches_plan")
                            if isinstance(final, dict)
                            else None
                        )
                    record = updates.get(
                        "record_transition",
                        torch.ones((self.train_num_envs_per_stage, 1), dtype=torch.bool),
                    )
                    valid = (
                        matches.reshape(record.shape[0], -1).bool().all(-1, keepdim=True)
                        if isinstance(matches, torch.Tensor)
                        else torch.zeros_like(record, dtype=torch.bool)
                    )
                    updates["target_grid_observed"] = valid.to(record.device)
                    updates["record_transition"] = record.bool() & valid.to(record.device)
        if getattr(self, "enable_rlt", False):
            updates["_stage_at_end"] = self._rlt_stage_now(stage_id)
        if discard:
            serial = self._episode_serial[stage_id]
            dones = env_output.dones
            ep_done = bool(dones is not None and torch.as_tensor(dones).any())
            infos = env_output.env_infos or {}
            infos_f = infos.get("final_info", infos) if isinstance(infos, dict) else {}
            disc_val = infos_f.get("episode_discarded") if isinstance(infos_f, dict) else None
            ep_disc = bool(disc_val is not None and torch.as_tensor(disc_val).any())
            updates["_episode_serial"] = serial
            updates["_episode_done"] = ep_done
            updates["_episode_discarded"] = ep_disc
            if ep_disc:
                self._discarded_serials[stage_id].add(serial)
                self.log_info(
                    f"[Discard] stage {stage_id} episode {serial}: kept out of training"
                )
            if ep_done:
                self._episode_serial[stage_id] = serial + 1
        if episode_end:
            updates["_early_epoch_end"] = True
        return updates or None

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        env_metrics = defaultdict(list)
        cfg = getattr(self, "cfg", None)
        rollout_cfg = _cfg_get(cfg, "rollout", None)
        env_train_cfg = _cfg_get(_cfg_get(cfg, "env", None), "train", None)
        pf_at = _cfg_get(rollout_cfg, "prefetch_at", None)
        pf_carry_mode = pf_at is not None
        ivpf = bool(_cfg_get(env_train_cfg, "intervene_prefetch", False))
        early_recv = bool(_cfg_get(rollout_cfg, "early_policy_recv", False)) and not self.env_decoupled_mode
        if getattr(self, "_pf_carry", None) is None:
            self._pf_carry = {}
        if getattr(self, "_pf_stale", None) is None:
            self._pf_stale = {}

        for epoch in range(self.rollout_epoch):
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            elif pf_carry_mode:
                env_outputs = self._bootstrap_and_send_train(
                    rollout_channel,
                    self._trajectory_step,
                    epoch,
                    skip_stage_ids=set(self._pf_carry.keys()),
                )
            else:
                env_outputs = self._bootstrap_and_send_train(
                    rollout_channel,
                    self._trajectory_step,
                    epoch,
                )
            # Preserve the pre-action boundary until chunk zero completes.
            initial_transitions = [
                self._build_env_transition(env_output) for env_output in env_outputs
            ]

            for stage_id in range(self.stage_num):
                await self._maybe_wait_env_delay(stage_id)

            episode_end = False
            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    stage_at_start = (
                        self._rlt_stage_now(stage_id) if self.enable_rlt else None
                    )
                    archive = self._episode_archive(stage_id)
                    archive_serial = (
                        archive.next_chunk() if archive is not None else None
                    )
                    archive_received = None
                    ivpf_future = None
                    pf_sent = [False]
                    self.__dict__.get("_row_scale_ref", {}).pop(stage_id, None)
                    pf_q = queue.Queue()

                    def _pf_send(_obs, _infos, _sid=stage_id, _cid=chunk_step_idx):
                        if pf_sent[0]:
                            return
                        pf_sent[0] = True
                        self._row_scale_prefix(_obs, _sid, pf_at)
                        sw = (
                            _infos.get("rlt_switch_flags")
                            if isinstance(_infos, dict)
                            else None
                        )
                        iv = (
                            _infos.get("intervene_flag")
                            if isinstance(_infos, dict)
                            else None
                        )
                        prepared_obs = self._train_policy_obs_spec.apply(
                            EnvOutput(obs=_obs).prepare_observations(_obs)
                        )
                        next_pf_key = self._next_prefetch_key(
                            self._trajectory_step,
                            epoch,
                            self._rank,
                            _sid,
                            _cid,
                            self.n_train_chunk_steps,
                            self.rollout_epoch,
                        )
                        req = PolicyInput(
                            obs=prepared_obs,
                            rlt_switch_flags=(
                                to_tensor(sw) if self.enable_rlt and sw is not None else None
                            ),
                            intervene_flags=(
                                to_tensor(iv) if self.enable_rlt and iv is not None else None
                            ),
                            env_parts=[None],
                            request_sizes=[self.train_num_envs_per_stage],
                            is_last=False,
                        )
                        _tq = time.time()
                        self._send_policy_input(
                            rollout_channel, req, _sid, next_pf_key
                        )
                        if os.environ.get("RLT_LAT_TRACE") == "1":
                            with open("/tmp/rlt_lat_env.log", "a") as _f:
                                _f.write("%.4f %.4f\n" % (_tq, time.time()))

                    def _pf_enqueue(_obs, _infos):
                        pf_q.put((_obs, _infos))

                    policy_output = None
                    smooth_intervene = getattr(self, "smooth_intervene", None)
                    if smooth_intervene is not None and smooth_intervene.is_active(stage_id) and not ivpf:
                        actions = await self._recv_actions(input_channel, stage_id)
                    else:
                        if smooth_intervene is not None and smooth_intervene.is_active(stage_id):
                            hold_actions = smooth_intervene.build_hold_actions(
                                stage_id, self.env_list[stage_id]
                            )
                            if getattr(self, "_ivpf_pool", None) is None:
                                self._ivpf_pool = concurrent.futures.ThreadPoolExecutor(
                                    max_workers=1
                                )
                            ivpf_future = self._ivpf_pool.submit(
                                self.env_interact_step,
                                hold_actions,
                                stage_id,
                                _pf_enqueue if pf_at is not None else None,
                                pf_at,
                            )

                        carried = self._pf_carry.pop(stage_id, None)
                        if carried is not None:
                            policy_output = carried
                        elif not self._expects_train_policy_payload() and not pf_carry_mode:
                            policy_output = await self._recv_actions(
                                input_channel, stage_id
                            )
                        else:
                            policy_output = await self._take_train_policy_output(
                                input_channel, stage_id
                            )

                        if archive is not None:
                            archive_received = self._archive_snapshot(policy_output)
                        if hasattr(policy_output, "forward_inputs"):
                            self._select_stage_head(policy_output, stage_at_start)
                            self._relabel_stage_onehot(policy_output, stage_at_start)
                            self._mask_time_scaled(policy_output, stage_at_start)
                            self._resample_policy_output(policy_output, stage_id)
                        actions = (
                            policy_output.actions
                            if hasattr(policy_output, "actions")
                            else policy_output
                        )

                    is_stale = bool(self._pf_stale.pop(stage_id, False))
                    if is_stale and smooth_intervene is not None:
                        actions = smooth_intervene.build_hold_actions(
                            stage_id, self.env_list[stage_id]
                        )

                    if smooth_intervene is not None:
                        smooth_intervene.remember_actions(stage_id, actions)
                    if archive is not None:
                        self._archive_inference(
                            archive,
                            archive_serial,
                            archive_received,
                            policy_output if policy_output is not None else actions,
                            ivpf_future is not None,
                            stage_at_start,
                        )

                    if self.enable_online_lerobot:
                        # Snapshot before stepping; environments may reuse obs buffers.
                        initial_observation = clone_nested_to_cpu(
                            env_outputs[stage_id].obs
                        )

                    if ivpf_future is None and pf_at is not None:
                        if getattr(self, "_ivpf_pool", None) is None:
                            self._ivpf_pool = concurrent.futures.ThreadPoolExecutor(
                                max_workers=1
                            )
                        ivpf_future = self._ivpf_pool.submit(
                            self.env_interact_step,
                            actions,
                            stage_id,
                            _pf_enqueue,
                            pf_at,
                        )

                    if ivpf_future is not None:
                        while True:
                            try:
                                _o, _i = pf_q.get(timeout=0.001)
                                _pf_send(_o, _i)
                            except queue.Empty:
                                pass
                            if early_recv and pf_sent[0]:
                                self._poll_early_policy_recv(input_channel, stage_id)
                                self._offer_early_chunk(stage_id)
                            if ivpf_future.done() and pf_q.empty():
                                break
                        env_output, env_info, chunk_step_data = ivpf_future.result()
                    else:
                        env_output, env_info, chunk_step_data = self.env_interact_step(
                            actions,
                            stage_id,
                        )

                    if stage_id in getattr(self, "_pending_chunks", {}):
                        episode_end = True

                    if self.enable_online_lerobot:
                        chunk_step_data["initial_observation"] = initial_observation
                    # Delay the next observation without blocking other worker tasks.
                    await self._maybe_wait_env_delay(stage_id)

                    reward_model_output = None
                    if reward_channel is not None:
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            stage_id=stage_id,
                            last_run=(
                                (
                                    epoch == self.rollout_epoch - 1
                                    and chunk_step_idx == self.n_train_chunk_steps - 1
                                )
                                or episode_end
                            ),
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    tianji_updates = self._build_tianji_post_exec_updates(
                        stage_id,
                        policy_output,
                        env_output,
                        is_stale=is_stale,
                        episode_end=episode_end,
                    )

                    if pf_sent[0]:
                        if (
                            env_output.dones is not None
                            and bool(torch.as_tensor(env_output.dones).any())
                        ):
                            self._pf_stale[stage_id] = True
                        is_epoch_closing = (
                            chunk_step_idx == self.n_train_chunk_steps - 1
                            or episode_end
                        )
                        self._publish_step(
                            rollout_channel,
                            env_output,
                            initial_transitions[stage_id],
                            reward_model_output,
                            chunk_step_data,
                            epoch,
                            chunk_step_idx,
                            stage_id,
                            extra_next_rlt_updates=tianji_updates,
                            send_completion_only=True,
                            is_last_override=is_epoch_closing,
                            has_cached_next_rlt=True,
                        )
                        if is_epoch_closing:
                            carried_next = await self._take_train_policy_output(
                                input_channel, stage_id
                            )
                            if episode_end:
                                self._pf_carry.pop(stage_id, None)
                                self._pf_stale.pop(stage_id, None)
                            else:
                                if archive is not None:
                                    self._archive_snapshot(carried_next, keep=True)
                                self._pf_carry[stage_id] = carried_next
                    elif tianji_updates is not None or episode_end:
                        self._publish_step(
                            rollout_channel,
                            env_output,
                            initial_transitions[stage_id],
                            reward_model_output,
                            chunk_step_data,
                            epoch,
                            chunk_step_idx,
                            stage_id,
                            extra_next_rlt_updates=tianji_updates,
                            is_last_override=(
                                True
                                if episode_end
                                else (chunk_step_idx == self.n_train_chunk_steps - 1)
                            ),
                        )
                    else:
                        self._publish_step(
                            rollout_channel,
                            env_output,
                            initial_transitions[stage_id],
                            reward_model_output,
                            chunk_step_data,
                            epoch,
                            chunk_step_idx,
                            stage_id,
                        )

                    if episode_end and stage_id in getattr(self, "_pending_chunks", {}):
                        env_output, reset_info = self._complete_episode_reset(stage_id)
                        env_info = {**env_info, **reset_info}

                    if (
                        get_env_attr(self.env_list[stage_id], "insert_delay_metrics")
                        is not None
                    ):
                        env_metrics["time/interact_delay"].append(
                            self.env_list[stage_id].insert_delay_metrics()
                        )
                    env_outputs[stage_id] = env_output
                    should_record = (
                        self.train_assignment.cfg.auto_reset
                        or self.train_assignment.cfg.ignore_terminations
                        or chunk_step_idx == self.n_train_chunk_steps - 1
                        or episode_end
                    )
                    if should_record:
                        self.record_env_metrics(env_metrics, env_info)
                    if episode_end:
                        break
                if episode_end:
                    break

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()
        self._trajectory_step += 1

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return {**env_metrics, **self.train_assignment.scoped(env_metrics)}

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.train_enable_offload:
                get_env_attr(env, "offload")()

        return env_metrics

    def _recv_eval_policy_output(self, input_channel: Channel, stage_id: int, **kwargs):
        joint_eval = bool(
            _cfg_get(
                _cfg_get(getattr(self, "cfg", None), "rollout", None),
                "joint_actor_eval",
                False,
            )
        )
        recv_kwargs: dict[str, Any] = {
            "group_name": self.cfg.rollout.group_name,
            "channel": input_channel,
            "tag": "eval_rollout_results",
            "route_key": stage_id if not self.env_decoupled_mode else None,
            "batch_size": self.eval_batch_size,
            "infer_batch_size_fn": (
                self._infer_rollout_batch_size
                if joint_eval or self.env_decoupled_mode
                else None
            ),
            "async_op": True,
            "decoupled_mode": self.env_decoupled_mode,
        }
        if joint_eval:
            recv_kwargs["merge_fn"] = self._merge_rollout_outputs
        recv_kwargs.update(kwargs)
        return self.recv_from(**recv_kwargs)

    def _poll_early_eval_recv(self, input_channel: Channel, stage_id: int) -> None:
        """Eval counterpart of ``_poll_early_policy_recv`` (rollout.early_policy_recv)."""
        pending = getattr(self, "_early_eval_recv", None)
        if pending is None:
            pending = self._early_eval_recv = {}
        state = pending.get(stage_id)
        if state is None:
            pending[stage_id] = {
                "work": self._recv_eval_policy_output(
                    input_channel, stage_id, async_op=True
                ),
                "output": None,
            }
        elif state["output"] is None and state["work"].done():
            state["output"] = state["work"].wait()
            if "target_action_grid" in (
                getattr(state["output"], "forward_inputs", None) or {}
            ):
                self._offer_early_chunk(stage_id, evaluation=True)

    def _take_eval_policy_output(self, input_channel: Channel, stage_id: int):
        state = (getattr(self, "_early_eval_recv", None) or {}).pop(stage_id, None)
        if state is None:
            return self._recv_eval_policy_output(input_channel, stage_id).wait()
        return state["output"] if state["output"] is not None else state["work"].wait()

    def _evaluate_step_prefetch(
        self,
        raw_actions,
        stage_id: int,
        rollout_channel: Channel,
        prefetch_at: int,
        *,
        send: bool,
        input_channel: Channel | None = None,
    ):
        """Run one eval chunk on a thread and send the next request at ``prefetch_at``."""
        if getattr(self, "_eval_pf_pool", None) is None:
            self._eval_pf_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        pending = queue.Queue()
        future = self._eval_pf_pool.submit(
            self.env_evaluate_step,
            raw_actions,
            stage_id,
            (lambda obs, infos: pending.put((obs, infos))) if send else None,
            prefetch_at if send else None,
        )
        sent = False
        early = (
            input_channel is not None
            and bool(
                _cfg_get(
                    _cfg_get(getattr(self, "cfg", None), "rollout", None),
                    "early_policy_recv",
                    False,
                )
            )
            and not self.env_decoupled_mode
        )
        while True:
            try:
                item = pending.get(timeout=0.001)
            except queue.Empty:
                item = None
            if early and sent:
                self._poll_early_eval_recv(input_channel, stage_id)
            if item is not None and not sent:
                obs, infos = item
                if stage_id in self.__dict__.get("_eval_row_scale_ref", {}):
                    self._row_scale_prefix(obs, stage_id, prefetch_at, evaluation=True)
                data = self._build_rollout_input_data(
                    {
                        "obs": obs,
                        "final_obs": None,
                        "rlt_switch_flags": (
                            to_tensor(infos.get("rlt_switch_flags"))
                            if self.enable_rlt
                            and isinstance(infos, dict)
                            and infos.get("rlt_switch_flags") is not None
                            else None
                        ),
                        "intervene_flags": (
                            to_tensor(infos.get("intervene_flag"))
                            if self.enable_rlt
                            and isinstance(infos, dict)
                            and infos.get("intervene_flag") is not None
                            else None
                        ),
                    }
                )
                self.send_to(
                    group_name=self.cfg.rollout.group_name,
                    channel=rollout_channel,
                    data=data,
                    split_fn=self._obs_split_fn,
                    mode="eval",
                    tag="rollout_results",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    decoupled_mode=self.env_decoupled_mode,
                )
                sent = True
            if future.done() and pending.empty():
                break
        env_output, env_info = future.result()
        return env_output, env_info, sent

    @Worker.timer("evaluate")
    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)
        early_stop_on_done = self.eval_assignment.cfg.get("early_stop_on_done", False)
        prefetch_at = _cfg_get(
            _cfg_get(getattr(self, "cfg", None), "rollout", None),
            "eval_prefetch_at",
            None,
        )
        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            if not self.eval_assignment.cfg.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                        env_infos=infos if isinstance(infos, dict) else None,
                    )
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=self._build_rollout_input_data(env_batch),
                        split_fn=self._obs_split_fn,
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

            eval_epoch_done = False
            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    policy_output = self._take_eval_policy_output(
                        input_channel, stage_id
                    )
                    if "target_action_grid" in (
                        getattr(policy_output, "forward_inputs", None) or {}
                    ):
                        self._select_stage_head(
                            policy_output,
                            self._rlt_stage_now(stage_id, evaluation=True),
                        )
                        self._resample_policy_output(
                            policy_output, stage_id, evaluation=True
                        )
                    raw_chunk_actions = (
                        policy_output.actions
                        if hasattr(policy_output, "actions")
                        else policy_output
                    )
                    if isinstance(raw_chunk_actions, torch.Tensor):
                        raw_chunk_actions = raw_chunk_actions.detach().cpu().numpy()
                    else:
                        raw_chunk_actions = np.asarray(raw_chunk_actions)

                    if self.eval_assignment.cfg.auto_reset:
                        last = (
                            eval_rollout_epoch == self.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        )
                    else:
                        last = eval_step == self.n_eval_chunk_steps - 1

                    sent = False
                    if prefetch_at is None:
                        env_output, env_info = self.env_evaluate_step(
                            raw_chunk_actions, stage_id
                        )
                    else:
                        env_output, env_info, sent = self._evaluate_step_prefetch(
                            raw_chunk_actions,
                            stage_id,
                            rollout_channel,
                            int(prefetch_at),
                            send=not last,
                            input_channel=input_channel,
                        )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if (
                        early_stop_on_done
                        and not sent
                        and env_output.dones is not None
                        and env_output.dones.any()
                        and eval_step < self.n_eval_chunk_steps - 1
                    ):
                        env_batch = self._build_rollout_input_data(env_output.to_dict())
                        env_batch["eval_epoch_done"] = True
                        self.send_to(
                            group_name=self.cfg.rollout.group_name,
                            channel=rollout_channel,
                            data=env_batch,
                            split_fn=self._obs_split_fn,
                            mode="eval",
                            tag="rollout_results",
                            route_key=stage_id if not self.env_decoupled_mode else None,
                            decoupled_mode=self.env_decoupled_mode,
                        )
                        eval_epoch_done = True
                        break

                    if last or sent:
                        continue
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=self._build_rollout_input_data(env_batch),
                        split_fn=self._obs_split_fn,
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

                if eval_epoch_done:
                    break

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                get_env_attr(self.eval_env_list[stage_id], "offload")()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return {**eval_metrics, **self.eval_assignment.scoped(eval_metrics)}
