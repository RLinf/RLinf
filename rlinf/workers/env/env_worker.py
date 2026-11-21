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
import gc
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.algorithms.rlt.transition import update_rlt_transitions
from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedLerobotTrajectoryBuilder,
    EmbodiedTrajectoryBuilder,
)
from rlinf.data.schema.embodied_types import (
    ChunkStepResult,
    EnvOutput,
    PolicyOutput,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.utils import get_env_attr, to_tensor
from rlinf.envs.wrappers import InsertDelay, RecordVideo
from rlinf.scheduler import Channel, Cluster, CommMapper, Worker
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import masked_stats, normalize_from_stats
from rlinf.utils.env_helpers import HistoryManager, SmoothInterveneController
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    split_dict_to_chunk,
    update_nested_cfg,
)
from rlinf.utils.obs_compression import compress_obs, is_compression_enabled
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
)


class EnvWorker(Worker):
    # Class-level default so the observation send path is safe even when the
    # instance is built without running ``__init__`` (e.g. ``object.__new__`` in
    # unit tests). ``None`` means "use the scheduler's default split"; when
    # ``env.obs_compression`` is enabled, ``__init__`` installs the compressing
    # ``split_fn`` (see ``_split_and_compress_obs``).
    _obs_split_fn = None

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
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        self.enable_rlt = OmegaConf.select(
            self.cfg, "algorithm.loss_type", default=""
        ) in {"rlt_ac", "rlt_td3"}
        # Optional lossless compression of image observations before they are
        # sent to the rollout workers. Disabled unless `env.obs_compression`
        # is present and `enable: true`. Compression runs inside a custom
        # `split_fn` (see `_split_and_compress_obs`) so it happens *after* the
        # channel splits the batch across ranks, keeping the scheduler's
        # batch-size inference and splitting operating on plain tensors.
        self.obs_compression_cfg = OmegaConf.select(
            self.cfg, "env.obs_compression", default=None
        )
        self._obs_split_fn = (
            self._split_and_compress_obs
            if is_compression_enabled(self.obs_compression_cfg)
            else None
        )

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
        self.env_infos_reward_keys = ("success", "episode", "final_info")
        if self.use_external_reward_model:
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)

        # Env configurations
        self.use_training_pipeline = self.cfg.runner.get("use_training_pipeline", False)
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
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1

        self.train_enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else False
        )
        self.eval_enable_offload = (
            eval_env_cfg.get("enable_offload", False)
            if eval_env_cfg is not None
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
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.actor_split_num = (
            1 if not self.enable_train else self.get_actor_split_num()
        )
        if self.use_training_pipeline and self.enable_train:
            self._init_pipeline_params()

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

        # env.train.discard_episodes (Tianji HIL "丢弃本条"): the chunks of an
        # episode wait here until it ends, so a discarded episode never reaches
        # the replay or any update (_tag_chunk_episode, _release_trajectories).
        self.discard_episodes = bool(
            self.enable_train and self.cfg.env.train.get("discard_episodes", False)
        )
        if self.discard_episodes:
            assert self.train_num_envs_per_stage == 1 and not self.use_training_pipeline, (
                "env.train.discard_episodes needs one env per stage and no training pipeline"
            )
            self._episode_serial = [0] * self.stage_num
            self._discarded_serials = [set() for _ in range(self.stage_num)]
            self._held_trajectories = [[] for _ in range(self.stage_num)]

        self.smooth_intervene = SmoothInterveneController.from_cfg(
            self.cfg,
            stage_num=self.stage_num,
            enable_train=self.enable_train,
            train_num_envs_per_stage=(
                self.train_num_envs_per_stage if self.enable_train else 0
            ),
        )

        if self.env_decoupled_mode:
            # Init the batch_router for env decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {}
            assert self._component_placement.get_world_size(
                "env"
            ) >= self._component_placement.get_world_size("rollout"), (
                "the world size of env must be greater than the world size of rollout in env_decoupled_mode"
            )

    def _prepare_trajectory_builders(
        self, trajectory_builders: list | None = None
    ) -> list:
        if self.enable_online_lerobot and trajectory_builders is not None:
            for stage_builder in trajectory_builders:
                stage_builder.rewards.clear()
            return trajectory_builders

        collect_only_success = bool(
            OmegaConf.select(
                self.cfg,
                "algorithm.dagger.online_lerobot.only_success",
                default=False,
            )
        )
        max_episode_length = self.cfg.env.train.max_episode_steps
        if self.enable_online_lerobot:
            return [
                EmbodiedLerobotTrajectoryBuilder(
                    max_episode_length=max_episode_length,
                    num_envs=self.train_num_envs_per_stage,
                    only_success=collect_only_success,
                    num_action_chunks=self.model_cfg.num_action_chunks,
                    action_dim=self.model_cfg.action_dim,
                )
                for _ in range(self.stage_num)
            ]
        return [
            EmbodiedTrajectoryBuilder(max_episode_length=max_episode_length)
            for _ in range(self.stage_num)
        ]

    def init_worker(self):
        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        if self.enable_train:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
            if self.train_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.env_list
                ), "train envs must have an offload method to enable offload!"
            self._release_at_episode_end = self._defer_episode_resets()

        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
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
        if self.enable_train:
            # train env
            train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
            if train_override_cfgs is not None:
                assert len(train_override_cfgs) > self._rank, (
                    f"{len(train_override_cfgs)=} > {self._rank=}"
                )

                general_train_override_cfg = OmegaConf.to_container(
                    self.cfg.env.train.get("override_cfg", {}), resolve=True
                )
                override_cfg = OmegaConf.to_container(
                    train_override_cfgs[self._rank], resolve=True
                ).copy()

                base_cfg = {}
                base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, override_cfg)
                setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))
            self._inject_realworld_reward_cfg(self.cfg.env.train)
        if self.enable_eval:
            eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
            if eval_override_cfgs is not None:
                assert len(eval_override_cfgs) > self._rank, (
                    f"{len(eval_override_cfgs)=} > {self._rank=}"
                )

                general_eval_override_cfg = OmegaConf.to_container(
                    self.cfg.env.eval.get("override_cfg", {}), resolve=True
                )
                eval_override_cfg = OmegaConf.to_container(
                    eval_override_cfgs[self._rank], resolve=True
                ).copy()
                base_eval_cfg = {}
                base_eval_cfg = update_nested_cfg(
                    base_eval_cfg, general_eval_override_cfg
                )
                base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
                setattr(
                    self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg)
                )
            self._inject_realworld_reward_cfg(self.cfg.env.eval)

    def _init_pipeline_params(self):
        actor_ws = self._component_placement.get_world_size("actor")
        logical_env_ws = self._world_size * self.stage_num
        self.shuffle_rollout = self.cfg.algorithm.get("shuffle_rollout", True)
        self.pipeline_stage_actor_splits = [
            CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs,
                src_world_size=logical_env_ws,
                dst_world_size=actor_ws,
                src_rank=self._rank * self.stage_num + stage_id,
            )
            for stage_id in range(self.stage_num)
        ]
        local_actor_ranks = {
            actor_rank
            for actor_splits in self.pipeline_stage_actor_splits
            for actor_rank, _ in actor_splits
        }
        self.pipeline_actor_env_ranks = {
            actor_rank: sorted(
                {
                    logical_src_rank // self.stage_num
                    for logical_src_rank, _ in CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.train.total_num_envs,
                        src_world_size=logical_env_ws,
                        dst_world_size=actor_ws,
                        dst_rank=actor_rank,
                    )
                }
            )
            for actor_rank in range(actor_ws)
        }
        self.pipeline_actor_keys = {
            actor_rank: CommMapper.build_channel_key(
                actor_rank, actor_rank, "pipeline_actor"
            )
            for actor_rank in local_actor_ranks
        }
        if self.shuffle_rollout:
            self.shuffle_generators = {
                actor_rank: torch.Generator().manual_seed(
                    self.cfg.actor.seed + actor_rank + self._rank * actor_ws
                )
                for actor_rank in local_actor_ranks
            }

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
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

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if (
                self.cfg.env.get("delay_sampler", None)
                and env_cfg is not self.cfg.env.eval
            ):
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
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _init_env(self):
        for i in range(self.stage_num):
            if self.enable_train:
                if self.cfg.env.train.auto_reset:
                    extracted_obs, _ = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    self.last_intervened_info_list.append((None, None))
                if self.train_enable_offload and self.cfg.env.train.get(
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
        import time as _t
        _t0 = _t.time()
        _prev = getattr(self, "_chunk_end_t", None)
        if _prev is not None:
            self._chunk_gaps = getattr(self, "_chunk_gaps", []) + [_t0 - _prev]
        """
        This function is used to interact with the environment.
        """
        exec_actions = prepare_actions(
            raw_chunk_actions=chunk_actions["raw_actions"]
            if isinstance(chunk_actions, dict)
            else chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
            env_cfg=self.cfg.env.train,
        )
        if isinstance(chunk_actions, dict):
            chunk_actions["actions"] = exec_actions
        else:
            chunk_actions = exec_actions

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(
                chunk_actions, on_step=on_step, on_step_at=on_step_at
            )
        )
        env_output, env_info, chunk_step_payload = self._chunk_env_output(
            obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list
        )
        if getattr(self, "_release_at_episode_end", False) and get_env_attr(
            self.env_list[stage_id], "auto_reset_pending"
        )():
            # The episode ended in this chunk; _complete_episode_reset rebuilds
            # this output once the episode is sent and the reset has run.
            self._pending_chunks[stage_id] = (
                obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list
            )
        # 块间空档统计:上一块执行结束 → 本块执行开始。每 30 块打印一次。
        self._chunk_end_t = _t.time()
        self._chunk_exec = getattr(self, "_chunk_exec", []) + [self._chunk_end_t - _t0]
        if len(self._chunk_exec) >= 30:
            import numpy as _np
            _g = _np.asarray(getattr(self, "_chunk_gaps", []) or [0.0])
            _e = _np.asarray(self._chunk_exec)
            _T = 1.0 / float(OmegaConf.select(self.cfg, "env.train.override_cfg.step_frequency", default=None) or self.cfg.env.train.get("step_frequency", 30))
            self.log_info(
                "[ChunkTiming] 执行 %.3fs/块(理论 %.3f) 块间空档 均值 %.3fs 最大 %.3fs 超过一周期 %d/%d 块 → 步频 %.1f Hz"
                % (_e.mean(), chunk_actions.shape[1] * _T, _g.mean(), _g.max(),
                   int((_g > _T).sum()), len(_g),
                   chunk_actions.shape[1] / (_e.mean() + _g.mean()))
            )
            self._chunk_exec = []
            self._chunk_gaps = []
        return env_output, env_info, chunk_step_payload

    def _chunk_env_output(
        self, obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
        """The EnvOutput, episode metrics and payload of one chunk_step result."""
        env_info = {}
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
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
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
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            # A deferred auto-reset (defer_auto_resets) has no final_info yet:
            # the chunk's own infos hold the same labels.
            final_info = infos.get("final_info", infos)
            if "intervene_action" in final_info:
                intervene_actions = final_info["intervene_action"]
                intervene_flags = final_info["intervene_flag"]

        iv_kind = infos["iv_kind"] if "iv_kind" in infos else None
        exec_actions = infos["exec_action"] if "exec_action" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any() and "final_info" in infos:
            if "iv_kind" in infos["final_info"] and "exec_action" in infos["final_info"]:
                iv_kind = infos["final_info"]["iv_kind"]
                exec_actions = infos["final_info"]["exec_action"]
        if iv_kind is not None and iv_kind.dim() != 2:          # 只接受块级 [B, C];逐步形状说明没经过 chunk_step 汇总
            iv_kind, exec_actions = None, None
        action_info = infos.get("final_info", infos) if self.cfg.env.train.auto_reset and chunk_dones.any() else infos
        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
            rlt_switch_flags=rlt_switch_flags,
            iv_kind=iv_kind,
            exec_actions=exec_actions,
            **{key: action_info.get(key) for key in (
                "exec_action_available", "intervene_action_available", "transition_padding"
            )},
            # rtj joint Stage2: the same labels in the second space.
            exec_actions_joint=action_info.get("exec_action_joint") if exec_actions is not None else None,
            intervene_actions_tcp=action_info.get("intervene_action_tcp") if exec_actions is not None else None,
        )
        chunk_step_payload = {
            "chunk_actions": exec_actions,
            "obs_list": obs_list,
            "terminations": chunk_terminations,
            "truncations": chunk_truncations,
            "infos_list": infos_list,
        }
        return env_output, env_info, chunk_step_payload

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int, on_step=None, on_step_at=None
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            env_cfg=self.cfg.env.eval,
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(
                chunk_actions, on_step=on_step, on_step_at=on_step_at
            )
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
        if self.cfg.env.eval.auto_reset:
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
            rlt_switch_flags=rlt_switch_flags,
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

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

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    @staticmethod
    def _infer_rollout_batch_size(data: Any) -> int:
        """Infer batch dim for routed shards; supports PolicyOutput and plain tensor payloads.

        When the channel carries a non-``PolicyOutput`` shard (e.g. reward tensor or eval
        actions) into a rollout recv, avoid assuming dataclass fields and delegate or use
        the leading dimension of dense arrays.
        """

        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return int(data.shape[0])
        if isinstance(data, PolicyOutput):
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(data, field_name, None)
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
            forward_inputs = getattr(data, "forward_inputs", None)
            if forward_inputs:
                first_tensor = next(iter(forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return int(first_tensor.shape[0])
            raise ValueError("Cannot infer batch size from rollout result.")
        from rlinf.scheduler import infer_batch_size

        return infer_batch_size(data)

    @Worker.timer("compute_bootstrap_rewards")
    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
        reward_model_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        if reward_model_output is not None:
            reward_model_output = reward_model_output.to(rewards.dtype)
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight * reward_model_output
            )

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video:
                    flush_video = get_env_attr(self.env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video:
                    flush_video = get_env_attr(self.eval_env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                if not self.cfg.env.eval.auto_reset:
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
        if dones is not None and getattr(dones, "ndim", 0) > 1:
            dones = dones[:, -1]
            reward_input.update({"dones": dones})

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

    def assign_history_reward(self, stage_id: int, reward_model_output: torch.Tensor):
        reward_assign_lengths = [
            min(
                history_buffer_length[env_id]
                for history_buffer_length in self.history_lengths[stage_id].values()
            )
            for env_id in range(self.train_num_envs_per_stage)
        ]
        rollout_rewards = self.trajectory_builders[stage_id].rewards
        rollout_rewards_length = len(rollout_rewards)
        reward_assign_lengths = [
            min(reward_assign_length, rollout_rewards_length)
            for reward_assign_length in reward_assign_lengths
        ]
        if not any(reward_assign_lengths):
            return
        reward = (self.reward_weight * reward_model_output).to(
            rollout_rewards[-1].dtype
        )
        for env_id, reward_assign_length in enumerate(reward_assign_lengths):
            for reward_assign_step in range(2, reward_assign_length + 1):
                rollout_rewards[-reward_assign_step][env_id] += reward[env_id]

    @Worker.timer("env/bootstrap_step")
    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.model_cfg.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                if self.enable_online_lerobot:
                    trajectory_builders = getattr(self, "trajectory_builders", None)
                    if trajectory_builders is not None:
                        trajectory_builders[stage_id].reset_episode_buffers()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    env_infos=infos if isinstance(infos, dict) else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                # Real hardware evolves while the synchronous learner updates.
                # Refresh only before a new inference request, never relabel an
                # already-prefetched chunk or change a recorded terminal frame.
                if self.cfg.rollout.get("prefetch_at", None) is None:
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
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def _build_rollout_input_data(self, env_batch: dict[str, Any]) -> dict[str, Any]:
        data = {
            "obs": env_batch["obs"],
            "final_obs": env_batch["final_obs"],
        }
        if self.enable_rlt:
            data["rlt_switch_flags"] = env_batch.get("rlt_switch_flags", None)
            data["intervene_flags"] = env_batch.get("intervene_flags", None)
        return data

    def _rlt_stage_now(self, stage_id: int, evaluation: bool = False):
        """当前阶段号(A1 键盘 wrapper 的 rlt_switch_flags),每个子环境一个;取不到返回 None。"""
        try:
            envs = self.eval_env_list if evaluation else self.env_list
            vals = envs[stage_id].env.call("get_wrapper_attr", "rlt_switch_flags")
            return [int(v) for v in vals]
        except Exception:
            return None

    def _select_stage_head(self, policy_output, stages) -> None:
        """切换块选头:推理时按"读观测那一刻"的阶段选了 actor 头;若本块执行起点的阶段已变(按键落在预取之后),
        改用 rollout 附带的新阶段头输出(forward_inputs.actions_by_stage,[B,S,C*D])。只在两边都是 actor 驾驶时替换;
        该键不进回放(每块 pop 掉)。实际执行值仍由 schema 2 的 exec_action 记录。"""
        fi = getattr(policy_output, "forward_inputs", None)
        if not fi:
            return
        by_stage = fi.pop("actions_by_stage", None)
        target_by_stage = fi.pop("target_actions_by_stage", None)
        if by_stage is None or stages is None or getattr(policy_output, "actions", None) is None:
            return
        acts = policy_output.actions
        for b_i, inf, st, new in self._stage_head_switches_of(fi, by_stage, stages):
            new = new.to(device=acts.device, dtype=acts.dtype)
            acts[b_i] = new.reshape(acts[b_i].shape)
            if target_by_stage is not None:
                target = target_by_stage[b_i, min(int(st), target_by_stage.shape[1]) - 1]
                fi["target_policy_action"][b_i] = target.to(fi["target_policy_action"])
            if isinstance(fi.get("policy_action_tcp"), torch.Tensor):
                # rtj joint Stage2: the new joint plan, its TCP rows in the critic slot.
                from rlinf.algorithms.rlt.rollout import split_actor_critic_actions

                plan = fi["policy_action"].clone()
                plan[b_i] = new.reshape(plan[b_i].shape).to(plan.dtype)
                fi["action"] = plan
                split_actor_critic_actions(fi, int(self.model_cfg.num_action_chunks))
            elif isinstance(fi.get("action"), torch.Tensor):
                fi["action"][b_i] = new.reshape(fi["action"][b_i].shape).to(fi["action"].dtype)
            self._stage_head_switches = getattr(self, "_stage_head_switches", 0) + 1
            self.log_info("[StageHead] 切换块改用新阶段头:推理时阶段 %d → 执行起点阶段 %d(累计 %d 次)" % (inf, int(st), self._stage_head_switches))

    @staticmethod
    def _stage_head_switches_of(fi, by_stage, stages):
        """(env index, inference stage, execution stage, that stage's head output)
        for each env whose stage changed after its observation was read."""
        n = int(os.environ.get("RLT_NUM_STAGES", "8"))
        pro = fi.get("proprio", None)
        if n <= 0 or not isinstance(pro, torch.Tensor) or pro.shape[-1] < n or pro.shape[0] != len(stages):
            return []
        oh = pro.reshape(pro.shape[0], -1)[:, -n:]
        infer_stage = torch.where(oh.sum(-1) > 0, oh.argmax(-1) + 1, torch.zeros_like(oh.argmax(-1)))
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

    def _offer_early_chunk(self, stage_id: int, evaluation: bool = False) -> None:
        """rt fork: give the env the next chunk as it will execute, once received.

        Same head choice (stage now) and prepare_actions as the chunk start; the
        env queues it behind the running chunk, and executes it only if the
        chunk it is handed at the turnover is identical.
        """
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
            by_stage, stages = fi.get("actions_by_stage"), self._rlt_stage_now(stage_id, evaluation=evaluation)
            if by_stage is not None and stages is not None:
                for b_i, _, _, new in self._stage_head_switches_of(fi, by_stage, stages):
                    acts[b_i] = new.to(device=acts.device, dtype=acts.dtype).reshape(acts[b_i].shape)
            if "target_actions_by_stage" in fi and stages is not None:
                for b_i, _, st, _ in self._stage_head_switches_of(fi, by_stage, stages):
                    fi["target_policy_action"][b_i] = fi["target_actions_by_stage"][b_i, min(int(st), fi["target_actions_by_stage"].shape[1]) - 1]
            acts = self._row_scaled_actions(output, acts, stage_id=stage_id, evaluation=evaluation)
            offer(chunk_actions=prepare_actions(
                raw_chunk_actions=acts,
                env_type=env_cfg.env_type,
                model_type=self.model_cfg.model_type,
                num_action_chunks=self.model_cfg.num_action_chunks,
                action_dim=self.model_cfg.action_dim,
                policy=self.model_cfg.get("policy_setup", None),
                wm_env_type=self.cfg.env.train.get("wm_env_type", None),
                env_cfg=env_cfg,
            ))
        except Exception as exc:  # noqa: BLE001 -- an offer is only an optimisation
            self.log_warning(f"[rt] next chunk not offered early: {exc}")

    @staticmethod
    def _relabel_stage_onehot(policy_output, stages) -> None:
        """把推理结果里的阶段 one-hot 改成"本块开始执行时"的阶段。

        状态观测是在上一块第 prefetch_at 步读的,阶段标签若也取那一刻,而阶段键在该步之后、
        本块开始之前按下,就会出现"标签没变、+1 已记在上一条"的错位:下一条样本跨阶段被截断、
        奖励却是 0,目标变成 0。按执行起点打标签后,标签变化、+1 奖励、截断总在同一条样本上。
        rollout 推理时 actor 看到的仍是读观测那一刻的标签,只影响换阶段那一块。"""
        if stages is None or policy_output is None:
            return
        fi = getattr(policy_output, "forward_inputs", None)
        if not fi:
            return
        n = int(os.environ.get("RLT_NUM_STAGES", "8"))
        if n <= 0:
            return
        # rtj joint Stage2: the critic's TCP proprio ends with the same one-hot.
        for key in ("proprio", "rlt_transition_proprio", "critic_proprio", "rlt_transition_critic_proprio"):
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

    def _row_scaled_actions(self, policy_output, actions, stage_id=None, evaluation=False):
        """11-04 speed-up (TIANJI_ROW_SCALE=k): executed row j is the VLA reference at model row
        RLT_REF_OFFSET + j*k, linear between rows. Deterministic in forward_inputs['ref_chunk'],
        so the early-offered chunk and the chunk handed at the turnover are identical.
        With ``stage_id`` the env of that stage also gets the dense rows (_offer_dense_rows)."""
        from rlinf.envs.realworld.tianji.tianji_contract import row_scale_from_env

        scale = row_scale_from_env()
        fi = getattr(policy_output, "forward_inputs", None) or {}
        if actions is not None and "target_action_grid" in fi:
            from rlinf.algorithms.rlt.action_grid import ActionGrid
            grid = ActionGrid.from_metadata(fi["target_action_grid"])
            ref = fi["ref_chunk"].detach().to(device="cpu", dtype=torch.float32)
            if evaluation or bool(self.cfg.runner.get("deploy_actor_updates", True)):
                target = fi["target_policy_action"].detach().to(ref).reshape(ref.shape[0], grid.target_steps, ref.shape[-1])
            else:
                target = grid.reference_targets(ref)
            fi["target_policy_action"] = target.reshape(target.shape[0], -1).to(fi["ref_chunk"])
            from rlinf.envs.realworld.tianji.tianji_kinematics import joint_chunk_rl_to_tcp_rl
            fi["target_policy_action_tcp"] = joint_chunk_rl_to_tcp_rl(fi["target_policy_action"].double(), grid.target_steps).to(fi["ref_chunk"])
            executed = grid.to_env(target)
            fi["policy_action"] = executed.reshape(executed.shape[0], -1).to(fi["ref_chunk"])
            fi["policy_action_tcp"] = joint_chunk_rl_to_tcp_rl(fi["policy_action"].double(), grid.env_steps).to(fi["ref_chunk"])
            fi["action"] = fi["policy_action_tcp"].clone()
            if stage_id is not None:
                self._offer_dense_rows(stage_id, ref, grid.offset, grid.env_steps, executed, targets=target, evaluation=evaluation)
            return executed.reshape(actions.shape).to(actions)
        if scale == 1.0 or actions is None:
            return actions
        fi = getattr(policy_output, "forward_inputs", None) or {}
        ref = fi.get("ref_chunk", None)
        if not isinstance(ref, torch.Tensor) or ref.dim() != 3:
            raise RuntimeError("TIANJI_ROW_SCALE needs forward_inputs['ref_chunk'] (batch, horizon, dim)")
        ref = ref.detach().to(device="cpu", dtype=torch.float32)
        batch, dim = ref.shape[0], ref.shape[2]
        rows = actions.numel() // (batch * dim)
        offset = float(int(os.environ.get("RLT_REF_OFFSET", "0")))
        resampled = self._resample_rows(ref, offset, scale, rows)
        executed = resampled.reshape(actions.shape).to(device=actions.device, dtype=actions.dtype)
        if stage_id is not None:
            self._offer_dense_rows(stage_id, ref, offset, rows, executed.reshape(batch, rows, dim), evaluation=evaluation)
        return executed

    def _offer_dense_rows(self, stage_id, ref, offset, rows, executed, targets=None, evaluation=False):
        """11-05 dense rows (TIANJI_DENSE_ROWS=1): every model row offset .. offset+k*rows-1 of the
        same reference, handed to the env beside the k-th rows it steps (offer and turnover alike);
        the backend submits them as that chunk's schedule at 1/(control_hz*k) s."""
        from rlinf.envs.realworld.tianji.tianji_contract import dense_rows_from_env

        k = dense_rows_from_env()
        if not k:
            return
        envs = self.eval_env_list if evaluation else self.env_list
        offer = getattr(envs[stage_id], "offer_dense_rows", None)
        if offer is None:
            return
        # ceil(k*rows) model rows (11-06: k=1.5 too); they fit the horizon whenever the env rows do.
        dense = (targets if targets is not None else
                 self._resample_rows(ref, offset, 1.0, int(-(-k * rows // 1)))).to(dtype=executed.dtype)
        offer(executed.detach().cpu(), dense.detach().cpu())

    @staticmethod
    def _resample_rows(ref, start, step, count):
        """ref (B, H, D) at fractional model rows start, start+step, ... (count rows), linear."""
        horizon = ref.shape[1]
        pos = start + step * torch.arange(count, dtype=torch.float64)
        if count and float(pos[-1]) > horizon - 1 + 1e-6:
            raise ValueError(f"resampled rows reach model row {float(pos[-1]):.2f} of a {horizon}-row reference")
        lo = pos.floor().long().clamp(0, horizon - 1)
        hi = (lo + 1).clamp(max=horizon - 1)
        w = (pos - lo.double()).to(ref.dtype).view(1, -1, 1)
        return ref[:, lo] * (1 - w) + ref[:, hi] * w

    def _resample_policy_output(self, policy_output, stage_id, evaluation=False) -> None:
        """Execute the resampled reference; keep it for the next request's RTC prefix."""
        if policy_output is None or getattr(policy_output, "actions", None) is None:
            return
        resampled = self._row_scaled_actions(policy_output, policy_output.actions, stage_id=stage_id, evaluation=evaluation)
        if resampled is policy_output.actions:
            return
        policy_output.actions = resampled.detach().clone()
        ref = policy_output.forward_inputs["ref_chunk"].detach().to(device="cpu", dtype=torch.float32)
        if "target_action_grid" in policy_output.forward_inputs:
            from rlinf.algorithms.rlt.action_grid import ActionGrid
            grid = ActionGrid.from_metadata(policy_output.forward_inputs["target_action_grid"])
            if grid.target_stride == 1:
                target = policy_output.forward_inputs["target_policy_action"].detach().cpu().reshape(ref.shape[0], grid.target_steps, ref.shape[-1])
                ref = grid.execution_reference(ref, target)
        cache = "_eval_row_scale_ref" if evaluation else "_row_scale_ref"
        self.__dict__.setdefault(cache, {})[stage_id] = ref.clone()

    def _row_scale_prefix(self, obs, stage_id, prefetch_at, evaluation=False) -> None:
        """The next request's RTC prefix at model rate: the reference rows the in-flight executed rows
        cover, [offset + (prefetch_at+1)*k, ... + offset). A held placeholder chunk (no reference)
        stretches its env rows to that length."""
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
            idx = (torch.arange(offset, dtype=torch.float64) / scale).floor().long().clamp(max=prev.shape[1] - 1)
            obs["rtc_prev_actions"] = prev[:, idx]
            return
        start = offset + (int(prefetch_at) + 1) * scale
        obs["rtc_prev_actions"] = self._resample_rows(ref, start, 1.0, offset).to(dtype=prev.dtype)

    @staticmethod
    def _mask_time_scaled(policy_output, stages) -> None:
        """11-04 experiment B: chunks that start in a time-scaled stage
        (TIANJI_STAGE_TIME_SCALE) run faster than the model's 30 Hz timeline,
        so they stay out of the replay and every loss (record_transition)."""
        if stages is None or policy_output is None:
            return
        from rlinf.envs.realworld.tianji.tianji_contract import row_scale_from_env, stage_time_scale_from_env

        scales = stage_time_scale_from_env()
        resampled = row_scale_from_env() != 1.0
        fi = getattr(policy_output, "forward_inputs", None)
        record = fi.get("record_transition", None) if (scales or resampled) and fi else None
        if not isinstance(record, torch.Tensor) or record.shape[0] != len(stages):
            return
        # Preserve the existing VLA-only collection/training eligibility.
        # An aligned network does not opt a speed-scaled rollout into learning.
        keep = torch.tensor(
            [not resampled and scales.get(int(s), 1.0) == 1.0 for s in stages],
            dtype=torch.bool, device=record.device
        ).reshape(-1, *([1] * (record.dim() - 1)))
        fi["record_transition"] = record.to(torch.bool) & keep

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

    def _send_train_bootstrap(
        self,
        rollout_channel: Channel,
        env_outputs: list[EnvOutput],
        skip_stage_ids: set[int] | None = None,
    ) -> None:
        skip_stage_ids = skip_stage_ids or set()
        for stage_id in range(self.stage_num):
            if stage_id in skip_stage_ids:
                continue
            env_output: EnvOutput = env_outputs[stage_id]
            env_batch = env_output.to_dict()
            self.send_to(
                group_name=self.cfg.rollout.group_name,
                channel=rollout_channel,
                data=self._build_rollout_input_data(env_batch),
                split_fn=self._obs_split_fn,
                mode="train",
                tag="rollout_results",
                route_key=stage_id if not self.env_decoupled_mode else None,
                decoupled_mode=self.env_decoupled_mode,
            )

    def _bootstrap_and_send_train(
        self,
        rollout_channel: Channel,
        skip_stage_ids: set[int] | None = None,
    ) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(
            rollout_channel,
            env_outputs,
            skip_stage_ids=skip_stage_ids,
        )
        return env_outputs

    def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel,
            skip_stage_ids=self.smooth_intervene.active_stage_ids(),
        )

    def record_env_metrics(
        self,
        env_metrics: dict[str, list],
        env_info: dict[str, Any],
    ):
        for key, value in env_info.items():
            env_metrics.setdefault(key, []).append(value)

    def _episode_archive(self, stage_id):
        """The training env's episode archive (tianji_episode_archive), or None.

        Recording only: it copies what this loop already holds and never feeds
        the trajectories, the replay or any update.
        """
        archives = self.__dict__.setdefault("_episode_archives", {})
        if stage_id not in archives:
            archive = None
            if self.enable_train and getattr(self, "enable_rlt", False):
                try:
                    getter = get_env_attr(self.env_list[stage_id], "episode_archive")
                    archive = getter() if callable(getter) else None
                    if archive is not None:
                        archive.set_run_info(self._archive_run_info())
                except Exception as exc:  # noqa: BLE001 -- recording never stops training
                    self.log_warning(f"[EpisodeArchive] unavailable: {exc}")
                    archive = None
            archives[stage_id] = archive
        return archives[stage_id]

    def _archive_run_info(self) -> dict:
        def plain(path):
            try:
                value = OmegaConf.select(self.cfg, path, default=None)
                return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
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
        """(output, copy of its tensors), before routing edits them in place.

        A carried output (keep) is routed at the end of an epoch and executed
        as the next epoch's first chunk; its copy waits here until then.
        """
        kept = self.__dict__.setdefault("_archive_snapshots", {})
        if not keep:
            hit = kept.pop(id(policy_output), None)
            if hit is not None and hit[0] is policy_output:
                return hit
        try:
            from rlinf.envs.realworld.tianji.tianji_episode_archive import snapshot_inference

            snapshot = (policy_output, snapshot_inference(policy_output))
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"[EpisodeArchive] inference not copied: {exc}")
            snapshot = (policy_output, None)
        if keep:
            kept[id(policy_output)] = snapshot
        return snapshot

    def _archive_inference(self, archive, serial, received, policy_output, hold_first, stages):
        """The chunk's inference and what of it the env executes."""
        try:
            fi = getattr(policy_output, "forward_inputs", None) or {}
            if received is None:
                use = "no_inference"  # a smooth-intervene placeholder
            elif received[0] is not policy_output:
                use = "replaced_by_hold"  # a prefetch from the previous episode
            elif hold_first:
                use = "during_takeover"  # the chunk started under human control
            else:
                use = "policy_chunk"
            archive.record_inference(
                serial,
                received[1] if received is not None else None,
                {"actions": policy_output.actions, "action": fi.get("action"),
                 "proprio": fi.get("proprio"), "critic_proprio": fi.get("critic_proprio"),
                 "target_policy_action": fi.get("target_policy_action"),
                 "target_policy_action_tcp": fi.get("target_policy_action_tcp"),
                 "target_action_grid": fi.get("target_action_grid")},
                {"inference_use": use, "stage_at_start": stages},
            )
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"[EpisodeArchive] inference not recorded: {exc}")

    def _ensure_schema2_keys(self, fi):
        """回放 schema 2 的新键(iv_kind / switch_step / data_schema)必须每块都有,否则按第一块键集堆叠会 KeyError 或静默丢键。
        这里先放默认值(占位块、过期块、首块都覆盖),真实值在 _apply_exec_record 里覆盖最后一行。"""
        if not getattr(self, "enable_rlt", False) or not isinstance(fi, dict) or "action" not in fi:
            return
        fi.pop("target_actions_by_stage", None)
        fi.pop("actions_by_stage", None)          # 只用于选头,不进回放(否则键集与旧数据不一致)
        B = int(fi["action"].shape[0])
        C = int(self.cfg.actor.model.num_action_chunks)
        from rlinf.algorithms.rlt.action_grid import ActionGrid
        grid = ActionGrid.from_config(self.cfg.actor.model)
        if grid.aligned:
            fi.setdefault("target_grid_observed", torch.zeros(B, 1, dtype=torch.bool))
            fi.setdefault("target_action_grid", grid.metadata(B))
            fi.setdefault("target_policy_action", torch.zeros(B, grid.target_steps * int(self.cfg.actor.model.action_dim)))
            fi.setdefault("target_model_action", torch.zeros_like(fi["target_policy_action"]))
            if "policy_action_tcp" in fi:
                fi.setdefault("target_policy_action_tcp", torch.zeros(B, grid.target_steps * int(self.cfg.actor.model.critic_action_dim)))
        fi.setdefault("iv_kind", torch.zeros(B, C, dtype=torch.int8))
        fi.setdefault("switch_step", torch.zeros(B, C, dtype=torch.int8))
        fi.setdefault("data_schema", torch.full((B, 1), 2, dtype=torch.int8))
        # The policy's own plan for this chunk, kept while _apply_exec_record
        # replaces "action" with the execution: a takeover chunk's Q label uses
        # it (algorithm.takeover_q_uses_policy_action).
        fi.setdefault("policy_action", fi.get("model_action", fi["action"]).clone())
        if getattr(self, "discard_episodes", False):
            # Episode of the chunk once executed (_tag_chunk_episode); removed
            # before the trajectory leaves this worker.
            fi.setdefault("episode_serial", torch.full((B, 1), -1, dtype=torch.int64))
        if self.cfg.algorithm.get("human_bc_target", "fitted") == "direct":
            # Placeholder/unexecuted chunks never become Q or human BC labels.
            fi.setdefault("exec_action_available", torch.zeros(B, C, dtype=torch.bool))
            fi.setdefault("human_bc_available", torch.zeros(B, C, dtype=torch.bool))
            # The actor's space (rtj joint Stage2: joints, while "action" is TCP).
            fi.setdefault("human_bc_actions", torch.zeros_like(fi["policy_action"]))
            fi.setdefault("transition_padding", torch.zeros(B, C, dtype=torch.bool))
        if "policy_action_tcp" in fi:
            # rtj joint Stage2: the same labels in the second space.
            fi.setdefault("exec_action_joint", torch.zeros_like(fi["policy_action"]))
            fi.setdefault("human_bc_actions_tcp", torch.zeros_like(fi["action"]))

    def _apply_exec_record(self, stage_id: int, env_output: EnvOutput):
        """块执行完:actions 栏记实际下发动作 + 逐步介入类别 + 块内切阶段标记(schema 2);
        环境没提供时退回原逻辑(只把介入步替换成人的动作)。"""
        builder = self.trajectory_builders[stage_id]
        if env_output.exec_actions is not None and env_output.iv_kind is not None:
            sw = None
            flags = env_output.rlt_switch_flags
            if flags is not None and flags.dim() == 2 and flags.shape[1] == env_output.iv_kind.shape[1]:
                from rlinf.algorithms.rlt.human_bc import switch_steps_from_flags
                sw = switch_steps_from_flags(flags)
            builder.update_last_exec(
                env_output.exec_actions, env_output.intervene_flags, env_output.iv_kind, sw,
                exec_action_available=env_output.exec_action_available,
                human_bc_available=env_output.intervene_action_available,
                human_bc_actions=env_output.intervene_actions if env_output.intervene_action_available is not None else None,
                transition_padding=env_output.transition_padding,
                exec_action_joint=env_output.exec_actions_joint,
                human_bc_actions_tcp=env_output.intervene_actions_tcp,
            )
        elif env_output.intervene_actions is not None:
            builder.update_last_actions(env_output.intervene_actions, env_output.intervene_flags)
        from rlinf.algorithms.rlt.action_grid import ActionGrid
        grid = ActionGrid.from_config(self.cfg.actor.model)
        if grid.aligned and builder.forward_inputs:
            info = env_output.env_infos or {}
            matches = info.get("target_grid_matches_plan") if isinstance(info, dict) else None
            if matches is None and isinstance(info, dict):
                final = info.get("final_info")
                matches = final.get("target_grid_matches_plan") if isinstance(final, dict) else None
            fi = builder.forward_inputs[-1]
            record = fi["record_transition"]
            valid = (matches.reshape(record.shape[0], -1).bool().all(-1, keepdim=True)
                     if isinstance(matches, torch.Tensor) else torch.zeros_like(record, dtype=torch.bool))
            fi["target_grid_observed"] = valid.to(record.device)
            fi["record_transition"] = record.bool() & valid.to(record.device)
        if getattr(self, "discard_episodes", False):
            self._tag_chunk_episode(stage_id, env_output)

    def _tag_chunk_episode(self, stage_id: int, env_output: EnvOutput):
        """Label the chunk just executed with its episode; close it on a done.

        The episode ends in the chunk whose steps report done. A Discard from
        the HIL panel arrives as ``episode_discarded`` on that chunk's steps.
        """
        builder = self.trajectory_builders[stage_id]
        if not builder.forward_inputs:
            return
        serial = self._episode_serial[stage_id]
        fi = builder.forward_inputs[-1]
        fi["episode_serial"] = torch.full_like(fi["episode_serial"], serial)
        dones = env_output.dones
        if dones is None or not bool(torch.as_tensor(dones).any()):
            return
        infos = env_output.env_infos or {}
        infos = infos.get("final_info", infos)
        discarded = infos.get("episode_discarded")
        if discarded is not None and bool(torch.as_tensor(discarded).any()):
            self._discarded_serials[stage_id].add(serial)
            self.log_info(f"[Discard] stage {stage_id} episode {serial}: kept out of training")
        self._episode_serial[stage_id] = serial + 1

    def _release_trajectories(self, stage_id: int, trajectories: list) -> list:
        """Trajectories ready for the learner; the rest wait for their episode.

        A trajectory leaves once none of its transitions belongs to the open
        episode. Chunks of a discarded episode are taken out of every loss
        (record_transition), and a trajectory with nothing else is dropped.
        """
        if not getattr(self, "discard_episodes", False):
            return trajectories
        held = self._held_trajectories[stage_id]
        held.extend(trajectories)
        open_serial = self._episode_serial[stage_id]
        discarded = torch.tensor(sorted(self._discarded_serials[stage_id]), dtype=torch.int64)
        ready = []
        while held:
            fi = held[0].forward_inputs
            serial = fi["episode_serial"]
            # Transitions only: the last entry is the next chunk's bootstrap.
            steps = held[0].rewards.shape[0] if held[0].rewards is not None else serial.shape[0]
            if bool((serial[:steps] == open_serial).any()):
                break
            trajectory = held.pop(0)
            fi.pop("episode_serial")
            dropped = torch.isin(serial, discarded)
            if not bool(dropped.any()):
                ready.append(trajectory)
            elif bool(dropped[:steps].all()):
                self.log_info(f"[Discard] stage {stage_id}: dropped a trajectory of {steps} chunks")
            else:
                record = fi["record_transition"]
                fi["record_transition"] = record & ~dropped.reshape(record.shape)
                self.log_info(
                    f"[Discard] stage {stage_id}: masked {int(dropped[:steps].sum())} of {steps} chunks"
                )
                ready.append(trajectory)
        return ready

    def _defer_episode_resets(self) -> bool:
        """Release a finished episode to training before the reset after it.

        discard_episodes holds an episode's chunks until it ends, but its last
        chunk only returned after the auto-reset, which waits for the
        operator's next Start: an episode reached training once the next one
        ran, its last epoch once the next one ended, and the last episode of
        a session never did. Deferred, the epoch ends with the episode, which
        is sent before the reset runs (_run_interact_once). Needs the prefetch
        timetable (the next chunk's observation leaves within the chunk), one
        env, one rollout epoch and an unwrapped env.
        """
        self._pending_chunks = {}
        if not (
            self.discard_episodes
            and self.stage_num == 1
            and self.rollout_epoch == 1
            and self.cfg.env.train.auto_reset
            and self.cfg.rollout.get("prefetch_at", None) is not None
            and not self.enable_online_lerobot
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

    async def _send_finished_episode(self, channel: Channel, timeout_s: float = 30.0):
        """Send what an episode end released, oldest first, and wait (bounded)
        until the channel holds it: the robot only waits for the next Start."""
        deferred, self._deferred_traj = getattr(self, "_deferred_traj", None), None
        builders = list(deferred[0]) if deferred else []
        works = []
        for builders_of_epoch in (builders, self.trajectory_builders):
            for stage_id, builder in enumerate(builders_of_epoch):
                for trajectory in self._release_trajectories(
                    stage_id, builder.to_splited_trajectories(self.actor_split_num)
                ):
                    works.append(channel.put(trajectory, async_op=True))
                builder.clear()
        deadline = time.monotonic() + timeout_s
        while works and not all(work.done() for work in works):
            if time.monotonic() > deadline:
                self.log_warning(f"[EpisodeEnd] {len(works)} trajectories still in flight after {timeout_s}s")
                break
            await asyncio.sleep(0.01)

    def _complete_episode_reset(self, stage_id: int) -> tuple[EnvOutput, dict[str, Any]]:
        """Run the deferred auto-reset (it waits for the operator's next Start)
        and rebuild the episode's last chunk output exactly as chunk_step
        returns it without deferral."""
        obs_list, rewards, terminations, truncations, infos_list = self._pending_chunks.pop(stage_id)
        complete = get_env_attr(self.env_list[stage_id], "complete_auto_reset")
        pool = getattr(self, "_ivpf_pool", None)
        # The executor thread that ran the chunk runs its reset, as before.
        obs_list[-1], infos_list[-1] = pool.submit(complete).result() if pool is not None else complete()
        env_output, env_info, _ = self._chunk_env_output(
            obs_list, rewards, terminations, truncations, infos_list
        )
        # The operator's wait is neither chunk execution nor a chunk gap.
        self._chunk_end_t = time.time()
        return env_output, env_info

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    @Worker.timer("env/send_rollout_trajectories")
    async def send_rollout_trajectories(
        self,
        trajectory_builder: EmbodiedTrajectoryBuilder,
        channel: Channel,
        stage_id: int | None = None,
    ):
        trajectories: list[Trajectory] = trajectory_builder.to_splited_trajectories(
            self.actor_split_num
        )
        trajectory_builder.clear()
        if stage_id is not None:
            trajectories = self._release_trajectories(stage_id, trajectories)
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)
        del trajectories
        gc.collect()

    @Worker.timer("env/send_lerobot_episodes")
    async def send_lerobot_episodes(
        self, episodes: list[list[dict]], channel: Channel
    ) -> None:
        if not episodes:
            return
        if self.actor_split_num <= 1:
            chunks = [episodes]
        else:
            chunks = split_list(
                episodes,
                self.actor_split_num,
                enforce_divisible_batch=False,
            )
        for chunk in chunks:
            if not chunk:
                continue
            channel.put(chunk, async_op=True)

    @Worker.timer("run_interact_once")
    def _recv_train_policy_output(self, input_channel: Channel, stage_id: int, **kwargs):
        return self.recv_from(
            group_name=self.cfg.rollout.group_name,
            channel=input_channel,
            tag="train_rollout_results",
            route_key=stage_id if not self.env_decoupled_mode else None,
            batch_size=self.train_batch_size,
            merge_fn=PolicyOutput.merge,
            infer_batch_size_fn=self._infer_rollout_batch_size,
            decoupled_mode=self.env_decoupled_mode,
            **kwargs,
        )

    def _poll_early_policy_recv(self, input_channel: Channel, stage_id: int):
        """rollout.early_policy_recv:预取请求一发出就开始接收它的结果,收齐后立即合并,
        块边界只取现成结果。收发仍一一对应、顺序不变;天机 30 Hz 段尾只剩 45 ms 滑行,
        块间的接收与合并(约 7 ms)会让每块交接时几乎停住。"""
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
                # The result is in the env worker (wall clock and the host monotonic
                # clock of the episode archive, to align the two).
                with open("/tmp/rlt_lat_env_recv.log", "a") as _f:
                    _f.write("%.4f %d\n" % (time.time(), time.monotonic_ns()))

    def _take_train_policy_output(self, input_channel: Channel, stage_id: int):
        state = (getattr(self, "_early_policy_recv", None) or {}).pop(stage_id, None)
        if state is None:
            return self._recv_train_policy_output(input_channel, stage_id)
        return state["output"] if state["output"] is not None else state["work"].wait()

    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.trajectory_builders = self._prepare_trajectory_builders(
            getattr(self, "trajectory_builders", None)
        )
        env_metrics = defaultdict(list)
        rlt_pending_obs: list[dict[str, Any] | None] = [None] * self.stage_num

        # 预取时间表(rollout.prefetch_at)开启时,epoch 边界不停车:
        # 上个 epoch 最后一块在第 prefetch_at 步发出的观测,其推理结果就是紧接着该执行的下一块。
        # 原逻辑把它只用来收尾记账,新 epoch 再发一次 bootstrap 观测、从头等一次推理,
        # 机器人每 600 步停约 0.9s。这里把它承接为新 epoch 第一块直接执行。
        _pf_carry_mode = self.cfg.rollout.get("prefetch_at", None) is not None
        if getattr(self, "_pf_carry", None) is None:
            self._pf_carry = {}
        sent_at_episode_end = False
        for epoch in range(self.rollout_epoch):
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            elif _pf_carry_mode:
                # 有承接结果的 stage 不发 bootstrap(auto_reset 下 bootstrap_step 只是取上次观测)
                env_outputs = self._bootstrap_and_send_train(
                    rollout_channel,
                    skip_stage_ids=set(self._pf_carry.keys()),
                )
            else:
                env_outputs = self._bootstrap_and_send_train(
                    rollout_channel,
                    skip_stage_ids=self.smooth_intervene.active_stage_ids(),
                )

            for stage_id in range(self.stage_num):
                await self._maybe_wait_env_delay(stage_id)

            # True once a chunk ends an episode whose reset is deferred
            # (_defer_episode_resets): the epoch ends with that chunk.
            episode_end = False
            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    _stage_at_start = self._rlt_stage_now(stage_id) if self.enable_rlt else None
                    self._apply_exec_record(stage_id, env_output)

                    reward_model_output = None
                    if reward_channel is not None and chunk_step_idx != 0:
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            stage_id=stage_id,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    # Episode archive (recording only, outside the data path): the
                    # serial the env tags this chunk with, and the inference as received.
                    _archive = self._episode_archive(stage_id)
                    _archive_serial = _archive.next_chunk() if _archive is not None else None
                    _archive_received = None
                    _ivpf = bool(self.cfg.env.train.get("intervene_prefetch", False))
                    _ivpf_future = None
                    # ---- 统一预取时间表(rollout.prefetch_at)----
                    # 每块执行到第 prefetch_at 步就把该步观测发给 rollout,推理与剩余步并行;
                    # 新块在下一个块边界开始执行。配合 RLT_REF_OFFSET = 块长-1-prefetch_at,
                    # actor 的动作与执行时刻对齐,不会往回拽。
                    # 介入块与策略块用同一个时间表,保证两类样本"状态↔参考块"的偏移一致。
                    _pf_at = self.cfg.rollout.get("prefetch_at", None)
                    _pf_sent = [False]
                    self.__dict__.get("_row_scale_ref", {}).pop(stage_id, None)
                    import queue as _pfq_mod
                    _pf_q = _pfq_mod.Queue()

                    def _pf_send(_obs, _infos, _sid=stage_id):
                        if _pf_sent[0]:
                            return
                        _pf_sent[0] = True
                        # rollout 只用到 obs / final_obs / rlt_switch_flags / intervene_flags
                        self._row_scale_prefix(_obs, _sid, _pf_at)
                        _data = {"obs": _obs, "final_obs": None}
                        if self.enable_rlt:
                            _sw = _infos.get("rlt_switch_flags") if isinstance(_infos, dict) else None
                            _iv = _infos.get("intervene_flag") if isinstance(_infos, dict) else None
                            _data["rlt_switch_flags"] = to_tensor(_sw) if _sw is not None else None
                            _data["intervene_flags"] = to_tensor(_iv) if _iv is not None else None
                        _tq = time.time()
                        self.send_to(
                            group_name=self.cfg.rollout.group_name,
                            channel=rollout_channel,
                            data=_data,
                            split_fn=self._obs_split_fn,
                            mode="train",
                            tag="rollout_results",
                            route_key=_sid if not self.env_decoupled_mode else None,
                            decoupled_mode=self.env_decoupled_mode,
                        )
                        if os.environ.get("RLT_LAT_TRACE") == "1":
                            with open("/tmp/rlt_lat_env.log", "a") as _f:
                                _f.write("%.4f %.4f\n" % (_tq, time.time()))

                    def _pf_enqueue(_obs, _infos):
                        # 后台执行线程里只入队,真正的发送在主线程完成(本地 channel 不是线程安全的)
                        _pf_q.put((_obs, _infos))

                    if self.smooth_intervene.is_active(stage_id) and not _ivpf:
                        policy_output = self.smooth_intervene.build_dummy_policy_output(
                            stage_id,
                            env=self.env_list[stage_id],
                            curr_obs=env_output.obs,
                        )
                    else:
                        if self.smooth_intervene.is_active(stage_id):
                            # ---- 介入预取(env.train.intervene_prefetch)----
                            # 介入期间执行的是人的主臂动作,策略动作会被介入 wrapper 覆盖,
                            # 所以不必等推理返回再执行。这里用保持动作占位,在后台线程里立刻开始
                            # 执行本块;主线程同时等"真实"推理结果,拿来做记录。
                            # 原 smooth_intervene 用替身输出跳过推理,会把 record_transition 和
                            # z_rl 一起清零,实测 98% 的介入块因此被丢出训练;这里两者都是真的。
                            # 记账顺序与原逻辑完全一致,仅把"执行"和"等待推理"重叠。
                            _hold = self.smooth_intervene.build_dummy_policy_output(
                                stage_id,
                                env=self.env_list[stage_id],
                                curr_obs=env_output.obs,
                            )
                            if getattr(self, "_ivpf_pool", None) is None:
                                import concurrent.futures as _cf
                                self._ivpf_pool = _cf.ThreadPoolExecutor(max_workers=1)
                            _ivpf_future = self._ivpf_pool.submit(
                                self.env_interact_step, _hold.actions, stage_id,
                                _pf_enqueue if _pf_at is not None else None, _pf_at,
                            )
                        _carried = self._pf_carry.pop(stage_id, None)
                        if _carried is not None:
                            policy_output = _carried      # 上个 epoch 末尾已收到的预取结果
                        else:
                            policy_output = self._take_train_policy_output(
                                input_channel, stage_id
                            )
                        if _archive is not None:
                            _archive_received = self._archive_snapshot(policy_output)
                        self._select_stage_head(policy_output, _stage_at_start)
                        self._relabel_stage_onehot(policy_output, _stage_at_start)
                        self._mask_time_scaled(policy_output, _stage_at_start)
                        self._resample_policy_output(policy_output, stage_id)
                        self.smooth_intervene.remember_policy_output(
                            stage_id, policy_output
                        )
                    # 预取的边界情况:上一块里这一段结束了(s/f/r)并已自动复位,
                    # 但那一块第 prefetch_at 步的观测早已发出,本次收到的是"基于上一段末尾画面"算的块。
                    # 照样收下(保持收发一一对应,否则此后每块都会错位),但不执行、不记录:原地保持一块。
                    _pfs = getattr(self, "_pf_stale", None)
                    if _pfs is not None and _pfs.get(stage_id, False):
                        _pfs[stage_id] = False
                        policy_output = self.smooth_intervene.build_dummy_policy_output(
                            stage_id,
                            env=self.env_list[stage_id],
                            curr_obs=env_output.obs,
                        )
                    if _archive is not None:
                        self._archive_inference(
                            _archive, _archive_serial, _archive_received, policy_output,
                            _ivpf_future is not None, _stage_at_start,
                        )
                    if _ivpf_future is None and _pf_at is not None:
                        # 策略块同样先开跑再记账:记账(append_step_result / update_rlt_transitions /
                        # 上个 epoch 轨迹发送)与执行重叠,块与块之间只剩一次 recv
                        if getattr(self, "_ivpf_pool", None) is None:
                            import concurrent.futures as _cf
                            self._ivpf_pool = _cf.ThreadPoolExecutor(max_workers=1)
                        _ivpf_future = self._ivpf_pool.submit(
                            self.env_interact_step, policy_output.actions, stage_id,
                            _pf_enqueue, _pf_at,
                        )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, policy_output.bootstrap_values, reward_model_output
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=policy_output.forward_inputs.get("action", None),
                        prev_logprobs=(
                            policy_output.prev_logprobs
                            if self.collect_prev_infos
                            else None
                        ),
                        prev_values=(
                            policy_output.prev_values
                            if self.collect_prev_infos
                            else None
                        ),
                        forward_inputs=policy_output.forward_inputs,
                        versions=policy_output.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )

                    self._ensure_schema2_keys(chunk_step_result.forward_inputs)
                    self.trajectory_builders[stage_id].append_step_result(
                        chunk_step_result
                    )
                    if (
                        self.reward_mode == "history_buffer"
                        and self.history_reward_assign
                        and reward_model_output is not None
                    ):
                        self.assign_history_reward(stage_id, reward_model_output)
                    if policy_output.intervene_flags is not None:
                        self.trajectory_builders[
                            stage_id
                        ].mark_last_step_with_intervene_flags(
                            policy_output.intervene_flags
                        )
                    if self.enable_rlt and self.collect_transitions:
                        update_rlt_transitions(
                            stage_id,
                            rlt_pending_obs,
                            self.trajectory_builders,
                            policy_output,
                            cache_current=True,
                            intervene_actions=env_output.intervene_actions,
                            intervene_flags=env_output.intervene_flags,
                        )

                    if _ivpf_future is not None:
                        _deferred = getattr(self, "_deferred_traj", None)
                        if _deferred:
                            self._deferred_traj = None
                            _builders, _ch = _deferred
                            for _sid, _b in enumerate(_builders):
                                # 不调 gc.collect():它持有 GIL,会卡住执行线程
                                # discard_episodes: an open episode's chunks wait for its end
                                for _tr in self._release_trajectories(
                                    _sid, _b.to_splited_trajectories(self.actor_split_num)
                                ):
                                    _ch.put(_tr, async_op=True)
                                _b.clear()
                        _early = bool(
                            self.cfg.rollout.get("early_policy_recv", False)
                        ) and not self.env_decoupled_mode
                        while True:
                            try:
                                # 1 ms:块结束后尽快开下一块(天机 30 Hz 段尾只剩 45 ms 滑行)
                                _o, _i = _pf_q.get(timeout=0.001)
                                _pf_send(_o, _i)
                            except _pfq_mod.Empty:
                                pass
                            if _early and _pf_sent[0]:
                                self._poll_early_policy_recv(input_channel, stage_id)
                                self._offer_early_chunk(stage_id)
                            if _ivpf_future.done() and _pf_q.empty():
                                break
                        env_output, env_info, chunk_step_payload = _ivpf_future.result()
                    else:
                        env_output, env_info, chunk_step_payload = self.env_interact_step(
                            policy_output.actions,
                            stage_id,
                            on_step=_pf_send if _pf_at is not None else None,
                            on_step_at=_pf_at,
                        )
                    if stage_id in getattr(self, "_pending_chunks", {}):
                        episode_end = True
                    # Emulated observation latency: wait before the obs goes out,
                    # without blocking the other coroutines in this worker.
                    await self._maybe_wait_env_delay(stage_id)
                    stage_builder = self.trajectory_builders[stage_id]
                    if isinstance(stage_builder, EmbodiedLerobotTrajectoryBuilder):
                        stage_builder.append_chunk_episode_data(
                            policy_output=policy_output,
                            **chunk_step_payload,
                        )
                    env_batch = env_output.to_dict()
                    skip_rollout_send = self.smooth_intervene.on_chunk_done(
                        stage_id,
                        env_output.intervene_flags,
                        env_output.dones,
                    )
                    if _ivpf:
                        skip_rollout_send = False     # 介入预取:介入期间照常发观测,下一块才有真实推理可记录
                    if _pf_sent[0]:
                        skip_rollout_send = True      # 本块观测已在块内提前发出
                        if bool(torch.as_tensor(env_output.dones).any()):
                            if getattr(self, "_pf_stale", None) is None:
                                self._pf_stale = {}
                            self._pf_stale[stage_id] = True   # 下一块收到的是过时结果,见上
                    if not skip_rollout_send:
                        self.send_to(
                            group_name=self.cfg.rollout.group_name,
                            channel=rollout_channel,
                            data=self._build_rollout_input_data(env_batch),
                            split_fn=self._obs_split_fn,
                            mode="train",
                            tag="rollout_results",
                            route_key=stage_id if not self.env_decoupled_mode else None,
                            decoupled_mode=self.env_decoupled_mode,
                        )
                    if (
                        get_env_attr(self.env_list[stage_id], "insert_delay_metrics")
                        is not None
                    ):
                        env_metrics["time/interact_delay"].append(
                            self.env_list[stage_id].insert_delay_metrics()
                        )
                    if self.collect_transitions and not self.enable_rlt:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.trajectory_builders[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    should_record = (
                        self.cfg.env.train.auto_reset
                        or self.cfg.env.train.ignore_terminations
                        or chunk_step_idx == self.n_train_chunk_steps - 1
                    )
                    if should_record:
                        self.record_env_metrics(env_metrics, env_info)
                    if episode_end:
                        break
                if episode_end:
                    break

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                self._apply_exec_record(stage_id, env_output)

                reward_model_output = None
                if reward_channel is not None:
                    last_run = epoch == self.rollout_epoch - 1
                    reward_model_output = self.get_reward_model_output(
                        env_output,
                        send_channel=reward_channel,
                        recv_channel=input_channel,
                        stage_id=stage_id,
                        last_run=last_run,
                    )
                    if reward_model_output is not None:
                        env_metrics["reward_model_output"].append(
                            reward_model_output.detach().float().reshape(-1).cpu()
                        )
                if self.smooth_intervene.is_active(stage_id) and not _pf_carry_mode:
                    policy_output = self.smooth_intervene.build_dummy_policy_output(
                        stage_id,
                        env=self.env_list[stage_id],
                        curr_obs=env_output.obs,
                    )
                else:
                    # 承接模式下最后一块必定已在块内发出预取观测(介入块也发),这里收的是真实结果:
                    # 既作最后一条 transition 的 next_obs,又留给下个 epoch 第一块直接执行。
                    # (原逻辑介入中用替身收尾,next_obs 的 z_rl 是零)
                    policy_output = self._take_train_policy_output(
                        input_channel, stage_id
                    )
                    if _pf_carry_mode and self._episode_archive(stage_id) is not None:
                        # Routed now, executed as the next epoch's first chunk.
                        self._archive_snapshot(policy_output, keep=True)
                    # 此刻没有块在执行,当前阶段 = 下一块执行起点的阶段
                    if self.enable_rlt:
                        _st_now = self._rlt_stage_now(stage_id)
                        self._select_stage_head(policy_output, _st_now)
                        self._relabel_stage_onehot(policy_output, _st_now)
                    self.smooth_intervene.remember_policy_output(
                        stage_id, policy_output
                    )
                    if _pf_carry_mode:
                        self._pf_carry[stage_id] = policy_output
                rewards = self.compute_bootstrap_rewards(
                    env_output, policy_output.bootstrap_values, reward_model_output
                )
                final_actions = policy_output.forward_inputs.get("action", None)
                final_forward_inputs = policy_output.forward_inputs
                if (
                    OmegaConf.select(self.cfg, "algorithm.loss_type", default="")
                    == "embodied_dagger"
                ):
                    final_actions = None
                    final_forward_inputs = {}

                chunk_step_result = ChunkStepResult(
                    actions=final_actions,
                    prev_logprobs=(
                        policy_output.prev_logprobs if self.collect_prev_infos else None
                    ),
                    prev_values=(
                        policy_output.prev_values if self.collect_prev_infos else None
                    ),
                    forward_inputs=final_forward_inputs,
                    versions=None if _pf_carry_mode else policy_output.versions,
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self._ensure_schema2_keys(chunk_step_result.forward_inputs)
                self.trajectory_builders[stage_id].append_step_result(chunk_step_result)
                if (
                    self.reward_mode == "history_buffer"
                    and self.history_reward_assign
                    and reward_model_output is not None
                ):
                    self.assign_history_reward(stage_id, reward_model_output)
                if self.enable_rlt and self.collect_transitions:
                    update_rlt_transitions(
                        stage_id,
                        rlt_pending_obs,
                        self.trajectory_builders,
                        policy_output,
                        cache_current=False,
                        intervene_actions=env_output.intervene_actions,
                        intervene_flags=env_output.intervene_flags,
                    )

            if episode_end:
                # Hand the finished episode to training, then run the reset
                # deferred at its end, which waits for the operator's next Start.
                if actor_channel is not None:
                    await self._send_finished_episode(actor_channel)
                    sent_at_episode_end = True
                for stage_id in list(self._pending_chunks):
                    env_outputs[stage_id], env_info = self._complete_episode_reset(stage_id)
                    self.record_env_metrics(env_metrics, env_info)

            if self.use_training_pipeline and actor_channel is not None:
                await self.send_rollout_trajectories_pipeline(
                    self.trajectory_builders, actor_channel
                )
                self.trajectory_builders = self._prepare_trajectory_builders(
                    getattr(self, "trajectory_builders", None)
                )

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if (
            not self.use_training_pipeline
            and actor_channel is not None
            and not sent_at_episode_end
        ):
            if self.enable_online_lerobot:
                for stage_id in range(self.stage_num):
                    episodes = self.trajectory_builders[stage_id].drain_episodes()
                    await self.send_lerobot_episodes(episodes, actor_channel)
            elif _pf_carry_mode:
                # 推迟到下个 epoch 第一块执行期间发送(见块循环),不占 epoch 边界时间
                self._deferred_traj = (list(self.trajectory_builders), actor_channel)
            else:
                for stage_id in range(self.stage_num):
                    await self.send_rollout_trajectories(
                        self.trajectory_builders[stage_id], actor_channel, stage_id
                    )

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.train_enable_offload:
                get_env_attr(env, "offload")()

        return env_metrics

    def _recv_eval_policy_output(self, input_channel: Channel, stage_id: int, **kwargs):
        joint_eval = self.cfg.rollout.get("joint_actor_eval", False)
        return self.recv_from(
            group_name=self.cfg.rollout.group_name,
            channel=input_channel,
            tag="eval_rollout_results",
            route_key=stage_id if not self.env_decoupled_mode else None,
            batch_size=self.eval_batch_size,
            merge_fn=PolicyOutput.merge if joint_eval else None,
            infer_batch_size_fn=self._infer_rollout_batch_size
            if joint_eval or self.env_decoupled_mode
            else None,
            decoupled_mode=self.env_decoupled_mode,
            **kwargs,
        )

    def _poll_early_eval_recv(self, input_channel: Channel, stage_id: int):
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
            if "target_action_grid" in (getattr(state["output"], "forward_inputs", None) or {}):
                self._offer_early_chunk(stage_id, evaluation=True)

    def _take_eval_policy_output(self, input_channel: Channel, stage_id: int):
        state = (getattr(self, "_early_eval_recv", None) or {}).pop(stage_id, None)
        if state is None:
            return self._recv_eval_policy_output(input_channel, stage_id)
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
        """Run one eval chunk on a thread and send the next request at ``prefetch_at``.

        Same timetable as training (rollout.prefetch_at): the request
        observation leaves while this chunk keeps executing, so the next chunk
        starts right at this chunk's end. Returns (env_output, env_info, sent);
        an env that sent nothing mid-chunk (episode ended, failed read) leaves
        the chunk-end observation to the caller. With ``input_channel`` and
        rollout.early_policy_recv, the reply is received while this chunk runs.
        """
        import concurrent.futures
        import queue

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
            and bool(self.cfg.rollout.get("early_policy_recv", False))
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
                data = {"obs": obs, "final_obs": None}
                if self.enable_rlt:
                    switch = infos.get("rlt_switch_flags") if isinstance(infos, dict) else None
                    flags = infos.get("intervene_flag") if isinstance(infos, dict) else None
                    data["rlt_switch_flags"] = to_tensor(switch) if switch is not None else None
                    data["intervene_flags"] = to_tensor(flags) if flags is not None else None
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
        prefetch_at = self.cfg.rollout.get("eval_prefetch_at", None)
        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
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

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    policy_output = self._take_eval_policy_output(input_channel, stage_id)
                    if "target_action_grid" in (getattr(policy_output, "forward_inputs", None) or {}):
                        self._select_stage_head(policy_output, self._rlt_stage_now(stage_id, evaluation=True))
                        self._resample_policy_output(policy_output, stage_id, evaluation=True)
                    raw_chunk_actions = (
                        policy_output.actions
                        if hasattr(policy_output, "actions")
                        else policy_output
                    )
                    if isinstance(raw_chunk_actions, torch.Tensor):
                        raw_chunk_actions = raw_chunk_actions.detach().cpu().numpy()
                    else:
                        raw_chunk_actions = np.asarray(raw_chunk_actions)
                    if self.cfg.env.eval.auto_reset:
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

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                get_env_attr(self.eval_env_list[stage_id], "offload")()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def compute_advantages_and_returns(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Advantages/returns are rollout-level quantities, so compute them before
        # splitting. After this point each channel item is an actor micro-batch that can
        # be trained directly without reconstructing the full rollout batch on actor.
        assert not (
            self.use_training_pipeline and self.cfg.algorithm.adv_type == "opd"
        ), (
            "OPD does not support runner.use_training_pipeline=True because "
            "teacher_logprobs are computed on actor workers after rollout."
        )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": rollout_batch["rewards"],
            "dones": rollout_batch["dones"],
            "values": rollout_batch.get("prev_values", None),
            "prev_logprobs": rollout_batch.get("prev_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": rollout_batch.get("loss_mask", None),
            "loss_mask_sum": rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": self.cfg.algorithm.get("normalize_advantages", True)
            and not self.use_training_pipeline,
        }
        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        return rollout_batch

    def prepare_pipeline_batch(self, trajectory: Trajectory) -> dict[str, torch.Tensor]:
        batch = convert_trajectories_to_batch([trajectory])
        batch = preprocess_embodied_batch(
            batch,
            rollout_epoch=1,
            auto_reset=self.cfg.env.train.auto_reset,
            ignore_terminations=self.cfg.env.train.ignore_terminations,
            reward_type=self.cfg.algorithm.reward_type,
            filter_rewards=self.cfg.algorithm.get("filter_rewards", False),
            group_size=self.cfg.algorithm.group_size,
            rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound", None),
            rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound", None),
        )
        return self.compute_advantages_and_returns(batch)

    def pack_pipeline_micro_batches(
        self, batch: dict[str, torch.Tensor], actor_rank: int
    ) -> list[dict]:
        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        if self.shuffle_rollout:
            shuffle_id = torch.randperm(
                batch_size, generator=self.shuffle_generators[actor_rank]
            )
        else:
            shuffle_id = torch.arange(batch_size)

        flatten_batch = flatten_embodied_batch(batch, shuffle_id)
        micro_batch_size = self.cfg.actor.micro_batch_size
        assert batch_size % micro_batch_size == 0, (
            f"Batch size {batch_size} is not divisible by micro_batch_size {micro_batch_size}."
        )
        num_micro_batches = batch_size // micro_batch_size
        micro_batches = split_dict_to_chunk(flatten_batch, num_micro_batches, dim=0)
        return [pack_batch(micro_batch) for micro_batch in micro_batches]

    async def send_rollout_trajectories_pipeline(
        self,
        trajectory_builders: list[EmbodiedTrajectoryBuilder],
        channel: Channel,
    ) -> None:
        pending_batches: list[tuple[int, dict[str, torch.Tensor]]] = []
        batches_by_actor_rank: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(
            list
        )

        with self.worker_timer("prepare_micro_batches"):
            for stage_id, trajectory_builder in enumerate(trajectory_builders):
                actor_splits = self.pipeline_stage_actor_splits[stage_id]
                trajectories = trajectory_builder.to_splited_trajectories_by_sizes(
                    [split_size for _, split_size in actor_splits]
                )

                for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                    batch = self.prepare_pipeline_batch(trajectory)
                    pending_batches.append((actor_rank, batch))
                    batches_by_actor_rank[actor_rank].append(batch)

            if self.cfg.algorithm.get("normalize_advantages", True):
                for actor_rank, batches in sorted(batches_by_actor_rank.items()):
                    local_adv_stats = sum(
                        masked_stats(batch["advantages"], batch.get("loss_mask"))
                        for batch in batches
                    )
                    env_ranks = self.pipeline_actor_env_ranks[actor_rank]
                    global_adv_stats = sum(
                        self.broadcast(
                            local_adv_stats if self._rank == src_rank else None,
                            groups=[(self._group_name, env_ranks)],
                            src=(self._group_name, src_rank),
                        )
                        for src_rank in env_ranks
                    )
                    for batch in batches:
                        batch["advantages"] = normalize_from_stats(
                            batch["advantages"], global_adv_stats
                        )

            for actor_rank, batch in pending_batches:
                for micro_batch in self.pack_pipeline_micro_batches(batch, actor_rank):
                    channel.put(
                        micro_batch,
                        key=self.pipeline_actor_keys[actor_rank],
                        async_op=True,
                    )
