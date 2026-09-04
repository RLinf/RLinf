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
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers import InsertDelay, RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.env_helpers import HistoryManager, SmoothInterveneController
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    update_nested_cfg,
)
from rlinf.utils.obs_compression import compress_obs, is_compression_enabled
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    # Class-level default so the observation send path is safe even when the
    # instance is built without running ``__init__`` (e.g. ``object.__new__`` in
    # unit tests). ``None`` means "use the scheduler's default split"; when
    # ``env.obs_compression`` is enabled, ``__init__`` installs the compressing
    # ``split_fn`` (see ``_split_and_compress_obs``).
    _obs_split_fn = None
    _policy_input_split_fn = None
    collect_final_values = True

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
        self.env_infos_reward_keys = ("success", "episode", "final_info")
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
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
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
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
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
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

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
            ),
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
        self, raw_actions: torch.Tensor, stage_id: int
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
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
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
            transition=EnvTransition(rlt_switch_flags=rlt_switch_flags),
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
        """Infer the leading batch dimension of rollout or reward payloads."""

        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return int(data.shape[0])
        from rlinf.scheduler import infer_batch_size

        return infer_batch_size(data)

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
        data = {
            "obs": env_batch["obs"],
            "final_obs": env_batch["final_obs"],
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

    def _build_policy_input(
        self,
        env_output: EnvOutput,
    ) -> PolicyInput:
        """Build the model input without trajectory-owned environment data."""
        return PolicyInput(
            obs=env_output.prepare_observations(env_output.obs),
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
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=rollout_channel,
            data=policy_input,
            mode="train",
            tag="policy_final" if policy_input.is_last else "policy",
            route_key=stage_id if not self.env_decoupled_mode else None,
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
    ) -> EnvPart | None:
        """Publish one environment outcome and schedule its next model input."""
        # The policy and environment halves of a chunk share this identity.
        key = TrajectoryKey(
            self._trajectory_step, epoch_id, self._rank, stage_id, chunk_id
        )
        source = TrajectorySource(key=key, size=self.train_num_envs_per_stage)
        is_last = chunk_id == self.n_train_chunk_steps - 1
        has_terminal_state = env_output.dones is not None and bool(
            env_output.dones.any()
        )
        # A terminal forward pass supplies value/transition data, not actions.
        needs_terminal = (
            self.collect_final_values
            and not self.enable_online_lerobot
            and (is_last or has_terminal_state)
        )

        next_obs_override = None
        if has_terminal_state and env_output.final_obs is not None:
            next_obs_override = env_output.final_obs
        elif is_last and self.env_decoupled_mode:
            # The next decoupled request may contain a reset observation.
            next_obs_override = env_output.obs

        next_key = None
        if not is_last:
            next_key = TrajectoryKey(
                self._trajectory_step,
                epoch_id,
                self._rank,
                stage_id,
                chunk_id + 1,
            )
        continue_smooth = self.smooth_intervene.on_chunk_done(
            stage_id, env_output.intervene_flags, env_output.dones
        )
        if continue_smooth and not is_last:
            policy_input = self.smooth_intervene.build_external_policy_input(
                stage_id,
                env=self.env_list[stage_id],
                obs=env_output.prepare_observations(env_output.obs),
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
                env_output.prepare_observations(next_obs_override)
                if next_obs_override is not None
                else None
            ),
            requires_inference=needs_terminal,
            # Only chunk zero owns the boundary before the first action.
            initial_transition=initial_transition if chunk_id == 0 else None,
        )
        policy_input.env_parts = [env_part]
        policy_input.is_last = is_last
        if is_last and self.env_decoupled_mode:
            return env_part
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
        previous_env_parts: dict[int, EnvPart] | None = None,
    ) -> None:
        for stage_id in range(self.stage_num):
            # Bootstrap starts chunk zero; no earlier transition exists yet.
            key = TrajectoryKey(step_id, epoch_id, self._rank, stage_id, 0)
            if self.smooth_intervene.is_active(stage_id):
                policy_input = self.smooth_intervene.build_external_policy_input(
                    stage_id,
                    env=self.env_list[stage_id],
                    obs=env_outputs[stage_id].prepare_observations(
                        env_outputs[stage_id].obs
                    ),
                )
            else:
                policy_input = self._build_policy_input(env_outputs[stage_id])
            policy_input.env_parts = [
                previous_env_parts.get(stage_id)
                if previous_env_parts is not None
                else None
            ]
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
        previous_env_parts: dict[int, EnvPart] | None = None,
    ) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(
            rollout_channel,
            env_outputs,
            step_id,
            epoch_id,
            previous_env_parts,
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

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    @staticmethod
    def _infer_action_batch_size(actions: torch.Tensor) -> int:
        """Infer the batch dimension of a policy response."""
        return int(actions.shape[0])

    def _recv_actions(self, input_channel: Channel, stage_id: int) -> torch.Tensor:
        """Receive actions using the route paired with the policy request."""
        return self.recv_from(
            group_name=self.cfg.rollout.group_name,
            channel=input_channel,
            tag="train_policy" if self.env_decoupled_mode else "policy",
            route_key=stage_id if not self.env_decoupled_mode else None,
            batch_size=self.train_batch_size,
            infer_batch_size_fn=self._infer_action_batch_size,
            decoupled_mode=self.env_decoupled_mode,
        )

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
        final_env_parts: dict[int, EnvPart] = {}

        for epoch in range(self.rollout_epoch):
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            else:
                env_outputs = self._bootstrap_and_send_train(
                    rollout_channel,
                    self._trajectory_step,
                    epoch,
                    final_env_parts,
                )
            final_env_parts = {}
            # Preserve the pre-action boundary until chunk zero completes.
            initial_transitions = [
                self._build_env_transition(env_output) for env_output in env_outputs
            ]

            for stage_id in range(self.stage_num):
                await self._maybe_wait_env_delay(stage_id)

            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    actions = self._recv_actions(input_channel, stage_id)
                    self.smooth_intervene.remember_actions(stage_id, actions)

                    env_output, env_info, chunk_step_data = self.env_interact_step(
                        actions, stage_id
                    )
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
                                epoch == self.rollout_epoch - 1
                                and chunk_step_idx == self.n_train_chunk_steps - 1
                            ),
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    env_part = self._publish_step(
                        rollout_channel,
                        env_output,
                        initial_transitions[stage_id],
                        reward_model_output,
                        chunk_step_data,
                        epoch,
                        chunk_step_idx,
                        stage_id,
                    )
                    if env_part is not None:
                        final_env_parts[stage_id] = env_part
                    if (
                        get_env_attr(self.env_list[stage_id], "insert_delay_metrics")
                        is not None
                    ):
                        env_metrics["time/interact_delay"].append(
                            self.env_list[stage_id].insert_delay_metrics()
                        )
                    env_outputs[stage_id] = env_output
                    should_record = (
                        self.cfg.env.train.auto_reset
                        or self.cfg.env.train.ignore_terminations
                        or chunk_step_idx == self.n_train_chunk_steps - 1
                    )
                    if should_record:
                        self.record_env_metrics(env_metrics, env_info)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()
        self._trajectory_step += 1
        if self.env_decoupled_mode:
            self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
                rollout_channel,
                self._trajectory_step,
                0,
                final_env_parts,
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

    @Worker.timer("evaluate")
    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)
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
                    policy_output = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.eval_batch_size,
                        infer_batch_size_fn=self._infer_rollout_batch_size
                        if self.env_decoupled_mode
                        else None,
                        decoupled_mode=self.env_decoupled_mode,
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
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch == self.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
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
