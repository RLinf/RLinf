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
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    split_dict,
    split_dict_to_chunk,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
)
from rlinf.workers.env.history_manager import HistoryManager


class EnvWorker(Worker):
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
        self.last_policy_info_list = []
        self.eval_policy_info_list = []
        self._prefetched_train_bootstrap: list[EnvOutput] | None = None
        self.rollout_epoch = self.cfg.algorithm.get("rollout_epoch", 1)
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num

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
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.eval
        self.enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else eval_env_cfg.get("enable_offload", False)
        )
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        self.eval_action_exec_chunks = int(
            self.cfg.env.eval.get(
                "action_exec_chunks", self.cfg.actor.model.num_action_chunks
            )
        )
        if self.eval_action_exec_chunks <= 0:
            raise ValueError(
                "env.eval.action_exec_chunks must be positive, got "
                f"{self.eval_action_exec_chunks}"
            )
        if (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            % self.eval_action_exec_chunks
            != 0
        ):
            raise ValueError(
                "env.eval.max_steps_per_rollout_epoch must be divisible by "
                "env.eval.action_exec_chunks, got "
                f"{self.cfg.env.eval.max_steps_per_rollout_epoch} and "
                f"{self.eval_action_exec_chunks}"
            )
        if not self.only_eval:
            if train_env_cfg is None:
                raise ValueError(
                    "env.train config is required when runner.only_eval=False."
                )
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
        self.n_train_chunk_steps = 0
        if not self.only_eval:
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.cfg.actor.model.num_action_chunks
            )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.eval_action_exec_chunks
        )
        self.actor_split_num = self.get_actor_split_num()

        if not self.only_eval:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
            self.rlt_local_policy_state: list[dict[str, torch.Tensor]] = []
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
            self.eval_rlt_local_policy_state: list[dict[str, torch.Tensor]] = []

    def init_worker(self):
        self.dst_rank_map = self._setup_dst_rank_map()
        self.src_rank_map = self._setup_src_rank_map()

        self.log_info(f"Env worker initialized with dst_rank_map: {self.dst_rank_map}")
        self.log_info(f"Env worker initialized with src_rank_map: {self.src_rank_map}")

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        if not self.only_eval:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )
            self.log_info(
                "Eval action scheduling: "
                f"model_num_action_chunks={self.cfg.actor.model.num_action_chunks}, "
                f"action_exec_chunks={self.eval_action_exec_chunks}, "
                f"n_eval_chunk_steps={self.n_eval_chunk_steps}, "
                f"expected_env_steps="
                f"{self.n_eval_chunk_steps * self.eval_action_exec_chunks}, "
                f"max_steps_per_rollout_epoch="
                f"{self.cfg.env.eval.max_steps_per_rollout_epoch}"
            )
            self.eval_policy_info_list = [
                self._init_rlt_local_policy_state(i, mode="eval")
                for i in range(self.stage_num)
            ]

        if not self.only_eval:
            self._init_env()
            if self.reward_mode == "history_buffer":
                self.train_history_managers = [
                    HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                    for _ in range(self.stage_num)
                ]
                self.history_lengths = [{} for _ in range(self.stage_num)]

    def update_env_cfg(self):
        if not self.only_eval:
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
            base_eval_cfg = update_nested_cfg(base_eval_cfg, general_eval_override_cfg)
            base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
            setattr(self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg))
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

    def _setup_dst_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        """Compute destination rank map for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout/reward layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving results from rollout and reward workers.

        Returns:
            Destination rank map for this env worker.
            The key is the channel name (e.g. "rollout_train", "reward_train", "rollout_eval"), and the value is a ordered list of tuples of (dst_rank, batch_size).
        """
        dst_rank_map = {}
        if not self.only_eval:
            dst_rank_map = {
                "rollout_train": CommMapper.get_dst_ranks(
                    batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                    src_world_size=self._component_placement.get_world_size("env"),
                    dst_world_size=self._component_placement.get_world_size("rollout"),
                    src_rank=self._rank,
                ),
            }
            if self.cfg.get("reward", {}).get("use_reward_model", False):
                dst_rank_map.update(
                    {
                        "reward_train": CommMapper.get_dst_ranks(
                            batch_size=self.cfg.env.train.total_num_envs
                            // self.stage_num,
                            src_world_size=self._component_placement.get_world_size(
                                "env"
                            ),
                            dst_world_size=self._component_placement.get_world_size(
                                "reward"
                            ),
                            src_rank=self._rank,
                        ),
                    }
                )

        if self.enable_eval:
            dst_rank_map.update(
                {
                    "rollout_eval": CommMapper.get_dst_ranks(
                        batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size("env"),
                        dst_world_size=self._component_placement.get_world_size(
                            "rollout"
                        ),
                        src_rank=self._rank,
                    ),
                }
            )
        return dst_rank_map

    def _setup_src_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        """Compute source rank map for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout/reward layouts.
        The returned ranks are used as communication counterparts for both receiving results from rollout and reward workers and sending action chunks.

        Returns:
            Source rank map for this env worker.
            The key is the channel name (e.g. "rollout_train", "reward_train", "rollout_eval"), and the value is a ordered list of tuples of (src_rank, batch_size).
        """
        src_rank_map = {}
        if not self.only_eval:
            src_rank_map = {
                "rollout_train": CommMapper.get_src_ranks(
                    batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                    src_world_size=self._component_placement.get_world_size("rollout"),
                    dst_world_size=self._component_placement.get_world_size("env"),
                    dst_rank=self._rank,
                ),
            }
            if self.cfg.get("reward", {}).get("use_reward_model", False):
                src_rank_map.update(
                    {
                        "reward_train": CommMapper.get_src_ranks(
                            batch_size=self.cfg.env.train.total_num_envs
                            // self.stage_num,
                            src_world_size=self._component_placement.get_world_size(
                                "reward"
                            ),
                            dst_world_size=self._component_placement.get_world_size(
                                "env"
                            ),
                            dst_rank=self._rank,
                        ),
                    }
                )
        if self.enable_eval:
            src_rank_map.update(
                {
                    "rollout_eval": CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size(
                            "rollout"
                        ),
                        dst_world_size=self._component_placement.get_world_size("env"),
                        dst_rank=self._rank,
                    ),
                }
            )
        return src_rank_map

    def _init_env(self):
        for i in range(self.stage_num):
            if self.cfg.env.train.auto_reset:
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))
                self.last_policy_info_list.append(
                    self._init_rlt_local_policy_state(i)
                )
            if self.enable_offload and hasattr(self.env_list[i], "offload"):
                self.env_list[i].offload()

    def _rlt_stage2_td3_enabled(self) -> bool:
        return (
            self.cfg.algorithm.get("loss_type", None) == "rlt_td3"
            and self.cfg.actor.model.get("model_type", None) == "rlt_stage2"
        )

    def _rlt_stage2_intervention_mode(self) -> str:
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        return str(intervention_cfg.get("mode", "local_correction"))

    def _rlt_stage2_intervention_enabled(self) -> bool:
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        return self._rlt_stage2_td3_enabled() and bool(
            intervention_cfg.get("enable", False)
        ) and self._rlt_stage2_intervention_mode() in {
            "local_correction",
            "human_override",
        }

    def _rlt_stage2_local_correction_enabled(self) -> bool:
        return (
            self._rlt_stage2_intervention_enabled()
            and self._rlt_stage2_intervention_mode() == "local_correction"
        )

    def _rlt_stage2_policy_info_enabled(self) -> bool:
        return self._rlt_stage2_local_correction_enabled()

    def _rlt_policy_env_type(self, mode: Literal["train", "eval"]) -> str:
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        return str(env_cfg.get("env_type", "")).lower()

    def _init_rlt_local_policy_state(
        self, stage_id: int, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, torch.Tensor] | None:
        if not self._rlt_stage2_policy_info_enabled():
            return None

        batch_size = (
            self.train_num_envs_per_stage
            if mode == "train"
            else self.eval_num_envs_per_stage
        )
        state = {
            "intervention_region": torch.zeros(batch_size, dtype=torch.bool),
            "expert_takeover": torch.zeros(batch_size, dtype=torch.bool),
            "deviation": torch.zeros(batch_size, dtype=torch.bool),
            "deviation_count": torch.zeros(batch_size, dtype=torch.int64),
            "takeover_left": torch.zeros(batch_size, dtype=torch.int64),
            "takeover_used": torch.zeros(batch_size, dtype=torch.int64),
            "prev_yz_error": torch.full((batch_size,), float("nan"), dtype=torch.float32),
            "prev_hole_x": torch.full((batch_size,), float("nan"), dtype=torch.float32),
        }
        states = (
            self.rlt_local_policy_state
            if mode == "train"
            else self.eval_rlt_local_policy_state
        )
        while len(states) <= stage_id:
            states.append({})
        states[stage_id] = state
        return self._export_rlt_local_policy_info(state)

    @staticmethod
    def _export_rlt_local_policy_info(
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            "expert_takeover": state["expert_takeover"][:, None],
            "deviation": state["deviation"][:, None],
            "deviation_count": state["deviation_count"].to(torch.float32)[:, None],
            "takeover_left": state["takeover_left"].to(torch.float32)[:, None],
            "takeover_used": state["takeover_used"].to(torch.float32)[:, None],
        }

    @staticmethod
    def _select_rlt_policy_source_info(
        infos: dict[str, Any],
        required_keys: list[str],
    ) -> dict[str, Any]:
        if all(key in infos for key in required_keys):
            return infos
        final_info = infos.get("final_info")
        if isinstance(final_info, dict) and all(
            key in final_info for key in required_keys
        ):
            return final_info
        missing = [key for key in required_keys if key not in infos]
        raise RuntimeError(
            "RLT intervention control is enabled, but ManiSkill info is missing "
            f"required keys {missing}. This usually means the env wrapper is not "
            "using the aligned peg-insertion info path."
        )

    @staticmethod
    def _unwrap_env(env: Any) -> Any:
        while hasattr(env, "env"):
            env = env.env
        return getattr(env, "unwrapped", env)

    @staticmethod
    def _lookup_rlt_policy_info_value(infos: dict[str, Any], key: str) -> Any:
        if key in infos:
            return infos[key]
        policy_info = infos.get("policy_info")
        if isinstance(policy_info, dict) and key in policy_info:
            return policy_info[key]
        final_info = infos.get("final_info")
        if isinstance(final_info, dict):
            if key in final_info:
                return final_info[key]
            final_policy_info = final_info.get("policy_info")
            if isinstance(final_policy_info, dict) and key in final_policy_info:
                return final_policy_info[key]
        return None

    @staticmethod
    def _coerce_rlt_bool_info(
        value: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() == 1:
            return torch.full(
                (batch_size,),
                bool(tensor.reshape(-1)[0].item()),
                dtype=torch.bool,
                device=device,
            )
        tensor = tensor.reshape(batch_size, -1)
        return tensor.to(torch.bool).any(dim=1)

    @staticmethod
    def _coerce_rlt_int_info(
        value: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.int64, device=device)
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() == 1:
            return torch.full(
                (batch_size,),
                int(tensor.reshape(-1)[0].item()),
                dtype=torch.int64,
                device=device,
            )
        tensor = tensor.reshape(batch_size, -1)
        return tensor[:, -1].to(torch.int64)

    def _update_rlt_generic_policy_state(
        self,
        infos: dict[str, Any],
        chunk_dones: torch.Tensor,
        stage_id: int,
        mode: Literal["train", "eval"],
    ) -> dict[str, torch.Tensor]:
        states = (
            self.rlt_local_policy_state
            if mode == "train"
            else self.eval_rlt_local_policy_state
        )
        state = states[stage_id]
        device = chunk_dones.device
        done_any = chunk_dones.any(dim=1).to(device)
        batch_size = int(done_any.shape[0])
        for key, value in state.items():
            state[key] = value.to(device)

        expert_takeover = self._coerce_rlt_bool_info(
            self._lookup_rlt_policy_info_value(infos, "expert_takeover"),
            batch_size=batch_size,
            device=device,
        )
        deviation = self._coerce_rlt_bool_info(
            self._lookup_rlt_policy_info_value(infos, "deviation"),
            batch_size=batch_size,
            device=device,
        )
        intervention_region = self._coerce_rlt_bool_info(
            self._lookup_rlt_policy_info_value(infos, "intervention_region"),
            batch_size=batch_size,
            device=device,
        )
        state["expert_takeover"] = torch.where(
            done_any,
            torch.zeros_like(expert_takeover),
            expert_takeover,
        )
        state["deviation"] = torch.where(done_any, torch.zeros_like(deviation), deviation)
        state["intervention_region"] = torch.where(
            done_any,
            torch.zeros_like(intervention_region),
            intervention_region,
        )
        for key in ("deviation_count", "takeover_left", "takeover_used"):
            state[key] = torch.where(
                done_any,
                torch.zeros_like(state[key]),
                self._coerce_rlt_int_info(
                    self._lookup_rlt_policy_info_value(infos, key),
                    batch_size=batch_size,
                    device=device,
                ),
            )
        state["prev_yz_error"] = torch.full_like(state["prev_yz_error"], float("nan"))
        state["prev_hole_x"] = torch.full_like(state["prev_hole_x"], float("nan"))
        return self._export_rlt_local_policy_info(state)

    def _update_rlt_local_policy_state(
        self,
        infos: dict[str, Any] | None,
        chunk_dones: torch.Tensor,
        stage_id: int,
        mode: Literal["train", "eval"] = "train",
    ) -> dict[str, torch.Tensor] | None:
        states = (
            self.rlt_local_policy_state
            if mode == "train"
            else self.eval_rlt_local_policy_state
        )
        if (
            not self._rlt_stage2_policy_info_enabled()
            or infos is None
            or not states
        ):
            return None

        if self._rlt_policy_env_type(mode) != "maniskill":
            return self._update_rlt_generic_policy_state(
                infos,
                chunk_dones,
                stage_id,
                mode,
            )

        required_keys = [
            "consecutive_grasp_current",
            "prealigned_current",
            "partial_insert_current",
            "success_current",
            "peg_head_goal_yz_dist",
            "peg_body_goal_yz_dist",
            "peg_head_hole_x",
            "peg_head_hole_abs_y",
            "peg_head_hole_abs_z",
        ]
        infos = self._select_rlt_policy_source_info(infos, required_keys)

        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        intervention_enabled = self._rlt_stage2_local_correction_enabled()
        state = states[stage_id]
        device = infos["peg_head_hole_x"].device
        if "intervention_region" not in state:
            state["intervention_region"] = torch.zeros(
                state["expert_takeover"].shape, dtype=torch.bool
            )
        for key, value in state.items():
            state[key] = value.to(device)

        done_any = chunk_dones.any(dim=1).to(device)

        success = infos["success_current"].to(torch.bool)
        grasp = infos["consecutive_grasp_current"].to(torch.bool)
        prealigned = infos["prealigned_current"].to(torch.bool)
        partial_insert = infos["partial_insert_current"].to(torch.bool)
        yz_error = torch.maximum(
            infos["peg_head_goal_yz_dist"].to(torch.float32),
            infos["peg_body_goal_yz_dist"].to(torch.float32),
        )
        hole_x = infos["peg_head_hole_x"].to(torch.float32)
        abs_y = infos["peg_head_hole_abs_y"].to(torch.float32)
        abs_z = infos["peg_head_hole_abs_z"].to(torch.float32)

        hole_radii = None
        env_list = self.env_list if mode == "train" else self.eval_env_list
        unwrapped = self._unwrap_env(env_list[stage_id])
        if hasattr(unwrapped, "box_hole_radii"):
            hole_radii = unwrapped.box_hole_radii.to(device, dtype=torch.float32)
        if hole_radii is None:
            fallback_hole_radius = intervention_cfg.get(
                "fallback_hole_radius",
                0.035,
            )
            hole_radii = torch.full_like(abs_y, float(fallback_hole_radius))

        intervention_entry = torch.zeros_like(success)
        intervention_hold = torch.zeros_like(success)
        if intervention_enabled:
            intervention_near_hole_x_min = float(
                intervention_cfg.get("near_hole_x_min", -0.05)
            )
            intervention_exit_hole_x_min = float(
                intervention_cfg.get("exit_hole_x_min", -0.12)
            )
            intervention_yz_margin = float(
                intervention_cfg.get("near_hole_yz_margin", 1.5)
            )
            intervention_yz = (
                (yz_error <= intervention_yz_margin * hole_radii)
                & (abs_y <= intervention_yz_margin * hole_radii)
                & (abs_z <= intervention_yz_margin * hole_radii)
            ) | prealigned | partial_insert
            intervention_near_hole = hole_x >= intervention_near_hole_x_min
            intervention_entry = (
                grasp & intervention_near_hole & intervention_yz & (~success)
            )
            intervention_hold = (
                state["intervention_region"]
                & (~success)
                & (hole_x >= intervention_exit_hole_x_min)
            )
            intervention_region = intervention_entry | intervention_hold
        else:
            intervention_region = torch.zeros_like(success)

        has_prev_yz = torch.isfinite(state["prev_yz_error"])
        has_prev_x = torch.isfinite(state["prev_hole_x"])
        progress_eps = float(intervention_cfg.get("progress_eps", 0.002))
        yz_error_eps = float(intervention_cfg.get("yz_error_eps", 0.002))
        safe_yz_margin = float(intervention_cfg.get("safe_yz_margin", 1.25))
        yz_worse = has_prev_yz & (yz_error > state["prev_yz_error"] + yz_error_eps)
        no_x_progress = has_prev_x & (hole_x <= state["prev_hole_x"] + progress_eps)
        safe_yz = (abs_y <= safe_yz_margin * hole_radii) & (
            abs_z <= safe_yz_margin * hole_radii
        )
        moved_away_from_hole = (
            has_prev_x
            & state["intervention_region"]
            & (hole_x < state["prev_hole_x"] - progress_eps)
        )
        lost_grasp = (~grasp) & state["intervention_region"]
        deviation = intervention_region & (
            yz_worse
            | no_x_progress
            | (~safe_yz)
            | lost_grasp
            | moved_away_from_hole
        )

        patience = int(intervention_cfg.get("deviation_patience", 2))
        state["deviation_count"] = torch.where(
            deviation,
            state["deviation_count"] + 1,
            torch.zeros_like(state["deviation_count"]),
        )

        takeover_chunks = int(intervention_cfg.get("takeover_chunks", 5))
        takeover_max_chunks = int(intervention_cfg.get("takeover_max_chunks", 10))
        if takeover_chunks <= 0 or takeover_max_chunks < takeover_chunks:
            raise ValueError(
                "algorithm.intervention must satisfy "
                "0 < takeover_chunks <= takeover_max_chunks, got "
                f"{takeover_chunks=} and {takeover_max_chunks=}."
            )

        previous_takeover = state["expert_takeover"]
        takeover_used_after_chunk = torch.where(
            previous_takeover,
            state["takeover_used"] + 1,
            state["takeover_used"],
        )
        recovered = intervention_region & grasp & safe_yz & (~deviation)
        keep_for_min_chunks = previous_takeover & (
            takeover_used_after_chunk < takeover_chunks
        )
        extend_until_recovered = (
            previous_takeover
            & (~recovered)
            & (takeover_used_after_chunk < takeover_max_chunks)
        )
        if mode == "train" and intervention_enabled:
            trigger = (
                intervention_region
                & (~previous_takeover)
                & (state["deviation_count"] >= patience)
            )
        else:
            trigger = torch.zeros_like(intervention_region)
        next_takeover = (trigger | keep_for_min_chunks | extend_until_recovered) & (
            ~success
        )
        next_takeover = torch.where(
            done_any, torch.zeros_like(next_takeover), next_takeover
        )
        released_takeover = previous_takeover & (~next_takeover)

        state["takeover_used"] = torch.where(
            trigger,
            torch.zeros_like(state["takeover_used"]),
            takeover_used_after_chunk,
        )
        state["takeover_used"] = torch.where(
            next_takeover,
            state["takeover_used"],
            torch.zeros_like(state["takeover_left"]),
        )
        remaining_to_min = torch.clamp(takeover_chunks - state["takeover_used"], min=0)
        remaining_to_max = torch.clamp(
            takeover_max_chunks - state["takeover_used"], min=0
        )
        state["takeover_left"] = torch.where(
            next_takeover,
            torch.where(
                state["takeover_used"] < takeover_chunks,
                remaining_to_min,
                remaining_to_max,
            ),
            torch.zeros_like(state["takeover_used"]),
        )
        state["expert_takeover"] = next_takeover
        state["intervention_region"] = torch.where(
            done_any, torch.zeros_like(intervention_region), intervention_region
        )
        state["deviation"] = torch.where(
            done_any, torch.zeros_like(deviation), deviation
        )
        state["deviation_count"] = torch.where(
            done_any | (~intervention_region) | trigger | released_takeover,
            torch.zeros_like(state["deviation_count"]),
            state["deviation_count"],
        )
        state["prev_yz_error"] = torch.where(
            state["intervention_region"],
            yz_error,
            torch.full_like(yz_error, float("nan")),
        )
        state["prev_hole_x"] = torch.where(
            state["intervention_region"],
            hole_x,
            torch.full_like(hole_x, float("nan")),
        )

        return self._export_rlt_local_policy_info(state)

    @staticmethod
    def _shape_str(value) -> str:
        return "None" if value is None else str(tuple(getattr(value, "shape", ())))

    def _validate_env_action_chunk(
        self,
        chunk_actions,
        *,
        mode: Literal["train", "eval"],
        expected_chunks: int,
    ) -> None:
        expected_action_dim = int(self.cfg.actor.model.action_dim)
        if (
            not hasattr(chunk_actions, "shape")
            or len(chunk_actions.shape) != 3
            or int(chunk_actions.shape[1]) != int(expected_chunks)
            or int(chunk_actions.shape[2]) != expected_action_dim
        ):
            raise ValueError(
                f"Invalid {mode} env action chunk shape before chunk_step: "
                f"expected [B, {expected_chunks}, {expected_action_dim}], got "
                f"{self._shape_str(chunk_actions)}. Refuse to execute actions; "
                "check actor.model.num_action_chunks/action_dim, "
                "env.eval.action_exec_chunks, policy_setup, and action preparation."
            )

    def _build_rlt_step_obs(
        self,
        start_obs: dict[str, Any] | None,
        obs_list,
    ) -> dict[str, Any] | None:
        if start_obs is None or not isinstance(obs_list, (list, tuple)) or not obs_list:
            return None

        stride = int(self.cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
        if stride <= 0:
            return None

        step_obs_list = [start_obs, *obs_list]
        offsets = self._rlt_sparse_step_obs_offsets(len(step_obs_list))
        step_obs: dict[str, Any] = {}
        batch_size = self._infer_obs_batch_size(step_obs_list[0])
        for key in step_obs_list[0].keys():
            if not offsets:
                continue
            values = [step_obs_list[offset].get(key, None) for offset in offsets]
            first_non_none = next((value for value in values if value is not None), None)
            if first_non_none is None:
                step_obs[key] = None
            elif isinstance(first_non_none, torch.Tensor):
                if any(value is None for value in values):
                    raise ValueError(
                        f"Inconsistent RLT step_obs key {key!r}: "
                        "tensor values contain None."
                    )
                values = [
                    value.to(first_non_none.device)
                    if value.device != first_non_none.device
                    else value
                    for value in values
                ]
                step_obs[key] = torch.stack(values, dim=0)
            elif isinstance(first_non_none, list):
                step_obs[key] = values
            else:
                step_obs[key] = values
        step_obs["_rlt_step_offsets"] = torch.tensor(
            offsets,
            dtype=torch.long,
        )[:, None].expand(len(offsets), batch_size).contiguous()
        return step_obs

    def _rlt_sparse_step_obs_offsets(self, step_count: int) -> list[int]:
        stride = int(self.cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        if stride <= 0 or chunk_len <= 0:
            return []

        offsets = set()
        offset = 0
        while True:
            offset = (offset + stride) % chunk_len
            if offset == 0 or offset in offsets:
                break
            if offset < step_count:
                offsets.add(offset)
        return sorted(offsets)

    @staticmethod
    def _infer_obs_batch_size(obs: dict[str, Any]) -> int:
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer RLT step_obs batch size from observation.")

    @staticmethod
    def _is_rlt_stage2_td3_cfg(cfg) -> bool:
        return (
            cfg.algorithm.get("loss_type", None) == "rlt_td3"
            and cfg.actor.model.get("model_type", None) == "rlt_stage2"
        )

    def _append_rlt_step_trace_to_previous_action(
        self,
        stage_id: int,
        rollout_result: RolloutResult,
    ) -> None:
        if not self._is_rlt_stage2_td3_cfg(self.cfg):
            return
        if rollout_result.rlt_step_trace:
            self.rollout_results[stage_id].append_rlt_step_trace(
                rollout_result.rlt_step_trace
            )

    def _build_chunk_step_result(
        self,
        rollout_result: RolloutResult,
        env_output: EnvOutput,
        rewards: torch.Tensor | None,
        *,
        final_forward_inputs: dict[str, Any] | None = None,
        include_action: bool = True,
    ) -> ChunkStepResult:
        forward_inputs = (
            rollout_result.forward_inputs
            if final_forward_inputs is None
            else final_forward_inputs
        )
        return ChunkStepResult(
            actions=(
                rollout_result.forward_inputs.get("action", None)
                if include_action
                else None
            ),
            prev_logprobs=(
                rollout_result.prev_logprobs if self.collect_prev_infos else None
            ),
            prev_values=rollout_result.prev_values if self.collect_prev_infos else None,
            forward_inputs=forward_inputs,
            versions=rollout_result.versions,
            dones=env_output.dones,
            truncations=env_output.truncations,
            terminations=env_output.terminations,
            rewards=rewards,
        )

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self,
        chunk_actions: torch.Tensor,
        stage_id: int,
        start_obs: dict[str, Any] | None = None,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        self._validate_env_action_chunk(
            chunk_actions,
            mode="train",
            expected_chunks=int(self.cfg.actor.model.num_action_chunks),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
            step_obs = (
                self._build_rlt_step_obs(start_obs, obs_list)
                if self._is_rlt_stage2_td3_cfg(self.cfg)
                else None
            )
        else:
            step_obs = None
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
                    if isinstance(infos, dict) and "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if isinstance(infos, dict) and "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if isinstance(infos, dict) and "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][
                        chunk_dones.any(dim=1)
                    ].cpu()

        intervene_actions = (
            infos["intervene_action"]
            if isinstance(infos, dict) and "intervene_action" in infos
            else None
        )
        intervene_flags = (
            infos["intervene_flag"]
            if isinstance(infos, dict) and "intervene_flag" in infos
            else None
        )
        if (
            self.cfg.env.train.auto_reset
            and chunk_dones.any()
            and isinstance(infos, dict)
            and "final_info" in infos
        ):
            final_info = infos["final_info"]
            if "intervene_action" in final_info:
                intervene_actions = final_info["intervene_action"]
                intervene_flags = final_info["intervene_flag"]

        policy_info = self._update_rlt_local_policy_state(infos, chunk_dones, stage_id)
        if policy_info is not None:
            env_info["deviation_rate"] = (
                policy_info["deviation"].float().mean().reshape(1).cpu()
            )

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            step_obs=step_obs,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
            policy_info=policy_info,
        )
        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        if chunk_actions.shape[1] < self.eval_action_exec_chunks:
            raise ValueError(
                f"Policy produced only {chunk_actions.shape[1]} action steps, "
                f"but env.eval.action_exec_chunks={self.eval_action_exec_chunks}."
            )
        chunk_actions = chunk_actions[:, : self.eval_action_exec_chunks]
        self._validate_env_action_chunk(
            chunk_actions,
            mode="eval",
            expected_chunks=self.eval_action_exec_chunks,
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
            if isinstance(infos, dict) and "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif isinstance(infos, dict) and "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        policy_info = self._update_rlt_local_policy_state(
            infos, chunk_dones, stage_id, mode="eval"
        )
        if policy_info is not None:
            policy_info["expert_takeover"] = torch.zeros_like(
                policy_info["expert_takeover"]
            )
            self.eval_policy_info_list[stage_id] = policy_info
            env_info["deviation_rate"] = (
                policy_info["deviation"].float().mean().reshape(1).cpu()
            )

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            policy_info=policy_info,
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

    @Worker.timer("env/recv_actions")
    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        """Receive and merge chunked actions for the current env worker.

        The method fetches one action shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env action chunks.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            Concatenated action chunk array with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    @Worker.timer("recv_rollout_results")
    def recv_rollout_results(
        self, input_channel: Channel, mode="train"
    ) -> RolloutResult:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        rollout_results: list[RolloutResult] = []

        def _infer_rollout_batch_size(rollout_result: RolloutResult) -> int:
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(rollout_result, field_name, None)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            if rollout_result.forward_inputs:
                first_tensor = next(iter(rollout_result.forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return first_tensor.shape[0]
            raise ValueError("Cannot infer batch size from rollout result.")

        for src_rank, expected_size in src_ranks_and_sizes:
            rollout_result = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_rollout_results"
                ),
            )

            actual_size = _infer_rollout_batch_size(rollout_result)
            assert actual_size == expected_size, (
                f"Expected rollout result size {expected_size} from rollout rank {src_rank}, "
                f"got batch size {actual_size}."
            )

            rollout_results.append(rollout_result)

        return RolloutResult.merge_rollout_results(rollout_results)

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
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                if (
                    not self.cfg.env.eval.auto_reset
                    and not self.cfg.env.eval.use_fixed_reset_state_ids
                ):
                    self.eval_env_list[i].update_reset_state_ids()

    @Worker.timer("env/send_obs")
    def send_env_batch(
        self,
        rollout_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            rollout_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline stage.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_rank_map[f"rollout_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batch = dict(env_batch)
        step_obs = env_batch.pop("step_obs", None)
        env_batches = split_dict(env_batch, split_sizes)
        step_obs_batches = self._split_rlt_step_obs(step_obs, split_sizes)
        for env_batch_i, step_obs_i in zip(
            env_batches, step_obs_batches, strict=True
        ):
            env_batch_i["step_obs"] = step_obs_i
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            rollout_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    @staticmethod
    def _split_rlt_step_obs(
        step_obs: dict[str, Any] | None,
        split_sizes: list[int],
    ) -> list[dict[str, Any] | None]:
        if step_obs is None:
            return [None for _ in split_sizes]

        def split_value(value: Any) -> list[Any]:
            if isinstance(value, torch.Tensor):
                return [
                    split_value.contiguous()
                    for split_value in torch.split(value, split_sizes, dim=1)
                ]
            if isinstance(value, list):
                split_values: list[Any] = []
                begin = 0
                for size in split_sizes:
                    split_values.append(
                        [step_values[begin : begin + size] for step_values in value]
                    )
                    begin += size
                return split_values
            if isinstance(value, dict):
                split_dicts: list[dict[str, Any]] = [
                    {} for _ in range(len(split_sizes))
                ]
                for sub_key, sub_value in value.items():
                    sub_splits = split_value(sub_value)
                    for idx, sub_split in enumerate(sub_splits):
                        split_dicts[idx][sub_key] = sub_split
                return split_dicts
            return [value for _ in split_sizes]

        step_obs_batches: list[dict[str, Any]] = [{} for _ in split_sizes]
        for key, value in step_obs.items():
            value_splits = split_value(value)
            for idx, value_split in enumerate(value_splits):
                step_obs_batches[idx][key] = value_split
        return step_obs_batches

    def send_reward_input(
        self,
        send_channel: Channel,
        reward_input: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ):
        dst_ranks_and_sizes = self.dst_rank_map[f"reward_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        reward_input_batches = split_dict(reward_input, split_sizes)
        for (rank, _), reward_input_i in zip(dst_ranks_and_sizes, reward_input_batches):
            send_channel.put(
                item=reward_input_i,
                key=CommMapper.build_channel_key(
                    self._rank, rank, extra=f"{mode}_reward_input"
                ),
                async_op=True,
            )

    @Worker.timer("recv_reward_results")
    def recv_reward_results(self, recv_channel: Channel) -> torch.Tensor:
        reward_results: list[torch.Tensor] = []
        src_ranks_and_sizes = self.src_rank_map["reward_train"]
        for src_rank, expected_size in src_ranks_and_sizes:
            rewards = recv_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra="reward_output"
                ),
            )
            if rewards is None:
                rewards = torch.zeros(expected_size, dtype=torch.float32)
            actual_size = rewards.shape[0]
            assert actual_size == expected_size, (
                f"Expected reward result size {expected_size} from reward rank {src_rank}, "
                f"got batch size {actual_size}."
            )
            reward_results.append(rewards)
        return torch.cat(reward_results, dim=0)

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
        self.send_reward_input(send_channel=send_channel, reward_input=reward_input)
        reward_output = self.recv_reward_results(recv_channel=recv_channel)
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
        rollout_rewards = self.rollout_results[stage_id].rewards
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
                .repeat(1, self.cfg.actor.model.num_action_chunks)
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
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    intervene_actions=None,
                    intervene_flags=None,
                    policy_info=self._init_rlt_local_policy_state(stage_id),
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                    policy_info=self.last_policy_info_list[stage_id],
                )
                env_outputs.append(env_output)

        return env_outputs

    def _send_train_bootstrap(
        self, rollout_channel: Channel, env_outputs: list[EnvOutput]
    ) -> None:
        for stage_id in range(self.stage_num):
            env_output: EnvOutput = env_outputs[stage_id]
            env_batch = env_output.to_dict()
            self.send_env_batch(
                rollout_channel,
                {
                    "obs": env_batch["obs"],
                    "final_obs": env_batch["final_obs"],
                    "step_obs": env_batch["step_obs"],
                    "policy_info": env_batch["policy_info"],
                },
            )

    def _bootstrap_and_send_train(self, rollout_channel: Channel) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(rollout_channel, env_outputs)
        return env_outputs

    def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel
        )

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any], epoch: int
    ):
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]
        self.last_policy_info_list = [
            env_output.policy_info for env_output in env_output_list
        ]

    @Worker.timer("env/send_rollout_trajectories")
    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: list[Trajectory] = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        rollout_result.clear()
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)
        del trajectories
        gc.collect()

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
            )
            for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)

        for epoch in range(self.rollout_epoch):
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            else:
                env_outputs = self._bootstrap_and_send_train(rollout_channel)

            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

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

                    rollout_result = self.recv_rollout_results(
                        input_channel, mode="train"
                    )
                    intervention_flags = rollout_result.forward_inputs.get(
                        "intervention_flags", None
                    )
                    if intervention_flags is not None:
                        env_metrics["expert_intervention_actual_rate"].append(
                            intervention_flags.detach().float().reshape(-1).cpu()
                        )
                    intervention_requested = rollout_result.forward_inputs.get(
                        "intervention_requested", None
                    )
                    if intervention_requested is not None:
                        env_metrics["expert_intervention_requested_rate"].append(
                            intervention_requested.detach().float().reshape(-1).cpu()
                        )
                    ready_for_online = rollout_result.forward_inputs.get(
                        "ready_for_online", None
                    )
                    if ready_for_online is not None:
                        env_metrics["rlt_ready_for_online"].append(
                            ready_for_online.detach().float().reshape(-1).cpu()
                        )
                    student_control = rollout_result.forward_inputs.get(
                        "student_control", None
                    )
                    if student_control is not None:
                        env_metrics["student_control_rate"].append(
                            student_control.detach().float().reshape(-1).cpu()
                        )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    self._append_rlt_step_trace_to_previous_action(
                        stage_id, rollout_result
                    )
                    chunk_step_result = self._build_chunk_step_result(
                        rollout_result,
                        env_output,
                        rewards,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    if (
                        self.reward_mode == "history_buffer"
                        and self.history_reward_assign
                        and reward_model_output is not None
                    ):
                        self.assign_history_reward(stage_id, reward_model_output)
                    if rollout_result.save_flags is not None:
                        self.rollout_results[stage_id].mark_last_step_with_flags(
                            rollout_result.save_flags
                        )

                    env_output, env_info = self.env_interact_step(
                        rollout_result.actions, stage_id, start_obs=curr_obs
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                            "step_obs": env_batch["step_obs"],
                            "policy_info": env_batch["policy_info"],
                        },
                    )
                    if self.collect_transitions:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

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
                rollout_result = self.recv_rollout_results(input_channel, mode="train")
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values, reward_model_output
                )
                self._append_rlt_step_trace_to_previous_action(
                    stage_id, rollout_result
                )
                final_forward_inputs = (
                    rollout_result.forward_inputs
                    if self._is_rlt_stage2_td3_cfg(self.cfg)
                    else {}
                )
                chunk_step_result = self._build_chunk_step_result(
                    rollout_result,
                    env_output,
                    rewards,
                    final_forward_inputs=final_forward_inputs,
                    include_action=False,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)
                if (
                    self.reward_mode == "history_buffer"
                    and self.history_reward_assign
                    and reward_model_output is not None
                ):
                    self.assign_history_reward(stage_id, reward_model_output)

            if self.use_training_pipeline and actor_channel is not None:
                for stage_id in range(self.stage_num):
                    await self.send_rollout_trajectories_pipeline(
                        self.rollout_results[stage_id], actor_channel
                    )
                self.rollout_results: list[EmbodiedRolloutResult] = [
                    EmbodiedRolloutResult(
                        max_episode_length=self.cfg.env.train.max_episode_steps,
                    )
                    for _ in range(self.stage_num)
                ]

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if not self.use_training_pipeline and actor_channel is not None:
            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )
            # reduce memory peak
            self.rollout_results = []
            gc.collect()

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
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
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
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                            "step_obs": env_batch["step_obs"],
                            "policy_info": env_batch["policy_info"],
                        },
                        mode="eval",
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch
                            == self.cfg.algorithm.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                            "step_obs": env_batch["step_obs"],
                            "policy_info": env_batch["policy_info"],
                        },
                        mode="eval",
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

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
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": rollout_batch["rewards"],
            "dones": rollout_batch["dones"],
            "values": rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": rollout_batch.get("loss_mask", None),
            "loss_mask_sum": rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": self.cfg.algorithm.get(
                "normalize_advantages", True
            ),
        }
        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        return rollout_batch

    def prepare_micro_batches(
        self, trajectory: Trajectory
    ) -> list[dict[str, torch.Tensor]]:
        # In training pipeline mode, send ready-to-train actor micro-batches instead of
        # full rollout trajectories. This keeps nested observations out of the channel
        # payload so packed tensors can use the channel tensor fast path.
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

        batch = self.compute_advantages_and_returns(batch)

        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        flatten_batch = flatten_embodied_batch(batch, torch.arange(batch_size))
        micro_batch_size = self.cfg.actor.micro_batch_size
        assert batch_size % micro_batch_size == 0, (
            f"Batch size {batch_size} is not divisible by micro_batch_size {micro_batch_size}."
        )
        num_micro_batches = batch_size // micro_batch_size
        micro_batches = split_dict_to_chunk(flatten_batch, num_micro_batches, dim=0)
        return [pack_batch(micro_batch) for micro_batch in micro_batches]

    async def send_rollout_trajectories_pipeline(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ) -> None:
        trajectories: list[Trajectory] = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        for trajectory in trajectories:
            with self.worker_timer("prepare_micro_batches"):
                micro_batches = self.prepare_micro_batches(trajectory)
                for micro_batch in micro_batches:
                    channel.put(micro_batch, async_op=True)
