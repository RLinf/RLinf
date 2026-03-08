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

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.placement import HybridComponentPlacement


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
        self.rollout_epoch = self.cfg.algorithm.get("rollout_epoch", 1)
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        # Env configurations
        self.enable_offload = self.cfg.env.train.get("enable_offload", False)
        # VLM planner integration (optional stage-aware subtask updates).
        # When subtask_interval > 0 the env worker calls the VLM planner every
        # subtask_interval chunk steps and updates env.task_description.
        self._subtask_interval: int = int(self.cfg.env.train.get("subtask_interval", 0))
        self._steps_since_subtask_update: int = 0
        self._vlm_planner = None  # set via set_vlm_planner() after construction

        # TOPReward dense reward state
        self._top_reward_enabled: bool = bool(
            self.cfg.env.train.get("top_reward_enabled", False)
        )
        self._top_reward_max_frames: int = int(
            self.cfg.env.train.get("top_reward_max_frames", 16)
        )
        self._episode_frames: list[np.ndarray] = []
        self._prev_top_score: float = 0.0
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

    def init_worker(self):
        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }

        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num
            )
        self.log_info(f"Env worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Env worker initialized with src_ranks: {self.src_ranks}")
        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        if not self.only_eval:
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
        if self.enable_eval:
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )

        if not self.only_eval:
            self._init_env()

    def set_vlm_planner(self, planner_handle) -> None:
        """Inject a VLMPlannerWorker Ray handle for stage-aware subtask updates.

        Call this from the runner after both workers have been created, passing
        the Ray remote handle (e.g. ``vlm_planner_actor``).  When
        ``subtask_interval > 0`` and this handle is set, the env worker will
        call ``planner_handle.get_next_subtask.remote()`` every
        ``subtask_interval`` chunk steps and update ``env.task_description``.

        Args:
            planner_handle: A VLMPlannerWorker Ray actor handle, or None to
                disable VLM-driven subtask updates.
        """
        self._vlm_planner = planner_handle
        self.log_info(
            f"[EnvWorker] VLM planner handle set (subtask_interval="
            f"{self._subtask_interval})."
        )

    def _maybe_update_subtask(self, stage_id: int) -> None:
        """Optionally call the VLM planner to refresh the current subtask.

        Called each chunk step from :meth:`interact`.  When
        ``subtask_interval > 0`` and a VLM planner handle is available, polls
        the planner and writes the new subtask description into the env so that
        the next observation includes the updated ``task_descriptions`` text.

        Args:
            stage_id: Pipeline stage index (used to select the correct env
                from ``self.env_list``).
        """
        if self._subtask_interval <= 0 or self._vlm_planner is None:
            return

        self._steps_since_subtask_update += 1
        if self._steps_since_subtask_update < self._subtask_interval:
            return

        self._steps_since_subtask_update = 0
        env = self.env_list[stage_id]

        # Collect the most recent observation image for the planner.
        import ray

        obs = getattr(env, "last_obs", None) or {}
        images = []
        main_images = obs.get("main_images", None)
        if main_images is not None:
            if isinstance(main_images, torch.Tensor):
                main_images = main_images.numpy()
            # main_images shape: (B, H, W, 3) or (1, H, W, 3); take first frame.
            if main_images.ndim == 4:
                images.append(main_images[0])
            else:
                images.append(main_images)

        memory_ref = self._vlm_planner.get_memory_text.remote()
        memory = ray.get(memory_ref)

        subtask_ref = self._vlm_planner.get_next_subtask.remote(images, memory)
        new_subtask: str = ray.get(subtask_ref)

        if new_subtask and hasattr(env, "task_description"):
            env.task_description = new_subtask
            self.log_info(
                f"[EnvWorker] Subtask updated for stage {stage_id}: '{new_subtask}'"
            )

    def _compute_top_reward(self, env_output: EnvOutput, stage_id: int) -> EnvOutput:
        """Compute TOPReward dense progress reward and inject into env_output.

        Extracts the current frame, queries the VLM planner for a progress
        score, and writes ``score_t - score_{t-1}`` into the reward tensor.

        Args:
            env_output: The env output from ``env_interact_step``.
            stage_id: Pipeline stage index.

        Returns:
            Modified env_output with dense reward injected.
        """
        if not self._top_reward_enabled or self._vlm_planner is None:
            return env_output

        # Extract current frame from observation.
        main_images = env_output.obs.get("main_images", None)
        if main_images is not None:
            if isinstance(main_images, torch.Tensor):
                frame = main_images[0].cpu().numpy()  # (H, W, 3)
            else:
                frame = np.asarray(main_images[0])
            self._episode_frames.append(frame)

        # Trim to max frames.
        if len(self._episode_frames) > self._top_reward_max_frames:
            self._episode_frames = self._episode_frames[-self._top_reward_max_frames :]

        # Get instruction from env.
        env = self.env_list[stage_id]
        instruction = getattr(env, "task_description", "")
        if hasattr(env, "env") and hasattr(env.env, "task_description"):
            instruction = env.env.task_description

        score_ref = self._vlm_planner.compute_top_reward.remote(
            self._episode_frames, instruction
        )
        score_t = ray.get(score_ref)

        reward = float(score_t) - self._prev_top_score
        self._prev_top_score = float(score_t)

        # Inject reward into the last chunk position.
        if env_output.rewards is not None:
            env_output.rewards[:, -1] = reward
        self.log_info(f"[EnvWorker] TOPReward: score={score_t:.4f}, delta={reward:.4f}")

        return env_output

    def _reset_top_reward_state(self) -> None:
        """Reset TOPReward episode state for a new episode."""
        self._episode_frames = []
        self._prev_top_score = 0.0

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
                    stats_sample_ratio=getattr(
                        env_cfg.data_collection, "stats_sample_ratio", 0.1
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute rollout peer ranks for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(rollout_rank, batch_size)`` tuples this env worker should send
            env outputs to.
        """
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute rollout source ranks and sizes for receiving action chunks."""
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            dst_rank=self._rank,
        )

    def _init_env(self):
        if self.cfg.env.train.auto_reset:
            for i in range(self.stage_num):
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))

                if self.enable_offload and hasattr(self.env_list[i], "offload"):
                    self.env_list[i].offload()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
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
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
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
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )

        # Inject TOPReward dense reward if enabled.
        env_output = self._compute_top_reward(env_output, stage_id)

        # Reset TOPReward state on episode done.
        if self._top_reward_enabled and chunk_dones[:, -1].any():
            self._reset_top_reward_state()

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
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
        )
        return env_output, env_info

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
        src_ranks_and_sizes = self.src_ranks[mode]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(src_rank, self._rank, extra=mode),
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
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    def split_env_batch(
        self,
        env_batch: dict[str, Any],
        sizes: list[int],
        mode: Literal["train", "eval"],
    ) -> list[dict[str, Any]]:
        """Split one env batch dict into size-specified sub-batches along dim-0.

        Tensor values are chunked on dim-0; list values are sliced proportionally;
        nested dict values are split recursively.

        Args:
            env_batch: Env output dictionary produced by ``EnvOutput.to_dict``.
            sizes: Batch sizes for each destination rank.
            mode: Rollout mode used for list-length validation.

        Returns:
            A list of split env batches, one item per destination rank.
        """
        count = len(sizes)
        total_size = sum(sizes)
        splitted_env_batches = [{} for _ in range(count)]
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == total_size, (
                    f"Tensor field '{key}' expected batch size {total_size}, got {value.shape[0]}."
                )
                splitted_values = torch.split(value, sizes, dim=0)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_values[i].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage} "
                        f"(train_num_envs_per_stage), got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage} "
                        f"(eval_num_envs_per_stage), got {length}"
                    )
                assert length == total_size, (
                    f"List field '{key}' expected length {total_size}, got {length}."
                )
                begin = 0
                for i, size in enumerate(sizes):
                    splitted_env_batches[i][key] = value[begin : begin + size]
                    begin += size
            elif isinstance(value, dict):
                splitted_sub_batches = self.split_env_batch(value, sizes, mode)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_sub_batches[i]
            else:
                for i in range(count):
                    splitted_env_batches[i][key] = value

        return splitted_env_batches

    def send_env_batch(
        self,
        output_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            output_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline stage.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = self.split_env_batch(env_batch, split_sizes, mode)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            output_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=mode),
            )

    def bootstrap_step(self) -> list[EnvOutput]:
        # Reset subtask counter at the start of each rollout epoch.
        self._steps_since_subtask_update = 0
        # Reset TOPReward episode state for new rollout epoch.
        if self._top_reward_enabled:
            self._reset_top_reward_state()

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
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
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
                )
                env_outputs.append(env_output)

        return env_outputs

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

    @Worker.timer("interact")
    def interact(self, input_channel: Channel, output_channel: Channel):
        env_metrics = defaultdict(list)
        for epoch in range(self.rollout_epoch):
            env_outputs = self.bootstrap_step()
            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_outputs[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(input_channel)
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self._maybe_update_subtask(stage_id)
                    self.send_env_batch(output_channel, env_output.to_dict())
                    env_outputs[stage_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        eval_metrics = defaultdict(list)

        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            for stage_id in range(self.stage_num):
                self.eval_env_list[stage_id].is_start = True
                extracted_obs, infos = self.eval_env_list[stage_id].reset()
                env_output = EnvOutput(
                    obs=extracted_obs,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                )
                self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

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
                    if eval_step == self.n_eval_chunk_steps - 1:
                        continue
                    self.send_env_batch(
                        output_channel, env_output.to_dict(), mode="eval"
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
