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

import copy
import json
import os
from typing import Optional, Union

import gym
import habitat
import numpy as np
import torch
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat_baselines.config.default import get_config
from hydra.core.global_hydra import GlobalHydra

from rlinf.envs.habitat.extensions import measures
from rlinf.envs.habitat.extensions.utils import (
    observations_to_image,
    resize_observation_images,
)
from rlinf.envs.habitat.venv import HabitatRLEnv, ReconfigureSubprocEnv
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    save_rollout_video,
    to_tensor,
)

measures.pass_format_check()


@registry.register_task_action
class NoOpAction(SimulatorTaskAction):
    """Register manually defined No-operation action for habitat env."""

    def step(self, *args, **kwargs):
        return self._sim.get_sensor_observations()


class HabitatEnv(gym.Env):
    def __init__(
        self, cfg, num_envs, seed_offset, total_num_processes, worker_info=None
    ):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = cfg.max_steps_per_rollout_epoch
        self.ignore_terminations = cfg.ignore_terminations
        self.dones_once = np.zeros(self.num_envs, dtype=bool)
        self.record_first_done_infos = None

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)

        self._init_env()

        self.metrics_cfg = cfg.metrics_cfg
        self.video_cfg = cfg.video_cfg
        self.render_images = {}
        self.current_raw_obs = None

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]

        # Truncate chunk if it contains "stop" and pad with "no_op"
        for env_idx, chunk_action in enumerate(chunk_actions):
            stop_idx = np.where(chunk_action == "stop")[0]
            if len(stop_idx) > 0:
                stop_idx = stop_idx[0] + 1
                truncated_chunk = chunk_action[:stop_idx].copy()
                chunk_actions[env_idx] = np.concatenate(
                    [truncated_chunk, ["no_op"] * (chunk_size - len(truncated_chunk))]
                )

        # Truncate chunk if it would exceed max_episode_steps and pad with "no_op"
        for env_idx, elapsed_step in enumerate(self.elapsed_steps):
            if elapsed_step + chunk_size >= self.max_episode_steps:
                reserved_idx = self.max_episode_steps - elapsed_step
                truncated_chunk = chunk_actions[env_idx][:reserved_idx].copy()
                truncated_chunk[reserved_idx - 1] = "stop"
                chunk_actions[env_idx] = np.concatenate(
                    [
                        truncated_chunk,
                        ["no_op"] * (chunk_size - len(truncated_chunk)),
                    ]
                )

        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        # [num_envs, chunk_steps]
        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = raw_chunk_terminations.any(dim=1)

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = raw_chunk_truncations.any(dim=1)
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def step(self, actions=None):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        self._elapsed_steps += 1

        # After excuting "stop" action, habitat env needs reset to process the next action
        # Replace "stop" with "no_op" before stepping the underlying env
        # to avoid unable to process the next action.
        actions = actions.astype("U20")
        is_stop = actions == "stop"
        actions[is_stop] = "no_op"

        raw_obs, _reward, terminations, info_lists = self.env.step(actions)

        # If some envs execute "no_op", manually normalize depth observations
        # according to Habitat's depth sensor config.
        self._normalize_depth(actions, raw_obs)

        terminations[is_stop] = True
        # TODO: what if termination means failure? (e.g. robot falling down)
        step_reward = self._calc_step_reward(terminations)
        infos = list_of_dict_to_dict_of_list(info_lists)
        infos = self._record_metrics(infos)

        truncations = self.elapsed_steps >= self.max_episode_steps
        dones_for_metric_save = terminations | truncations
        # Only save episode metrics once: at the first time an env becomes done.
        metric_save_masks = dones_for_metric_save & (~self.dones_once)
        if metric_save_masks.any():
            self._save_metrics(infos, metric_save_masks)

        self.current_raw_obs = raw_obs
        obs = self._wrap_obs(raw_obs)

        if self.video_cfg.save_video:
            episode_ids = self.env.get_current_episode_ids()
            for i in range(len(raw_obs)):
                frame = observations_to_image(raw_obs[i], info_lists[i])
                frame = resize_observation_images(frame, frame["rgb"].shape[0])
                frame_concat = np.concatenate(
                    (frame["rgb"], frame["depth"], frame["top_down_map"]), axis=1
                )
                key = f"episode_{episode_ids[i]}"
                if key not in self.render_images:
                    self.render_images[key] = []
                self.render_images[key].append(frame_concat)

        if self.ignore_terminations:
            terminations[:] = False
        dones = terminations | truncations

        if dones.any() and self.auto_reset:
            if self.video_cfg.save_video:
                self.flush_video(dones=dones)

            final_infos = (
                self.record_first_done_infos
                if self.record_first_done_infos is not None
                else infos
            )
            obs, infos = self._handle_auto_reset(dones, obs, final_infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        raw_obs = self.env.reset(env_idx)
        self._elapsed_steps[env_idx] = 0
        self.dones_once[env_idx] = False
        if (
            self.record_first_done_infos is not None
            and "episode" in self.record_first_done_infos
        ):
            episode = self.record_first_done_infos["episode"]
            # Construct a mask to mark the envs in this reset.
            device = next(iter(episode.values())).device
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            mask[env_idx] = True
            for k, v in episode.items():
                v[mask] = torch.zeros_like(v)
        infos = {}

        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs

        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]
        obs = self._wrap_obs(self.current_raw_obs)

        return obs, infos

    def flush_video(
        self, video_sub_dir: Optional[str] = None, dones: Optional[np.ndarray] = None
    ):
        output_dir = self.video_cfg.video_base_dir
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        if dones is None:
            dones_episode_ids = np.array(self.env.get_current_episode_ids())
        else:
            dones_episode_ids = np.array(self.env.get_current_episode_ids())[dones]

        for episode_ids in dones_episode_ids:
            video_name = f"episode_{episode_ids}"
            if video_name in self.render_images:
                save_rollout_video(
                    self.render_images[video_name],
                    output_dir=output_dir,
                    video_name=video_name,
                    fps=self.video_cfg.fps,
                )
                self.render_images[video_name] = []

    def update_reset_state_ids(self):
        pass

    def _normalize_depth(self, actions, raw_obs):
        """Normalize depth for envs whose action is 'no_op', following
        Habitat's depth sensor configuration.
        """
        is_no_op = actions == "no_op"
        if not np.any(is_no_op):
            return

        env_config = self.env.get_env_attr("config")[0]
        depth_cfg = env_config.simulator.agents.main_agent.sim_sensors.depth_sensor
        if not getattr(depth_cfg, "normalize_depth", False):
            return

        min_depth = float(depth_cfg.min_depth)
        max_depth = float(depth_cfg.max_depth)

        for env_idx, flag_no_op in enumerate(is_no_op):
            if not flag_no_op:
                continue
            obs = raw_obs[env_idx]
            if "depth" not in obs:
                continue
            depth = obs["depth"]
            depth = np.clip(depth, min_depth, max_depth)
            depth = (depth - min_depth) / (max_depth - min_depth)
            obs["depth"] = depth
            raw_obs[env_idx] = obs

    def _wrap_obs(self, obs_list):
        image_list = []
        task_descs = []
        token_list = []
        task_descs = []
        for obs in obs_list:
            image_list.append(observations_to_image(obs))
            # VLN-CE observations usually carry instruction in one of these fields.
            inst = ""
            token = []
            if isinstance(obs, dict):
                if "instruction" in obs and isinstance(obs["instruction"], dict):
                    inst = str(obs["instruction"].get("text", ""))
                elif "instruction_text" in obs:
                    inst = str(obs["instruction_text"])
                elif "text" in obs:
                    inst = str(obs["text"])
                token = obs["instruction"].get("tokens", [])
            task_descs.append(inst)
            token_list.append(token)

        image_tensor = to_tensor(list_of_dict_to_dict_of_list(image_list))
        episode_ids = self.env.get_current_episode_ids()

        obs = {}
        obs["main_images"] = image_tensor["rgb"].clone()  # [N_ENV, H, W, C]
        obs["task_descriptions"] = task_descs
        obs["states"] = torch.tensor(episode_ids, dtype=torch.int64)
        obs["wrist_images"] = token_list

        if "depth" in image_tensor:
            depth_tensor = image_tensor["depth"].clone()
            obs["extra_view_images"] = depth_tensor.unsqueeze(1)  # [N_ENV, 1, H, W, C]

        return obs

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(env_idx=env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _record_metrics(self, infos):
        episode_info = {}
        episode_info["distance_to_goal"] = infos["distance_to_goal"].copy()
        episode_info["success"] = infos["success"].copy()
        episode_info["spl"] = infos["spl"].copy()
        episode_info["trajectory_Length"] = infos["trajectory_Length"].copy()
        episode_info["oracle_success"] = infos["oracle_success"].copy()
        episode_info["oracle_navigation_error"] = infos[
            "oracle_navigation_error"
        ].copy()
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _save_metrics(self, infos, metric_save_masks):
        """Save metrics by episode_id when env first done."""
        mask = torch.from_numpy(metric_save_masks)  # [num_envs]
        self.dones_once[metric_save_masks] = True
        episode = infos["episode"]

        if self.record_first_done_infos is None:
            self.record_first_done_infos = {
                "episode": {k: torch.zeros_like(v) for k, v in episode.items()}
            }

        # Update the envs that become done in this step
        for k, v in episode.items():
            cached_v = self.record_first_done_infos["episode"][k]
            m = mask.to(v.device)
            cached_v[m] = v[m]

        # Save metrics by episode_id when env first done
        episode_ids = self.env.get_current_episode_ids()
        for i in range(len(metric_save_masks)):
            if metric_save_masks[i]:
                episode_id = episode_ids[i]
                metrics_dict = {}
                for k, v in episode.items():
                    if torch.is_tensor(v):
                        if v.dim() == 1:
                            metrics_dict[k] = v[i].item()
                        else:
                            metrics_dict[k] = v[i].cpu().numpy().tolist()
                    elif isinstance(v, (list, np.ndarray)):
                        metrics_dict[k] = v[i]
                    else:
                        metrics_dict[k] = v
                metrics_file = os.path.join(
                    self.metrics_cfg.metrics_base_dir, f"episode_{episode_id}.json"
                )
                os.makedirs(self.metrics_cfg.metrics_base_dir, exist_ok=True)
                with open(metrics_file, "w") as f:
                    json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    def _init_env(self):
        env_fns = self._get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def _get_env_fns(self):
        env_fn_params = self._get_env_fn_params()
        env_fns = []

        for param in env_fn_params:

            def env_fn(p=param):
                config_path = p["config_path"]
                episode_ids = p["episode_ids"]
                seed = p["seed"]

                config = get_config(config_path)

                dataset = habitat.datasets.make_dataset(
                    config.habitat.dataset.type,
                    config=config.habitat.dataset,
                )

                dataset.episodes = [
                    ep for ep in dataset.episodes if ep.episode_id in episode_ids
                ]

                env = HabitatRLEnv(config=config, dataset=dataset)
                env.seed(seed)
                return env

            env_fns.append(env_fn)

        return env_fns

    def _get_env_fn_params(self):
        env_fn_params = []

        # Habitat uses hydra to load the config,
        # but the hydra maybe initialized somewhere else,
        # so we need to clear it to avoid conflicts
        hydra_initialized = GlobalHydra.instance().is_initialized()
        if hydra_initialized:
            GlobalHydra.instance().clear()

        config_path = self.cfg.init_params.config_path
        habitat_config = get_config(config_path)

        habitat_dataset = habitat.datasets.make_dataset(
            habitat_config.habitat.dataset.type,
            config=habitat_config.habitat.dataset,
        )

        episode_ids = self._build_ordered_episodes(habitat_dataset)

        num_episodes = len(episode_ids)
        episodes_per_env = num_episodes // self.num_envs

        episode_ranges = []
        start = 0
        for i in range(self.num_envs - 1):
            episode_ranges.append((start, start + episodes_per_env))
            start += episodes_per_env
        episode_ranges.append((start, num_episodes))

        for env_id in range(self.num_envs):
            start, end = episode_ranges[env_id]
            assigned_ids = episode_ids[start:end]

            env_fn_params.append(
                {
                    "config_path": config_path,
                    "episode_ids": assigned_ids,
                    "seed": self.seed + env_id,
                }
            )

        return env_fn_params

    def _build_ordered_episodes(self, dataset):
        """
        rearrange the episode ids to be consecutive for each scene
        """
        scene_ids = []
        episode_ids = []
        scene_id_to_idx = {}  # scene_id(str) -> scene_idx(int)
        scene_to_episodes = {}  # scene_idx(int) -> episode_ids(list[int])

        for episode in dataset.episodes:
            sid = episode.scene_id
            eid = episode.episode_id
            if sid not in scene_id_to_idx:
                scene_idx = len(scene_ids)
                scene_id_to_idx[sid] = scene_idx
                scene_ids.append(sid)
                scene_to_episodes[scene_idx] = []
            else:
                scene_idx = scene_id_to_idx[sid]
            scene_to_episodes[scene_idx].append(eid)

        for scene_idx in range(len(scene_ids)):
            episode_ids.extend(scene_to_episodes[scene_idx])

        return episode_ids
