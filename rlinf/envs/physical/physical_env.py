import numpy as np
import torch
import gymnasium as gym
import copy
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from .franka.franka_env import FrankaEnv, FrankaRobotConfig
import rlinf.envs.physical.franka.tasks
from rlinf.envs.utils import to_tensor, list_of_dict_to_dict_of_list
from typing import OrderedDict


class PhysicalEnv(gym.Env):
    def __init__(self, cfg, seed_offset, total_num_processes):
        self.cfg = cfg
        self.env_cfg = FrankaRobotConfig(
            robot_ip="0.0.0.0"
        )
        self.video_cfg = cfg.video_cfg

        self.seed = cfg.seed + seed_offset
        self.num_envs = self.cfg.num_envs
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = cfg.num_group
        self.group_size = cfg.group_size

        if self.num_envs != 1:
            raise NotImplementedError(
                f"Currently, the code only supports 1 env, but {self.num_envs=} is received."
            )
        self._init_env()

        
        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._init_reset_state_ids()

    def _init_env(self):
        env_fns = []
        for _ in range(self.num_envs):
            def env_fn():
                env = gym.make(id=self.cfg.init_params.id, config=self.env_cfg)
                return env
            env_fns.append(env_fn)
        
        self.env = SyncVectorEnv(env_fns)
        self.task_descriptions = list(self.env.call("task_description"))
    
    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2  # TODO
    
    @property
    def is_start(self):
        return self._is_start
    
    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps
    
    def _init_metrics(self):
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
    
    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
    
    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = to_tensor(episode_info)
        return infos
        
    
    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):

        # TODO: handle partial reset
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        
        extracted_obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos
    
    def _wrap_obs(self, raw_obs):
        """
        raw_obs: Dict of list
        """
        obs = dict()
        
        # Process states
        full_states = []
        raw_states = OrderedDict(sorted(raw_obs["state"].items()))
        for key, value in raw_states.items():
            full_states.append(value)
        full_states = np.concatenate(full_states, axis=-1)
        obs["states"] = full_states

        # Process images
        obs["images"] = dict()
        for camera_name in raw_obs["frames"]:
            image_numpy: np.ndarray = raw_obs["frames"][camera_name] # [B, H, W, C]
            obs["images"][camera_name] = np.moveaxis(image_numpy, 3, 1) # [B, C, H, W]

        obs = to_tensor(obs)
        obs["task_descriptions"] = self.task_descriptions
        return obs
    
    def step(self, actions=None, auto_reset=True):
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            obs, infos = self.reset(
                reset_state_ids=self.reset_state_ids
                if self.use_fixed_reset_state_ids
                else None
            )
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)

            return obs, None, to_tensor(terminations), to_tensor(truncations), infos

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        terminations = np.array(_reward) == 1
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps

        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(_reward)

        if self.video_cfg.save_video:
            raise NotImplementedError(f"Not support save video yet")
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "task": self.task_descriptions,
            }
            self.add_new_frames(raw_obs, plot_infos)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )
    

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones.cpu().numpy(), extracted_obs, infos
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
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
    
    
    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos
    

    def _calc_step_reward(self, reward):
        return reward
    
    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids
    
    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        )



