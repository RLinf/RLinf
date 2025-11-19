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

from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rlinf.envs.utils import to_tensor
from rlinf.envs.worldmodel.dataset import NpyTrajectoryDatasetWrapper
from rlinf.envs.worldmodel.worldmodel_env import WorldModelEnv
from rlinf.models.worldmodel.evac_model import EvacModelInference


class ChunkWorldModelEnv(WorldModelEnv):
    """
    A Gym environment that wraps a chunk-based world model for reinforcement learning.

    This environment extends WorldModelEnv to support chunk-based inference,
    where multiple steps are processed at once for efficiency. It's designed
    for models like Evac that only support multi-step batch inference.

    Key differences from WorldModelEnv:
    - Only supports chunk_step, not single step
    - Processes entire chunks of actions at once
    - Maintains action buffer for trajectory computation
    """

    def __init__(self, cfg, seed_offset: int, total_num_processes, record_metrics=True):
        """
        Initializes the ChunkWorldModelEnv.

        Args:
            cfg: The configuration object containing environment settings.
            seed_offset: An offset added to the base seed.
            total_num_processes: The total number of parallel processes.
            record_metrics: Whether to track episode-level metrics.
        """
        # Check if this is an Evac model configuration
        model_type = cfg.get("model_type", "evac")

        if model_type == "evac":
            # Prepare config for EvacModelInference
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)

            # Setup dataset config for NpyTrajectoryDatasetWrapper
            if hasattr(cfg, "initial_image_path"):
                dataset_cfg = {
                    "data_dir": cfg.initial_image_path,
                    "start_select_policy": "first_frame",
                    "target_select_policy": "last_frame",
                    "camera_names": ["image"],
                }
            else:
                # Use existing dataset_cfg if available
                raise NotImplementedError
            cfg_dict["dataset_cfg"] = dataset_cfg

            # Setup backend config for EvacModelInference
            backend_cfg = {
                "batch_size": cfg.num_envs,
                "num_prompt_frames": cfg.n_previous,
                "gen_num_image_each_step": cfg.chunk,
                "max_episode_steps": cfg.max_episode_steps,
                "chunk": cfg.chunk,
                "n_previous": cfg.n_previous,
                "sample_size": cfg.sample_size,
                "world_model_cfg": cfg.world_model_cfg,
                "reward_model_cfg": cfg.reward_model_cfg,
                "action_predictor_cfg": cfg.action_predictor_cfg,
            }
            cfg_dict["backend_cfg"] = backend_cfg

            # Convert back to OmegaConf for parent class
            cfg_modified = OmegaConf.create(cfg_dict)
        else:
            cfg_modified = cfg

        # Initialize parent class
        super().__init__(cfg_modified, seed_offset, total_num_processes, record_metrics)

        # Chunk-specific configuration
        self.chunk = cfg.chunk
        self.n_previous = cfg.n_previous
        self.sample_size = tuple(cfg.sample_size)

        # Override dataset and env if using EvacModelInference
        if model_type == "evac":
            # Override dataset with NpyTrajectoryDatasetWrapper
            self.task_dataset = NpyTrajectoryDatasetWrapper(**dataset_cfg)
            self.total_num_group_envs = len(self.task_dataset)

            # Override the env with EvacModelInference
            self.env = EvacModelInference(backend_cfg, self.task_dataset, self.device)

        # Action buffer for trajectory computation
        self.action_buffer = None

        # Current observations (maintained separately for chunk processing)
        self.current_obs = None

        # Task descriptions for Evac model
        self.task_descriptions = [""] * self.num_envs

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
    ):
        """
        Resets the environment to initial states and returns observations.

        Args:
            seed: Optional seed or list of seeds for deterministic resets.
            options: Optional dictionary containing reset options.

        Returns:
            tuple: A tuple containing:
                - obs: The initial observations from the reset environment
                - info: Additional information about the reset state
        """
        # For EvacModelInference, extract init_ee_pose and task descriptions from dataset
        if isinstance(self.env, EvacModelInference):
            # Extract init_ee_poses and task descriptions from dataset
            num_envs = self.num_envs
            if len(self.task_dataset) < num_envs:
                raise ValueError(
                    f"Not enough episodes in dataset. Found {len(self.task_dataset)}, need {num_envs}"
                )

            # Set random seed if provided
            if seed is not None:
                if isinstance(seed, list):
                    np.random.seed(seed[0])
                else:
                    np.random.seed(seed)

            # Randomly select episode indices
            episode_indices = np.random.choice(
                len(self.task_dataset), size=num_envs, replace=False
            )

            # Extract init_ee_poses and task descriptions
            init_ee_poses = []
            task_descriptions = []
            for episode_idx in episode_indices:
                episode_data = self.task_dataset[episode_idx]
                first_frame = episode_data["start_items"][0]

                task_desc = episode_data.get("task", "")
                task_descriptions.append(str(task_desc))

                if "observation.state" in first_frame:
                    init_ee_pose = first_frame["observation.state"]
                    if isinstance(init_ee_pose, torch.Tensor):
                        init_ee_pose = init_ee_pose.numpy()
                    init_ee_poses.append(init_ee_pose)
                else:
                    init_ee_poses.append(None)

            options["init_ee_pose"] = init_ee_poses
            self.task_descriptions = task_descriptions

        obs, info = self.env.reset(seed=seed, options=options)

        # Convert obs to tensor format for chunk processing
        # obs is a list of dicts, we need to extract images
        if isinstance(obs, list):
            # Get the last observation from the list
            last_obs = obs[-1]
            # Extract images - handle both camera name and "full_image" key
            if "full_image" in last_obs.get("images_and_states", {}):
                images = last_obs["images_and_states"][
                    "full_image"
                ]  # [num_envs, H, W, 3]
            else:
                camera_name = self.camera_names[0]  # Assuming single camera for now
                images = last_obs["images_and_states"][
                    camera_name
                ]  # [num_envs, H, W, 3]

            # Convert to [num_envs, 3, 1, n_previous, H, W] format
            num_envs = images.shape[0]
            # Convert from HWC to CHW and normalize
            images = images.permute(0, 3, 1, 2).float() / 255.0  # [num_envs, 3, H, W]
            # Normalize to [-1, 1]
            images = (images - 0.5) / 0.5
            # Resize if needed
            if images.shape[2:] != self.sample_size:
                images = F.interpolate(
                    images, size=self.sample_size, mode="bilinear", align_corners=False
                )
            # Add temporal dimension: [num_envs, 3, 1, n_previous, H, W]
            images = (
                images.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, self.n_previous, 1, 1)
            )
            self.current_obs = images.to(self.device)
        else:
            # Handle single observation case
            self.current_obs = obs

        # Initialize action buffer from init_ee_pose if available
        if "init_ee_pose" in options:
            init_ee_poses = options["init_ee_pose"]
        else:
            # Try to get from obs
            if isinstance(obs, list) and len(obs) > 0:
                last_obs = obs[-1]
                if "state" in last_obs.get("images_and_states", {}):
                    states = last_obs["images_and_states"]["state"]
                    if isinstance(states, torch.Tensor):
                        init_ee_poses = [
                            states[i].cpu().numpy()[:8] for i in range(self.num_envs)
                        ]
                    else:
                        init_ee_poses = [
                            states[i][:8] if len(states[i]) >= 8 else None
                            for i in range(self.num_envs)
                        ]
                else:
                    init_ee_poses = [None] * self.num_envs
            else:
                init_ee_poses = [None] * self.num_envs

        # Initialize action buffer
        init_actions = []
        for init_ee_pose in init_ee_poses:
            if init_ee_pose is not None:
                if isinstance(init_ee_pose, torch.Tensor):
                    init_ee_pose = init_ee_pose.numpy()
                init_ee_pose = init_ee_pose.flatten()
                # Duplicate to 16-dim for both arms
                init_action = np.concatenate([init_ee_pose, init_ee_pose], axis=0)
            else:
                # Default action
                init_action = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0] * 2, dtype=np.float32
                )
            init_actions.append(init_action)

        init_actions_array = np.stack(init_actions, axis=0)  # [num_envs, 16]
        self.action_buffer = (
            torch.from_numpy(init_actions_array)
            .unsqueeze(1)
            .repeat(1, self.n_previous + self.chunk, 1)
            .to(self.device)
        )

        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()

        return obs[-1] if isinstance(obs, list) else obs, info

    def step(
        self, actions: Union[torch.Tensor, np.ndarray] = None, auto_reset=True
    ) -> tuple:
        """
        Executes one environment step (not supported for chunk-based env).

        Args:
            actions: The actions to execute (not used for chunk env).
            auto_reset: Whether to automatically reset.

        Raises:
            NotImplementedError: Single step is not supported for chunk-based environments.
        """
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            obs, infos = self.reset()
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            return obs, None, to_tensor(terminations), to_tensor(truncations), infos
        raise NotImplementedError(
            "Single step is not supported for ChunkWorldModelEnv. Use chunk_step instead."
        )

    def policy_output_to_abs_action(self, policy_output):
        """
        Convert policy output to absolute action using action predictor.

        Args:
            policy_output: Policy output [num_envs, chunk, 7]

        Returns:
            Absolute actions [num_envs, chunk, 16]
        """
        policy_output = torch.tensor(policy_output, device=self.device).float()
        pre_ee_pose = self.action_buffer[:, -self.chunk - 1, :8]
        ee_pose_list = torch.zeros((self.num_envs, self.chunk, 8), device=self.device)

        num_envs, chunk, _ = policy_output.shape
        # Access action_predictor from env (EvacModelInference)
        action_predictor = getattr(self.env, "action_predictor", None)
        if action_predictor is None:
            raise ValueError(
                "action_predictor not found in env. Make sure EvacModelInference is used."
            )

        for i in range(chunk):
            ee_pose_list[:, i] = action_predictor.get_ee_pose(
                policy_output[:, i], pre_ee_pose
            )
            pre_ee_pose = ee_pose_list[:, i]

        # Duplicate to 16-dim for both arms
        ee_pose_list = torch.cat([ee_pose_list, ee_pose_list], dim=-1)

        # Update action buffer
        self.action_buffer[:, -self.chunk - 1 : -1] = ee_pose_list
        abs_action = self.action_buffer[:, -self.chunk - 1 : -1]

        return abs_action

    def chunk_step(
        self, policy_output_action: Union[torch.Tensor, np.ndarray]
    ) -> tuple:
        """
        Executes a chunk of actions for all environments.

        Args:
            policy_output_action: Policy output actions [num_envs, chunk, 7]

        Returns:
            tuple: A tuple containing:
                - extracted_obs: The observations after executing all chunked actions
                - chunk_rewards: Tensor of rewards [num_envs, chunk]
                - chunk_terminations: Boolean tensor [num_envs, chunk]
                - chunk_truncations: Boolean tensor [num_envs, chunk]
                - info: Dictionary containing information about the final state
        """
        # Convert policy output to absolute actions
        self.policy_output_to_abs_action(policy_output_action)

        # Update elapsed steps
        self._elapsed_steps += self.chunk

        # Perform chunk inference
        with torch.cuda.amp.autocast(dtype=torch.float16):
            new_obs, chunk_rewards = self.env.chunk_infer(
                self.current_obs, self.action_buffer
            )

        # Update current observations
        self.current_obs = new_obs

        # Wrap observations
        extracted_obs = self._wrap_obs()

        # Calculate step rewards
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        # Handle terminations/truncations
        raw_chunk_terminations = deepcopy(chunk_rewards_tensors)
        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        truncations = torch.tensor(self.elapsed_steps >= self.cfg.max_episode_steps).to(
            self.device
        )

        if truncations.any():
            raw_chunk_truncations[:, -1] = truncations

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}

        infos = self._record_metrics(chunk_rewards_tensors.sum(dim=1), infos)

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        # Trim action buffer
        self.action_buffer = self.action_buffer[:, -self.chunk - self.n_previous :, :]

        return (
            extracted_obs,
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _wrap_obs(self):
        """Wrap observation to match expected format."""

        # Extract the last frame
        b, c, v, t, h, w = self.current_obs.shape
        last_frame = self.current_obs[:, :, 0, -1, :, :]  # [b, 3, h, w]

        full_image = last_frame.permute(0, 2, 3, 1)  # [b, H, W, 3]

        # Denormalize from [-1, 1] to [0, 255]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255)

        # Resize if needed
        if full_image.shape[1:3] != self.sample_size:
            full_image = full_image.permute(0, 3, 1, 2)
            full_image = F.interpolate(
                full_image, size=self.sample_size, mode="bilinear", align_corners=False
            )
            full_image = full_image.permute(0, 2, 3, 1)

        full_image = full_image.to(torch.uint8)

        states = self.action_buffer[:, -1]  # [num_envs, 16]

        # Get task descriptions
        task_descriptions = getattr(self, "task_descriptions", [""] * self.num_envs)

        obs = {
            "images_and_states": {
                "full_image": full_image,
                "state": states,
            },
            "task_descriptions": task_descriptions,
        }

        return obs

    def _calc_step_reward(self, chunk_rewards):
        """Calculate step reward from chunk rewards."""
        reward_diffs = torch.zeros(
            (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
        )
        for i in range(self.chunk):
            reward_diffs[:, i] = chunk_rewards[:, i] - self.prev_step_reward
            self.prev_step_reward = chunk_rewards[:, i]

        if self.use_rel_reward:
            return reward_diffs
        else:
            return chunk_rewards


if __name__ == "__main__":
    import os

    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir

    # Set required environment variable
    os.environ.setdefault(
        "EMBODIED_PATH", "/mnt/mnt/public/jzn/workspace/RLinf/examples/embodiment"
    )

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    config_dir = "/mnt/mnt/public/jzn/workspace/RLinf/examples/embodiment/config"
    config_name = "libero_spatial_evac_grpo_openvlaoft_impl"

    print(f"Loading config: {config_name} from {config_dir}")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["eval"]

    # Create environment using ChunkWorldModelEnv
    env = ChunkWorldModelEnv(cfg, seed_offset=0, total_num_processes=1)

    # Reset environment
    obs, info = env.reset()
    print("\nAfter reset:")
    print(f"  obs keys: {list(obs.keys())}")
    if "images_and_states" in obs:
        print(f"  images_and_states keys: {list(obs['images_and_states'].keys())}")
        if "full_image" in obs["images_and_states"]:
            print(f"  full_image shape: {obs['images_and_states']['full_image'].shape}")
        if "state" in obs["images_and_states"]:
            print(f"  state shape: {obs['images_and_states']['state'].shape}")
    print(f"  task_descriptions: {obs.get('task_descriptions', [])}")

    # Test chunk_step
    print("\n" + "-" * 80)
    print("Testing chunk_step...")
    chunk_steps = env.chunk
    num_envs = cfg.num_envs

    def read_from_npy():
        path = "/mnt/mnt/public/jzn/workspace/RLinf/reward_model/reward_data/embodiment_converted/train_data/step_0_seed_0_traj_0.npy"
        action = np.load(path, allow_pickle=True)[0]["delta_action"]
        return action

    init_action = read_from_npy()

    # Create chunk actions: [num_envs, chunk_steps, action_dim]
    chunk_actions = np.tile(init_action, (num_envs, chunk_steps, 1))

    print(f"Running {10} chunk steps...")
    for i in range(10):
        obs, reward, term, trunc, info = env.chunk_step(chunk_actions)

        print(f"\nStep {i + 1}:")
        print(f"  action_buffer shape: {env.action_buffer.shape}")
        print(f"  current_obs shape: {env.current_obs.shape}")
        print(f"  reward shape: {reward.shape}, mean: {reward.mean().item():.4f}")
        print(f"  term shape: {term.shape}, any: {term.any().item()}")
        print(f"  trunc shape: {trunc.shape}, any: {trunc.any().item()}")
        if "episode" in info:
            print(f"  episode info keys: {list(info['episode'].keys())}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
