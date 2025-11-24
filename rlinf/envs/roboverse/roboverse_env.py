import gym
import numpy as np
import torch
import copy
import os
from typing import List, Optional, Union

import metasim  # noqa: F401
from metasim.task.registry import get_task_class
from metasim.task.rl_task import RLTaskEnv
from metasim.scenario.cameras import PinholeCameraCfg
from pytorch3d import transforms
from rlinf.envs.libero.utils import (
    get_benchmark_overridden,
    get_libero_image,
    get_libero_wrist_image,
    list_of_dict_to_dict_of_list,
    put_info_on_image,
    quat2axisangle,
    save_rollout_video,
    tile_images,
    to_tensor,
)
class RoboVerseEnv(gym.Env):
    def __init__(self, cfg, rank, world_size):
        # default codes
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.seed = self.cfg.seed + rank
        self._is_start = True
        self.num_envs = self.cfg.num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.cfg.num_group
        self.use_fixed_reset_state_ids = False # cfg.use_fixed_reset_state_ids
        self.total_num_group_envs = self.num_envs // self.group_size # TODO: need check

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []

        # RoboVerse Specific Codes
        simulator = self.cfg.simulator_backend
        task_cls = get_task_class(self.cfg.task_name)
        # assert issubclass(task_cls, RLTaskEnv), "Task class must be a subclass of RLTaskEnv"
        robot = self.cfg.robot_name
        self.robot_name = robot

        # Camera configuration (can be overridden in config)
        # Default: match Maniskill's render_camera settings (512x512)
        # Safely get init_params or use defaults
        if hasattr(self.cfg, 'init_params'):
            camera_width = getattr(self.cfg.init_params, 'camera_width', getattr(self.cfg, 'camera_width', 512))
            camera_height = getattr(self.cfg.init_params, 'camera_height', getattr(self.cfg, 'camera_height', 512))
            camera_pos = getattr(self.cfg.init_params, 'camera_pos', getattr(self.cfg, 'camera_pos', (1.5, -1.5, 1.5)))
            camera_look_at = getattr(self.cfg.init_params, 'camera_look_at', getattr(self.cfg, 'camera_look_at', (0.0, 0.0, 0.5)))
        else:
            camera_width = getattr(self.cfg, 'camera_width', 512)
            camera_height = getattr(self.cfg, 'camera_height', 512)
            camera_pos = getattr(self.cfg, 'camera_pos', (1.5, -1.5, 1.5))
            camera_look_at = getattr(self.cfg, 'camera_look_at', (0.0, 0.0, 0.5))
        
        scenario = task_cls.scenario.update(
            robots=[robot],
            simulator=simulator,
            num_envs=self.cfg.num_envs,
            headless=self.cfg.headless,
            env_spacing=2.5,
            cameras=[PinholeCameraCfg(
                name="camera",
                width=camera_width,
                height=camera_height,
                pos=camera_pos,
                look_at=camera_look_at,
        )],
        )
        self.scenario = scenario

        self.env = task_cls(scenario=scenario, device="cuda")

        self.reset_state_ids_all = self.get_reset_state_ids_all()
        # Initialize fixed reset ids if requested
        if self.use_fixed_reset_state_ids:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
            self.reset_state_ids = reset_state_ids.repeat(self.group_size)
        else:
            self.reset_state_ids = None

        self.solver = 'pyroki'
        self._setup_ik()
        self.ee_body_idx = None
        self.ee_body_name = self.scenario.robots[0].ee_body_name

        self.task_descriptions = [self.env.task_desc for _ in range(self.num_envs)]

        # action coordinate frame: 'local' (default) or 'world'
        self.action_coordinate_frame = getattr(cfg, 'action_coordinate_frame', 'world')
        
        # Initialize last_obs to None (will be set after first reset)
        self.last_obs = None

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (len(reset_state_ids) % self.world_size)
        self._generator_ordered.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.world_size, -1)
        return reset_state_ids
    
    def _init_metrics(self):
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
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0


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

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []

    def reset(
        self,
        env_idx: Optional[Union[int, List[int], np.ndarray]] = None,
        reset_state_ids=None,
        options: Optional[dict] = {},
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        # self._reconfigure(reset_state_ids, env_idx) # TODO

        for _ in range(10): # TODO ???
            zero_actions = np.zeros((self.num_envs, 9))
            # if self.last_obs is not None:
            #     # Convert zero end-effector delta to dictionary format
            #     zero_action_tensor = torch.zeros((self.num_envs, 9), device=self.device)
            #     zero_actions = self._convert_action(zero_action_tensor)
            #     # Serialize to make it pickle-able for multiprocessing
            #     zero_actions = self._serialize_actions_dict(zero_actions)
            # else:
            #     # Fallback: create minimal zero action dict if last_obs not available
            #     # This should not normally happen, but provide a safe fallback
            #     zero_actions = {}
            #     for env_id in range(self.num_envs):
            #         zero_actions[env_id] = {
            #             self.robot_name: {
            #                 "dof_pos_target": {}
            #             }
            #         }
            raw_obs, reward, terminated, time_out, info_list = self.env.step(zero_actions)

        obs = raw_obs
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        
        # Reset step counter for new episode
        self._step_counter = 0
        
        infos = {}
        return obs, infos

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

    def step(self, actions=None, auto_reset=True):
        """
        Step through the environment for one timestep.

        Args:
            actions (np.ndarray or torch.Tensor): Actions for each env.
        Returns:
            obs (torch.Tensor): Wrapped observations.
            rewards (torch.Tensor): Step rewards.
            terminations (torch.Tensor): Done due to success/failure.
            truncations (torch.Tensor): Done due to time limit.
            infos (dict): Additional information.
        """
        # 1. deal with reset
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            raw_obs, infos = self.reset(
                reset_state_ids=self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self.is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            self.last_obs = raw_obs
            obs = raw_obs.cameras["camera"].rgb.permute(0,3,1,2)
            obs_dict = {
                "images": obs,
                "task_descriptions": self.task_descriptions,
            }
            return obs_dict, None, to_tensor(terminations), to_tensor(truncations), infos

        # 2. convert end-effector commands to robot actions
        actions = torch.tensor(actions)
        actions = self._convert_action(actions)
        
        # Convert actions dict to serializable format (for multiprocessing)
        # Ensure all CUDA tensors are moved to CPU and converted to numpy
        if isinstance(actions, dict):
            actions = self._serialize_actions_dict(actions)
        elif isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        # 3. interact with metasim
        raw_obs, reward, terminated, time_out, info_list = self.env.step(actions)
        infos = info_list #list_of_dict_to_dict_of_list(info_list)
        # NOTE: we do not use the state-based observation and reward from metasim

        # 4. time steps & dones
        self._elapsed_steps += 1
        terminations = terminated.cpu().numpy()
        truncations = self._elapsed_steps >= self.cfg.max_episode_steps
        dones = terminations | truncations

        # 5. generate rewards
        step_reward = self._calc_step_reward(terminations)

        # 6. get image observation
        # states = self.env.handler.get_states()
        image_obs = raw_obs.cameras["camera"].rgb # b, c, h, w

        # 7. record video
        if getattr(self.video_cfg, "save_video", False):
            plot_infos = {
                "reward": step_reward,
                "done": terminations,
                "step": self._elapsed_steps,
            }
            self.add_new_frames(plot_infos, image_obs)

        image_obs = image_obs.permute(0,3,1,2)
        
        # 8. reward metrics
        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        # 9. auto-reset
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            image_obs, infos = self._handle_auto_reset(dones, image_obs, infos)
        
        obs_dict = {
            "images": image_obs,
            "task_descriptions": self.task_descriptions,
        }

        self.last_obs = raw_obs

        # 10. return information in Gym style
        return (
            obs_dict,
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
            image_obs, infos = self._handle_auto_reset(
                past_dones.cpu().numpy(), extracted_obs, infos
            )
            extracted_obs = {
                "images": image_obs,
                "task_descriptions": self.task_descriptions,
            }

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
        raw_obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )

        image_obs = raw_obs.cameras["camera"].rgb.permute(0,3,1,2)

        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return image_obs, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward
        
    def update_reset_state_ids(self):
        return # For single env, no need to change task
        if self.cfg.only_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        reset_state_ids = self.reset_state_ids_all[self.rank][
            self.start_idx : self.start_idx + num_reset_states
        ]
        self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids
    
    def add_new_frames(self, plot_infos, images):
        full_image = tile_images(images.cpu().numpy(), nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    # ---------------- IK ----------------
    def _setup_ik(self):
        from metasim.utils.ik_solver import setup_ik_solver
        self.robot_cfg = self.scenario.robots[0]
        self.ik_solver = setup_ik_solver(self.robot_cfg, self.solver)

    def _get_valid_joint_names(self):
        """
        Get valid joint names for the current robot configuration.
        This helps filter out joints that don't exist in the current setup.
        
        Different tasks may use different robot configurations, so joint names may vary.
        For example, 'pick_bbq_sause' might have 'panda_finger_joint1', but 'pick_milk' might not.
        
        Returns:
            set: Set of valid joint names, or None if unable to determine
        """
        valid_joints = None
        try:
            # Method 1: Try to get from handler (most reliable)
            if hasattr(self.env, 'handler') and hasattr(self.env.handler, 'get_joint_names'):
                joint_names = self.env.handler.get_joint_names(self.robot_name)
                if isinstance(joint_names, (list, tuple)):
                    valid_joints = set(joint_names)
            
            # Method 2: Try to get from robot state if available
            if valid_joints is None and self.last_obs is not None:
                if hasattr(self.last_obs, 'robots'):
                    rs = self.last_obs.robots.get(self.robot_name)
                    if rs is not None:
                        # Try joint_names attribute
                        if hasattr(rs, 'joint_names'):
                            valid_joints = set(rs.joint_names)
                        # Try to infer from joint_pos shape and reorder_idx
                        elif hasattr(rs, 'joint_pos') and hasattr(self.env, 'handler'):
                            try:
                                reorder_idx = self.env.handler.get_joint_reindex(self.robot_name)
                                # Get joint names by mapping indices
                                # This is a fallback if direct access isn't available
                                if len(reorder_idx) > 0:
                                    # We can't get names directly, but we know the count
                                    # This won't help with filtering, but at least we know the structure
                                    pass
                            except:
                                pass
            
            # Method 3: Try to get from robot config
            if valid_joints is None and hasattr(self, 'robot_cfg'):
                if hasattr(self.robot_cfg, 'joint_names'):
                    valid_joints = set(self.robot_cfg.joint_names)
                elif hasattr(self.robot_cfg, 'actuators'):
                    # Try to extract from actuators
                    joint_names = []
                    for actuator in self.robot_cfg.actuators:
                        if hasattr(actuator, 'joint_name'):
                            joint_names.append(actuator.joint_name)
                    if joint_names:
                        valid_joints = set(joint_names)
        except Exception as e:
            # If we can't get joint names, return None to skip filtering
            # This means we won't filter any joints (safer but may cause errors)
            pass
        
        return valid_joints
    
    def _serialize_actions_dict(self, actions_dict):
        """
        Convert actions dictionary to pickle-able format for multiprocessing.
        Converts all CUDA tensors to CPU numpy arrays and filters out invalid joint names.
        
        Args:
            actions_dict: Dictionary in format {env_id: {obj_name: {"dof_pos_target": {joint_name: tensor}}}}
        
        Returns:
            Serialized dictionary with all tensors converted to numpy arrays
        """
        if not isinstance(actions_dict, dict):
            return actions_dict
        
        # Get valid joint names to filter out non-existent joints
        valid_joint_names = self._get_valid_joint_names()
        
        serialized = {}
        for env_id, env_actions in actions_dict.items():
            serialized[env_id] = {}
            for obj_name, obj_actions in env_actions.items():
                serialized[env_id][obj_name] = {}
                for key, value in obj_actions.items():
                    if isinstance(value, dict):
                        # Handle nested dict (e.g., "dof_pos_target": {joint_name: tensor})
                        serialized[env_id][obj_name][key] = {}
                        for joint_name, joint_value in value.items():
                            # Filter out invalid joint names if we have a valid set
                            if valid_joint_names is not None and joint_name not in valid_joint_names:
                                continue  # Skip this joint if it doesn't exist
                            
                            if isinstance(joint_value, torch.Tensor):
                                serialized[env_id][obj_name][key][joint_name] = joint_value.detach().cpu().numpy()
                            else:
                                serialized[env_id][obj_name][key][joint_name] = joint_value
                    elif isinstance(value, torch.Tensor):
                        serialized[env_id][obj_name][key] = value.detach().cpu().numpy()
                    else:
                        serialized[env_id][obj_name][key] = value
        return serialized

    # ---------------- EE control + IK ----------------
    def _convert_action(self, action):
        # Use CUDA device for all IK operations (cuRobo requires CUDA tensors)
        device = self.device
        """Δ-pose (local) -> target EE pose -> cuRobo IK -> joint targets."""
        num_envs = action.shape[0]

        # 2) Robot state (TensorState -> tensors)
        rs = self.last_obs.robots[self.robot_name]

        # IK solver expects original joint order, but state uses alphabetical order
        reorder_idx = self.env.handler.get_joint_reindex(self.robot_name)
        inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        joint_pos_raw = rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)
        curr_robot_q = joint_pos_raw[:, inverse_reorder_idx].to(device).float()
        robot_ee_state = (rs.body_state if isinstance(rs.body_state, torch.Tensor) else torch.tensor(rs.body_state)).to(device).float()
        robot_root_state = (rs.root_state if isinstance(rs.root_state, torch.Tensor) else torch.tensor(rs.root_state)).to(device).float()

        if self.ee_body_idx is None:
            self.ee_body_idx = rs.body_names.index(self.ee_body_name)
        ee_p_world = robot_ee_state[:, self.ee_body_idx, 0:3]
        ee_q_world = robot_ee_state[:, self.ee_body_idx, 3:7]
        # print(f"EE position in world: {ee_p_world}")

        # Base pose
        robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]
        # print(f"Robot position in world: {robot_pos}")
        # Local frame transform
        # Ensure all intermediate calculations use float32
        inv_base_q = transforms.quaternion_invert(robot_quat).to(dtype=torch.float32)
        curr_ee_pos_local = transforms.quaternion_apply(inv_base_q, ee_p_world - robot_pos).to(dtype=torch.float32)
        curr_ee_quat_local = transforms.quaternion_multiply(inv_base_q, ee_q_world).to(dtype=torch.float32)

        # 3) Apply deltas
        # Ensure action is on the correct device (CUDA) and float32 type
        action = action.to(device).float()
        ee_pos_delta_raw = action[:num_envs, :3]
        ee_rot_delta_raw = action[:num_envs, 3:-1]
        gripper_open = action[:num_envs, -1]

        # Coordinate frame conversion: if action is in world frame, convert to local frame
        if self.action_coordinate_frame == 'world':
            # Convert world frame delta to local frame
            # Position: rotate world delta by inverse base quaternion
            ee_pos_delta_world = ee_pos_delta_raw
            ee_pos_delta = transforms.quaternion_apply(inv_base_q, ee_pos_delta_world).to(dtype=torch.float32)
            
            ee_rot_euler_world = ee_rot_delta_raw  # Euler angles (XYZ) in world frame
            # Convert Euler to rotation matrix, then to quaternion
            ee_rot_matrix_world = transforms.euler_angles_to_matrix(ee_rot_euler_world, "XYZ")
            ee_rot_quat_world = transforms.matrix_to_quaternion(ee_rot_matrix_world).to(dtype=torch.float32)
            
            ee_quat_delta = transforms.quaternion_multiply(inv_base_q, ee_rot_quat_world).to(dtype=torch.float32)
        else:
            # Default: action is already in local frame
            ee_pos_delta = ee_pos_delta_raw
            ee_quat_delta = transforms.matrix_to_quaternion(
                transforms.euler_angles_to_matrix(ee_rot_delta_raw, "XYZ")
            ).to(dtype=torch.float32)

        ee_pos_target = (curr_ee_pos_local + ee_pos_delta).to(device=device, dtype=torch.float32)
        ee_quat_target = transforms.quaternion_multiply(curr_ee_quat_local, ee_quat_delta).to(device=device, dtype=torch.float32)
        
        curr_robot_q = curr_robot_q.to(device=device, dtype=torch.float32)

        # 4) IK (seed = current q)
        q_solution, ik_succ = self.ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)

        # 5) Gripper control
        from metasim.utils.ik_solver import process_gripper_command
        gripper_widths = process_gripper_command(gripper_open, self.robot_cfg, device)

        # Compose robot command
        actions = self.ik_solver.compose_joint_action(q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True)
        return actions

    @property
    def device(self):
        return "cuda:0"
