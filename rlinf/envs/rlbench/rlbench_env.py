# parallel_rlbench_env.py
import os
import time
import copy
import random
import numpy as np
from typing import List, Optional, Union, Tuple
import multiprocessing as mp
import imageio
import gym
import torch
from scipy.spatial.transform import Rotation as R
from rlinf.envs.utils import to_tensor
from rlinf.envs.libero.utils import tile_images

# rlbench imports
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig


# ----------------------------
# Worker process
# ----------------------------
def _make_obs_config(image_size=(128, 128), front_only=True):
    cam = CameraConfig(rgb=True, depth=False, mask=False, image_size=image_size)
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    if front_only:
        obs_config.front_camera = cam
    else:
        # enable several default cameras (front + wrist) - adjust as needed
        obs_config.front_camera = cam
        obs_config.wrist_camera = cam
        obs_config.left_shoulder_camera = cam
        obs_config.right_shoulder_camera = cam
        obs_config.overhead_camera = cam
    return obs_config


def worker_process(
    child_conn: mp.connection.Connection,
    task_name: str,
    seed: int,
    headless: bool = True,
    image_size: Tuple[int, int] = (128, 128),
):
    try:
        # build env & task
        obs_config = _make_obs_config(image_size=image_size, front_only=True)
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaIK(absolute_mode=False),
            gripper_action_mode=Discrete(),
        )
        env = Environment(
            action_mode, obs_config=obs_config, headless=headless, robot_setup="panda"
        )
        env.launch()
        # Convert task name string to task class
        # In multiprocessing, globals() may not contain imported classes, so use getattr directly
        import rlbench.tasks as tasks_module

        TaskClass = getattr(tasks_module, task_name, None)
        if TaskClass is None:
            raise ValueError(
                f"Task class '{task_name}' not found. "
                f"Make sure the task name matches the class name (e.g., 'ReachTarget', 'StackBlocks'). "
                f"Available tasks can be found in rlbench.tasks module."
            )
        task = env.get_task(TaskClass)
        np.random.seed(seed)

        # initial reset
        reset_ret = task.reset()
        # some versions return (desc, obs); support both
        if isinstance(reset_ret, tuple) and len(reset_ret) >= 2:
            current_obs = reset_ret[1]
        else:
            current_obs = reset_ret

        while True:
            cmd, data = child_conn.recv()
            if cmd == "reset":
                ret = task.reset()
                if isinstance(ret, tuple) and len(ret) >= 2:
                    obs = ret[1]
                else:
                    obs = ret
                # send front_rgb (uint8)
                child_conn.send(obs.front_rgb.copy().astype(np.uint8))
                current_obs = obs

            elif cmd == "step":
                action = data  # np array
                ret = task.step(action)
                # task.step often returns (obs, reward, terminate)
                if len(ret) == 3:
                    obs, reward, terminate = ret
                else:
                    # fallback
                    obs = ret[0]
                    reward = float(ret[1]) if len(ret) > 1 else 0.0
                    terminate = bool(ret[-1]) if len(ret) > 2 else False
                # return front_rgb, reward, terminate
                child_conn.send(
                    (
                        obs.front_rgb.copy().astype(np.uint8),
                        float(reward),
                        bool(terminate),
                    )
                )
                current_obs = obs

            elif cmd == "get_description":
                # Get a random description from task._task_descriptions
                if hasattr(task, "_task_descriptions") and task._task_descriptions:
                    description = random.choice(task._task_descriptions)
                else:
                    # Fallback if _task_descriptions is not available
                    description = ""
                child_conn.send(description)

            elif cmd == "close":
                try:
                    env.shutdown()
                except Exception:
                    pass
                child_conn.close()
                break

            else:
                child_conn.send(("unknown_cmd", None))
    except Exception as e:
        # Propagate exception info to parent before exiting
        try:
            child_conn.send(("worker_error", str(e)))
        except Exception:
            pass
        raise


# ----------------------------
# RLBench Env
# ----------------------------
class RLBenchEnv(gym.Env):
    """
    Parallel multi-task RLBench environment with chunk_step and episode recording.
    - Spawns N worker processes; each worker runs one RLBench task instance.
    - Public API: reset(), step(actions), chunk_step(chunk_actions), flush_video(), close()
    """

    def __init__(
        self, cfg, seed_offset=0, total_num_processes=None, record_metrics=True
    ):
        """
        cfg: OmegaConf config object with environment parameters
        seed_offset: offset for seed
        total_num_processes: total number of processes across all groups
        record_metrics: whether to record metrics
        """
        from omegaconf import OmegaConf

        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes or cfg.num_envs
        self.record_metrics = record_metrics

        # Extract config values
        self.num_envs = cfg.num_envs
        self.seed = cfg.seed + seed_offset
        self.headless = True
        self.image_size = (
            tuple(cfg.image_size)
            if isinstance(cfg.image_size, list)
            else cfg.image_size
        )
        self.max_episode_steps = cfg.max_episode_steps
        self.auto_reset = cfg.auto_reset
        self.video_cfg = cfg.video_cfg

        if getattr(cfg, "save_video", False) and not self.video_cfg.save_video:
            self.video_cfg.save_video = True
        self.video_base_dir = cfg.video_base_dir
        self.task_names = (
            cfg.task_names if hasattr(cfg, "task_names") else ["ReachTarget"]
        )

        # # Action limits for relative mode (delta actions)
        # # Position delta: limit to [-0.1, 0.1] meters (10cm) to avoid IK failures
        # # Rotation delta: limit to [-0.3, 0.3] radians (~17 degrees)
        # # Gripper: limit to [0, 1] for binary control
        # self.action_pos_limit = getattr(cfg, "action_pos_limit", 0.01)  # meters
        # self.action_rot_limit = getattr(cfg, "action_rot_limit", 0.01)  # radians
        # self.action_gripper_min = getattr(cfg, "action_gripper_min", 0.0)
        # self.action_gripper_max = getattr(cfg, "action_gripper_max", 1.0)

        os.makedirs(self.video_base_dir, exist_ok=True)

        # pipes/workers
        self.parent_conns = []
        self.workers = []
        # Store task names for each env for debugging
        self.env_task_names = []
        for i in range(self.num_envs):
            parent_conn, child_conn = mp.Pipe()
            task_name = self.task_names[i % len(self.task_names)]
            self.env_task_names.append(task_name)
            # Use different seeds for each env to ensure different initializations
            env_seed = self.seed + i
            p = mp.Process(
                target=worker_process,
                args=(child_conn, task_name, env_seed, self.headless, self.image_size),
                daemon=True,
            )
            p.start()
            self.parent_conns.append(parent_conn)
            self.workers.append(p)

        # Debug: print task assignment
        print(f"[RLBenchEnv] Task assignment: {self.env_task_names}")

        # metrics
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)

        # video frames storage per env
        self.render_images = [[] for _ in range(self.num_envs)]
        self.video_cnt = 0

        # current raw obs cache (store last returned image)
        self.current_obs_images = [None] * self.num_envs

        # startup flag: require reset before step
        self._is_start = True

    # -------------------------
    # Helper: send reset to specific envs
    # -------------------------
    def reset(self, env_idx: Optional[Union[int, List[int], np.ndarray]] = None):
        """
        Reset specified envs (or all if env_idx is None).
        Returns: obs_images: np.array shape [len(env_idx), H, W, 3]
        """
        if env_idx is None:
            idxs = list(range(self.num_envs))
        elif isinstance(env_idx, (int, np.integer)):
            idxs = [int(env_idx)]
        else:
            idxs = list(env_idx)

        for i in idxs:
            self.parent_conns[i].send(("reset", None))

        obs_list = []
        for i in idxs:
            ret = self.parent_conns[i].recv()
            # Check for worker error: error messages are tuples with string as first element
            # Normal returns are single numpy arrays (image)
            if (
                isinstance(ret, tuple)
                and len(ret) == 2
                and isinstance(ret[0], str)
                and ret[0] == "worker_error"
            ):
                raise RuntimeError(f"Worker {i} error: {ret[1]}")
            obs_img = np.array(ret)
            obs_list.append(obs_img)
            # metrics reset for that env
            self.prev_step_reward[i] = 0.0
            self.success_once[i] = False
            self.fail_once[i] = False
            self.returns[i] = 0.0
            self._elapsed_steps[i] = 0
            self.current_obs_images[i] = obs_img
            if self.video_cfg.save_video:
                self.render_images[i] = [obs_img.copy()]

        if len(obs_list) == 1:
            return obs_list[0]
        else:
            return np.stack(obs_list, axis=0)

    def get_description(self, target_length: Optional[int] = None):
        """
        Get task descriptions for all environments.
        Args:
            target_length: If provided, repeat descriptions to match this length.
                          If None, returns descriptions for num_envs.
        Returns a list of descriptions, one for each environment (or repeated to target_length).
        """
        # Send get_description command to all workers
        for i in range(self.num_envs):
            self.parent_conns[i].send(("get_description", None))

        # Receive descriptions from all workers
        descriptions = []
        for i in range(self.num_envs):
            ret = self.parent_conns[i].recv()
            # Check for worker error
            if (
                isinstance(ret, tuple)
                and len(ret) == 2
                and isinstance(ret[0], str)
                and ret[0] == "worker_error"
            ):
                raise RuntimeError(f"Worker {i} error: {ret[1]}")
            descriptions.append(ret)

        # If target_length is specified and different from num_envs, repeat descriptions
        if target_length is not None and target_length != len(descriptions):
            if target_length > len(descriptions):
                # Repeat descriptions to match target_length
                repeat_times = target_length // len(descriptions)
                remainder = target_length % len(descriptions)
                descriptions = descriptions * repeat_times + descriptions[:remainder]
            else:
                # Truncate if target_length is smaller (shouldn't happen normally)
                descriptions = descriptions[:target_length]

        return descriptions

    # -------------------------
    # Step (single-step for all envs)
    # -------------------------
    def step(
        self, actions: Optional[np.ndarray] = None, auto_reset: Optional[bool] = None
    ):
        """
        actions: array-like [num_envs, action_dim], dtype float
        Returns:
            obs_dict: dict with keys "images" (torch.Tensor [num_envs,H,W,3]) and "task_descriptions" (list[str])
            rewards: torch.Tensor [num_envs], dtype float32
            terminations: torch.Tensor [num_envs], dtype bool
            truncations: torch.Tensor [num_envs], dtype bool
            infos: dict (contains episode metrics)
        """
        if actions is None:
            actions = np.zeros((self.num_envs, 7))
        if self._is_start:
            obs = self.reset()
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            infos = self._make_episode_info({})
            obs_dict = {
                "images": torch.tensor(obs).permute(0, 3, 1, 2),
                "task_descriptions": self.get_description(target_length=self.num_envs),
            }
            rewards_tensor = torch.zeros(self.num_envs, dtype=torch.float32)
            terminations_tensor = torch.tensor(terminations, dtype=torch.bool)
            truncations_tensor = torch.tensor(truncations, dtype=torch.bool)
            return (
                obs_dict,
                rewards_tensor,
                terminations_tensor,
                truncations_tensor,
                infos,
            )

        if isinstance(actions, np.ndarray) is False:
            actions = np.array(actions)

        assert actions.shape[0] == self.num_envs, (
            "actions must have shape [num_envs, action_dim]"
        )

        # # Clip actions to reasonable limits to avoid IK failures
        # # Position delta (first 3 dims): limit to [-action_pos_limit, action_pos_limit] meters
        # actions[:, :3] = np.clip(actions[:, :3], -self.action_pos_limit, self.action_pos_limit)
        # # Rotation delta (next 3 dims, Euler angles): limit to [-action_rot_limit, action_rot_limit] radians
        # actions[:, 3:6] = np.clip(actions[:, 3:6], -self.action_rot_limit, self.action_rot_limit)
        # # Gripper (last dim): limit to [action_gripper_min, action_gripper_max]
        # actions[:, 6:] = np.clip(actions[:, 6:], self.action_gripper_min, self.action_gripper_max)

        # convert rotations in actions to quaternions
        quaternions = R.from_euler("xyz", actions[:, 3:6]).as_quat()
        actions = np.concatenate([actions[:, :3], quaternions, actions[:, 6:]], axis=1)

        # send step to all workers
        for i in range(self.num_envs):
            self.parent_conns[i].send(("step", actions[i]))

        results = [self.parent_conns[i].recv() for i in range(self.num_envs)]
        obs_list, rewards, terminations = [], [], []
        for i, res in enumerate(results):
            # Check for worker error: error messages are tuples with string as first element
            # Normal returns are (img_array, reward, terminate) where img_array is numpy array
            if (
                isinstance(res, tuple)
                and len(res) == 2
                and isinstance(res[0], str)
                and res[0] == "worker_error"
            ):
                raise RuntimeError(f"Worker {i} error: {res[1]}")
            img, reward, terminate = res
            obs_list.append(img)
            rewards.append(float(reward))
            terminations.append(bool(terminate))
            # metrics update
            self.returns[i] += float(reward)
            self._elapsed_steps[i] += 1
            if terminate:
                self.success_once[i] = True

            # video frame append
            if self.video_cfg.save_video:
                self.render_images[i].append(img.copy())

            # update current obs
            self.current_obs_images[i] = img

        obs = np.stack(obs_list, axis=0)
        rewards = np.array(rewards, dtype=np.float32)
        terminations = np.array(terminations, dtype=bool)
        truncations = self._elapsed_steps >= self.max_episode_steps

        # build infos similar to Libero: include episode metrics
        infos = self._make_episode_info({})

        dones = terminations | truncations
        do_auto_reset = (
            self.auto_reset if auto_reset is None else auto_reset
        ) and self.auto_reset

        if dones.any() and do_auto_reset:
            obs_after_reset, infos_after = self._handle_auto_reset(dones, obs, infos)
            obs = obs_after_reset
            infos = infos_after

        # Convert to torch tensors and build obs dict
        obs_dict = {
            "images": torch.tensor(obs).permute(0, 3, 1, 2),
            "task_descriptions": self.get_description(target_length=self.num_envs),
        }
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminations_tensor = torch.tensor(terminations, dtype=torch.bool)
        truncations_tensor = torch.tensor(truncations, dtype=torch.bool)

        # return in Libero style: obs, reward, terminations, truncations, infos
        return obs_dict, rewards_tensor, terminations_tensor, truncations_tensor, infos

    # -------------------------
    # chunk_step (process chunk_actions on main process by repeated step calls)
    # -------------------------
    def chunk_step(self, chunk_actions: np.ndarray):
        """
        chunk_actions: [num_envs, chunk_steps, action_dim]
        returns:
            final_obs: last observation after chunk (dict with "images" and "task_descriptions")
            chunk_rewards: torch.Tensor [num_envs, chunk_steps], dtype float32
            chunk_terminations: torch.Tensor [num_envs, chunk_steps], dtype bool
            chunk_truncations: torch.Tensor [num_envs, chunk_steps], dtype bool
            infos: dict (contains episode metrics possibly updated)
        """
        n_envs = self.num_envs
        chunk_size = chunk_actions.shape[1]
        chunk_rewards_list = []
        chunk_term_list = []
        chunk_trunc_list = []

        final_obs = None
        infos = None

        # iterate micro-steps
        for t in range(chunk_size):
            actions_t = chunk_actions[:, t, :]
            obs, rewards, terminations, truncations, infos = self.step(
                actions_t, auto_reset=False
            )
            chunk_rewards_list.append(rewards)
            chunk_term_list.append(terminations)
            chunk_trunc_list.append(truncations)
            final_obs = obs

        # stack into arrays [num_envs, chunk_steps]
        chunk_rewards = torch.stack(chunk_rewards_list, dim=1)
        chunk_terms = torch.stack(chunk_term_list, axis=1)
        chunk_truncs = torch.stack(chunk_trunc_list, axis=1)

        # Determine past dones across chunk
        past_terminations = chunk_terms.any(axis=1)
        past_truncations = chunk_truncs.any(axis=1)
        past_dones = np.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            # emulate Libero behavior: do auto reset for envs that had done at any step in chunk
            # final_obs is already a dict from step(), pass it directly
            final_obs, infos = self._handle_auto_reset(past_dones, final_obs, infos)
        else:
            # Ensure infos is always valid, even if no done occurred
            if infos is None or "episode" not in infos:
                infos = self._make_episode_info({})

        # Build chunk termination/truncation indicators in Libero style:
        # If auto_reset or ignore_terminations -> zeros except last step indicates if any termination occurred in chunk.
        # Here we mimic Libero: only set last step True when any happened (for termination/truncation)
        chunk_terms[:, -1] = past_terminations
        chunk_truncs[:, -1] = past_truncations

        return final_obs, chunk_rewards, chunk_terms, chunk_truncs, infos

    # -------------------------
    # Auto reset handler - similar behavior as Libero
    # -------------------------
    def _handle_auto_reset(self, dones: np.ndarray, _final_obs: dict, infos: dict):
        """
        dones: boolean array length num_envs indicating which envs ended
        _final_obs: last observations (dict with "images" and "task_descriptions") before reset
        Returns: (obs_after_reset, infos) - and infos will include 'final_observation','final_info' etc.
        """
        # Deep copy final_obs dict to preserve it
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        # Copy episode info from infos to final_info (similar to MetaWorld, Behavior)
        # This ensures final_info["episode"] exists for env_worker.py
        final_info = {}
        if "episode" in infos:
            # Deep copy episode info to preserve metrics before reset
            final_info["episode"] = {}
            for key, value in infos["episode"].items():
                if isinstance(value, torch.Tensor):
                    final_info["episode"][key] = value.clone()
                elif isinstance(value, np.ndarray):
                    final_info["episode"][key] = value.copy()
                else:
                    final_info["episode"][key] = copy.deepcopy(value)

        # reset those envs
        obs_after_np = self.reset(env_idx=env_idx)
        # Convert to dict format with torch tensor
        obs_after = {
            "images": torch.tensor(obs_after_np).permute(0, 3, 1, 2),
            "task_descriptions": self.get_description(target_length=self.num_envs),
        }
        # assemble infos similar to Libero:
        infos_out = {}
        infos_out["final_observation"] = final_obs
        infos_out["final_info"] = final_info
        infos_out["_final_info"] = dones
        infos_out["_final_observation"] = dones
        infos_out["_elapsed_steps"] = dones
        # also include episode metrics (success_once, returns, episode_len)
        infos_out.update(self._make_episode_info({}))
        return obs_after, infos_out

    # -------------------------
    # Episode metrics packaging
    # -------------------------
    def _make_episode_info(self, extra_info: dict):
        episode_info = {}
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self._elapsed_steps.copy()
        # avoid division by zero
        avg_reward = np.zeros_like(self.returns)
        nonzero_mask = episode_info["episode_len"] > 0
        avg_reward[nonzero_mask] = (
            episode_info["return"][nonzero_mask]
            / episode_info["episode_len"][nonzero_mask]
        )
        episode_info["reward"] = avg_reward
        # Convert to torch.Tensor to match other environments (Libero, MetaWorld)
        infos = {"episode": to_tensor(episode_info)}
        infos.update(extra_info)
        return infos

    # -------------------------
    # Video utilities
    # -------------------------
    def add_new_frames(self, imgs: List[np.ndarray]):
        """Add frames from a step (imgs: list length num_envs of HWC uint8)"""
        for i, img in enumerate(imgs):
            if img is None:
                continue
            if self.video_cfg.save_video:
                self.render_images[i].append(img.copy())

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Save stored frames: combine all envs into one tiled video"""
        # Check if video saving is enabled (use self.video_cfg.save_video)
        if not self.video_cfg.save_video:
            # If video saving is disabled, just clear the buffer
            self.render_images = [[] for _ in range(self.num_envs)]
            return

        # Find the maximum number of frames across all envs
        max_frames = (
            max(len(frames) for frames in self.render_images)
            if self.render_images
            else 0
        )
        if max_frames == 0:
            # No frames to save
            self.render_images = [[] for _ in range(self.num_envs)]
            return

        # Combine frames from all envs into tiled images
        tiled_frames = []
        for frame_idx in range(max_frames):
            # Collect one frame from each env (use last frame if env has fewer frames)
            env_frames = []
            for env_idx in range(self.num_envs):
                frames = self.render_images[env_idx]
                if frame_idx < len(frames):
                    env_frames.append(frames[frame_idx])
                elif len(frames) > 0:
                    # Use last frame if this env has fewer frames
                    env_frames.append(frames[-1])
                else:
                    # If env has no frames, create a black image
                    # Use the size from the first available frame
                    if len(env_frames) > 0:
                        h, w = env_frames[0].shape[:2]
                        env_frames.append(np.zeros((h, w, 3), dtype=np.uint8))

            # Tile all env frames into one image
            if len(env_frames) > 0:
                # Calculate nrows for a roughly square grid
                nrows = int(np.sqrt(self.num_envs))
                tiled_frame = tile_images(env_frames, nrows=nrows)
                tiled_frames.append(tiled_frame)

        # Save the combined video
        seed_dir = os.path.join(self.video_base_dir, f"seed_{self.seed}")
        out_dir = (
            seed_dir if video_sub_dir is None else os.path.join(seed_dir, video_sub_dir)
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"combined_{self.video_cnt}.mp4")
        with imageio.get_writer(path, fps=20) as writer:
            for frame in tiled_frames:
                # ensure uint8 HWC
                writer.append_data(np.asarray(frame))
        print(f"[RLBenchEnv] Saved combined video: {path} (tiled {self.num_envs} envs)")

        # reset frames buffer
        self.render_images = [[] for _ in range(self.num_envs)]
        self.video_cnt += 1

    # -------------------------
    # Close
    # -------------------------
    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for w in self.workers:
            w.join(timeout=5)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    from omegaconf import OmegaConf

    # Example: Create config and initialize environment
    cfg_dict = {
        "task_names": ["ReachTarget"],
        "num_envs": 2,
        "seed": 0,
        "headless": True,
        "image_size": [128, 128],
        "max_episode_steps": 100,
        "auto_reset": True,
        "save_video": True,
        "video_base_dir": "./rlbench_videos",
    }
    cfg = OmegaConf.create(cfg_dict)

    env = RLBenchEnv(cfg, seed_offset=0, total_num_processes=2, record_metrics=False)

    # initial reset
    obs = env.reset()
    print("reset obs shape:", np.array(obs).shape)

    # random chunk actions example
    n_steps = 40
    action_dim = 7
    for t in range(n_steps):
        actions = np.zeros((env.num_envs, action_dim)).astype(np.float32)
        actions[..., 0] = -0.01
        obs, rewards, terms, truncs, infos = env.step(actions)
        print(f"t={t} rewards={rewards}, terms={terms}")

    # chunk_step example: 3 micro steps per call
    chunk = np.zeros((env.num_envs, 3, action_dim)).astype(np.float32)
    chunk[..., 0] = -0.01
    final_obs, chunk_rewards, chunk_terms, chunk_truncs, infos = env.chunk_step(chunk)
    print("chunk_rewards shape:", chunk_rewards.shape)

    env.flush_video("demo")
    env.close()
