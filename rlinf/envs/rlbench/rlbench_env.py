# parallel_rlbench_env.py
import os
import time
import copy
import numpy as np
from typing import List, Optional, Union, Tuple
import multiprocessing as mp
import imageio
import gym
from scipy.spatial.transform import Rotation as R

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

def worker_process(child_conn: mp.connection.Connection,
                   task_name: str,
                   seed: int,
                   headless: bool = True,
                   image_size: Tuple[int,int]=(128,128)):
    try:
        # build env & task
        obs_config = _make_obs_config(image_size=image_size, front_only=True)
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaIK(absolute_mode=False),
            gripper_action_mode=Discrete()
        )
        env = Environment(action_mode, obs_config=obs_config, headless=headless, robot_setup='panda')
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
                child_conn.send((obs.front_rgb.copy().astype(np.uint8), float(reward), bool(terminate)))
                current_obs = obs

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

    def __init__(self,
                cfg, seed_offset=0, total_num_processes=None, record_metrics=True):
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
        self.image_size = tuple(cfg.image_size) if isinstance(cfg.image_size, list) else cfg.image_size
        self.max_episode_steps = cfg.max_episode_steps
        self.auto_reset = cfg.auto_reset
        self.save_video = getattr(cfg, "save_video", True)
        self.video_base_dir = cfg.video_base_dir
        self.task_names = cfg.task_names if hasattr(cfg, "task_names") else ["ReachTarget"]
        
        os.makedirs(self.video_base_dir, exist_ok=True)

        # pipes/workers
        self.parent_conns = []
        self.workers = []
        for i in range(self.num_envs):
            parent_conn, child_conn = mp.Pipe()
            task_name = self.task_names[i % len(self.task_names)]
            p = mp.Process(
                target=worker_process,
                args=(child_conn, task_name, self.seed + i, self.headless, self.image_size),
                daemon=True,
            )
            p.start()
            self.parent_conns.append(parent_conn)
            self.workers.append(p)

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
            if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], str) and ret[0] == "worker_error":
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
            if self.save_video:
                self.render_images[i] = [obs_img.copy()]

        if len(obs_list) == 1:
            return obs_list[0]
        else:
            return np.stack(obs_list, axis=0)

    # -------------------------
    # Step (single-step for all envs)
    # -------------------------
    def step(self, actions: np.ndarray, auto_reset: Optional[bool] = None):
        """
        actions: array-like [num_envs, action_dim], dtype float
        Returns: obs_images (np array [num_envs,H,W,3]), rewards (np array), terminations (bool array), infos (dict)
        """
        if self._is_start:
            obs = self.reset()
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            infos = self._make_episode_info({})
            return obs, np.zeros(self.num_envs, dtype=np.float32), terminations, truncations, infos

        if isinstance(actions, np.ndarray) is False:
            actions = np.array(actions)

        assert actions.shape[0] == self.num_envs, "actions must have shape [num_envs, action_dim]"

        # convert rotations in actions to quaternions
        quaternions = R.from_euler('xyz', actions[:, 3:6]).as_quat()
        actions = np.concatenate([actions[:, :3], quaternions, actions[:, 6:]], axis=1)

        # send step to all workers
        for i in range(self.num_envs):
            self.parent_conns[i].send(("step", actions[i]))

        results = [self.parent_conns[i].recv() for i in range(self.num_envs)]
        obs_list, rewards, terminations = [], [], []
        for i, res in enumerate(results):
            # Check for worker error: error messages are tuples with string as first element
            # Normal returns are (img_array, reward, terminate) where img_array is numpy array
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == "worker_error":
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
            if self.save_video:
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
        do_auto_reset = (self.auto_reset if auto_reset is None else auto_reset) and self.auto_reset

        if dones.any() and do_auto_reset:
            obs_after_reset, infos_after = self._handle_auto_reset(dones, obs, infos)
            obs = obs_after_reset
            infos = infos_after

        # return in Libero style: obs, reward, terminations, truncations, infos
        return obs, rewards, terminations, truncations, infos

    # -------------------------
    # chunk_step (process chunk_actions on main process by repeated step calls)
    # -------------------------
    def chunk_step(self, chunk_actions: np.ndarray):
        """
        chunk_actions: [num_envs, chunk_steps, action_dim]
        returns:
            final_obs: last observation after chunk (np array [num_envs,...])
            chunk_rewards: np.array [num_envs, chunk_steps]
            chunk_terminations: np.bool array [num_envs, chunk_steps]
            chunk_truncations: np.bool array [num_envs, chunk_steps]
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
            obs, rewards, terminations, truncations, infos = self.step(actions_t, auto_reset=False)
            chunk_rewards_list.append(rewards)
            chunk_term_list.append(terminations)
            chunk_trunc_list.append(truncations)
            final_obs = obs

        # stack into arrays [num_envs, chunk_steps]
        chunk_rewards = np.stack(chunk_rewards_list, axis=1)
        chunk_terms = np.stack(chunk_term_list, axis=1)
        chunk_truncs = np.stack(chunk_trunc_list, axis=1)

        # Determine past dones across chunk
        past_terminations = chunk_terms.any(axis=1)
        past_truncations = chunk_truncs.any(axis=1)
        past_dones = np.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            # emulate Libero behavior: do auto reset for envs that had done at any step in chunk
            final_obs, infos = self._handle_auto_reset(past_dones, final_obs, infos)

        # Build chunk termination/truncation indicators in Libero style:
        # If auto_reset or ignore_terminations -> zeros except last step indicates if any termination occurred in chunk.
        # Here we mimic Libero: only set last step True when any happened (for termination/truncation)
        chunk_terminations = np.zeros_like(chunk_terms)
        chunk_truncations = np.zeros_like(chunk_truncs)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations[:, -1] = past_truncations

        return final_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos

    # -------------------------
    # Auto reset handler - similar behavior as Libero
    # -------------------------
    def _handle_auto_reset(self, dones: np.ndarray, _final_obs: np.ndarray, infos: dict):
        """
        dones: boolean array length num_envs indicating which envs ended
        _final_obs: last observations (possibly before reset)
        Returns: (obs_after_reset, infos) - and infos will include 'final_observation','final_info' etc.
        """
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = {}  # could copy more detailed info per env if available

        # reset those envs
        obs_after = self.reset(env_idx=env_idx)
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
        avg_reward[nonzero_mask] = episode_info["return"][nonzero_mask] / episode_info["episode_len"][nonzero_mask]
        episode_info["reward"] = avg_reward
        infos = {"episode": episode_info}
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
            if self.save_video:
                self.render_images[i].append(img.copy())

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Save stored frames for each env to ./video_base_dir/seed_X/video_sub_dir/env_i.mp4"""
        seed_dir = os.path.join(self.video_base_dir, f"seed_{self.seed}")
        out_dir = seed_dir if video_sub_dir is None else os.path.join(seed_dir, video_sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        for i, frames in enumerate(self.render_images):
            if len(frames) == 0:
                continue
            path = os.path.join(out_dir, f"env_{i}.mp4")
            with imageio.get_writer(path, fps=20) as writer:
                for f in frames:
                    # ensure uint8 HWC
                    writer.append_data(np.asarray(f))
            print(f"[RLBenchEnv] Saved video: {path}")
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
