import gc
import importlib
import os
import sys

import torch
import yaml
from envs import *

sys.path.append("../../")

import logging
import multiprocessing as mp
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
from description.utils.generate_episode_instructions import (
    generate_episode_descriptions,
)
from envs._GLOBAL_CONFIGS import *
from envs.utils.create_actor import UnStableError

LOG_LEVEL = os.getenv("VECTOR_ENV_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("concurrent.futures").setLevel(logging.WARNING)
logging.getLogger("curobo").setLevel(logging.ERROR)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def update_obs(observation):
    def safe_uint8(img):
        if img is None:
            return None

        if img.dtype != np.uint8:
            return (img * 255).clip(0, 255).astype(np.uint8)
        return img

    full_image = safe_uint8(observation["observation"]["head_camera"]["rgb"])
    left_wrist_image = safe_uint8(
        observation["observation"].get("left_camera", {}).get("rgb", None)
    )
    right_wrist_image = safe_uint8(
        observation["observation"].get("right_camera", {}).get("rgb", None)
    )
    state = observation["joint_action"]["vector"]

    return {
        "full_image": full_image,
        "left_wrist_image": left_wrist_image,
        "right_wrist_image": right_wrist_image,
        "state": state,
    }


class SubEnv:
    def __init__(
        self,
        env_id: int,
        task_name: str,
        args: dict,
        env_seed: int = None,
        instruction_type="seen",
        global_lock=None,
    ):
        self.env_id = env_id
        self.task_name = task_name
        self.args = args
        self.env_seed = env_seed
        if self.env_seed is None:
            self.env_seed = self.env_id
        self.instruction = None
        self.task = class_decorator(self.task_name)
        self.instruction_type = instruction_type
        self.global_lock = global_lock
        self.lock = threading.Lock()

    def setup_task(self):
        self.close()
        self.task = class_decorator(self.task_name)

        with self.global_lock:
            with self.lock:
                trial_seed = self.env_seed
                is_valid = False
                while not is_valid:
                    try:
                        task = class_decorator(self.task_name)
                        task.setup_demo(
                            now_ep_num=trial_seed,
                            seed=trial_seed,
                            is_test=True,
                            **self.args,
                        )
                        episode_info = task.get_info()
                        is_valid = True
                    except Exception:
                        task.close_env()
                        trial_seed += 1
                        continue
                    task.close_env()

                self.episode_info_list = [episode_info]

    def create_instruction(self):
        task_descriptions = generate_episode_descriptions(
            self.task_name, self.episode_info_list, 1, self.env_seed
        )
        instruction = np.random.choice(task_descriptions[0][self.instruction_type])
        return instruction

    def step(self, actions):
        if self.get_instruction() is None:
            self.reset(env_seed=None)

        with self.lock:
            import os
            import subprocess

            import numpy as np

            # --- 1. FFmpeg Pipeline Initialization ---
            if getattr(self.task, "eval_video_ffmpeg", None) is None:
                try:
                    video_dir = os.path.join(
                        os.getenv("ROBOTWIN_PATH", "."), "eval_video_log"
                    )
                    os.makedirs(video_dir, exist_ok=True)
                    ep_seed = getattr(self, "env_seed", int(time.time()))
                    video_save_path = os.path.join(
                        video_dir, f"seed_{ep_seed}_{int(time.time())}.mp4"
                    )

                    if (
                        hasattr(self.task, "now_obs")
                        and "observation" in self.task.now_obs
                        and "head_camera" in self.task.now_obs["observation"]
                    ):
                        h, w, _ = self.task.now_obs["observation"]["head_camera"][
                            "rgb"
                        ].shape
                    else:
                        h, w = 480, 640

                    ffmpeg_process = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-y",
                            "-f",
                            "rawvideo",
                            "-vcodec",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s",
                            f"{w}x{h}",
                            "-r",
                            "30",
                            "-i",
                            "-",
                            "-c:v",
                            os.getenv("VIDEO_CODEC", "libopenh264"),
                            "-pix_fmt",
                            "yuv420p",
                            video_save_path,
                        ],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )

                    self.task.eval_video_path = video_save_path
                    self.task.eval_video_ffmpeg = ffmpeg_process
                    print(f"\n[FFmpeg] Started recording: {video_save_path}\n")
                except Exception as e:
                    print(f"[FFmpeg] Init error: {e}")

            # --- 2. Execute Action Loop ---
            actions_np = np.array(actions)
            if len(actions_np.shape) == 3:
                actions_np = actions_np[0]

            for act in actions_np:
                self.task.take_action(act)

                if getattr(self.task, "eval_video_ffmpeg", None) is not None:
                    try:
                        fresh_obs = self.task.get_obs()
                        frame = fresh_obs["observation"]["head_camera"]["rgb"]

                        if frame.dtype != np.uint8:
                            frame = (frame * 255).clip(0, 255).astype(np.uint8)

                        if frame.flags.contiguous:
                            self.task.eval_video_ffmpeg.stdin.write(frame.tobytes())
                        else:
                            self.task.eval_video_ffmpeg.stdin.write(
                                np.ascontiguousarray(frame).tobytes()
                            )
                    except Exception:
                        pass

            info = {"success": getattr(self.task, "eval_success", False)}
            reward = np.array([1 if info["success"] else 0], dtype=np.float32)
            termination = np.array([1 if info["success"] else 0], dtype=np.int32)
            truncation = np.array(
                [1 if self.task.take_action_cnt >= self.task.step_lim else 0],
                dtype=np.int32,
            )

            # --- 3. Close Pipeline if Episode Done ---
            done = bool(termination[0]) or bool(truncation[0])
            if done and getattr(self.task, "eval_video_ffmpeg", None) is not None:
                try:
                    self.task.eval_video_ffmpeg.stdin.close()
                    self.task.eval_video_ffmpeg.wait(timeout=5)
                    print(f"\n[FFmpeg] Video saved: {self.task.eval_video_path}\n")
                except Exception:
                    pass
                self.task.eval_video_path = None
                self.task.eval_video_ffmpeg = None

            obs = update_obs(self.task.get_obs())
            obs["instruction"] = self.task.get_instruction()

        return {
            "obs": obs,
            "reward": reward,
            "terminated": termination,
            "truncated": truncation,
            "info": info,
        }

    def reset(self, env_seed=None):
        with self.global_lock:
            with self.lock:
                if getattr(self.task, "eval_video_ffmpeg", None) is not None:
                    try:
                        self.task.eval_video_ffmpeg.stdin.close()
                        self.task.eval_video_ffmpeg.wait(timeout=5)
                    except Exception:
                        pass
                    self.task.eval_video_path = None
                    self.task.eval_video_ffmpeg = None

                if self.task is not None:
                    self.task.close_env()
                if env_seed is not None:
                    self.env_seed = env_seed

                self.instruction = self.create_instruction()
                self.args["instruction"] = self.instruction

                trial_seed = self.env_seed
                is_valid = False
                while not is_valid:
                    try:
                        self.task.setup_demo(
                            now_ep_num=trial_seed, seed=trial_seed, **self.args
                        )
                        self.task.step_lim = self.args["step_lim"]
                        self.task.run_steps = 0
                        self.task.reward_step = 0
                        is_valid = True
                    except UnStableError as e:
                        logging.warning(
                            f"RoboTwin SubEnv {self.env_id} reset error with seed {trial_seed}, error: {e}, trying new seed: {trial_seed + 1}"
                        )
                        self.task.close_env()
                        trial_seed += 1
                        continue
                    except Exception as e:
                        logging.error(
                            f"RoboTwin SubEnv {self.env_id} reset error with seed {trial_seed}, error: {e}"
                        )
                        self.task.close_env()
                        raise

        return

    def get_obs(self):
        with self.lock:
            obs = self.task.get_obs()
            obs = update_obs(obs)
            obs["instruction"] = self.task.get_instruction()

        return obs

    def get_instruction(self):
        with self.lock:
            if self.task is None:
                return None
            return self.instruction

    def close(self, clear_cache=True):
        if getattr(self.task, "eval_video_ffmpeg", None) is not None:
            try:
                self.task.eval_video_ffmpeg.stdin.close()
                self.task.eval_video_ffmpeg.wait(timeout=5)
            except Exception:
                pass
            self.task.eval_video_path = None
            self.task.eval_video_ffmpeg = None

        if self.task is not None:
            with self.lock:
                self.task.close_env(clear_cache=clear_cache)

    def check_seed(self, seed):
        setup_demo_success = False
        play_once_success = False

        t1 = time.time()
        with self.global_lock:
            with self.lock:
                try:
                    self.task.setup_demo(now_ep_num=seed, seed=seed, **self.args)
                    setup_demo_success = True
                    _ = self.task.get_obs()
                    _ = self.task.play_once()
                    if self.task.plan_success and self.task.check_success():
                        play_once_success = True
                except Exception as e:
                    logging.warning(
                        f"RoboTwin SubEnv {self.env_id} check_seed error with seed {seed}, error: {e}"
                    )
        t2 = time.time()
        result = {
            "setup_demo_success": setup_demo_success,
            "play_once_success": play_once_success,
            "cost_time": t2 - t1,
        }
        return result


class VectorEnv(gym.Env):
    def __init__(
        self,
        task_config,
        n_envs,
        env_seeds=None,
        instruction_type="seen",
    ):
        self.env_seeds = env_seeds
        if self.env_seeds is not None:
            assert len(self.env_seeds) == n_envs
        assets_path = os.getenv("ASSETS_PATH")
        self.task_name = task_config.get("task_name")

        head_camera_type = "D435"
        rdt_step = 10
        args = task_config

        args["planner_backend"] = args.get(
            "planner_backend", "curobo"
        )  # Choices: [curobo, mplib]

        embodiment_type = args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        args["head_camera_h"] = _camera_config[head_camera_type]["h"]
        args["head_camera_w"] = _camera_config[head_camera_type]["w"]

        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise "No embodiment files"
            return robot_file

        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args

        if len(embodiment_type) == 1:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[1])
            )
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(
            args["right_robot_file"]
        )

        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = str(embodiment_type[0]) + "_" + str(embodiment_type[1])

        args["embodiment_name"] = embodiment_name

        args["rdt_step"] = rdt_step
        args["save_path"] += f"/{args['task_name']}_reward"

        args["n_envs"] = n_envs
        args["action_dim"] = 14

        args["eval_mode"] = True
        args["eval_video_log"] = False
        args["render_freq"] = 0

        self.args = args
        self.n_envs = n_envs

        self.envs = []
        self.instruction_type = instruction_type

        self.global_lock = threading.Lock()

        self.env_thread_pool = ThreadPoolExecutor(max_workers=n_envs)

        self._init_envs()

    def _init_envs(self):
        for i in range(self.n_envs):
            sub_env = SubEnv(
                env_id=i,
                task_name=self.task_name,
                args=self.args,
                env_seed=self.env_seeds[i] if self.env_seeds else None,
                instruction_type=self.instruction_type,
                global_lock=self.global_lock,
            )
            sub_env.setup_task()
            self.envs.append(sub_env)

    def transform(self, results):
        res_dict = defaultdict(list)
        for res in results:
            for k, v in res.items():
                res_dict[k].append(v)

        res_dict = dict(res_dict)
        return (
            res_dict["obs"],
            res_dict["reward"],
            res_dict["terminated"],
            res_dict["truncated"],
            res_dict["info"],
        )

    def step(self, actions):
        if len(self.envs) == 0:
            self._init_envs()

        step_futures = {}
        for i in range(self.n_envs):
            future = self.env_thread_pool.submit(self.envs[i].step, actions[i])
            step_futures[i] = future

        results = []
        for i in range(self.n_envs):
            future = step_futures[i]
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"SubEnv {i} step error: {e}")

        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
            self.transform(results)
        )

        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def reset(self, env_idx=None, env_seeds=None):
        if len(self.envs) == 0:
            self._init_envs()

        if env_idx is None:
            env_idx = list(range(self.n_envs))
        elif isinstance(env_idx, (list, tuple)):
            env_idx = list(env_idx)
        elif isinstance(env_idx, torch.Tensor):
            env_idx = env_idx.tolist()
        else:
            env_idx = [env_idx]

        reset_futures = {}
        for idx in env_idx:
            if 0 <= idx < self.n_envs:
                seed = None
                if env_seeds is not None and len(env_seeds) == len(env_idx):
                    seed_idx = env_idx.index(idx)
                    seed = env_seeds[seed_idx]

                future = self.env_thread_pool.submit(
                    self.envs[idx].reset, env_seed=seed
                )
                reset_futures[idx] = future

        for idx in env_idx:
            if 0 <= idx < self.n_envs:
                future = reset_futures[idx]
                try:
                    future.result(timeout=120)
                except Exception as e:
                    raise RuntimeError(f"SubEnv {idx} reset error: {e}")

    def get_obs(self):
        obs_venv = []
        for env in self.envs:
            obs_venv.append(env.get_obs())

        return obs_venv

    def close(self, clear_cache=True):
        for env in self.envs:
            env.close(clear_cache=clear_cache)

        if clear_cache:
            for env in self.envs:
                env = None
            self.envs = []
            gc.collect()
            torch.cuda.empty_cache()

    def check_seeds(self, seeds: list[int]):
        assert len(seeds) == self.n_envs
        check_futures = {}
        for i in range(self.n_envs):
            future = self.env_thread_pool.submit(self.envs[i].check_seed, seeds[i])
            check_futures[i] = future

        results = [None] * self.n_envs
        for future in as_completed(check_futures.values(), timeout=120):
            for idx, f in check_futures.items():
                if f == future:
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        raise RuntimeError(f"SubEnv {idx} check seed error: {e}")
                    break

        return results


if __name__ == "__main__":
    mp.set_start_method("spawn")  # solve CUDA compatibility problem
    task_name = "place_shoe"
    n_envs = 4
    steps = 30
    horizon = 10
    action_dim = 14
    times = 10
    env = VectorEnv(task_name, n_envs, horizon)
    actions = np.zeros((n_envs, horizon, action_dim))
    for t in range(times):
        prev_obs_venv, reward_venv, truncation, termination, info_venv = env.reset()
        for step in range(steps):
            actions += np.random.randn(n_envs, horizon, action_dim) * 0.05
            actions = np.clip(actions, 0, 1)
            obs_venv, reward_venv, truncation, termination, info_venv = env.step(
                actions
            )

            if step % 10 == 0:
                env.reset()
            elif step % 5 == 0:
                env.reset(env_idx=[0, 2])

            obs = env.get_obs()
        env.close()
