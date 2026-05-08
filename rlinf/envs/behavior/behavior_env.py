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

import gc
import inspect
import json
import os
import traceback
from multiprocessing import get_context
from threading import Thread

import gymnasium as gym
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.behavior.instance_loader import ActivityInstanceLoader
from rlinf.envs.behavior.utils import (
    apply_env_wrapper,
    apply_runtime_renderer_settings,
    convert_uint8_rgb,
    setup_omni_cfg,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

__all__ = ["BehaviorEnv"]


class BehaviorProcess:
    @staticmethod
    def process_loop(cfg: DictConfig, conn, num_envs: int):
        process = None
        try:
            process = BehaviorProcess(cfg, conn, num_envs)
            process.loop()
        except Exception:
            if process is not None and process.conn is not None:
                process.conn.send({"traceback": traceback.format_exc()})
        finally:
            if process is not None:
                if process.env is not None:
                    try:
                        process.env.close()
                    except Exception:
                        pass
                if process.conn is not None:
                    process.conn.close()

    def __init__(self, cfg: DictConfig, conn, num_envs: int):
        self.conn = conn
        from omnigibson.envs import VectorEnvironment

        omni_cfg = setup_omni_cfg(cfg)
        self.instance_loader = ActivityInstanceLoader.from_omni_cfg(omni_cfg)

        # create env and apply env wrapper if enabled
        omni_cfg_dict = OmegaConf.to_container(
            omni_cfg,
            resolve=True,
            throw_on_missing=True,
        )
        self.env = VectorEnvironment(num_envs, omni_cfg_dict)
        apply_runtime_renderer_settings()
        wrapper_name = OmegaConf.select(omni_cfg, "env.env_wrapper")
        self.env = apply_env_wrapper(self.env, wrapper_name)

        # Isaac Sim's `omni.kit.app` calls ``gc.disable()`` at startup.
        # OmniGibson has self-referential cycles and leaks memory when
        # cyclic GC is disabled. Since we do not need real-time performance,
        # enable cyclic GC here so that we do not encounter OOMs in long runs.
        gc.enable()

        step_signature = inspect.signature(self.env.step)
        step_params = step_signature.parameters.values()
        step_supports_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in step_params
        )
        self.step_supports_get_obs = (
            step_supports_kwargs or "get_obs" in step_signature.parameters
        )
        self.step_supports_render = (
            step_supports_kwargs or "render" in step_signature.parameters
        )
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )

        self.conn.send({"result": self.instance_loader.activity_name})

    def step_env(self, actions, need_obs: bool):
        if self.step_supports_get_obs and self.step_supports_render:
            raw_obs, step_rewards, terminations, truncations, infos = self.env.step(
                actions, get_obs=need_obs, render=need_obs
            )
        else:
            raw_obs, step_rewards, terminations, truncations, infos = self.env.step(
                actions
            )
        if not need_obs:
            # Normalize intermediate-step observations to None so downstream
            # code can skip parsing cleanly.
            raw_obs = None
        return (
            raw_obs,
            to_tensor(step_rewards),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def reset(self, payload):
        self.instance_loader.prepare_reset(self.env)
        raw_obs, infos = self.env.reset()
        self.conn.send({"result": (raw_obs, infos)})

    def chunk_step(self, chunk_actions):
        chunk_size = chunk_actions.shape[1]
        results = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            is_last = i == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            results.append(self.step_env(actions, need_obs=need_obs))
        results = tuple(zip(*results))
        self.conn.send({"result": results})

    def loop(self):
        cmd_handlers = {
            "reset": self.reset,
            "chunk_step": self.chunk_step,
        }
        while True:
            cmd, payload = self.conn.recv()
            if cmd in cmd_handlers:
                cmd_handlers[cmd](payload)
            elif cmd == "close":
                self.env.close()
                self.env = None
                self.conn.send({"result": None})
                self.conn.close()
                self.conn = None
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")


class ThreadWithResult(Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None
    ):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.result = None
        self.start()

    def run(self):
        if self._target:
            self.result = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self.result


class BehaviorProcessProxy:
    def __init__(self, cfg: DictConfig, num_env_shard: int):
        spawn_ctx = get_context("spawn")
        self.parent_conn, child_conn = spawn_ctx.Pipe()
        self.env_process = spawn_ctx.Process(
            target=BehaviorProcess.process_loop,
            args=(
                cfg,
                child_conn,
                num_env_shard,
            ),
            daemon=True,
        )
        self.env_process.start()
        child_conn.close()
        self.last_cmd = "initialize"

    def wait_ready_msg(self):
        msg = self.wait_for_subproc("initialize")
        return BehaviorProcessProxy.msg_postprocess(msg, "initialize")

    def call_subproc(self, cmd: str, payload=None, wait=False):
        assert self.last_cmd is None, (
            f"last cmd({self.last_cmd}) not finished before calling new cmd({cmd})"
        )
        self.parent_conn.send((cmd, payload))
        self.last_cmd = cmd
        if wait:
            result = self.parent_conn.recv()
            self.last_cmd = None
            return result

    def wait_for_subproc(self, cmd: str):
        assert self.last_cmd == cmd, (
            f"last cmd({self.last_cmd}) called not equal to the cmd to wait for({cmd})"
        )
        self.last_cmd = None
        return self.parent_conn.recv()

    @staticmethod
    def msg_postprocess(msg: dict, cmd: str):
        if msg.get("traceback", None) is not None:
            raise RuntimeError(
                f"Behavior subprocess env failed on command '{cmd}':\n{msg['traceback']}"
            )
        return msg["result"]

    def wait_for_close(self):
        assert self.last_cmd == "close", (
            f"last cmd({self.last_cmd}) called but wait for close"
        )
        if self.env_process.is_alive():
            self.env_process.join(timeout=2)
            if self.env_process.is_alive():
                self.env_process.terminate()
        self.env_process = None
        try:
            self.parent_conn.close()
            self.parent_conn = None
        except Exception:
            pass


class BehaviorEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg
        self.reward_coef = cfg.get("reward_coef", 1)

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.seed = self.cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self._is_start = True
        self.num_env_subprocess = int(self.cfg.get("num_env_subprocess", 1))
        self.num_env_shard = self._split_num_envs(
            self.num_envs, self.num_env_subprocess
        )

        self.logger = get_logger()

        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = torch.tensor(cfg.max_episode_steps)
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )
        if self.record_metrics:
            self._init_metrics()
        self._init_env()

    def _split_num_envs(self, num_envs: int, num_processes: int) -> int:
        """Split ``num_envs`` across ``num_processes`` shards as evenly as possible."""
        assert num_processes > 0, f"num_processes({num_processes}) must be positive"
        assert num_envs % num_processes == 0, (
            f"num_envs({num_envs}) must be divisible by num_processes({num_processes})"
        )
        return num_envs // num_processes

    def _load_tasks_cfg(self, activity_name: str):
        # Read task description

        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[activity_name]

    def _init_env(self):
        self.env_proxys = [
            BehaviorProcessProxy(
                self.cfg,
                self.num_env_shard,
            )
            for _ in range(self.num_env_subprocess)
        ]
        activity_names = [env_proxy.wait_ready_msg() for env_proxy in self.env_proxys]

        if len(set(activity_names)) != 1:
            raise RuntimeError(
                f"Behavior env subprocesses reported different activity_name: {activity_names}"
            )
        activity_name = activity_names[0]
        self._load_tasks_cfg(activity_name)

    def call_subprocs_shards(self, cmd: str, payloads: list | None = None):
        """Send the same command to every shard; recv in parallel to avoid pipe backpressure."""
        if payloads is None:
            payload_shards = [None] * len(self.env_proxys)
        else:
            assert len(payloads) == self.num_envs, (
                f"payload_shards length {len(payload_shards)} != num subprocesses {self.num_envs}"
            )
            s = self.num_env_shard
            payload_shards = [
                payloads[i * s : (i + 1) * s] for i in range(self.num_env_subprocess)
            ]

        if self.num_env_subprocess > 1:
            recv_threads = [
                ThreadWithResult(
                    target=env_proxy.call_subproc,
                    args=(cmd, payload_shard, True),
                    daemon=True,
                )
                for env_proxy, payload_shard in zip(self.env_proxys, payload_shards)
            ]
            all_msgs = [thread.join() for thread in recv_threads]
        else:
            for env_proxy, payload_shard in zip(self.env_proxys, payload_shards):
                env_proxy.call_subproc(cmd, payload_shard)
            all_msgs = [
                env_proxy.wait_for_subproc(cmd) for env_proxy in self.env_proxys
            ]
        return [BehaviorProcessProxy.msg_postprocess(msg, cmd) for msg in all_msgs]

    def env_reset(self):
        shard_results = self.call_subprocs_shards("reset")
        all_raw_obs, all_infos = [], []
        for raw_obs, infos in shard_results:
            all_raw_obs.extend(raw_obs)
            all_infos.extend(infos)
        return all_raw_obs, all_infos

    def merge_chunk_results(self, shard_results: list, chunk_size: int):
        def cat_parts(parts):
            tensors = [p if torch.is_tensor(p) else torch.as_tensor(p) for p in parts]
            return torch.cat(tensors, dim=0)

        merged_obs_lists = []
        merged_rewards = []
        merged_terms = []
        merged_trunc = []
        merged_infos = []
        for step_idx in range(chunk_size):
            is_last = step_idx == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            if need_obs:
                merged_obs_step = []
            else:
                merged_obs_step = None
            reward_parts = []
            termination_parts = []
            truncation_parts = []
            merged_infos_step = []
            for (
                raw_obs_list,
                raw_rewards_list,
                raw_terminations_list,
                raw_truncations_list,
                raw_infos_list,
            ) in shard_results:
                if need_obs:
                    assert raw_obs_list[step_idx] is not None, (
                        f"obs is None at step {step_idx}"
                    )
                    merged_obs_step.extend(raw_obs_list[step_idx])
                else:
                    assert raw_obs_list[step_idx] is None, (
                        f"obs is not None at step {step_idx}"
                    )
                reward_parts.append(raw_rewards_list[step_idx])
                termination_parts.append(raw_terminations_list[step_idx])
                truncation_parts.append(raw_truncations_list[step_idx])
                merged_infos_step.extend(raw_infos_list[step_idx])
            merged_obs_lists.append(merged_obs_step)
            merged_rewards.append(cat_parts(reward_parts))
            merged_terms.append(cat_parts(termination_parts))
            merged_trunc.append(cat_parts(truncation_parts))
            merged_infos.append(merged_infos_step)
        return (
            merged_obs_lists,
            merged_rewards,
            merged_terms,
            merged_trunc,
            merged_infos,
        )

    def env_chunk_step(self, chunk_actions):
        shard_results = self.call_subprocs_shards("chunk_step", chunk_actions)
        return self.merge_chunk_results(shard_results, chunk_actions.shape[1])

    def env_close(self):
        for env_proxy in self.env_proxys:
            env_proxy.call_subproc("close")
        for env_proxy in self.env_proxys:
            env_proxy.wait_for_close()

    def _extract_obs_image(self, raw_obs):
        state = None
        for sensor_data in raw_obs.values():
            assert isinstance(sensor_data, dict)
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = convert_uint8_rgb(v["rgb"])
                elif "right_realsense_link:Camera:0" in k:
                    right_image = convert_uint8_rgb(v["rgb"])
                elif "zed_link:Camera:0" in k:
                    zed_image = convert_uint8_rgb(v["rgb"])
                elif "proprio" in k:
                    state = v
        assert state is not None, (
            "state is not found in the observation which is required for the behavior training."
        )

        return {
            "main_images": zed_image,  # [H, W, C]
            "wrist_images": torch.stack(
                [left_image, right_image], axis=0
            ),  # [N_IMG, H, W, C]
            "state": state,
        }

    def _wrap_obs(self, obs_list):
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        obs = {
            "main_images": torch.stack(
                [obs["main_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, H, W, C]
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, N_IMG, H, W, C]
            "task_descriptions": [self.task_description for _ in range(self.num_envs)],
            "states": torch.stack(
                [obs["state"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, 32]
        }
        return obs

    def _calc_step_reward(self, reward):
        reward = self.reward_coef * reward
        return reward

    def reset(self):
        raw_obs, infos = self.env_reset()
        obs = self._wrap_obs(raw_obs)
        rewards = torch.zeros(self.num_envs, dtype=bool)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu()
        (
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_truncations_list,
            raw_infos_list,
        ) = self.env_chunk_step(chunk_actions)

        obs_list = []
        infos_list = []
        scaled_rewards_list = []
        merged_terminations_list = []
        info_done_flags = []
        for raw_obs, raw_rewards, raw_terminations, step_infos in zip(
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_infos_list,
        ):
            if raw_obs is None:
                obs_list.append(None)
            else:
                obs_list.append(self._wrap_obs(raw_obs))
            step_rewards = self._calc_step_reward(raw_rewards)
            infos_list.append(self._record_metrics(step_rewards, step_infos))
            if self.ignore_terminations:
                raw_terminations = torch.zeros_like(raw_terminations)
            merged_terminations_list.append(raw_terminations)
            scaled_rewards_list.append(step_rewards)
            # `raw_infos_list[i]` is a list of per-env info dicts for chunk step i.
            step_done = [
                self._extract_info_done(info) if isinstance(info, dict) else False
                for info in step_infos
            ]
            info_done_flags.append(torch.tensor(step_done, dtype=torch.bool))

        chunk_rewards = torch.stack(
            scaled_rewards_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_terminations = torch.stack(
            merged_terminations_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_truncations = torch.stack(
            raw_truncations_list, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_terminations.any(dim=1)
        past_truncations = raw_truncations.any(dim=1)

        # Some OmniGibson builds may report episode completion primarily via
        # `info["done"]` while leaving `terminations`/`truncations` booleans
        # as all-False for the whole chunk. RLinf's evaluation metrics gate on
        # `terminations|truncations`, so we fall back to info-done here.
        past_info_dones = torch.stack(info_done_flags, dim=1).any(dim=1)

        # If the config asks to ignore terminations, map info-done into
        # truncations; otherwise map it into terminations.
        if self.ignore_terminations:
            past_truncations = torch.logical_or(past_truncations, past_info_dones)
        else:
            past_terminations = torch.logical_or(past_terminations, past_info_dones)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    @property
    def device(self):
        return "cuda"

    @property
    def elapsed_steps(self):
        return self.max_episode_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
        else:
            mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.prev_step_reward[mask] = 0.0
        if self.record_metrics:
            self.success_once[mask] = False
            self.returns[mask] = 0

    def _record_metrics(self, rewards, infos):
        info_lists = []
        for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
            done_dict = info.get("done", {})
            episode_info = {
                "success": done_dict.get("success", False),
                "episode_length": info.get("episode_length", 0),
            }
            self.returns[env_idx] += reward
            self.success_once[env_idx] = self.success_once[env_idx] | done_dict.get(
                "success", False
            )
            episode_info["success_once"] = self.success_once[env_idx].clone()

            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )
            if self.ignore_terminations:
                episode_info["success_at_end"] = info["success"]

            info_lists.append(episode_info)

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        return infos

    @staticmethod
    def _extract_info_done(info: dict) -> bool:
        tc = info["done"]["termination_conditions"]
        return any(v["done"] for v in tc.values())

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = infos.copy()
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset()
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        # use for multi task training
        pass

    def close(self):
        self.env_close()
