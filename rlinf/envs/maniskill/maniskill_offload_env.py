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

import io
import traceback
from typing import Optional

import torch
import torch.multiprocessing as mp

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.envs.maniskill.utils import (
    cleanup_cuda_tensors,
    get_batch_rng_state,
    recursive_to_own,
    set_batch_rng_state,
)
from rlinf.envs.utils import recursive_to_device


class EnvOffloadMixin:
    def get_state(self) -> bytes:
        pass

    def load_state(self, state: bytes):
        pass


class _ManiskillEnvCore(ManiskillEnv, EnvOffloadMixin):
    def get_state(self) -> bytes:
        env_state = self.env.get_state()
        rng_state = {
            "_main_rng": self.env.unwrapped._main_rng,
            "_batched_main_rng": get_batch_rng_state(
                self.env.unwrapped._batched_main_rng
            ),
            "_main_seed": self.env.unwrapped._main_seed,
            "_episode_rng": self.env.unwrapped._episode_rng,
            "_batched_episode_rng": get_batch_rng_state(
                self.env.unwrapped._batched_episode_rng
            ),
            "_episode_seed": self.env.unwrapped._episode_seed,
        }
        action_space_state = {
            "action_space": self.env.unwrapped.action_space,
            "single_action_space": self.env.unwrapped.single_action_space,
            "_orig_single_action_space": self.env.unwrapped._orig_single_action_space,
        }
        physx_state = {
            "cuda_articulation_link_data": self.env.unwrapped.scene.px.cuda_articulation_link_data.torch().cpu(),
            "cuda_articulation_qacc": self.env.unwrapped.scene.px.cuda_articulation_qacc.torch().cpu(),
            "cuda_articulation_qf": self.env.unwrapped.scene.px.cuda_articulation_qf.torch().cpu(),
            "cuda_articulation_qpos": self.env.unwrapped.scene.px.cuda_articulation_qpos.torch().cpu(),
            "cuda_articulation_qvel": self.env.unwrapped.scene.px.cuda_articulation_qvel.torch().cpu(),
            "cuda_articulation_target_qpos": self.env.unwrapped.scene.px.cuda_articulation_target_qpos.torch().cpu(),
            "cuda_articulation_target_qvel": self.env.unwrapped.scene.px.cuda_articulation_target_qvel.torch().cpu(),
            "cuda_rigid_body_data": self.env.unwrapped.scene.px.cuda_rigid_body_data.torch().cpu(),
            "cuda_rigid_dynamic_data": self.env.unwrapped.scene.px.cuda_rigid_dynamic_data.torch().cpu(),
        }

        user_defined_task_reset_states = getattr(self.env, "task_reset_states", {})
        for key, value in user_defined_task_reset_states.items():
            if torch.is_tensor(value):
                user_defined_task_reset_states[key] = value.cpu()

        user_defined_task_metric_states = getattr(self.env, "task_metric_states", {})

        simulator_state = {
            "sim_state": env_state,
            "sim_timestep": self.env.unwrapped.scene.get_timestep(),
            "elapsed_steps": self.env.unwrapped._elapsed_steps.cpu(),
            "rng_state": rng_state,
            "action_space_state": action_space_state,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
            "is_start": self.is_start,
            "video_cnt": self.video_cnt,
            "_init_raw_obs": self.env.unwrapped._init_raw_obs,
            "agent_controller_state": recursive_to_device(
                self.env.agent.controller.get_state(), "cpu"
            ),
            "physx_state": physx_state,
            "user_defined_task_reset_states": user_defined_task_reset_states,
            "user_defined_task_metric_states": recursive_to_device(
                user_defined_task_metric_states, "cpu"
            ),
        }

        if self.record_metrics:
            simulator_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "fail_once": self.fail_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )
        buffer = io.BytesIO()
        torch.save(simulator_state, buffer)

        # force refresh GPU state
        self.env.unwrapped.scene._gpu_apply_all()
        self.env.unwrapped.scene.px.gpu_update_articulation_kinematics()
        self.env.unwrapped.scene._gpu_fetch_all()

        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        """Load simulator state from bytes buffer"""

        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        options = {
            "reconfigure": False,
        }
        for key, value in state["user_defined_task_reset_states"].items():
            options[key] = value.to(self.env.device)

        self.env.reset(seed=self.seed, options=options)

        self.env.set_state(state["sim_state"])

        self.env.unwrapped.scene.set_timestep(state["sim_timestep"])
        self.env.unwrapped._elapsed_steps = state["elapsed_steps"].to(
            self.env.unwrapped.device
        )
        controller_state = recursive_to_device(
            state["agent_controller_state"], self.env.device
        )
        self.env.agent.controller.set_state(controller_state)

        # Restore RNG state
        rng_state = state["rng_state"]
        self.env.unwrapped._main_rng = rng_state["_main_rng"]
        self.env.unwrapped._batched_main_rng = set_batch_rng_state(
            rng_state["_batched_main_rng"]
        )
        self.env.unwrapped._main_seed = rng_state["_main_seed"]
        self.env.unwrapped._episode_rng = rng_state["_episode_rng"]
        self.env.unwrapped._batched_episode_rng = set_batch_rng_state(
            rng_state["_batched_episode_rng"]
        )
        self.env.unwrapped._episode_seed = rng_state["_episode_seed"]

        # Restore simulator task state
        self.prev_step_reward = state["prev_step_reward"].to(self.device)
        self.reset_state_ids = state["reset_state_ids"].to(self.device)
        self._generator.set_state(state["generator_state"])
        self.is_start = state["is_start"]

        task_metric_states = recursive_to_device(
            state["user_defined_task_metric_states"], self.device
        )
        for key, value in task_metric_states.items():
            setattr(self.env, key, value)

        # Restore utils state
        self.video_cnt = state["video_cnt"]

        if self.record_metrics and "success_once" in state:
            self.success_once = state["success_once"].to(self.device)
            self.fail_once = state["fail_once"].to(self.device)
            self.returns = state["returns"].to(self.device)


def _maniskill_worker_main(
    cfg,
    num_envs,
    seed_offset,
    total_num_processes,
    worker_info,
    command_queue,
    result_queue,
    state_buffer: Optional[bytes],
):
    from rlinf.utils.omega_resolver import omegaconf_register

    omegaconf_register()

    try:
        env = _ManiskillEnvCore(
            cfg, num_envs, seed_offset, total_num_processes, worker_info
        )
        if state_buffer:
            env.load_state(state_buffer)

        result_queue.put({"status": "ready"})

        while True:
            command = command_queue.get()
            method_name = command.get("method")

            if method_name == "shutdown":
                break

            try:
                args = recursive_to_own(command.get("args", []))
                kwargs = recursive_to_own(command.get("kwargs", {}))

                if method_name == "__setattr__":
                    attr_name, attr_value = args
                    setattr(env, attr_name, attr_value)
                    result_queue.put({"status": "success", "data": None})
                    continue

                if not hasattr(env, method_name):
                    result_queue.put(
                        {
                            "status": "error",
                            "error": f"Method '{method_name}' not found",
                        }
                    )
                    continue

                method = getattr(env, method_name)
                if not callable(method):
                    result_queue.put(
                        {
                            "status": "error",
                            "error": f"Method '{method_name}' is not callable",
                        }
                    )
                    continue

                result = method(*args, **kwargs)
                result_queue.put(
                    {"status": "success", "data": recursive_to_own(result)}
                )
            except Exception as err:
                result_queue.put(
                    {
                        "status": "error",
                        "error": str(err),
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as err:
        result_queue.put(
            {
                "status": "error",
                "error": str(err),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        command_queue.close()
        result_queue.close()
        cleanup_cuda_tensors()


class ManiskillOffloadEnv(EnvOffloadMixin):
    """Proxy environment that runs ManiSkill core env in a spawned process.

    This class does not implement env APIs like `reset` or `chunk_step` directly.
    Missing method calls are resolved by `__getattr__`, which forwards them to the
    worker process via `_rpc(method_name, ...)`.

    Lifecycle:
    - `offload()`: optionally snapshots state (`get_state`) and shuts down worker.
    - `onload()`: starts worker again and restores from `state_buffer` if available.
    - `_rpc(...)`: auto-calls `onload()` when process is offloaded, so calls like
      `env.reset()` continue to work after offload.
    """

    _LOCAL_ATTRS = {
        "cfg",
        "num_envs",
        "seed_offset",
        "total_num_processes",
        "worker_info",
        "rpc_timeout_s",
        "context",
        "process",
        "command_queue",
        "result_queue",
        "state_buffer",
        "_has_valid_state_buffer",
    }

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
        rpc_timeout_s: int = 120,
    ):
        del record_metrics
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.rpc_timeout_s = rpc_timeout_s
        self.context: Optional[mp.context.BaseContext] = None
        self.process: Optional[mp.Process] = None
        self.command_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.state_buffer: Optional[bytes] = None
        self._has_valid_state_buffer = False
        self.onload()

    def start_env(self):
        if self.process is not None and self.process.is_alive():
            return

        self.context = mp.get_context("spawn")
        self.command_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.process = self.context.Process(
            target=_maniskill_worker_main,
            args=(
                self.cfg,
                self.num_envs,
                self.seed_offset,
                self.total_num_processes,
                self.worker_info,
                self.command_queue,
                self.result_queue,
                self.state_buffer,
            ),
        )
        self.process.start()
        try:
            result = self.result_queue.get(timeout=self.rpc_timeout_s)
        except Exception as err:
            self._force_shutdown()
            raise RuntimeError(
                f"Maniskill offload init timeout/failure: {err}"
            ) from err
        if result.get("status") != "ready":
            self._force_shutdown()
            raise RuntimeError(f"Maniskill offload init failed: {result}")

    def onload(self, force: bool = False):
        if not force and self.process is not None and self.process.is_alive():
            return
        self.start_env()
        # If no valid serialized state is available, bootstrap with one reset so
        # auto-reset training can safely call step/chunk_step after onload.
        if (not self._has_valid_state_buffer) and getattr(
            self.cfg, "auto_reset", False
        ):
            self._rpc("reset")

    def offload(self, keep_state: bool = True):
        if self.process is None or not self.process.is_alive():
            return

        if keep_state:
            try:
                self.state_buffer = self._rpc("get_state")
                self._has_valid_state_buffer = True
            except Exception:
                self.state_buffer = None
                self._has_valid_state_buffer = False
        else:
            self.state_buffer = None
            self._has_valid_state_buffer = False

        try:
            self.command_queue.put({"method": "shutdown"})
        finally:
            self._force_shutdown()

    def stop_env(self):
        self.offload(keep_state=True)

    def close(self):
        if self.process is not None and self.process.is_alive():
            try:
                self._rpc("close", timeout_s=30)
            except Exception:
                pass
        self.offload(keep_state=False)

    def get_state(self) -> bytes:
        return self._rpc("get_state")

    def load_state(self, state: bytes):
        self._rpc("load_state", args=[state])

    def _rpc(self, method_name, args=None, kwargs=None, timeout_s=None):
        if self.process is None or not self.process.is_alive():
            self.onload()
        payload = {
            "method": method_name,
            "args": recursive_to_own(args or []),
            "kwargs": recursive_to_own(kwargs or {}),
        }
        self.command_queue.put(payload)
        result = self.result_queue.get(timeout=timeout_s or self.rpc_timeout_s)
        result = recursive_to_own(result)
        if result.get("status") == "error":
            trace = result.get("traceback")
            if trace:
                raise RuntimeError(f"{result.get('error')}\n{trace}")
            raise RuntimeError(result.get("error", "Unknown offload RPC error"))
        return result.get("data")

    def _force_shutdown(self):
        if self.command_queue is not None:
            self.command_queue.close()
        if self.result_queue is not None:
            self.result_queue.close()
        if self.process is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        self.process = None
        self.command_queue = None
        self.result_queue = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        def method_proxy(*args, **kwargs):
            return self._rpc(name, args=args, kwargs=kwargs)

        return method_proxy

    def __setattr__(self, name, value):
        if name in self._LOCAL_ATTRS:
            super().__setattr__(name, value)
            return

        if name.startswith("_"):
            raise AttributeError(
                f"Cannot set private attribute '{name}' on offloaded environment"
            )
        self._rpc("__setattr__", args=[name, value])
