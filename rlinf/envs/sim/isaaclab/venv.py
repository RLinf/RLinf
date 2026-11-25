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

import queue
import time
from multiprocessing.connection import Connection

import torch.multiprocessing as mp

from .utils import CloudpickleWrapper


def _torch_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    obs_queue: mp.Queue,
    reset_idx_queue: mp.Queue,
):
    parent_remote.close()
    env_fn = env_fn_wrapper.x
    isaac_env, sim_app = env_fn()
    device = isaac_env.device
    try:
        while True:
            try:
                cmd = child_remote.recv()
            except EOFError:
                break
            if cmd == "reset":
                reset_index, reset_seed = reset_idx_queue.get()
                if reset_index is None:
                    reset_result = isaac_env.reset(seed=reset_seed)
                else:
                    reset_result = isaac_env.reset(
                        seed=reset_seed, env_ids=reset_index.to(device)
                    )
                obs_queue.put(reset_result)
            elif cmd == "step":
                input_action = action_queue.get()
                step_result = isaac_env.step(input_action)
                obs_queue.put(step_result)
            elif cmd == "close":
                break
            elif cmd == "device":
                child_remote.send(isaac_env.device)
            else:
                raise NotImplementedError(cmd)
    finally:
        child_remote.close()
        try:
            isaac_env.close()
        finally:
            sim_app.close()


class SubProcIsaacLabEnv:
    def __init__(self, env_fn, *, timeout: float | None = None):
        self.timeout = timeout
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)
        self.action_queue = ctx.Queue()
        self.obs_queue = ctx.Queue()
        self.reset_idx = ctx.Queue()
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.action_queue,
            self.obs_queue,
            self.reset_idx,
        )
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        self.isaac_lab_process.start()
        self.child_remote.close()

    def reset(self, seed=None, env_ids=None):
        self.parent_remote.send("reset")
        self.reset_idx.put((env_ids, seed))
        obs, info = self._receive()
        return obs, info

    def step(self, action):
        self.parent_remote.send("step")
        self.action_queue.put(action)
        return self._receive()

    def _receive(self):
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while True:
            try:
                return self.obs_queue.get(timeout=1)
            except queue.Empty:
                if not self.isaac_lab_process.is_alive():
                    self.close()
                    raise RuntimeError("Isaac Lab subprocess exited before replying")
                if deadline is not None and time.monotonic() >= deadline:
                    self._close(graceful=False)
                    raise TimeoutError("Isaac Lab subprocess timed out")

    def device(self):
        try:
            self.parent_remote.send("device")
            if not self.parent_remote.poll(self.timeout):
                raise TimeoutError("Isaac Lab startup timed out")
            return self.parent_remote.recv()
        except BaseException:
            self._close(graceful=False)
            raise

    def close(self):
        self._close(graceful=True)

    def _close(self, *, graceful: bool):
        process = self.isaac_lab_process
        if process is None:
            return
        try:
            if graceful and process.is_alive():
                try:
                    self.parent_remote.send("close")
                except (BrokenPipeError, EOFError, OSError):
                    pass
                else:
                    process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
            if process.is_alive():
                raise TimeoutError("Isaac Lab subprocess did not exit after kill")
            process.close()
            self.isaac_lab_process = None
        finally:
            self.parent_remote.close()
            for channel in (self.action_queue, self.obs_queue, self.reset_idx):
                channel.cancel_join_thread()
                channel.close()
