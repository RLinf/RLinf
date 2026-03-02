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

from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from .utils import CloudpickleWrapper


def _to_cpu(obj):
    """Recursively move tensors in nested structures to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(x) for x in obj)
    return obj


def _droid_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    obs_queue: mp.Queue,
    reset_idx_queue: mp.Queue,
):
    import numpy as np

    parent_remote.close()
    env_fn = env_fn_wrapper.x
    droid_env, sim_app = env_fn()
    device = getattr(droid_env, "device", None)
    if device is None and hasattr(droid_env, "unwrapped"):
        device = getattr(droid_env.unwrapped, "device", torch.device("cuda:0"))
    try:
        while True:
            try:
                cmd = child_remote.recv()
            except EOFError:
                child_remote.close()
                break
            if cmd == "reset":
                reset_index, reset_seed = reset_idx_queue.get()
                # DROID/sim_evals typically supports only full reset(seed=...)
                reset_result = droid_env.reset(seed=reset_seed)
                obs, info = reset_result
                obs_queue.put((_to_cpu(obs), _to_cpu(info)))
            elif cmd == "step":
                input_action = action_queue.get()
                if isinstance(input_action, np.ndarray):
                    input_action = torch.from_numpy(input_action).to(device)
                step_result = droid_env.step(input_action)
                obs, reward, term, trunc, info = step_result
                obs_queue.put((
                    _to_cpu(obs),
                    _to_cpu(reward),
                    _to_cpu(term),
                    _to_cpu(trunc),
                    _to_cpu(info),
                ))
            elif cmd == "close":
                droid_env.close()
                child_remote.close()
                sim_app.close()
                break
            elif cmd == "device":
                child_remote.send(device)
            else:
                child_remote.close()
                raise NotImplementedError(f"Unknown cmd: {cmd}")
    except KeyboardInterrupt:
        child_remote.close()
    finally:
        try:
            droid_env.close()
        except Exception as e:
            print(f"DROID env closed with error: {e}")
        try:
            sim_app.close()
        except Exception:
            pass


class SubProcDroidEnv:
    """Subprocess wrapper for DROID (Isaac Lab) environment."""

    def __init__(self, env_fn):
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
        self.proc = ctx.Process(target=_droid_worker, args=args, daemon=True)
        self.proc.start()
        self.child_remote.close()

    def reset(self, seed=None, env_ids=None):
        self.parent_remote.send("reset")
        self.reset_idx.put((env_ids, seed))
        obs, info = self.obs_queue.get()
        return obs, info

    def step(self, action):
        """action: (batch, action_dim) numpy or tensor."""
        self.parent_remote.send("step")
        self.action_queue.put(action)
        return self.obs_queue.get()

    def close(self):
        self.parent_remote.send("close")
        self.proc.join(timeout=5.0)
        if self.proc.is_alive():
            self.proc.terminate()

    def device(self):
        self.parent_remote.send("device")
        return self.parent_remote.recv()
