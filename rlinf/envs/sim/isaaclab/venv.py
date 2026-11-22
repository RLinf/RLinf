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

import contextlib
import traceback
from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from .utils import CloudpickleWrapper

_CHILD_POLL_INTERVAL_S = 1.0
_CHILD_CLOSE_TIMEOUT_S = 60.0


class _ChildFailure:
    """Traceback the subprocess sends in place of a result."""

    def __init__(self, traceback_text: str):
        self.traceback_text = traceback_text


def _torch_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    reset_idx_queue: mp.Queue,
):
    parent_remote.close()
    isaac_env = None
    try:
        env_fn = env_fn_wrapper.x
        isaac_env, sim_app = env_fn()
        device = isaac_env.device
        while True:
            try:
                cmd = child_remote.recv()
            except EOFError:
                child_remote.close()
                break
            if cmd == "reset":
                reset_index, reset_seed = reset_idx_queue.get()
                if reset_index is None:
                    reset_result = isaac_env.reset(seed=reset_seed)
                else:
                    reset_result = isaac_env.reset(
                        seed=reset_seed, env_ids=reset_index.to(device)
                    )
                child_remote.send(reset_result)
            elif cmd == "step":
                input_action = action_queue.get()
                step_result = isaac_env.step(input_action)
                child_remote.send(step_result)
            elif cmd == "close":
                isaac_env.close()
                child_remote.close()
                sim_app.close()
                break
            elif cmd == "device":
                child_remote.send(isaac_env.device)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        child_remote.close()
    except Exception:
        with contextlib.suppress(OSError):
            child_remote.send(_ChildFailure(traceback.format_exc()))
        raise
    finally:
        if isaac_env is not None:
            try:
                isaac_env.close()
            except Exception as e:
                print(f"IsaacLab Env Closed with error: {e}")


class SubProcIsaacLabEnv:
    def __init__(self, env_fn):
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)
        self.action_queue = ctx.Queue()
        self.reset_idx = ctx.Queue()
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.action_queue,
            self.reset_idx,
        )
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        self.isaac_lab_process.start()
        self.child_remote.close()

    def _exit_message(self) -> str:
        self.isaac_lab_process.join(timeout=_CHILD_POLL_INTERVAL_S)
        return (
            f"Expected a result from IsaacLab subprocess {self.isaac_lab_process.pid}, "
            f"but it stopped with exit code {self.isaac_lab_process.exitcode}."
        )

    def _send(self, cmd: str) -> None:
        try:
            self.parent_remote.send(cmd)
        except OSError:
            # The subprocess may have left its traceback in the pipe.
            self._recv()
            raise RuntimeError(self._exit_message()) from None

    def _recv(self):
        """Return the next result, raising if the subprocess failed or died.

        Results share the control pipe so that a dead subprocess ends any read,
        including one partway through a message, with EOF.
        """
        try:
            while not self.parent_remote.poll(_CHILD_POLL_INTERVAL_S):
                if (
                    not self.isaac_lab_process.is_alive()
                    and not self.parent_remote.poll()
                ):
                    raise EOFError
            result = self.parent_remote.recv()
        except (EOFError, OSError):
            raise RuntimeError(self._exit_message()) from None
        if isinstance(result, _ChildFailure):
            raise RuntimeError(
                f"IsaacLab subprocess {self.isaac_lab_process.pid} failed.\n"
                f"{result.traceback_text}"
            )
        return result

    def reset(self, seed=None, env_ids=None):
        self._send("reset")
        self.reset_idx.put((env_ids, seed))
        obs, info = self._recv()
        return obs, info

    def step(self, action: torch.Tensor):
        """
        action : (bs, action_dim)
        """
        self._send("step")
        self.action_queue.put(action)
        env_step_result = self._recv()
        return env_step_result

    def close(self):
        """Stop the subprocess. Calling this again has no effect."""
        if self.parent_remote.closed:
            return
        try:
            self.parent_remote.send("close")
        except OSError:
            pass
        self.isaac_lab_process.join(timeout=_CHILD_CLOSE_TIMEOUT_S)
        if self.isaac_lab_process.is_alive():
            self.isaac_lab_process.terminate()
            self.isaac_lab_process.join(timeout=_CHILD_CLOSE_TIMEOUT_S)
        if self.isaac_lab_process.is_alive():
            self.isaac_lab_process.kill()
            self.isaac_lab_process.join()
        self.parent_remote.close()

    def device(self):
        self._send("device")
        return self._recv()
