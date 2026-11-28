# Copyright 2026 The RLinf Authors.
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

"""CPU stand-ins for the Arena/Kit interpreter used by the environment tests."""

import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch


class Droid(gym.Env):
    """Three lanes that succeed, drop an object and time out at different steps."""

    device = torch.device("cpu")
    metadata = {"render_fps": 15}

    def __init__(self, num_envs=3, with_rewards=True, seed=0, limit=None, prompt=None):
        self.num_envs = num_envs
        self.seed_value = seed
        self.episode_limit = None
        self.starts = 0
        self.active = torch.ones(num_envs, dtype=torch.bool)
        self.limit = limit
        self.prompt = prompt or "Place the banana on the plate."
        self.steps = torch.zeros(num_envs, dtype=torch.long)
        self.gripper = torch.zeros(num_envs, 1)
        self.placement = torch.zeros(num_envs, dtype=torch.uint8)
        self.rng = torch.Generator().manual_seed(seed)
        self.success = self.steps.bool()
        self.termination_manager = SimpleNamespace(
            active_terms=["success", "drop"], get_term=lambda key: self.success
        )
        self.reward_manager = SimpleNamespace(
            active_terms=["lift"] if with_rewards else []
        )

    def configure_episode_limit(self, n):
        assert self.starts == 0, "Native episode limits are single-use"
        self.episode_limit = n

    def get_language_instruction(self):
        return self.prompt

    @property
    def episode_length_buf(self):
        return self.steps

    def _raw(self):
        image = self.steps[:, None, None, None].expand(-1, 2, 4, 3).byte().clone()
        image[:, 0, 0, 0] = self.placement
        return {
            "camera_obs": {
                "external_camera_rgb": image,
                "wrist_camera_rgb": image + 50,
            },
            "policy": {
                "joint_pos": self.steps[:, None] + torch.arange(7, dtype=torch.float32),
                "gripper_pos": self.gripper.clone(),
            },
        }

    def reset(self, *, seed=None, options=None):
        assert self.episode_limit is None or not self.starts, (
            "An episode-limited Arena allows one reset"
        )
        self.starts += self.num_envs
        self.placement = torch.randint(
            0, 128, (self.num_envs,), generator=self.rng, dtype=torch.uint8
        )
        self.active.fill_(True)
        self.steps.zero_()
        self.gripper.fill_((self.seed_value if seed is None else seed) / 100)
        return self._raw(), {}

    def step(self, action):
        self.steps += 1
        lane = torch.arange(self.num_envs)
        self.gripper = action[:, -1:]
        self.success = (lane == 0) & (self.steps == 1)
        term = self.success | ((lane == 1) & (self.steps == 2))
        trunc = (lane == 2) & (self.steps == 3)
        if self.limit:
            trunc |= self.limit.func(self, **self.limit.params)
        final = self._raw()
        term &= self.active
        trunc &= self.active
        if self.episode_limit is None:
            count = int((term | trunc).sum())
            self.placement[term | trunc] = torch.randint(
                0, 128, (count,), generator=self.rng, dtype=torch.uint8
            )
            self.starts += count
            self.steps[term | trunc] = 0
        else:
            self.active &= ~(term | trunc)
        return self._raw(), lane.float() + 1, term, trunc, {"final_obs": final}


def _run_native_standin():
    """Install vendor stand-ins in the simulator subprocess and run the real IPC loop."""

    Droid.metadata = {"render_fps": int(os.environ.get("ARENA_TEST_RENDER_FPS", "15"))}
    rerenders = int(os.environ.get("ARENA_TEST_RERENDERS", "0"))

    def definition(embodiment="droid_abs_joint_pos", **kwargs):
        return SimpleNamespace(
            embodiment=SimpleNamespace(
                registry_name=embodiment,
                camera_config=SimpleNamespace(fit_resolution=lambda h, w: None),
            )
        )

    class Builder:
        def __init__(self, environment, cfg):
            self.cfg = cfg

        def build_env_cfg(self):
            return SimpleNamespace(
                terminations=SimpleNamespace(),
                num_rerenders_on_reset=rerenders,
                recorders=SimpleNamespace(
                    dataset_export_mode=1, pick_and_place=object()
                ),
            ), {}

        def make_registered(self, cfg, kwargs):
            assert (
                cfg.recorders.dataset_export_mode == 0
                and cfg.recorders.pick_and_place is not None
            ), "Keep task metrics without creating HDF5 files"
            assert (
                cfg.num_rerenders_on_reset == max(1, rerenders)
                and self.cfg.env_cfg_override["compute_final_obs"]
            ), "Reset must refresh both cameras and preserve terminal observations"
            return gym.wrappers.OrderEnforcing(
                Droid(
                    self.cfg.num_envs,
                    seed=self.cfg.seed,
                    limit=getattr(cfg.terminations, "rlinf_step_limit", None),
                    prompt=self.cfg.language_instruction,
                )
            )

    def read_graph(path):
        if path.name != "banana_on_plate.yaml":
            raise FileNotFoundError(f"No Arena graph at {path}")
        return SimpleNamespace(to_arena_env=definition)

    class MapleTable:
        def build(self, cfg):
            return definition(getattr(cfg, "embodiment", "droid_abs_joint_pos"))

    environments = {"pick_and_place_maple_table": MapleTable}

    def launch(device, **kwargs):
        assert kwargs.get("multi_gpu") is False, (
            "Kit only selects the assigned CUDA device for single-GPU rendering"
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible and int(device.split(":")[1]) >= len(visible.split(",")):
            raise ValueError(
                f"Device {device} lies outside the worker GPU mask {visible}"
            )
        return SimpleNamespace(app=SimpleNamespace(close=lambda: None))

    modules = {
        "isaaclab.app": {"AppLauncher": launch},
        "isaaclab.managers": {
            "DatasetExportMode": SimpleNamespace(EXPORT_NONE=0),
            "TerminationTermCfg": SimpleNamespace,
        },
        "isaaclab_arena_environments": {"__file__": __file__},
        "isaaclab_arena.assets.registries": {
            "EnvironmentRegistry": lambda: SimpleNamespace(
                get_component_by_name=environments.__getitem__,
                get_environment_cfg_type=lambda cls: SimpleNamespace,
            )
        },
        "isaaclab_arena.environment_spec.arena_env_graph_spec": {
            "ArenaEnvGraphSpec": SimpleNamespace(from_yaml=read_graph)
        },
        "isaaclab_arena.environments.arena_env_builder": {"ArenaEnvBuilder": Builder},
        "isaaclab_arena.environments.arena_env_builder_cfg": {
            "ArenaEnvBuilderCfg": SimpleNamespace
        },
        "isaaclab_arena_openpi.policy.droid_adapter": {
            "Pi0DroidAdapter": SimpleNamespace(target_image_size=(224, 224))
        },
    }
    sys.modules.update(
        {name: SimpleNamespace(**members) for name, members in modules.items()}
    )
    runpy.run_path(sys.argv[-2], run_name="__main__")


def _run_fault_standin(mode):
    """Exercise the real supervisor and socket protocol with a stalled Kit helper."""
    import ctypes
    import signal
    import socket
    import struct
    import subprocess

    runtime = runpy.run_path(sys.argv[-2])
    sock = socket.socket(fileno=int(sys.argv[-1]))
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    runtime["receive_message"](sock)
    if mode != "startup":
        runtime["send_message"](sock, (True, 15))
    Path(os.environ["ARENA_FAULT_PIDS"]).write_text(f"{os.getpid()} {helper.pid}")
    if mode == "leader-exit":
        return
    if mode not in ("startup", "blocked-send"):
        runtime["receive_message"](sock)
    if mode in ("partial", "partial-eof"):
        sock.sendall(struct.pack("!Q", 100) + b".")
        if mode == "partial-eof":
            sock.close()
    elif mode == "interrupt":
        os.kill(os.getppid(), signal.SIGINT)
    # Simulate a native Kit call that blocks the interpreter, including its GIL.
    ctypes.PyDLL(None).sleep(120)


if __name__ == "__main__":
    if mode := os.environ.get("ARENA_FAULT"):
        _run_fault_standin(mode)
    else:
        _run_native_standin()
