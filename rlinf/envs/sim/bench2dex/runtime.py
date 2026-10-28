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

# This file contains code copied from the Bench2Dex project: https://github.com/Bench2Dex/Bench2Dex.
# The original code is licensed under the MIT License.

import gc
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rlinf.envs.sim.bench2dex.cameras import Bench2DexCameras
from rlinf.envs.sim.bench2dex.robots import JointTargets
from rlinf.envs.sim.bench2dex.scene import build_scene
from rlinf.envs.sim.bench2dex.success import TerminalSuccess

# One 20 Hz action spans three 60 Hz physics steps.
_PHYSICS_STEPS_PER_ACTION = 3
_RENDER_WARMUP_STEPS = 5
_RESET_SETTLE_STEPS = 60


def make_env(
    cfg: dict,
    ignore_terminations: bool,
) -> tuple["Bench2DexRuntime", Any]:
    """Launch Isaac Sim, then build the benchmark runtime.

    Returns:
        The runtime and the ``SimulationApp`` that hosts it.
    """
    os.environ.pop("DISPLAY", None)
    from isaaclab.app import AppLauncher

    # Render cameras like the official headless runs: GUI rendering kit, viewport on.
    app = AppLauncher(
        headless=True,
        enable_cameras=True,
        video=True,
        experience="isaaclab.python.rendering.kit",
    ).app
    try:
        env = Bench2DexRuntime(cfg, ignore_terminations)
    except Exception:
        app.close()
        raise
    return env, app


class Bench2DexRuntime:
    """Own benchmark scenes, joint control, cameras, and terminal success."""

    def __init__(
        self,
        cfg: dict,
        ignore_terminations: bool,
    ):
        import isaaclab.sim as sim_utils

        self.cfg = cfg
        self.bench2dex_path = Path(cfg["bench2dex_path"])
        self.assets_path = Path(cfg["assets_path"])
        self.ignore_terminations = ignore_terminations
        task_name = cfg["task_name"]
        with (self.bench2dex_path / "scenes" / f"{task_name}.yaml").open() as stream:
            self.task = yaml.safe_load(stream)
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=0.0166666,
                device="cuda:0",
                physx=sim_utils.PhysxCfg(
                    enable_ccd=True,
                    enable_stabilization=True,
                    bounce_threshold_velocity=0.01,
                    gpu_max_rigid_contact_count=2**23,
                    gpu_max_rigid_patch_count=2**22,
                ),
            )
        )
        self.device = torch.device(self.sim.device)
        self.physics_dt = self.sim.get_physics_dt()
        self.objects = {}
        self.cameras = None
        self.robot = None
        self._closed = False

    def reset(
        self, seed: int | None = None, env_ids: torch.Tensor | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore the benchmark scene, default joint state, and success history."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
        if self.objects:
            self._clear_scene()
        self.objects, self.compensate = build_scene(
            self.task, self.assets_path, self.cfg["robot_key"]
        )
        self.robot = self.objects["global_robot"]
        self.cameras = Bench2DexCameras(
            self.bench2dex_path / "configs" / "collect" / "default.yaml",
            self.cfg["robot_key"],
            self.cfg["camera_names"],
            self.device,
        )
        self.sim.reset()
        # Wait for PhysX to finish re-initializing on the GPU; reading articulation
        # views earlier can hit stale Fabric memory after many scene rebuilds.
        torch.cuda.synchronize(self.device)
        for obj in self.objects.values():
            data = obj.data
            root = data.default_root_state.clone()
            obj.write_root_pose_to_sim(root[:, :7])
            if (
                not obj.cfg.spawn.rigid_props
                or not obj.cfg.spawn.rigid_props.kinematic_enabled
            ):
                obj.write_root_velocity_to_sim(root[:, 7:])
            if hasattr(obj, "write_joint_state_to_sim"):
                obj.write_joint_state_to_sim(
                    data.default_joint_pos.clone(), data.default_joint_vel.clone()
                )
            obj.reset()
        self.targets = self.robot.data.default_joint_pos.clone()
        self.joints = JointTargets(
            self.cfg["robot_key"],
            self.bench2dex_path / "robots" / "active_dof_maps.yml",
            self.assets_path,
            list(self.robot.joint_names),
        )
        if len(self.joints.indices) != self.cfg["action_dim"]:
            raise ValueError(
                f"Robot has {len(self.joints.indices)} active joints, but action_dim is {self.cfg['action_dim']}."
            )
        self.cameras.initialize(self.robot)
        for _ in range(_RENDER_WARMUP_STEPS):
            self.sim.render()
        for _ in range(_RESET_SETTLE_STEPS):
            self._physics_step()
        self.metrics = TerminalSuccess(self.task["metrics"], self.physics_dt)
        self.obs = self._get_obs()
        return self.obs, {}

    def _physics_step(self):
        if self.compensate:
            view = self.robot.root_physx_view
            effort = (
                view.get_gravity_compensation_forces()
                + view.get_coriolis_and_centrifugal_compensation_forces()
            )
            self.robot.set_joint_effort_target(effort[:, : self.joints.full_dof])
        self.robot.set_joint_position_target(self.targets)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        for obj in self.objects.values():
            obj.update(self.physics_dt)

    def _get_obs(self) -> dict:
        self.cameras.sync()
        self.sim.render()
        images = self.cameras.capture(self.cfg["image_size"], self.physics_dt)
        obs = {
            "main_images": images[0],
            "states": self.robot.data.joint_pos[:, self.joints.indices].clone(),
            "task_descriptions": [self.task["description"]],
        }
        if len(images) > 1:
            obs["extra_view_images"] = torch.stack(images[1:], dim=1)
        return obs

    def _object_states(self) -> dict:
        states = {}
        for name, obj in self.objects.items():
            if name == "global_robot":
                continue
            data = obj.data
            root = data.root_state_w[0].cpu().numpy()
            state = {
                "pose_world": root[[0, 1, 2, 4, 5, 6, 3]],
                "lin_vel_world": root[7:10],
                "ang_vel_world": root[10:13],
            }
            if hasattr(obj, "write_joint_state_to_sim"):
                state.update(
                    joint_names=list(obj.joint_names),
                    qpos=data.joint_pos[0].cpu().numpy(),
                    qvel=data.joint_vel[0].cpu().numpy(),
                )
            states[name] = state
        return states

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Hold active joint targets for three physics steps and track stable success."""
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.targets = self.joints.expand(actions)
        if self.ignore_terminations or not self.metrics.success:
            for _ in range(_PHYSICS_STEPS_PER_ACTION):
                self._physics_step()
                self.metrics.update(self._object_states())
                if self.metrics.success and not self.ignore_terminations:
                    break
            self.obs = self._get_obs()
        success = self.metrics.success
        terminated = torch.tensor([success], device=self.device)
        return (
            self.obs,
            terminated.float(),
            terminated,
            torch.zeros(1, dtype=torch.bool, device=self.device),
            {
                "success_at_end": torch.tensor(
                    [self.metrics.instant_success], device=self.device
                ),
            },
        )

    def _clear_scene(self):
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.prims as prim_utils
        import omni.kit.app

        if self.cameras is not None:
            self.cameras.close()
        self.cameras = None
        self.objects.clear()
        self.robot = None
        gc.collect()
        # Isaac Lab's stop callback otherwise blocks until the timeline plays again.
        stop_guard = self.sim._disable_app_control_on_stop_handle
        self.sim._disable_app_control_on_stop_handle = True
        try:
            self.sim._timeline.stop()
            self.sim._timeline.commit()
            app = omni.kit.app.get_app()
            app.update()
            stage = sim_utils.get_current_stage()
            for path in (
                "/World/Objects",
                "/World/Sensors",
                "/World/defaultGroundPlane",
                "/World/DistantLight",
            ):
                if stage.GetPrimAtPath(path).IsValid():
                    prim_utils.delete_prim(path)
            app.update()
            app.update()
        finally:
            self.sim._disable_app_control_on_stop_handle = stop_guard

    def close(self) -> None:
        """Release scene resources before the owner closes SimulationApp."""
        if self._closed:
            return
        self._closed = True
        self._clear_scene()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
