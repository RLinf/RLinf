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

"""Batched MuJoCo Warp world.

``physics_step`` integrates once. ``forward`` refreshes derived quantities
and does not integrate. ``from_model`` accepts an ``MjModel`` that was
compiled outside this class.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

__all__ = ["MjWarpWorld"]


def _use_supported_solver(mj_model: object) -> None:
    import mujoco

    if mj_model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
        # MJWarp does not implement the PGS solver; CG is the closest
        # supported option.
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG


class MjWarpWorld:
    """One MuJoCo Warp model and its batched world data.

    ``qpos``, ``qvel``, and ``ctrl`` are writable tensors of shape
    ``(num_envs, nq)``, ``(num_envs, nv)``, and ``(num_envs, nu)``.
    ``model`` and ``data`` are the underlying MuJoCo Warp objects.
    ``mj_model`` is the native MuJoCo model.

    Args:
        mj_model: Native model. A PGS solver is replaced with CG.
        num_envs: Number of parallel worlds.
        device: ``cpu`` or ``cuda`` device.
    """

    def __init__(
        self,
        mj_model: object,
        num_envs: int,
        device: Union[torch.device, str],
    ) -> None:
        import mujoco_warp as mjw
        import warp as wp

        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"MjWarpWorld supports only CPU and CUDA devices, got {self.device}."
            )
        _use_supported_solver(mj_model)
        wp.init()
        self._wp = wp
        self._mjw = mjw
        self._wp_device = str(self.device)
        self.mj_model = mj_model

        with wp.ScopedDevice(self._wp_device):
            self.model = mjw.put_model(mj_model)
            self.data = mjw.make_data(mj_model, nworld=self.num_envs)

        self._qpos = wp.to_torch(self.data.qpos)
        self._qvel = wp.to_torch(self.data.qvel)
        self._ctrl = wp.to_torch(self.data.ctrl)
        self.nq = int(mj_model.nq)
        self.nv = int(mj_model.nv)
        self.nu = int(mj_model.nu)
        self.timestep = float(mj_model.opt.timestep)
        self.qpos0 = torch.as_tensor(
            np.array(mj_model.qpos0), dtype=torch.float32, device=self.device
        )
        ctrl_range = np.asarray(mj_model.actuator_ctrlrange, dtype=np.float32)
        if self.nu > 0:
            self.ctrl_low = torch.as_tensor(ctrl_range[:, 0], device=self.device)
            self.ctrl_high = torch.as_tensor(ctrl_range[:, 1], device=self.device)
        else:
            self.ctrl_low = torch.zeros(0, device=self.device)
            self.ctrl_high = torch.zeros(0, device=self.device)

    @classmethod
    def from_xml(
        cls,
        xml_path: str,
        num_envs: int,
        device: Union[torch.device, str],
    ) -> "MjWarpWorld":
        """Build a world from an MJCF file.

        Args:
            xml_path: Path to the MJCF file.
            num_envs: Number of parallel worlds.
            device: ``cpu`` or ``cuda`` device.

        Returns:
            A world whose native model was loaded from ``xml_path``.
        """
        import mujoco

        return cls.from_model(mujoco.MjModel.from_xml_path(xml_path), num_envs, device)

    @classmethod
    def from_model(
        cls,
        mj_model: object,
        num_envs: int,
        device: Union[torch.device, str],
    ) -> "MjWarpWorld":
        """Build a world from an already compiled native model.

        Args:
            mj_model: Native ``MjModel``. Its solver is switched to CG when it
                is PGS.
            num_envs: Number of parallel worlds.
            device: ``cpu`` or ``cuda`` device.

        Returns:
            A world that steps ``mj_model``.
        """
        return cls(mj_model, num_envs, device)

    @property
    def qpos(self) -> torch.Tensor:
        """Writable generalized positions, shape ``(num_envs, nq)``."""
        return self._qpos

    @property
    def qvel(self) -> torch.Tensor:
        """Writable generalized velocities, shape ``(num_envs, nv)``."""
        return self._qvel

    @property
    def ctrl(self) -> torch.Tensor:
        """Writable actuator controls, shape ``(num_envs, nu)``."""
        return self._ctrl

    def physics_step(self) -> None:
        """Integrate every world by one MuJoCo timestep."""
        with self._wp.ScopedDevice(self._wp_device):
            self._mjw.step(self.model, self.data)

    def forward(self) -> None:
        """Refresh derived quantities from ``qpos`` and ``qvel``.

        This does not advance time.
        """
        with self._wp.ScopedDevice(self._wp_device):
            self._mjw.forward(self.model, self.data)

    def reset_worlds(self, world_ids: torch.Tensor) -> None:
        """Reset physics state for ``world_ids``.

        The call clears the selected worlds. It does not write a new pose,
        velocity, or control.

        Args:
            world_ids: 1-D indices of the worlds to reset.

        Raises:
            IndexError: If an index is outside ``[0, num_envs)``.
        """
        world_ids = world_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if world_ids.numel() == 0:
            return
        if int(world_ids.min()) < 0 or int(world_ids.max()) >= self.num_envs:
            raise IndexError(
                f"world_ids must be in [0, {self.num_envs}), got {world_ids.tolist()}."
            )
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_mask[world_ids] = True
        with self._wp.ScopedDevice(self._wp_device):
            self._mjw.reset_data(
                self.model,
                self.data,
                reset=self._wp.from_torch(reset_mask, dtype=self._wp.bool),
            )
