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

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def camera_orientation(eye: np.ndarray, target: np.ndarray, chest: bool) -> np.ndarray:
    """Return camera axes in world coordinates using the benchmark look-at frame."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    dot = np.dot(up, forward)
    primary = up - forward * dot
    if chest:
        norm = np.linalg.norm(primary)
        up = primary / norm if norm >= 1e-9 else np.array([0.0, -1.0, 0.0])
        if dot < 0:
            up = -up
    else:
        secondary = np.array([0.0, -1.0, 0.0])
        if abs(np.dot(secondary, forward)) > 0.999:
            secondary = np.array([1.0, 0.0, 0.0])
        secondary -= forward * np.dot(secondary, forward)
        up = (1 - dot * dot) * primary + dot * dot * secondary
        if np.linalg.norm(up) < 1e-9:
            up = secondary
        up /= np.linalg.norm(up)
    right = np.cross(forward, -up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


class Bench2DexCameras:
    """Render calibrated world and robot-mounted RGB cameras."""

    def __init__(self, config_path: Path, robot_key: str, names: list[str], device):
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.prims as prim_utils
        import omni.usd
        from isaaclab.sensors import Camera, CameraCfg

        self.device = device
        self.cameras = {}
        self.specs = {}
        self.mounts = {}
        self.masks = {}
        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        profiles = config["robot_wrist_camera_profiles"][robot_key]
        specs = {spec["id"]: spec for spec in config["camera_list"]}
        for name in names:
            spec = copy.deepcopy(specs[name])
            spec.update(profiles.get(name, {}))
            spec["width"], spec["height"] = config["camera_pixel"].get(
                name, (spec["width"], spec["height"])
            )
            base_path = f"/World/Sensors/{name}"
            prim_utils.create_prim(base_path, "Xform")
            path = f"{base_path}/Sensor"
            kwargs = {
                "horizontal_aperture": spec.get("horizontal_aperture", 20.955),
                "clipping_range": tuple(spec["clipping_range"]),
            }
            if spec.get("camera_type") == "fisheye":
                intrinsic = spec["fisheye_camera_matrix"]
                spawn = sim_utils.FisheyeCameraCfg(
                    **kwargs,
                    projection_type="fisheyeKannalaBrandtK3",
                    focal_length=intrinsic[0][0]
                    * kwargs["horizontal_aperture"]
                    / spec["width"],
                )
            else:
                spawn = sim_utils.PinholeCameraCfg(
                    **kwargs, focal_length=spec["focal_length"]
                )
            self.cameras[name] = Camera(
                CameraCfg(
                    prim_path=path,
                    width=spec["width"],
                    height=spec["height"],
                    update_period=0,
                    data_types=["rgb"],
                    spawn=spawn,
                )
            )
            self.specs[name] = spec
            if spec.get("camera_type") == "fisheye":
                prim = sim_utils.get_current_stage().GetPrimAtPath(path)
                prim.ApplyAPI("OmniLensDistortionOpenCvFisheyeAPI")
                params = {
                    "fx": intrinsic[0][0],
                    "fy": intrinsic[1][1],
                    "cx": intrinsic[0][2],
                    "cy": intrinsic[1][2],
                    "imageSize": (spec["width"], spec["height"]),
                }
                params.update(
                    zip(
                        ("k1", "k2", "k3", "k4"),
                        spec["fisheye_distortion_coefficients"],
                    )
                )
                theta = np.radians(spec.get("fisheye_max_fov_deg", 180.0) / 2)
                radius = theta * (
                    1
                    + sum(
                        k * theta ** (2 * (i + 1))
                        for i, k in enumerate(spec["fisheye_distortion_coefficients"])
                    )
                )
                row, col = torch.meshgrid(
                    torch.arange(spec["height"], device=device),
                    torch.arange(spec["width"], device=device),
                    indexing="ij",
                )
                self.masks[name] = ((col - params["cx"]) / params["fx"]) ** 2 + (
                    (row - params["cy"]) / params["fy"]
                ) ** 2 <= radius**2
                for attr, value in params.items():
                    omni.usd.set_prop_val(
                        prim.GetAttribute(f"omni:lensdistortion:opencvFisheye:{attr}"),
                        value,
                    )

    def initialize(self, robot) -> None:
        """Bind mounted cameras to robot bodies after simulation initialization."""
        from isaaclab.utils.math import quat_from_euler_xyz, quat_from_matrix

        self.robot = robot
        for name, camera in self.cameras.items():
            spec = self.specs[name]
            if spec["mount_type"] == "robot_link":
                body_id = robot.body_names.index(spec["parent_link"])
                offset = torch.tensor([spec["offset_xyz"]], device=self.device)
                rpy = torch.tensor(spec["offset_rpy"], device=self.device)
                quat = quat_from_euler_xyz(*rpy).unsqueeze(0)
                self.mounts[name] = (body_id, offset, quat)
            else:
                pos = torch.tensor([spec["position"]], device=self.device)
                matrix = camera_orientation(
                    np.array(spec["position"]),
                    np.array(spec["target"]),
                    name == "cam_chest",
                )
                quat = quat_from_matrix(
                    torch.tensor(matrix, dtype=torch.float32, device=self.device)
                ).unsqueeze(0)
                camera.set_world_poses(
                    positions=pos, orientations=quat, convention="ros"
                )
            camera.reset()
        self.sync()

    def sync(self) -> None:
        """Update wrist poses before rendering the corresponding RGB frames."""
        from isaaclab.utils.math import combine_frame_transforms

        for name, (body_id, offset, rotation) in self.mounts.items():
            pos, quat = combine_frame_transforms(
                self.robot.data.body_pos_w[:, body_id],
                self.robot.data.body_quat_w[:, body_id],
                offset,
                rotation,
            )
            self.cameras[name].set_world_poses(
                positions=pos, orientations=quat, convention="ros"
            )

    def capture(self, image_size: list[int], dt: float) -> list[torch.Tensor]:
        """Read the latest rendered frames and resize to policy input dimensions."""
        frames = []
        for name, camera in self.cameras.items():
            camera.update(dt, force_recompute=True)
            rgb = camera.data.output["rgb"][..., :3]
            if name in self.masks:
                rgb = rgb * self.masks[name][None, :, :, None]
            frames.append(
                F.interpolate(
                    rgb.permute(0, 3, 1, 2).float(),
                    size=image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        return frames

    def close(self) -> None:
        """Detach annotators before the scene's camera prims are removed."""
        import isaaclab.sim as sim_utils
        import omni.replicator.core as rep

        stage = sim_utils.get_current_stage()
        camera_paths = []
        for camera in self.cameras.values():
            paths = camera.render_product_paths
            camera_paths.append(camera.cfg.prim_path.rsplit("/", 1)[0])
            for annotators in camera._rep_registry.values():
                for annotator, path in zip(annotators, paths):
                    annotator.detach([path])
            camera._rep_registry.clear()
            camera._clear_callbacks()
        self.cameras.clear()
        rep.vp_manager.destroy_hydra_textures()
        sim_utils.SimulationContext.instance().render()
        for path in camera_paths:
            stage.RemovePrim(path)
        self.mounts.clear()
        self.robot = None
