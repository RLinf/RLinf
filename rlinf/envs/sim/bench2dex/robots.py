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
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml


def load_robot_config(robot_key: str) -> dict:
    """Load robot poses, actuator parameters, and hand friction settings."""
    with Path(__file__).with_suffix(".yaml").open() as stream:
        configs = yaml.safe_load(stream)
    if robot_key not in configs:
        raise ValueError(f"Unknown Bench2Dex robot: {robot_key!r}.")
    return configs[robot_key]


def spawn_robot(robot_key: str, assets_path: Path, table_z: float) -> tuple:
    """Spawn a benchmark robot and apply its collision and drive settings."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from pxr import Usd, UsdPhysics, UsdShade

    spec = load_robot_config(robot_key)
    cfg = copy.deepcopy(spec["config"])
    spawn = cfg.pop("spawn")
    spawn["usd_path"] = str(assets_path / spawn["usd_path"])
    spawn["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(**spawn["rigid_props"])
    spawn["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
        **spawn["articulation_props"]
    )
    spawn["collision_props"] = sim_utils.CollisionPropertiesCfg(
        **spawn["collision_props"]
    )
    init_state = cfg.pop("init_state")
    init_state["pos"][2] += table_z - 0.75
    init_state["pos"] = tuple(init_state["pos"])
    init_state["rot"] = tuple(init_state["rot"])
    robot = Articulation(
        ArticulationCfg(
            **{key: value for key, value in cfg.items() if key != "actuators"},
            spawn=sim_utils.UsdFileCfg(**spawn),
            init_state=ArticulationCfg.InitialStateCfg(**init_state),
            actuators={
                name: ImplicitActuatorCfg(**params)
                for name, params in cfg["actuators"].items()
            },
        )
    )
    stage = sim_utils.get_current_stage()
    root = cfg["prim_path"]
    material_prim = stage.DefinePrim(f"{root}/HandFrictionMaterial", "Material")
    material = UsdShade.Material(material_prim)
    physics = UsdPhysics.MaterialAPI.Apply(material_prim)
    physics.CreateStaticFrictionAttr(spec["static_friction"])
    physics.CreateDynamicFrictionAttr(spec["dynamic_friction"])
    physics.CreateRestitutionAttr(0.0)
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        path = str(prim.GetPath())
        if not path.startswith(root) or "HandFrictionMaterial" in path:
            continue
        if prim.GetTypeName() in (
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
        ) and any(name in prim.GetName() for name in spec.get("force_drives", [])):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive and drive.GetTypeAttr().Get() == "acceleration":
                drive.GetTypeAttr().Set("force")
        if not any(name in path.lower() for name in spec["friction_keywords"]):
            continue
        if prim.IsInstanceProxy():
            continue
        hand_link = prim.GetTypeName() == "Xform" and not any(
            part in path.lower() for part in ("/visual", "/collision", "/joint")
        )
        if (
            hand_link
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or (
                "/collision" in path.lower()
                and prim.GetTypeName()
                in {"Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Cone"}
            )
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.strongerThanDescendants, "physics"
            )
    exclusion = (
        (-spec["exclude_half_x"], -0.55, table_z),
        (
            spec["exclude_half_x"],
            init_state["pos"][1] + spec["exclude_front_extent"],
            table_z + 0.2,
        ),
    )
    return robot, exclusion, spec["gravity_compensation"]


class JointTargets:
    """Map runtime active joints to full targets, including URDF mimic chains."""

    def __init__(
        self,
        robot_key: str,
        dof_maps_path: Path,
        assets_path: Path,
        joint_names: list[str],
    ):
        with dof_maps_path.open() as stream:
            spec = yaml.safe_load(stream)["robots"][robot_key]
        if set(joint_names) != set(spec["full_joint_names"]):
            raise ValueError(f"Joint names for {robot_key!r} do not match its DOF map.")
        active = set(spec["active_joint_names"])
        self.indices = [i for i, name in enumerate(joint_names) if name in active]
        self.full_dof = len(joint_names)
        self.mimic = []
        if spec.get("mimic_dof", 0):
            path = assets_path / spec["urdf_path"].removeprefix("Bench2Dex/")
            names = {name: i for i, name in enumerate(joint_names)}
            rules = {}
            for joint in ET.parse(path).getroot().findall(".//joint"):
                mimic = joint.find("mimic")
                name = joint.get("name")
                if mimic is not None and name in names and mimic.get("joint") in names:
                    rules[name] = (
                        mimic.get("joint"),
                        float(mimic.get("multiplier", "1.0")),
                        float(mimic.get("offset", "0.0")),
                    )
            resolved = set(names) - set(rules)
            while rules:
                ready = [name for name, rule in rules.items() if rule[0] in resolved]
                if not ready:
                    raise ValueError(f"Unresolved mimic joints: {list(rules)}.")
                for name in ready:
                    source, multiplier, offset = rules.pop(name)
                    self.mimic.append((names[name], names[source], multiplier, offset))
                    resolved.add(name)

    def expand(self, actions: torch.Tensor) -> torch.Tensor:
        """Expand active targets in simulation joint order."""
        targets = actions.new_zeros((1, self.full_dof))
        targets[:, self.indices] = actions
        for target, source, multiplier, offset in self.mimic:
            targets[:, target] = targets[:, source] * multiplier + offset
        return targets
