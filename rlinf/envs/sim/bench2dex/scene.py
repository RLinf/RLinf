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

import math
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlinf.envs.sim.bench2dex.robots import spawn_robot
from rlinf.utils.logging import get_logger

logger = get_logger()
Aabb = tuple[tuple[float, float, float], tuple[float, float, float]]
MAX_PLACE_ATTEMPTS = 80
MIN_SEPARATION = 0.01

_raw_bounds_cache: dict[str, Aabb | None] = {}


def _compute_bounds_uncached(usd_path: str) -> Aabb | None:
    """Return the unscaled USD world-space axis-aligned bounds."""
    from pxr import Usd, UsdGeom

    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as exc:
        logger.warning(f"Cannot open '{usd_path}' for bounds: {exc}")
        return None
    if stage is None:
        logger.warning(f"Cannot open '{usd_path}' for bounds.")
        return None
    up = UsdGeom.GetStageUpAxis(stage)
    if up != UsdGeom.Tokens.z:
        logger.warning(
            f"'{usd_path}' has upAxis='{up}' (not Z). You likely need rpy_deg: [90,0,0] to rotate Y-up to Z-up."
        )
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        children = stage.GetPseudoRoot().GetChildren()
        root = children[0] if children else stage.GetPseudoRoot()
    purposes = [UsdGeom.Tokens.default_]
    for tok in ("render", "proxy"):
        if hasattr(UsdGeom.Tokens, tok):
            purposes.append(getattr(UsdGeom.Tokens, tok))
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    bound = bbox.ComputeWorldBound(root)
    rng = (
        bound.ComputeAlignedRange()
        if hasattr(bound, "ComputeAlignedRange")
        else bound.GetRange()
    )
    lo = [float(rng.GetMin()[i]) for i in range(3)]
    hi = [float(rng.GetMax()[i]) for i in range(3)]
    if not all(math.isfinite(v) for v in lo + hi):
        logger.warning(f"Non-finite bounds for '{usd_path}'.")
        return None
    return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])


def _compute_bounds(
    usd_path: str, scale: tuple[float, float, float] | None
) -> Aabb | None:
    """Return the AABB of a USD asset, applying *scale* if given."""
    if usd_path not in _raw_bounds_cache:
        _raw_bounds_cache[usd_path] = _compute_bounds_uncached(usd_path)
    raw = _raw_bounds_cache[usd_path]
    if raw is None:
        return None
    lo_raw, hi_raw = raw
    lo = list(lo_raw)
    hi = list(hi_raw)
    if scale:
        for i in range(3):
            a, b = lo[i] * scale[i], hi[i] * scale[i]
            lo[i], hi[i] = min(a, b), max(a, b)
    return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])


def rpy_deg_to_quat(
    rpy_deg: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Convert roll/pitch/yaw in **degrees** to a (w, x, y, z) quaternion."""
    r, p, y = (math.radians(v) for v in rpy_deg)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _quat_rotate(
    q: tuple[float, float, float, float], p: tuple[float, float, float]
) -> tuple[float, float, float]:
    w, qx, qy, qz = q
    px, py, pz = p
    t = 2.0 * (qx * px + qy * py + qz * pz)
    s = w * w - (qx * qx + qy * qy + qz * qz)
    c = 2.0 * w
    return (
        s * px + t * qx + c * (qy * pz - qz * py),
        s * py + t * qy + c * (qz * px - qx * pz),
        s * pz + t * qz + c * (qx * py - qy * px),
    )


def _rotate_bounds(
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
    rpy_deg: tuple[float, float, float],
) -> tuple[float, float, float]:
    if all(v == 0 for v in rpy_deg):
        dx, dy = hi[0] - lo[0], hi[1] - lo[1]
        return lo[2], hi[2], 0.5 * math.hypot(dx, dy)
    q = rpy_deg_to_quat(rpy_deg)
    pts = [
        _quat_rotate(q, (x, y, z))
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
    xs = [v[0] for v in pts]
    ys = [v[1] for v in pts]
    zs = [v[2] for v in pts]
    return min(zs), max(zs), 0.5 * math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _normalize_aabb(aabb_like) -> Aabb | None:
    if not aabb_like or len(aabb_like) != 2:
        return None
    p0 = aabb_like[0]
    p1 = aabb_like[1]
    if len(p0) != 3 or len(p1) != 3:
        return None
    x0, y0, z0 = (float(v) for v in p0)
    x1, y1, z1 = (float(v) for v in p1)
    return (
        (min(x0, x1), min(y0, y1), min(z0, z1)),
        (max(x0, x1), max(y0, y1), max(z0, z1)),
    )


def _normalize_aabbs(items) -> list[Aabb]:
    out: list[Aabb] = []
    if not items:
        return out
    for item in items:
        norm = _normalize_aabb(item)
        if norm is not None:
            out.append(norm)
    return out


def _resolve_explicit_position(
    position_like: Any, *, obj_id: str, table_z: float, min_z: float
) -> tuple[float, float, float]:
    if isinstance(position_like, (str, bytes)) or not isinstance(
        position_like, Sequence
    ):
        raise TypeError(
            f"Object '{obj_id}' position must be a sequence of length 2 or 3."
        )
    values = list(position_like)
    if len(values) not in (2, 3):
        raise ValueError(f"Object '{obj_id}' position must have length 2 or 3.")
    try:
        x = float(values[0])
        y = float(values[1])
        if len(values) == 3:
            z = table_z + float(values[2])
        else:
            z = table_z - min_z
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Object '{obj_id}' position values must be numeric.") from exc
    return (x, y, z)


def _hits_forbidden(
    cx: float, cy: float, footprint: float, forbidden_aabbs: Sequence[Aabb]
) -> bool:
    expand = footprint + MIN_SEPARATION
    for (x0, y0, _z0), (x1, y1, _z1) in forbidden_aabbs:
        if x0 - expand <= cx <= x1 + expand and y0 - expand <= cy <= y1 + expand:
            return True
    return False


def _sample_position(
    aabb: Aabb,
    table_z: float,
    min_z: float,
    footprint: float,
    placed: list[tuple[str, float, float, float]],
    yaw_range: tuple[float, float],
    forbidden_aabbs: Sequence[Aabb],
) -> tuple[tuple[float, float, float], float]:
    (x0, y0, _z0), (x1, y1, _z1) = aabb
    valid_xy = None
    for _ in range(MAX_PLACE_ATTEMPTS):
        cx = random.uniform(x0, x1)
        cy = random.uniform(y0, y1)
        if _hits_forbidden(cx, cy, footprint, forbidden_aabbs):
            continue
        collides = False
        for _, px, py, pr in placed:
            gap = footprint + pr + MIN_SEPARATION
            if (cx - px) ** 2 + (cy - py) ** 2 < gap**2:
                collides = True
                break
        if not collides:
            valid_xy = (cx, cy)
            break
    if valid_xy is None:
        raise ValueError(
            f"Failed to find a valid placement after {MAX_PLACE_ATTEMPTS} attempts (zone=[({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f})], footprint={footprint:.4f}, placed={len(placed)} objects, forbidden_aabbs={len(forbidden_aabbs)})."
        )
    z = table_z - min_z
    yaw = random.uniform(yaw_range[0], yaw_range[1])
    return (valid_xy[0], valid_xy[1], z), yaw


DEFAULT_ARTICULATION_FIX_ROOT_LINK = False
_OPTIONAL_ACTUATOR_FLOAT_KEYS = ("effort_limit_sim", "velocity_limit_sim", "armature")
_REQUIRED_ACTUATOR_FLOAT_KEYS = ("stiffness", "damping")
_ALLOWED_ACTUATOR_KEYS = frozenset(
    {
        "joint_names_expr",
        *_OPTIONAL_ACTUATOR_FLOAT_KEYS,
        *_REQUIRED_ACTUATOR_FLOAT_KEYS,
    }
)
_ALLOWED_ARTICULATION_SPAWN_KEYS = frozenset(
    {
        "articulation_props",
        "mass_props",
        "actuators",
        "joint_limits",
        "joint_pos",
    }
)
_ALLOWED_ARTICULATION_PROP_KEYS = frozenset({"fix_root_link"})
_ALLOWED_MASS_PROP_KEYS = frozenset({"density"})
_ALLOWED_JOINT_LIMIT_KEYS = frozenset({"lower", "upper", "unit"})


def _as_float(value: Any, *, obj_id: str, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Object '{obj_id}' {field_name} must be numeric.") from exc


def _as_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{label} must be a boolean.")


def _normalize_joint_names_expr(
    value: Any, *, obj_id: str, group_name: str
) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            f"Object '{obj_id}' actuator group '{group_name}' joint_names_expr must be a sequence of strings."
        )
    normalized: list[str] = []
    for expr in value:
        if not isinstance(expr, str) or not expr:
            raise TypeError(
                f"Object '{obj_id}' actuator group '{group_name}' joint_names_expr entries must be non-empty strings."
            )
        try:
            re.compile(expr)
        except re.error as exc:
            raise ValueError(
                f"Object '{obj_id}' actuator group '{group_name}' has invalid joint_names_expr '{expr}': {exc}."
            ) from exc
        normalized.append(expr)
    if not normalized:
        raise TypeError(
            f"Object '{obj_id}' actuator group '{group_name}' joint_names_expr must not be empty."
        )
    return normalized


def _normalize_actuator_group(
    group_name: str, spec: Mapping[str, Any], *, obj_id: str
) -> dict[str, Any]:
    unknown_keys = sorted(set(spec.keys()) - _ALLOWED_ACTUATOR_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Object '{obj_id}' actuator group '{group_name}' has unsupported keys: {', '.join(unknown_keys)}."
        )
    if "joint_names_expr" not in spec:
        raise TypeError(
            f"Object '{obj_id}' actuator group '{group_name}' must define joint_names_expr."
        )
    normalized: dict[str, Any] = {
        "joint_names_expr": _normalize_joint_names_expr(
            spec["joint_names_expr"], obj_id=obj_id, group_name=group_name
        )
    }
    for field_name in _REQUIRED_ACTUATOR_FLOAT_KEYS:
        if field_name not in spec:
            raise TypeError(
                f"Object '{obj_id}' actuator group '{group_name}' must define {field_name}."
            )
        normalized[field_name] = _as_float(
            spec[field_name],
            obj_id=obj_id,
            field_name=f"actuator group '{group_name}' {field_name}",
        )
    for field_name in _OPTIONAL_ACTUATOR_FLOAT_KEYS:
        if spec.get(field_name) is not None:
            normalized[field_name] = _as_float(
                spec[field_name],
                obj_id=obj_id,
                field_name=f"actuator group '{group_name}' {field_name}",
            )
    return normalized


def _normalize_actuator_specs_mapping(
    actuators: Any, *, obj_id: str, source_label: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(actuators, Mapping):
        raise TypeError(
            f"Object '{obj_id}' {source_label} must be a mapping of actuator group names to configs."
        )
    normalized: dict[str, dict[str, Any]] = {}
    for group_name, spec in actuators.items():
        if not isinstance(group_name, str) or not group_name:
            raise TypeError(
                f"Object '{obj_id}' {source_label} group names must be non-empty strings."
            )
        if not isinstance(spec, Mapping):
            raise TypeError(
                f"Object '{obj_id}' {source_label} group '{group_name}' must be a mapping."
            )
        normalized[group_name] = _normalize_actuator_group(
            group_name, spec, obj_id=obj_id
        )
    return normalized


def _normalize_joint_limits(
    joint_limits: Any, *, obj_id: str, source_label: str
) -> dict[str, dict[str, float]]:
    if not isinstance(joint_limits, Mapping):
        raise TypeError(
            f"Object '{obj_id}' {source_label} must be a mapping of joint names to limit configs."
        )
    normalized: dict[str, dict[str, float]] = {}
    for joint_name, spec in joint_limits.items():
        if not isinstance(joint_name, str) or not joint_name:
            raise TypeError(
                f"Object '{obj_id}' {source_label} joint names must be non-empty strings."
            )
        if not isinstance(spec, Mapping):
            raise TypeError(
                f"Object '{obj_id}' {source_label}.{joint_name} must be a mapping."
            )
        unknown_keys = sorted(set(spec.keys()) - _ALLOWED_JOINT_LIMIT_KEYS)
        if unknown_keys:
            raise ValueError(
                f"Object '{obj_id}' {source_label}.{joint_name} has unsupported keys: {', '.join(unknown_keys)}."
            )
        if "lower" not in spec or "upper" not in spec:
            raise TypeError(
                f"Object '{obj_id}' {source_label}.{joint_name} must define lower and upper."
            )
        lower = _as_float(
            spec["lower"],
            obj_id=obj_id,
            field_name=f"{source_label}.{joint_name}.lower",
        )
        upper = _as_float(
            spec["upper"],
            obj_id=obj_id,
            field_name=f"{source_label}.{joint_name}.upper",
        )
        if spec.get("unit", "rad") != "rad":
            raise ValueError(
                f"Object '{obj_id}' {source_label}.{joint_name}.unit must be 'rad', got {spec['unit']!r}."
            )
        if lower > upper:
            raise ValueError(
                f"Object '{obj_id}' {source_label}.{joint_name} lower must be <= upper."
            )
        normalized[joint_name] = {"lower": lower, "upper": upper}
    return normalized


def _normalize_joint_pos_mapping(
    joint_pos: Any, *, obj_id: str, source_label: str
) -> dict[str, float]:
    if not isinstance(joint_pos, Mapping):
        raise TypeError(
            f"Object '{obj_id}' {source_label} must be a mapping of joint names to values."
        )
    normalized: dict[str, float] = {}
    for joint_name, value in joint_pos.items():
        if not isinstance(joint_name, str) or not joint_name:
            raise TypeError(
                f"Object '{obj_id}' {source_label} keys must be non-empty strings."
            )
        normalized[joint_name] = _as_float(
            value, obj_id=obj_id, field_name=f"{source_label}.{joint_name}"
        )
    return normalized


def _normalize_spawn_config(
    spawn: Any, *, obj_id: str, source_label: str
) -> dict[str, Any]:
    if spawn is None:
        return {}
    if not isinstance(spawn, Mapping):
        raise TypeError(f"Object '{obj_id}' {source_label} must be a mapping.")
    unknown_keys = sorted(set(spawn.keys()) - _ALLOWED_ARTICULATION_SPAWN_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Object '{obj_id}' {source_label} has unsupported keys: {', '.join(unknown_keys)}."
        )
    normalized: dict[str, Any] = {}
    articulation_props = spawn.get("articulation_props")
    if articulation_props is not None:
        if not isinstance(articulation_props, Mapping):
            raise TypeError(
                f"Object '{obj_id}' {source_label}.articulation_props must be a mapping."
            )
        unknown_keys = sorted(
            set(articulation_props.keys()) - _ALLOWED_ARTICULATION_PROP_KEYS
        )
        if unknown_keys:
            raise ValueError(
                f"Object '{obj_id}' {source_label}.articulation_props has unsupported keys: {', '.join(unknown_keys)}."
            )
        normalized["articulation_props"] = {}
        if "fix_root_link" in articulation_props:
            normalized["articulation_props"]["fix_root_link"] = _as_bool(
                articulation_props["fix_root_link"],
                label=f"Object '{obj_id}' {source_label}.articulation_props.fix_root_link",
            )
    mass_props = spawn.get("mass_props")
    if mass_props is not None:
        if not isinstance(mass_props, Mapping):
            raise TypeError(
                f"Object '{obj_id}' {source_label}.mass_props must be a mapping."
            )
        unknown_keys = sorted(set(mass_props.keys()) - _ALLOWED_MASS_PROP_KEYS)
        if unknown_keys:
            raise ValueError(
                f"Object '{obj_id}' {source_label}.mass_props has unsupported keys: {', '.join(unknown_keys)}."
            )
        normalized["mass_props"] = {}
        if mass_props.get("density") is not None:
            normalized["mass_props"]["density"] = _as_float(
                mass_props["density"],
                obj_id=obj_id,
                field_name=f"{source_label}.mass_props.density",
            )
    if "actuators" in spawn:
        normalized["actuators"] = _normalize_actuator_specs_mapping(
            spawn["actuators"], obj_id=obj_id, source_label=f"{source_label}.actuators"
        )
    if "joint_limits" in spawn:
        normalized["joint_limits"] = _normalize_joint_limits(
            spawn["joint_limits"],
            obj_id=obj_id,
            source_label=f"{source_label}.joint_limits",
        )
    if "joint_pos" in spawn:
        normalized["joint_pos"] = _normalize_joint_pos_mapping(
            spawn["joint_pos"], obj_id=obj_id, source_label=f"{source_label}.joint_pos"
        )
    return normalized


def _joint_is_covered_by_actuators(
    joint_name: str, actuator_specs: Mapping[str, Mapping[str, Any]]
) -> bool:
    for spec in actuator_specs.values():
        for expr in spec.get("joint_names_expr", []):
            if re.fullmatch(expr, joint_name):
                return True
    return False


def _validate_joint_pos_coverage(
    obj_id: str,
    joint_pos: Mapping[str, float],
    actuator_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    if joint_pos:
        uncovered_joints = [
            joint_name
            for joint_name in joint_pos.keys()
            if not _joint_is_covered_by_actuators(joint_name, actuator_specs)
        ]
        if uncovered_joints:
            raise ValueError(
                f"Object '{obj_id}' joint_pos joints are not covered by any actuator group: {', '.join(uncovered_joints)}."
            )


def resolve_articulation_spawn_config(
    asset_spec: Mapping[str, Any], obj: Mapping[str, Any]
) -> dict[str, Any]:
    obj_id = str(obj.get("id", "<unknown>"))
    asset_key = str(obj.get("asset", "<unknown>"))
    asset_spawn = _normalize_spawn_config(
        asset_spec.get("spawn"),
        obj_id=obj_id,
        source_label=f"asset '{asset_key}' spawn",
    )
    merged: dict[str, Any] = {
        "articulation_props": {"fix_root_link": DEFAULT_ARTICULATION_FIX_ROOT_LINK},
        "mass_props": asset_spawn["mass_props"],
        "actuators": asset_spawn["actuators"],
    }
    if "articulation_props" in asset_spawn:
        merged["articulation_props"].update(asset_spawn["articulation_props"])
    if "joint_limits" in asset_spawn:
        merged["joint_limits"] = asset_spawn["joint_limits"]
    joint_pos: dict[str, float] = {}
    if "joint_pos" in asset_spawn:
        joint_pos.update(asset_spawn["joint_pos"])
    if obj.get("joint_pos") is not None:
        joint_pos.update(
            _normalize_joint_pos_mapping(
                obj["joint_pos"], obj_id=obj_id, source_label="joint_pos"
            )
        )
    if joint_pos:
        merged["joint_pos"] = joint_pos
    _validate_joint_pos_coverage(obj_id, joint_pos, merged["actuators"])
    return merged


def build_articulation_actuator_cfgs_from_specs(
    actuator_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build ImplicitActuatorCfg instances from normalized actuator specs."""
    from isaaclab.actuators import ImplicitActuatorCfg

    return {
        group_name: ImplicitActuatorCfg(**dict(spec_kwargs))
        for group_name, spec_kwargs in actuator_specs.items()
    }


def _load_sim_utils():
    import isaaclab.sim as sim_utils

    return sim_utils


def _build_usd_cfg(
    path: str,
    scale: tuple[float, float, float] | None,
    body: str,
    spawn: Mapping | None = None,
) -> Any:
    sim_utils = _load_sim_utils()
    usd_kw: dict = {"usd_path": path}
    if scale is not None:
        usd_kw["scale"] = scale
    if body == "dynamic":
        rigid_props_cfg = (
            spawn.get("rigid_props", {}) if isinstance(spawn, Mapping) else {}
        )
        kinematic = bool(rigid_props_cfg.get("kinematic_enabled", False))
        disable_gravity = bool(rigid_props_cfg.get("disable_gravity", kinematic))
        usd_kw["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=kinematic,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=disable_gravity,
        )
        usd_kw["collision_props"] = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True
        )
        mass_props = spawn.get("mass_props", {}) if isinstance(spawn, Mapping) else {}
        if mass_props:
            mass_val = mass_props.get("mass", None)
            density_val = mass_props.get("density", None)
            mp_kwargs = {}
            if mass_val is not None:
                mp_kwargs["mass"] = float(mass_val)
            if density_val is not None:
                mp_kwargs["density"] = float(density_val)
            if mp_kwargs:
                usd_kw["mass_props"] = sim_utils.MassPropertiesCfg(**mp_kwargs)
    elif body == "articulation":
        articulation_props = (
            spawn.get("articulation_props", {}) if isinstance(spawn, Mapping) else {}
        )
        mass_props = spawn.get("mass_props", {}) if isinstance(spawn, Mapping) else {}
        usd_kw["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            fix_root_link=articulation_props.get(
                "fix_root_link", DEFAULT_ARTICULATION_FIX_ROOT_LINK
            ),
        )
        usd_kw["mass_props"] = sim_utils.MassPropertiesCfg(
            density=mass_props["density"]
        )
        usd_kw["collision_props"] = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True
        )
    else:
        raise ValueError(f"Unsupported body_type '{body}'.")
    return sim_utils.UsdFileCfg(**usd_kw)


def _apply_joint_limits(prim_path, limits):
    import isaaclab.sim as sim_utils
    from pxr import Sdf

    pending = dict(limits or {})
    for prim in sim_utils.get_current_stage().Traverse():
        if not str(prim.GetPath()).startswith(prim_path):
            continue
        name = prim.GetName()
        if prim.GetTypeName() not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            continue
        if name not in pending:
            continue
        spec = pending.pop(name)
        lower, upper = spec["lower"], spec["upper"]
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            lower, upper = math.degrees(lower), math.degrees(upper)
        prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(lower)
        prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(upper)
    if pending:
        raise ValueError(
            f"Joint limit overrides did not match joints: {list(pending)}."
        )


def build_scene(task: dict, assets_path: Path, robot_key: str) -> tuple:
    """Build a benchmark layout with native Isaac Lab assets."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import (
        Articulation,
        ArticulationCfg,
        RigidObject,
        RigidObjectCfg,
    )
    from pxr import UsdLux

    table = task.get("table", {})
    sx, sy, _ = table.get("size", [2.2, 1.1, 0.04])
    table_z = table.get("height", 0.75)
    color = tuple(table.get("color", [0.45, 0.30, 0.20]))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)
    light = sim_utils.DistantLightCfg(
        intensity=2500.0, color=(0.8, 0.8, 0.8), angle=1.0
    )
    light.func("/World/DistantLight", light, orientation=rpy_deg_to_quat((0, -60, 0)))
    stage = sim_utils.get_current_stage()
    UsdLux.ShadowAPI.Apply(
        stage.GetPrimAtPath("/World/DistantLight")
    ).CreateShadowEnableAttr(True)
    stage.DefinePrim("/World/Objects", "Xform")
    for name, size, position in (
        ("Table", (sx, sy - 0.25, table_z), (0.0, 0.125, table_z / 2)),
        ("RobotSupportTable", (1.7, 0.2, 0.75), (0.0, -0.45, 0.375)),
    ):
        cfg = sim_utils.MeshCuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color if name == "Table" else (0.45, 0.30, 0.20)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        cfg.func(f"/World/Objects/{name}", cfg, translation=position)
    robot, exclusion, compensate = spawn_robot(robot_key, assets_path, table_z)
    objects = {"global_robot": robot}
    paths = {}
    layout = task.get("layout", {})
    zone_name = layout.get("zone")
    zone = task.get("zones", {}).get(zone_name, {})
    aabb = _normalize_aabb(zone.get("aabb"))
    forbidden = _normalize_aabbs(zone.get("exclude_aabbs", []))
    if zone_name == "tabletop":
        aabb = ((-sx / 2, -sy / 2 + 0.25, table_z), (sx / 2, sy / 2, table_z + 0.1))
        forbidden.append(exclusion)
    placed = []
    for obj in task.get("objects", []):
        obj_id = obj["id"]
        asset = task["assets"][obj["asset"]]
        path = str(
            (assets_path / asset["path"].split("dex2bench_dataset/", 1)[1]).resolve()
        )
        scale = tuple(asset["scale"]) if "scale" in asset else None
        bounds = _compute_bounds(path, scale)
        if bounds is None:
            raise ValueError(f"Cannot determine asset bounds: {path}.")
        lo, hi = bounds
        rpy = tuple(obj.get("rpy_deg", (0, 0, 0)))
        min_z, _, footprint = _rotate_bounds(lo, hi, (rpy[0], rpy[1], 0))
        if "position" in obj:
            pos = _resolve_explicit_position(
                obj["position"], obj_id=obj_id, table_z=table_z, min_z=min_z
            )
            yaw = rpy[2]
        elif obj_id in layout.get("objects", []) and aabb is not None:
            pos, yaw = _sample_position(
                aabb,
                table_z,
                min_z,
                footprint,
                placed,
                tuple(layout.get("yaw_range", (0, 360))),
                forbidden,
            )
            yaw += rpy[2]
        else:
            raise ValueError(f"No position or layout zone for {obj_id!r}.")
        quat = rpy_deg_to_quat((rpy[0], rpy[1], yaw))
        prim = f"/World/Objects/{obj_id.replace(' ', '_')}"
        paths[obj_id] = prim
        body = asset.get("body_type", "dynamic").lower()
        if body == "articulation":
            spawn = resolve_articulation_spawn_config(asset, obj)
        else:
            spawn = asset.get("spawn", {})
        usd = _build_usd_cfg(path, scale, body, spawn)
        if body == "dynamic":
            objects[obj_id] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim,
                    spawn=usd,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
                )
            )
        elif body == "articulation":
            init_state = {"pos": pos, "rot": quat}
            if "joint_pos" in spawn:
                init_state["joint_pos"] = spawn["joint_pos"]
            objects[obj_id] = Articulation(
                ArticulationCfg(
                    prim_path=prim,
                    spawn=usd,
                    init_state=ArticulationCfg.InitialStateCfg(**init_state),
                    actuators=build_articulation_actuator_cfgs_from_specs(
                        spawn["actuators"]
                    ),
                )
            )
            _apply_joint_limits(prim, spawn.get("joint_limits"))
        if "color" in obj:
            material_path = f"{prim}/ObjectMaterial"
            material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(obj["color"]), roughness=0.5, metallic=0.0
            )
            material.func(material_path, material)
            sim_utils.bind_visual_material(
                prim, material_path, stronger_than_descendants=True
            )
        placed.append((obj_id, pos[0], pos[1], footprint))
    # Activation also zeroes the body's sleep threshold, matching the official scene.
    for pair in task.get("metrics", {}).get("contact_pairs", []):
        sim_utils.activate_contact_sensors(paths[pair["sensor"]], threshold=0.0)
    return objects, compensate
