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

"""Isaac/HSSD process patches used by the SIMPLE runtime."""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock, Timeout


def use_pip_isaac_sim() -> None:
    """Let the pip-installed Isaac Sim locate its own Kit in this process.

    RLinf launch scripts export ``ISAAC_PATH``, ``EXP_PATH`` and
    ``CARB_APP_PATH`` for a standalone Isaac Sim, with placeholder defaults.
    SIMPLE runs on the ``isaacsim`` wheel pinned in ``simple.txt``, which
    derives these paths only when they are unset.
    """
    for key in ("ISAAC_PATH", "EXP_PATH", "CARB_APP_PATH"):
        os.environ.pop(key, None)
    # Isaac Sim must start non-interactively in a Ray actor.
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def install_articulation_reuse_guard(native_env: Any) -> None:
    """Keep SIMPLE from rebuilding an unchanged Isaac articulation on reset."""
    isaac = native_env.isaac
    original_add = isaac.add_articulated_object
    asset_keys: dict[str, tuple[str, str]] = {}

    def add_articulated_object_once(obj_name: str, obj_info: Any) -> None:
        asset_name = str(obj_info.asset.name)
        asset_key = (str(obj_info.asset.uid), str(obj_info.asset.usd_path))
        if asset_name in isaac.articulated_objects:
            if asset_keys.get(asset_name) != asset_key:
                raise RuntimeError(
                    f"Cannot replace SIMPLE articulated object {asset_name!r} "
                    "without rebuilding its Isaac physics views."
                )
            return
        original_add(obj_name, obj_info)
        asset_keys[asset_name] = asset_key

    isaac.add_articulated_object = add_articulated_object_once


def link_kit_cache(portable_root: str) -> FileLock:
    """Point a per-process Kit root at a warm, exclusively locked cache slot.

    Kit stores compiled RTX shaders and MDL materials under
    ``<portable-root>/cache``; rebuilding them adds about 90 s to every
    simulator start. Data and logs stay in the per-process root. Each process
    holds its slot lock until it exits, so live simulators never share a slot.
    """
    default_root = Path.home() / ".cache" / "rlinf" / "simple-kit"
    cache_root = Path(os.environ.get("RLINF_SIMPLE_KIT_CACHE_DIR", default_root))
    cache_root.mkdir(parents=True, exist_ok=True)
    for slot in itertools.count():
        lock = FileLock(str(cache_root / f"slot-{slot}.lock"))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            continue
        cache_dir = cache_root / f"slot-{slot}"
        cache_dir.mkdir(exist_ok=True)
        (Path(portable_root) / "cache").symlink_to(cache_dir, target_is_directory=True)
        return lock


class IsaacFrameGate:
    """Mirror MuJoCo into Isaac only for observations RLinf consumes.

    In ``mujoco_isaac`` mode Isaac only renders: every 200 Hz physics substep
    copies MuJoCo poses into Isaac and renders an RTX frame, although RLinf
    reads one frame per action chunk. ``sync_states`` overwrites every mirrored
    pose and zeroes velocities, so skipping intermediate mirrors leaves the next
    rendered state unchanged. Deferred steps return MuJoCo joint positions with
    the previous frames; ``observe`` mirrors and renders the current state.
    """

    def __init__(self, native_env: Any) -> None:
        self._env = native_env
        self._isaac_step = native_env.isaac.step
        self._get_obs = native_env._get_obs
        self._deferred = False
        self._frames: dict[str, Any] = {}
        native_env.isaac.step = self._step_isaac
        native_env._get_obs = self._observe_or_defer

    @contextmanager
    def deferred(self) -> Iterator[None]:
        """Step physics without mirroring or rendering Isaac."""
        self._deferred = True
        try:
            yield
        finally:
            self._deferred = False

    def observe(self) -> dict[str, Any]:
        """Mirror the current MuJoCo state into Isaac once and render it."""
        if self._deferred:
            raise RuntimeError("Cannot render Isaac inside a deferred block.")
        self._isaac_step(self._env.mujoco)
        return self._observe_or_defer()

    def _step_isaac(self, *args: Any, **kwargs: Any) -> Any:
        if not self._deferred:
            return self._isaac_step(*args, **kwargs)

    def _observe_or_defer(self) -> dict[str, Any]:
        if self._deferred:
            qpos = list(self._env.mujoco.get_robot_qpos().values())
            return {"joint_qpos": np.asarray(qpos, dtype=np.float32), **self._frames}
        observation = self._get_obs()
        self._frames = {k: v for k, v in observation.items() if k != "joint_qpos"}
        return observation


def resolve_hssd_scene_dir(state: dict[str, Any]) -> Path | None:
    """Resolve the preinstalled HSSD scene used by relative USD references."""
    scene = state.get("dr_state_dict", {}).get("scene", {})
    if not str(scene.get("uid", "")).startswith("hssd:"):
        return None

    data_dir = scene.get("data_dir")
    if not isinstance(data_dir, str):
        raise ValueError("SIMPLE HSSD reset state is missing scene.data_dir.")

    from simple.utils import resolve_data_path

    scene_dir = Path(resolve_data_path(data_dir, auto_download=False))
    props_dir = scene_dir / "props"
    if not props_dir.is_dir():
        raise FileNotFoundError(
            f"SIMPLE HSSD material assets are missing under {props_dir}."
        )
    return scene_dir


def anchor_hssd_material_references(scene_dir: Path) -> Any:
    """Anchor HSSD material references to their containing scene layer."""
    from pxr import Sdf

    scene_path = scene_dir / f"{scene_dir.name}.usd"
    layer = Sdf.Layer.FindOrOpen(str(scene_path))
    if layer is None:
        raise FileNotFoundError(f"Cannot open SIMPLE HSSD scene layer {scene_path}.")
    props_dir = scene_dir / "props"
    material_files = {
        path.relative_to(props_dir).as_posix().casefold(): path.relative_to(
            props_dir
        ).as_posix()
        for path in props_dir.rglob("*")
        if path.is_file()
    }
    material_refs = [
        asset_path
        for asset_path in layer.GetCompositionAssetDependencies()
        if "/props/" in asset_path
    ]
    if not material_refs:
        raise ValueError(f"SIMPLE HSSD scene has no material references: {scene_path}.")
    for asset_path in material_refs:
        requested_path = asset_path.rsplit("/props/", 1)[1]
        actual_path = material_files.get(requested_path.casefold())
        if actual_path is None:
            raise FileNotFoundError(
                f"SIMPLE HSSD material {requested_path} is missing under {props_dir}."
            )
        anchored_path = f"./props/{actual_path}"
        if not layer.UpdateCompositionAssetDependency(asset_path, anchored_path):
            raise RuntimeError(
                f"Cannot update SIMPLE HSSD material reference {asset_path}."
            )
    return layer
