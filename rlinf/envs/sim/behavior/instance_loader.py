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

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from rlinf.envs.sim.behavior.utils import sync_robot_after_pose_override

TASK_INSTANCE_FILE_SUFFIX = "_template-tro_state.json"
TASK_INSTANCE_TEMPLATE_FILE_SUFFIX = "_template.json"
SUPPORTED_INSTANCE_RESAMPLE_MODES = ("disabled", "offline", "online")
SUPPORTED_INSTANCE_FILE_FORMATS = ("template", "tro_state")


@dataclass(frozen=True)
class ActivityInstanceFile:
    """Metadata for one cached BEHAVIOR activity instance file."""

    instance_id: int
    path: str
    file_format: str


def parse_activity_instance_filename(
    filename: str,
    activity_name: str,
    instance_file_format: str,
) -> tuple[int, int] | None:
    """Parse definition and instance ids from a cached instance filename.

    Args:
        filename: Candidate cached instance filename.
        activity_name: Expected BEHAVIOR activity name.
        instance_file_format: Expected cached file format. Must be ``template`` or
            ``tro_state``.

    Returns:
        A ``(definition_id, instance_id)`` tuple when the filename matches the
        expected activity and format. Returns ``None`` when it does not match.

    Raises:
        ValueError: If ``instance_file_format`` is unsupported.
    """
    if instance_file_format == "template":
        suffix = TASK_INSTANCE_TEMPLATE_FILE_SUFFIX
    elif instance_file_format == "tro_state":
        suffix = TASK_INSTANCE_FILE_SUFFIX
    else:
        raise ValueError(
            f"Unsupported cached instance format: {instance_file_format!r}."
        )

    if not filename.endswith(suffix):
        return None

    infix = f"_task_{activity_name}_"
    if infix not in filename:
        return None

    stem = filename[: -len(suffix)]
    _, suffix_stem = stem.split(infix, 1)
    definition_and_instance = suffix_stem.split("_")
    if len(definition_and_instance) != 2:
        return None

    definition_id, instance_id = definition_and_instance
    if not definition_id.isdigit() or not instance_id.isdigit():
        return None

    return int(definition_id), int(instance_id)


def discover_activity_instance_files(
    activity_instance_dir: str | os.PathLike[str],
    activity_name: str,
    activity_definition_id: int,
    instance_file_format: str,
) -> list[ActivityInstanceFile]:
    """Discover cached instance files for one BEHAVIOR activity.

    Args:
        activity_instance_dir: Directory containing cached instance JSON files.
        activity_name: Expected BEHAVIOR activity name.
        activity_definition_id: Expected BEHAVIOR activity definition id.
        instance_file_format: Cached instance file format to load.

    Returns:
        A sorted list of discovered cached instance files.

    Raises:
        ValueError: If the directory does not exist, contains duplicate instance
            ids for the requested format, or contains no matching files.
    """
    instance_dir = Path(activity_instance_dir)
    if not instance_dir.is_dir():
        raise ValueError(
            f"activity_instance_dir must be an existing directory, got: {instance_dir}"
        )

    instance_files = {}
    for entry in instance_dir.iterdir():
        if not entry.is_file():
            continue

        parsed = parse_activity_instance_filename(
            entry.name,
            activity_name=activity_name,
            instance_file_format=instance_file_format,
        )
        if parsed is None:
            continue

        definition_id, instance_id = parsed
        if definition_id != activity_definition_id:
            continue
        if instance_id in instance_files:
            raise ValueError(
                "Duplicate activity instance id "
                f"{instance_id} found in {instance_dir} for format "
                f"{instance_file_format!r}."
            )

        instance_files[instance_id] = ActivityInstanceFile(
            instance_id=instance_id,
            path=str(entry),
            file_format=instance_file_format,
        )

    if not instance_files:
        raise ValueError(
            "No cached BEHAVIOR task instances were found in "
            f"{instance_dir} for activity_name={activity_name}, "
            f"activity_definition_id={activity_definition_id}, "
            f"instance_file_format={instance_file_format!r}."
        )

    return [instance_files[k] for k in sorted(instance_files)]


def load_activity_instance_tro_state(
    env,
    instance_id: int,
    tro_file_path: str,
    reset_scene: bool = False,
) -> None:
    """Apply a cached task instance to scene 0, save its initial state, and optionally reset."""
    _apply_tro_state(env, 0, instance_id, tro_file_path)
    _settle_instance_scenes(env, [0])
    if reset_scene:
        env.scenes[0].reset()


def _apply_tro_state(env, env_idx: int, instance_id: int, tro_file_path: str) -> None:
    """Place one instance before the shared physics step for the batch."""
    from omnigibson.tasks.behavior_task import get_presampled_robot_poses
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    with open(tro_file_path, "r", encoding="utf-8") as f:
        tro_state = recursively_convert_to_torch(json.load(f))

    scene = env.scenes[env_idx]
    robot = env.task.get_agent(env, env_idx)
    object_scope = env.task.object_scopes[env_idx]
    robot_poses = tro_state.pop("robot_poses", None)
    for tro_key, state in tro_state.items():
        entity = object_scope.get(tro_key)
        assert entity is not None, (
            f"Cached task-relevant object {tro_key!r} is not present in the current "
            f"object_scope while loading {tro_file_path}."
        )
        if entity is robot:
            continue
        root_link = state.get("root_link") if isinstance(state, dict) else None
        if (
            scene.idx != 0
            and isinstance(root_link, dict)
            and "pos" in root_link
            and "ori" in root_link
        ):
            root_link["pos"], root_link["ori"] = (
                scene.convert_scene_relative_pose_to_world(
                    root_link["pos"], root_link["ori"]
                )
            )
        entity.load_state(state, serialized=False)

    if robot_poses is not None:
        robot.reset()
        robot_pose = get_presampled_robot_poses(
            robot_poses, robot.model, env.task.presampled_pose_key
        )[0]
        robot.set_position_orientation(
            robot_pose["position"],
            robot_pose["orientation"],
            frame="scene",
        )
        sync_robot_after_pose_override(robot)
    scene.write_task_metadata(key="robot_poses", data=robot_poses)
    if env.num_envs == 1:
        env.task.activity_instance_id = instance_id


def _settle_instance_scenes(env, env_indices):
    # Physics is shared. Settle once after placing the whole batch, then save each starting layout.
    import omnigibson as og
    from omnigibson.prims.entity_prim import EntityPrim

    og.sim.update_handles()
    for _ in range(25):
        og.sim.step_physics()
        for idx in env_indices:
            for entity in env.task.object_scopes[idx].values():
                if isinstance(entity, EntityPrim):
                    entity.keep_still()
    for idx in env_indices:
        env.scenes[idx].update_initial_file()


class ActivityInstanceLoader:
    """Prepare BEHAVIOR reset-time task instances for one vectorized env."""

    def __init__(
        self,
        omni_cfg: DictConfig,
        activity_name: str,
        activity_instance_id: int,
        instance_resample_mode: str,
        activity_instances: tuple[ActivityInstanceFile, ...],
    ):
        self.omni_cfg = omni_cfg
        self.activity_name = activity_name
        self.activity_instance_id = activity_instance_id
        self.instance_resample_mode = instance_resample_mode
        self.activity_instances = activity_instances

    @classmethod
    def from_omni_cfg(cls, omni_cfg: DictConfig) -> "ActivityInstanceLoader":
        """Build an instance loader from OmniGibson task config.

        Args:
            omni_cfg: Full OmniGibson config used to construct the BEHAVIOR env.

        Returns:
            A configured activity instance loader.

        Raises:
            ValueError: If the instance-resample configuration is invalid.
        """
        activity_name = OmegaConf.select(omni_cfg, "task.activity_name")
        activity_definition_id = OmegaConf.select(
            omni_cfg, "task.activity_definition_id"
        )
        activity_instance_id = OmegaConf.select(omni_cfg, "task.activity_instance_id")
        activity_instance_dir = OmegaConf.select(omni_cfg, "task.activity_instance_dir")
        instance_resample_mode = OmegaConf.select(
            omni_cfg, "task.instance_resample_mode"
        )
        instance_file_format = OmegaConf.select(omni_cfg, "task.instance_file_format")
        online_object_sampling = OmegaConf.select(
            omni_cfg, "task.online_object_sampling"
        )
        use_presampled_robot_pose = OmegaConf.select(
            omni_cfg, "task.use_presampled_robot_pose"
        )

        if not isinstance(instance_resample_mode, str):
            raise ValueError(
                f"task.instance_resample_mode must be a string, got {instance_resample_mode!r}."
            )
        instance_resample_mode = instance_resample_mode.lower()
        if instance_resample_mode not in SUPPORTED_INSTANCE_RESAMPLE_MODES:
            raise ValueError(
                "task.instance_resample_mode must be one of "
                f"{SUPPORTED_INSTANCE_RESAMPLE_MODES}, got {instance_resample_mode!r}."
            )

        if instance_file_format is not None:
            if not isinstance(instance_file_format, str):
                raise ValueError(
                    f"task.instance_file_format must be a string, got {instance_file_format!r}."
                )
            instance_file_format = instance_file_format.lower()
            if instance_file_format not in SUPPORTED_INSTANCE_FILE_FORMATS:
                raise ValueError(
                    "task.instance_file_format must be one of "
                    f"{SUPPORTED_INSTANCE_FILE_FORMATS}, got {instance_file_format!r}."
                )

        if instance_resample_mode == "online":
            if activity_instance_dir is not None:
                raise ValueError(
                    "task.activity_instance_dir is incompatible with "
                    "task.instance_resample_mode='online'."
                )
            if not online_object_sampling:
                raise ValueError(
                    "task.instance_resample_mode='online' requires "
                    "task.online_object_sampling to be True."
                )
            if use_presampled_robot_pose:
                raise ValueError(
                    "task.instance_resample_mode='online' requires "
                    "task.use_presampled_robot_pose to be False."
                )

        if activity_instance_dir is None:
            if instance_resample_mode == "offline":
                raise ValueError(
                    "task.activity_instance_dir must be set when "
                    "task.instance_resample_mode is 'offline'."
                )
            return cls(
                omni_cfg=omni_cfg,
                activity_name=activity_name,
                activity_instance_id=activity_instance_id,
                instance_resample_mode=instance_resample_mode,
                activity_instances=(),
            )

        if online_object_sampling:
            raise ValueError(
                "task.activity_instance_dir only supports cached offline instances. "
                "Please disable task.online_object_sampling."
            )
        if instance_file_format is None:
            raise ValueError(
                "task.instance_file_format must be set to 'template' or "
                "'tro_state' when task.activity_instance_dir is set."
            )

        activity_instances = tuple(
            discover_activity_instance_files(
                activity_instance_dir=activity_instance_dir,
                activity_name=activity_name,
                activity_definition_id=activity_definition_id,
                instance_file_format=instance_file_format,
            )
        )
        if instance_resample_mode == "disabled":
            instance_ids = {entry.instance_id for entry in activity_instances}
            if activity_instance_id not in instance_ids:
                raise ValueError(
                    f"task.activity_instance_id={activity_instance_id} is not present in "
                    f"task.activity_instance_dir={activity_instance_dir}."
                )

        return cls(
            omni_cfg=omni_cfg,
            activity_name=activity_name,
            activity_instance_id=activity_instance_id,
            instance_resample_mode=instance_resample_mode,
            activity_instances=activity_instances,
        )

    def prepare_reset(self, vec_env, reset_indices=None) -> None:
        """Apply any reset-time task-instance mutation required by the config.

        Args:
            vec_env: Batched OmniGibson environment.
            reset_indices: Local scene indices to prepare, or all scenes when omitted.
        """
        indices = range(vec_env.num_envs) if reset_indices is None else reset_indices
        if len(indices) == 0:
            return
        if self.instance_resample_mode == "online":
            import omnigibson as og

            # The 3.9.3 online sampler is bound to a single scene and stops physics on exit.
            vec_env.scenes[0].reset()
            try:
                success, feedback = vec_env.task.sampler.sample_states(
                    vec_env.task.compiled_task
                )
            finally:
                if og.sim.is_stopped():
                    og.sim.play()
            if not success:
                raise RuntimeError(
                    f"Could not resample {self.activity_name}: {feedback}"
                )
            return

        if not self.activity_instances:
            return

        if self.instance_resample_mode == "offline":
            instance_files = [random.choice(self.activity_instances) for _ in indices]
        else:
            instance_file = self._get_activity_instance(self.activity_instance_id)
            instance_files = [instance_file] * len(indices)

        self._apply_instance_files(vec_env, instance_files, indices)

    def _get_activity_instance(self, instance_id: int) -> ActivityInstanceFile:
        for instance_file in self.activity_instances:
            if instance_file.instance_id == instance_id:
                return instance_file
        raise ValueError(f"Activity instance id {instance_id} was not discovered.")

    def _apply_instance_files(
        self,
        vec_env,
        instance_files: list[ActivityInstanceFile],
        env_indices,
    ) -> None:
        file_format = instance_files[0].file_format
        if any(
            instance_file.file_format != file_format for instance_file in instance_files
        ):
            raise ValueError(
                "Mixed cached instance formats in a single reset are not supported."
            )
        if file_format == "template":
            self._load_template_instances(vec_env, instance_files, env_indices)
            return
        if file_format == "tro_state":
            self._load_tro_state_instances(vec_env, instance_files, env_indices)
            return
        raise ValueError(f"Unsupported cached instance format: {file_format}")

    def _load_template_instances(
        self,
        vec_env,
        instance_files: list[ActivityInstanceFile],
        env_indices,
    ) -> None:
        # SDK reload() recreates the whole batch. Restore the template in its own scene.
        for idx, instance_file in zip(env_indices, instance_files, strict=True):
            scene = vec_env.scenes[idx]
            with open(instance_file.path, encoding="utf-8") as file:
                template = json.load(file)
            # Full-house templates can omit the robot and include rooms excluded by the config.
            # Apply the same object selection used when OmniGibson constructs the scene.
            task_metadata = template["metadata"]["task"]
            init_info = template["objects_info"]["init_info"]
            objects = {
                name: info
                for name, info in init_info.items()
                if scene._should_load_object(info, task_metadata)
            }
            registry = template["state"]["registry"]
            registry["object_registry"] = {
                name: state
                for name, state in registry["object_registry"].items()
                if name in objects
            }
            robot = vec_env.task.get_agent(vec_env, idx)
            robot.reset()
            objects.setdefault(robot.name, robot.get_init_info())
            template["objects_info"]["init_info"] = objects
            scene.restore(template)
            vec_env.task.assign_object_scope_with_cache(vec_env, idx)
            scene.update_initial_file()
            if vec_env.num_envs == 1:
                vec_env.task.activity_instance_id = instance_file.instance_id

    def _load_tro_state_instances(
        self,
        vec_env,
        instance_files: list[ActivityInstanceFile],
        env_indices,
    ) -> None:
        for idx, instance_file in zip(env_indices, instance_files, strict=True):
            _apply_tro_state(
                vec_env, idx, instance_file.instance_id, instance_file.path
            )
        _settle_instance_scenes(vec_env, env_indices)
