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

from omegaconf import DictConfig, ListConfig, OmegaConf

from rlinf.envs.behavior.utils import (
    clear_robot_grasp_state,
    reset_robot_joint_state_to_reset_pose,
    sync_robot_after_pose_override,
)

TASK_INSTANCE_FILE_SUFFIX = "_template-tro_state.json"
TASK_INSTANCE_TEMPLATE_FILE_SUFFIX = "_template.json"
SUPPORTED_INSTANCE_RESAMPLE_MODES = ("disabled", "offline", "online")
SUPPORTED_INSTANCE_FILE_FORMATS = ("template", "tro_state")
RLINF_REPLAY_METADATA_KEY = "rlinf_replay"
DEFAULT_MIDROLLOUT_RESTORE_SETTLE_STEPS = 120  # about 1 second at 120 Hz


def parse_activity_instance_ids(value) -> list[int] | None:
    """Parse a BEHAVIOR activity instance id spec.

    Supported forms are an integer, a list/tuple of integers or range strings,
    or a comma-separated string such as ``"1000-1255,1300"``. Ranges are
    inclusive on both ends.
    """
    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]

    values = value if isinstance(value, (list, tuple)) else [value]
    instance_ids: list[int] = []
    for item in values:
        if isinstance(item, int):
            instance_ids.append(int(item))
            continue
        if not isinstance(item, str):
            raise ValueError(
                "task.activity_instance_id entries must be integers or range "
                f"strings, got {item!r}."
            )
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                start, end = int(left), int(right)
                if start > end:
                    raise ValueError(
                        f"Invalid task.activity_instance_id range {part!r}: "
                        "start > end."
                    )
                instance_ids.extend(range(start, end + 1))
            else:
                instance_ids.append(int(part))

    if not instance_ids:
        raise ValueError("task.activity_instance_id must not be empty.")
    return instance_ids


def _is_agent_tro_key(tro_key: str, entity=None) -> bool:
    return tro_key.startswith("agent.") or getattr(entity, "synset", None) in (
        "agent",
        "agent.n.01",
    )


def _object_for_saved_prim_path(env, saved_prim_path: str):
    basename = saved_prim_path.rstrip("/").split("/")[-1]
    for entity in env.task.object_scope.values():
        if not getattr(entity, "exists", False):
            continue
        prim_path = getattr(entity, "prim_path", "")
        if prim_path.rstrip("/").split("/")[-1] == basename:
            return entity
    return env.scene.object_registry("prim_path", saved_prim_path, None)


def _as_pose_tensor(value):
    if hasattr(value, "to"):
        return value

    import torch

    return torch.as_tensor(value, dtype=torch.float32)


def _restore_robot_joint_positions(robot, saved_positions, preserve_base: bool = True):
    """Restore articulation joints while keeping the restored base pose stable."""
    import torch

    saved_positions = torch.as_tensor(saved_positions)
    if preserve_base:
        current_positions = robot.get_joint_positions()
        if current_positions is not None:
            current_positions = torch.as_tensor(
                current_positions,
                dtype=saved_positions.dtype,
                device=saved_positions.device,
            )
            if current_positions.shape == saved_positions.shape:
                saved_positions = saved_positions.clone()
                saved_positions[:6] = current_positions[:6]

    robot.set_joint_positions(positions=saved_positions, drive=False)
    robot.set_joint_velocities(
        velocities=torch.zeros_like(saved_positions), drive=False
    )
    robot.keep_still()


def _sync_robot_controller_no_op_goals(robot) -> None:
    """Set controller goals to the restored state so the next action starts cleanly."""
    controllers = getattr(robot, "controllers", None)
    get_control_dict = getattr(robot, "get_control_dict", None)
    if not controllers or not callable(get_control_dict):
        return

    control_dict = get_control_dict()
    for controller in controllers.values():
        compute_no_op_goal = getattr(controller, "compute_no_op_goal", None)
        if callable(compute_no_op_goal):
            controller._goal = compute_no_op_goal(control_dict=control_dict)


def _midrollout_restore_settle_steps() -> int:
    value = os.environ.get("RLINF_BEHAVIOR_TRO_RESTORE_SETTLE_STEPS")
    if value is None:
        return DEFAULT_MIDROLLOUT_RESTORE_SETTLE_STEPS
    steps = int(value)
    if steps < 0:
        raise ValueError(
            "RLINF_BEHAVIOR_TRO_RESTORE_SETTLE_STEPS must be non-negative, "
            f"got {steps}."
        )
    return steps


def _settle_restored_midrollout_state(env, robot, saved_positions) -> None:
    """Hold the restored robot state for a few physics ticks before rollout."""
    import omnigibson as og

    for _ in range(_midrollout_restore_settle_steps()):
        _restore_robot_joint_positions(robot, saved_positions)
        _sync_robot_controller_no_op_goals(robot)
        og.sim.step_physics()
        _restore_robot_joint_positions(robot, saved_positions)
        _sync_robot_controller_no_op_goals(robot)
        for entity in env.task.object_scope.values():
            if entity.exists and not entity.is_system:
                entity.keep_still()


def _rebase_agent_grasp_paths(env, agent_state: dict, robot_pose: dict | None) -> dict:
    agent_state = dict(agent_state)
    if robot_pose is not None and isinstance(agent_state.get("root_link"), dict):
        root_link = dict(agent_state["root_link"])
        root_pos = _as_pose_tensor(robot_pose["position"])
        root_ori = _as_pose_tensor(robot_pose["orientation"])
        if getattr(env.scene, "idx", 0) != 0:
            root_pos, root_ori = env.scene.convert_scene_relative_pose_to_world(
                root_pos, root_ori
            )
        root_link["pos"] = root_pos
        root_link["ori"] = root_ori
        agent_state["root_link"] = root_link

    grasp_params = agent_state.get("ag_obj_constraint_params")
    if not isinstance(grasp_params, dict):
        return agent_state

    rebased_params = {}
    for arm, arm_params in grasp_params.items():
        if not arm_params:
            rebased_params[arm] = arm_params
            continue

        arm_params = dict(arm_params)
        obj = _object_for_saved_prim_path(env, arm_params["ag_obj_prim_path"])
        if obj is None:
            rebased_params[arm] = {}
            continue

        link_name = arm_params["ag_link_prim_path"].rstrip("/").split("/")[-1]
        link = getattr(obj, "links", {}).get(link_name)
        if link is None:
            rebased_params[arm] = {}
            continue

        arm_params["ag_obj_prim_path"] = obj.prim_path
        arm_params["ag_link_prim_path"] = link.prim_path
        rebased_params[arm] = arm_params

    agent_state["ag_obj_constraint_params"] = rebased_params
    return agent_state


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


def _restore_ag_state(robot, scene, ag_state: dict) -> None:
    """Restore assisted-grasping constraints from saved state."""
    for arm in robot.arm_names:
        if robot._ag_obj_constraints.get(arm) is not None:
            robot.release_grasp_immediately(arm=arm)

    for arm, ag_info in ag_state.items():
        ag_obj = scene.object_registry("prim_path", ag_info["ag_obj_prim_path"])
        if ag_obj is None:
            continue
        link_name = ag_info["ag_link_prim_path"].split("/")[-1]
        ag_link = ag_obj.links.get(link_name)
        if ag_link is None:
            continue
        robot._establish_grasp_rigid(
            arm=arm,
            ag_data=(ag_obj, ag_link),
            contact_pos=ag_info.get("contact_pos"),
        )


def load_activity_instance_tro_state(
    env,
    instance_id: int,
    tro_file_path: str,
    reset_scene: bool = False,
) -> None:
    """Apply a cached tro_state file to an existing OmniGibson env.

    Args:
        env: OmniGibson environment to mutate in place.
        instance_id: Activity instance id represented by ``tro_file_path``.
        tro_file_path: Path to a ``*_template-tro_state.json`` file.
        reset_scene: Whether to call ``env.scene.reset()`` after applying the
            cached state.
    """
    import omnigibson as og
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    env.task.activity_instance_id = instance_id
    with open(tro_file_path, "r", encoding="utf-8") as f:
        raw_tro_state = json.load(f)
    replay_metadata = raw_tro_state.pop(RLINF_REPLAY_METADATA_KEY, None)
    tro_state = recursively_convert_to_torch(raw_tro_state)

    robot = env.task.get_agent(env)
    robot_name = getattr(robot, "model_name", getattr(robot, "model", None))
    assert robot_name is not None, (
        "Robot model name is required to load task instances."
    )
    robot_poses = tro_state.pop("robot_poses", None)
    robot_pose = None
    if robot_poses is not None:
        assert robot_name in robot_poses, (
            f"{robot_name} presampled pose is not found in {tro_file_path}"
        )
        robot_pose = robot_poses[robot_name][0]
    robot_joints = tro_state.pop("robot_joints", None)
    ag_state = tro_state.pop("ag_state", None)
    is_midrollout = robot_joints is not None
    load_agent_state = os.environ.get("RLINF_BEHAVIOR_TRO_LOAD_AGENT_STATE", "1") != "0"
    agent_states = {
        tro_key: tro_state.pop(tro_key)
        for tro_key in list(tro_state)
        if _is_agent_tro_key(tro_key)
    }
    if not load_agent_state:
        agent_states = {}

    clear_robot_grasp_state(robot)

    for tro_key, state in tro_state.items():
        entity = env.task.object_scope.get(tro_key)
        assert entity is not None, (
            f"Cached task-relevant object {tro_key!r} is not present in the current "
            f"object_scope while loading {tro_file_path}."
        )
        if _is_agent_tro_key(tro_key, entity):
            continue
        if (
            getattr(env.scene, "idx", 0) != 0
            and isinstance(state, dict)
            and isinstance(state.get("root_link"), dict)
            and "pos" in state["root_link"]
            and "ori" in state["root_link"]
        ):
            rebased_state = dict(state)
            rebased_root_link = dict(state["root_link"])
            rebased_pos, rebased_ori = env.scene.convert_scene_relative_pose_to_world(
                rebased_root_link["pos"],
                rebased_root_link["ori"],
            )
            rebased_root_link["pos"] = rebased_pos
            rebased_root_link["ori"] = rebased_ori
            rebased_state["root_link"] = rebased_root_link
            state = rebased_state
        entity.load_state(state, serialized=False)

    loaded_agent_state = False
    if agent_states:
        agent_state = _rebase_agent_grasp_paths(
            env, next(iter(agent_states.values())), robot_pose
        )
        robot.load_state(agent_state, serialized=False)
        loaded_agent_state = True

    if robot_pose is not None:
        robot.set_position_orientation(
            robot_pose["position"],
            robot_pose["orientation"],
            frame="scene",
        )
        if is_midrollout and robot_name in robot_joints:
            saved_positions = robot_joints[robot_name]["joint_positions"]
            _restore_robot_joint_positions(robot, saved_positions)
            _sync_robot_controller_no_op_goals(robot)
        env.scene.write_task_metadata(key="robot_poses", data=robot_poses)
    else:
        env.scene.write_task_metadata(key="robot_poses", data=None)
    env.scene.write_task_metadata(key=RLINF_REPLAY_METADATA_KEY, data=replay_metadata)

    if not loaded_agent_state and not is_midrollout:
        reset_robot_joint_state_to_reset_pose(robot, preserve_base_pose=True)
    sync_robot_after_pose_override(robot)

    if not is_midrollout:
        for _ in range(25):
            og.sim.step_physics()
            for entity in env.task.object_scope.values():
                if entity.exists and not entity.is_system:
                    entity.keep_still()

    if is_midrollout and ag_state is not None:
        _restore_ag_state(robot, env.scene, ag_state)
        _sync_robot_controller_no_op_goals(robot)
    if is_midrollout and robot_name in robot_joints:
        _settle_restored_midrollout_state(
            env,
            robot,
            robot_joints[robot_name]["joint_positions"],
        )
        _sync_robot_controller_no_op_goals(robot)

    env.scene.update_initial_file()
    if reset_scene:
        env.scene.reset()


class ActivityInstanceLoader:
    """Prepare BEHAVIOR reset-time task instances for one vectorized env."""

    def __init__(
        self,
        omni_cfg: DictConfig,
        activity_name: str,
        activity_instance_id: int,
        instance_resample_mode: str,
        activity_instances: tuple[ActivityInstanceFile, ...],
        seed: int | None = None,
    ):
        self.omni_cfg = omni_cfg
        self.activity_name = activity_name
        self.activity_instance_id = activity_instance_id
        self.instance_resample_mode = instance_resample_mode
        self.activity_instances = activity_instances
        self._rng = random.Random(seed)

    @classmethod
    def from_omni_cfg(
        cls, omni_cfg: DictConfig, seed_offset: int = 0
    ) -> "ActivityInstanceLoader":
        """Build an instance loader from OmniGibson task config.

        Args:
            omni_cfg: Full OmniGibson config used to construct the BEHAVIOR env.
            seed_offset: Added to the config seed to derive the sampling RNG seed,
                so each env shard can sample activity instances deterministically
                and independently.

        Returns:
            A configured activity instance loader.

        Raises:
            ValueError: If the instance-resample configuration is invalid.
        """
        seed = int(OmegaConf.select(omni_cfg, "seed", default=0) or 0) + int(
            seed_offset
        )
        activity_name = OmegaConf.select(omni_cfg, "task.activity_name")
        activity_definition_id = OmegaConf.select(
            omni_cfg, "task.activity_definition_id"
        )
        activity_instance_id = OmegaConf.select(omni_cfg, "task.activity_instance_id")
        parsed_instance_ids = parse_activity_instance_ids(activity_instance_id)
        if parsed_instance_ids is None:
            requested_instance_ids = None
        elif len(parsed_instance_ids) == 1:
            requested_instance_ids = None
            activity_instance_id = parsed_instance_ids[0]
        else:
            requested_instance_ids = parsed_instance_ids
            activity_instance_id = requested_instance_ids[0]
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
            OmegaConf.update(
                omni_cfg,
                "task.activity_instance_id",
                activity_instance_id,
                merge=False,
            )
            return cls(
                omni_cfg=omni_cfg,
                activity_name=activity_name,
                activity_instance_id=activity_instance_id,
                instance_resample_mode=instance_resample_mode,
                activity_instances=(),
                seed=seed,
            )

        if activity_instance_dir is None:
            if instance_resample_mode == "offline":
                raise ValueError(
                    "task.activity_instance_dir must be set when "
                    "task.instance_resample_mode is 'offline'."
                )
            OmegaConf.update(
                omni_cfg,
                "task.activity_instance_id",
                activity_instance_id,
                merge=False,
            )
            return cls(
                omni_cfg=omni_cfg,
                activity_name=activity_name,
                activity_instance_id=activity_instance_id,
                instance_resample_mode=instance_resample_mode,
                activity_instances=(),
                seed=seed,
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
            if requested_instance_ids is not None:
                raise ValueError(
                    "task.instance_resample_mode='disabled' requires exactly one "
                    "task.activity_instance_id."
                )
            instance_ids = {entry.instance_id for entry in activity_instances}
            if activity_instance_id not in instance_ids:
                raise ValueError(
                    f"task.activity_instance_id={activity_instance_id} is not present in "
                    f"task.activity_instance_dir={activity_instance_dir}."
                )
        elif requested_instance_ids is not None:
            by_id = {
                entry.instance_id: entry for entry in activity_instances
            }
            activity_instances = tuple(by_id[i] for i in requested_instance_ids)

        # Challenge tro_state instances are applied after construction; bootstrap
        # OmniGibson from the complete seed template (instance 0) first. Otherwise
        # OmniGibson derives `scene_instance` from `activity_instance_id` and looks
        # for `..._0_<id>_template.json`, which only exists for instance 0.
        bootstrap_activity_instance_id = (
            0 if instance_file_format == "tro_state" else activity_instance_id
        )
        OmegaConf.update(
            omni_cfg,
            "task.activity_instance_id",
            bootstrap_activity_instance_id,
            merge=False,
        )

        return cls(
            omni_cfg=omni_cfg,
            activity_name=activity_name,
            activity_instance_id=activity_instance_id,
            instance_resample_mode=instance_resample_mode,
            activity_instances=activity_instances,
            seed=seed,
        )

    def prepare_reset(
        self,
        vec_env,
        instance_ids: list[int] | None = None,
        group_size: int = 1,
    ) -> None:
        """Apply any reset-time task-instance mutation required by the config.

        Args:
            vec_env: Vectorized OmniGibson environment whose child envs should be
                updated before ``vec_env.reset()``.
            instance_ids: Optional per-env cached instance ids to load for this
                reset. Used by demonstration replay initialization so the
                simulator reset matches the replayed episode.
            group_size: Number of trajectories sharing one sampled reset instance.
                GRPO uses this to keep each group on the same initial condition.
        """
        group_size = int(group_size)
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}.")
        if instance_ids is not None and len(instance_ids) != len(vec_env.envs):
            raise ValueError(
                "Number of requested instance ids must match the number of "
                f"vectorized environments, got {len(instance_ids)} and {len(vec_env.envs)}."
            )
        if instance_ids is None and len(vec_env.envs) % group_size != 0:
            raise ValueError(
                "Number of vectorized environments must be divisible by group_size "
                f"for grouped BEHAVIOR reset, got {len(vec_env.envs)} and {group_size}."
            )

        if self.instance_resample_mode == "online":
            if instance_ids is not None:
                raise ValueError(
                    "Per-episode replay instance ids are not supported with "
                    "task.instance_resample_mode='online'."
                )
            task_cfg = OmegaConf.select(self.omni_cfg, "task")
            for env in vec_env.envs:
                env.update_task(task_config=task_cfg)
            return

        if not self.activity_instances:
            if instance_ids is not None:
                raise ValueError(
                    "Per-episode replay instance ids require cached activity instances; "
                    "set task.activity_instance_dir and use offline cached instances."
                )
            return

        if instance_ids is not None:
            instance_files = [self._get_activity_instance(i) for i in instance_ids]
        elif self.instance_resample_mode == "offline":
            group_count = len(vec_env.envs) // group_size
            group_files = self._sample_activity_instances(group_count)
            instance_files = [
                instance_file
                for instance_file in group_files
                for _ in range(group_size)
            ]
        else:
            instance_file = self._get_activity_instance(self.activity_instance_id)
            instance_files = [instance_file] * len(vec_env.envs)

        self._apply_instance_files(vec_env, instance_files)

    def _sample_activity_instances(self, count: int) -> list:
        """Sample ``count`` activity instances with replacement."""
        if count <= 0:
            return []
        instances = list(self.activity_instances)
        if count <= len(instances):
            return self._rng.sample(instances, k=count)
        return [self._rng.choice(instances) for _ in range(count)]

    def _get_activity_instance(self, instance_id: int) -> ActivityInstanceFile:
        for instance_file in self.activity_instances:
            if instance_file.instance_id == instance_id:
                return instance_file
        raise ValueError(f"Activity instance id {instance_id} was not discovered.")

    def _apply_instance_files(
        self,
        vec_env,
        instance_files: list[ActivityInstanceFile],
    ) -> None:
        if len(instance_files) != len(vec_env.envs):
            raise ValueError(
                "Number of cached activity instance files must match the number of "
                f"vectorized environments, got {len(instance_files)} and {len(vec_env.envs)}."
            )

        file_format = instance_files[0].file_format
        if any(
            instance_file.file_format != file_format for instance_file in instance_files
        ):
            raise ValueError(
                "Mixed cached instance formats in a single reset are not supported."
            )
        if file_format == "template":
            self._load_template_instances(vec_env, instance_files)
            return
        if file_format == "tro_state":
            self._load_tro_state_instances(vec_env, instance_files)
            return
        raise ValueError(f"Unsupported cached instance format: {file_format}")

    def _load_template_instances(
        self,
        vec_env,
        instance_files: list[ActivityInstanceFile],
    ) -> None:
        import omnigibson as og

        if not og.sim.is_stopped():
            og.sim.stop()

        for env, instance_file in zip(vec_env.envs, instance_files, strict=True):
            env.reload(self._build_reload_config(instance_file))

        og.sim.play()
        for env in vec_env.envs:
            env.post_play_load()

    def _load_tro_state_instances(
        self,
        vec_env,
        instance_files: list[ActivityInstanceFile],
    ) -> None:
        for env, instance_file in zip(vec_env.envs, instance_files, strict=True):
            load_activity_instance_tro_state(
                env,
                instance_id=instance_file.instance_id,
                tro_file_path=instance_file.path,
                reset_scene=False,
            )

    def _build_reload_config(self, instance_file: ActivityInstanceFile) -> dict:
        cfg = OmegaConf.create(OmegaConf.to_container(self.omni_cfg, resolve=False))
        OmegaConf.update(cfg, "task.activity_instance_id", instance_file.instance_id)
        OmegaConf.update(cfg, "task.activity_instance_dir", None, merge=False)
        OmegaConf.update(cfg, "scene.scene_file", instance_file.path, merge=False)
        OmegaConf.update(cfg, "scene.scene_instance", None, merge=False)
        return OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        )
