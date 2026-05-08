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

import argparse
import json
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from rlinf.envs.behavior.utils import setup_omni_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_NUM_TRIALS = 50
DEFAULT_SETTLE_STEPS = 10
DEFAULT_ROBOT_STAGING_POSITION = (-50.0, -50.0, -50.0)
SUPPORTED_OUTPUT_FORMATS = ("template", "tro_state")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BEHAVIOR cached-instance generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate BEHAVIOR cached activity instances from an RLinf "
            "behavior_r1pro-style yaml."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "examples/embodiment/config/env/behavior_r1pro.yaml"),
        help="Path to an RLinf BEHAVIOR env yaml with an omni_config section.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        required=True,
        choices=SUPPORTED_OUTPUT_FORMATS,
        help="Cached instance format to generate.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        required=True,
        help="First activity_instance_id to generate.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        required=True,
        help="Last activity_instance_id to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initial activity_instance_id seed used to bootstrap the sampler.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Maximum sampling trials per activity_instance_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write cached instances into. Defaults to "
            "omni_config.task.activity_instance_dir when set, otherwise RLinf's "
            "default challenge-task-instances directory."
        ),
    )
    parser.add_argument(
        "--robot-staging-position",
        type=float,
        nargs=3,
        default=DEFAULT_ROBOT_STAGING_POSITION,
        metavar=("X", "Y", "Z"),
        help="Temporary robot position used during online object sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached instance files.",
    )
    return parser.parse_args()


def load_env_cfg(config_path: str) -> DictConfig:
    """Load an RLinf BEHAVIOR env yaml.

    Args:
        config_path: Path to the yaml file.

    Returns:
        The loaded OmegaConf config.

    Raises:
        ValueError: If the yaml does not contain an ``omni_config`` section.
    """
    env_cfg = OmegaConf.load(config_path)
    if OmegaConf.select(env_cfg, "omni_config") is None:
        raise ValueError(
            f"{config_path} must be an RLinf BEHAVIOR env yaml with an omni_config section."
        )
    return env_cfg


def build_sampling_omni_cfg(
    env_cfg: DictConfig,
    seed: int,
    robot_staging_position: tuple[float, float, float],
) -> DictConfig:
    """Build the OmniGibson config used for cached-instance sampling.

    Args:
        env_cfg: RLinf BEHAVIOR env yaml.
        seed: Initial ``activity_instance_id`` used to bootstrap sampling.
        robot_staging_position: Temporary robot position during sampling.

    Returns:
        The OmniGibson config used to create the sampling environment.
    """
    omni_cfg = setup_omni_cfg(env_cfg)
    OmegaConf.update(omni_cfg, "env.automatic_reset", False)
    OmegaConf.update(omni_cfg, "task.activity_instance_id", seed)
    OmegaConf.update(omni_cfg, "task.activity_instance_dir", None, merge=False)
    OmegaConf.update(omni_cfg, "task.instance_resample_mode", "disabled")
    OmegaConf.update(omni_cfg, "task.online_object_sampling", True)
    OmegaConf.update(omni_cfg, "task.use_presampled_robot_pose", False)
    OmegaConf.update(omni_cfg, "scene.scene_instance", None, merge=False)
    OmegaConf.update(omni_cfg, "scene.scene_file", None, merge=False)
    for idx, _ in enumerate(OmegaConf.select(omni_cfg, "robots", default=[])):
        OmegaConf.update(
            omni_cfg,
            f"robots[{idx}].position",
            list(robot_staging_position),
            merge=False,
        )
    return omni_cfg


def resolve_output_dir(omni_cfg: DictConfig, explicit_output_dir: str | None) -> Path:
    """Resolve the directory used to save cached instances.

    Args:
        omni_cfg: OmniGibson config used for sampling.
        explicit_output_dir: CLI override directory.

    Returns:
        The directory where cached instances should be written.

    Raises:
        ValueError: If no output directory can be resolved.
    """
    if explicit_output_dir is not None:
        return Path(explicit_output_dir)

    activity_instance_dir = OmegaConf.select(omni_cfg, "task.activity_instance_dir")
    if activity_instance_dir is not None:
        return Path(activity_instance_dir)

    from omnigibson.utils.asset_utils import get_task_instance_path

    scene_model = OmegaConf.select(omni_cfg, "scene.scene_model")
    activity_name = OmegaConf.select(omni_cfg, "task.activity_name")
    task_instance_root = get_task_instance_path(scene_model)
    if task_instance_root is None:
        raise ValueError(
            "Could not resolve a default cached-instance directory for "
            f"scene_model={scene_model!r}. Pass --output-dir explicitly."
        )
    return (
        Path(task_instance_root)
        / "json"
        / f"{scene_model}_task_{activity_name}_instances"
    )


def configure_sampling_macros() -> None:
    """Apply the OmniGibson sampling macro settings used by BEHAVIOR scripts."""
    from omnigibson.macros import macros

    macros.systems.micro_particle_system.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY = 0.5
    macros.systems.macro_particle_system.MACRO_PARTICLE_SYSTEM_MAX_DENSITY = 200.0
    macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 5
    macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 5


def build_output_path(task, output_dir: Path, output_format: str) -> Path:
    """Build the output file path for the current task instance.

    Args:
        task: Active OmniGibson BEHAVIOR task.
        output_dir: Directory that stores cached instances.
        output_format: Cached instance format.

    Returns:
        The full output path for the current ``activity_instance_id``.

    Raises:
        ValueError: If ``output_format`` is unsupported.
    """
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output_format {output_format!r}. Expected one of {SUPPORTED_OUTPUT_FORMATS}."
        )

    filename = task.get_cached_activity_scene_filename(
        scene_model=task.scene_name,
        activity_name=task.activity_name,
        activity_definition_id=task.activity_definition_id,
        activity_instance_id=task.activity_instance_id,
    )
    suffix = ".json" if output_format == "template" else "-tro_state.json"
    return output_dir / f"{filename}{suffix}"


def dump_tro_state(env, output_path: Path, overwrite: bool) -> None:
    """Write the current task-relevant state to ``output_path``.

    Args:
        env: Active OmniGibson environment.
        output_path: Target ``*_template-tro_state.json`` file.
        overwrite: Whether existing files may be overwritten.

    Raises:
        FileExistsError: If ``output_path`` exists and ``overwrite`` is false.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing cached instance: {output_path}"
        )

    from omnigibson.utils.config_utils import TorchEncoder

    tro_state = {
        bddl_name: bddl_inst.dump_state(serialized=False)
        for bddl_name, bddl_inst in env.task.object_scope.items()
        if bddl_inst.exists and getattr(bddl_inst, "synset", None) != "agent"
    }
    robot_poses = env.scene.get_task_metadata(key="robot_poses")
    if robot_poses is not None:
        tro_state["robot_poses"] = robot_poses

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tro_state, f, cls=TorchEncoder, indent=4)


def save_activity_instance(
    env,
    output_dir: Path,
    output_format: str,
    overwrite: bool,
) -> Path:
    """Save the current sampled instance in the requested format.

    Args:
        env: Active OmniGibson environment.
        output_dir: Directory that stores cached instances.
        output_format: Cached instance format.
        overwrite: Whether existing files may be overwritten.

    Returns:
        The written output path.

    Raises:
        FileExistsError: If the target file exists and ``overwrite`` is false.
    """
    output_path = build_output_path(env.task, output_dir, output_format)
    if output_format == "template":
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing cached instance: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        env.task.save_task(
            env=env,
            save_dir=str(output_dir),
            override=True,
            task_relevant_only=False,
        )
    else:
        dump_tro_state(env, output_path=output_path, overwrite=overwrite)
    return output_path


def generate_activity_instances(
    env,
    output_dir: Path,
    output_format: str,
    start_idx: int,
    end_idx: int,
    num_trials: int,
    overwrite: bool,
) -> None:
    """Generate cached BEHAVIOR instances by repeated online sampling.

    Args:
        env: Active OmniGibson environment configured for online object sampling.
        output_dir: Directory that stores cached instances.
        output_format: Cached instance format.
        start_idx: First activity instance id to generate.
        end_idx: Last activity instance id to generate.
        num_trials: Maximum trials per instance id.
        overwrite: Whether existing files may be overwritten.

    Raises:
        RuntimeError: If any requested instance id could not be generated.
        ValueError: If ``start_idx`` / ``end_idx`` are invalid.
    """
    if start_idx > end_idx:
        raise ValueError(
            f"start_idx must be <= end_idx, got {start_idx} and {end_idx}."
        )
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}.")

    import omnigibson as og
    from omnigibson.objects import DatasetObject
    from omnigibson.sampling.utils import validate_task
    from omnigibson.utils.asset_utils import get_dataset_path

    default_scene_path = (
        Path(get_dataset_path("behavior-1k-assets"))
        / "scenes"
        / env.task.scene_name
        / "json"
        / f"{env.task.scene_name}_best.json"
    )
    if not default_scene_path.is_file():
        raise ValueError(f"Stable scene json does not exist: {default_scene_path}")

    with default_scene_path.open("r", encoding="utf-8") as f:
        default_scene_dict = json.load(f)

    env.task.sampler._parse_inroom_object_room_assignment()
    env.task.sampler._build_sampling_order()

    for system in env.scene.active_systems.values():
        system.remove_all_particles()
    og.sim.step()
    initial_state = og.sim.dump_state()

    failed_instance_ids = []
    for instance_id in range(start_idx, end_idx + 1):
        env.task.activity_instance_id = instance_id
        output_path = build_output_path(env.task, output_dir, output_format)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing cached instance: {output_path}"
            )

        print(
            f"Sampling activity_instance_id={instance_id} -> {output_path.name}",
            flush=True,
        )
        for trial_idx in range(num_trials):
            og.sim.load_state(initial_state)
            og.sim.step()

            error_msg = env.task.sampler._sample_initial_conditions_final()
            if error_msg is not None:
                print(
                    f"  trial {trial_idx}: sampling failed: {error_msg}",
                    flush=True,
                )
                continue

            for _ in range(DEFAULT_SETTLE_STEPS):
                og.sim.step()

            for entity in env.task.object_scope.values():
                if isinstance(entity, DatasetObject):
                    entity.keep_still()

            for _ in range(DEFAULT_SETTLE_STEPS):
                og.sim.step()

            task_final_state = env.scene.dump_state()
            validated, error_msg = validate_task(
                env.task,
                {"state": task_final_state},
                default_scene_dict,
            )
            if not validated:
                print(
                    f"  trial {trial_idx}: validation failed: {error_msg}",
                    flush=True,
                )
                continue

            env.scene.load_state(task_final_state)
            env.scene.update_initial_file()
            saved_path = save_activity_instance(
                env,
                output_dir=output_dir,
                output_format=output_format,
                overwrite=overwrite,
            )
            print(f"  trial {trial_idx}: saved {saved_path}", flush=True)
            break
        else:
            failed_instance_ids.append(instance_id)

    if failed_instance_ids:
        raise RuntimeError(
            "Failed to generate cached instances for "
            f"activity_instance_id(s): {failed_instance_ids}"
        )


def main() -> None:
    """Run BEHAVIOR cached-instance generation."""
    args = parse_args()
    env_cfg = load_env_cfg(args.config)
    configured_output_dir = OmegaConf.select(
        env_cfg,
        "omni_config.task.activity_instance_dir",
    )

    print("Building OmniGibson sampling config...", flush=True)
    omni_cfg = build_sampling_omni_cfg(
        env_cfg=env_cfg,
        seed=args.seed,
        robot_staging_position=tuple(args.robot_staging_position),
    )
    output_dir = resolve_output_dir(
        omni_cfg,
        args.output_dir if args.output_dir is not None else configured_output_dir,
    )
    print(f"Cached instances will be written to: {output_dir}", flush=True)

    configure_sampling_macros()

    import omnigibson as og

    env = None
    try:
        omni_cfg_dict = OmegaConf.to_container(
            omni_cfg,
            resolve=True,
            throw_on_missing=True,
        )
        env = og.Environment(omni_cfg_dict)
        generate_activity_instances(
            env,
            output_dir=output_dir,
            output_format=args.output_format,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            num_trials=args.num_trials,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"Error during instance generation: {e}", flush=True)
    finally:
        if env is not None:
            env.close()
        og.shutdown()


if __name__ == "__main__":
    main()
