import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym

import numpy as np
import torch

import mani_skill.envs
from mani_skill import ASSET_DIR
from mani_skill.utils import io_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from mshab.utils.bench.interact_scene_builder import ReplicaCADInteractSceneBuilder
from mshab.utils.io import NoIndent, NoIndentSupportingJSONEncoder
from mshab.utils.profile import Profiler
from mshab.utils.time import NonOverlappingTimeProfiler


MS_SIM_CONFIG = dict(sim_freq=100, control_freq=20)


def make_habitat_env(config_yaml_name="interact", concur_render=True, auto_sleep=True):
    import habitat  # type: ignore
    import habitat_sim  # type: ignore
    from gym import spaces as old_gym_spaces  # type: ignore
    from habitat_baselines.common.habitat_env_factory import HabitatVectorEnvFactory  # type: ignore

    from gymnasium import spaces

    def batch_spaces(b_spaces):
        if all(isinstance(s, old_gym_spaces.Box) for s in b_spaces):
            low = np.stack([s.low for s in b_spaces])
            high = np.stack([s.high for s in b_spaces])
            return spaces.Box(low=low, high=high, dtype=b_spaces[0].dtype)

        elif all(isinstance(s, old_gym_spaces.Discrete) for s in b_spaces):
            return spaces.MultiDiscrete([s.n for s in b_spaces])

        elif all(isinstance(s, old_gym_spaces.Dict) for s in b_spaces):
            return spaces.Dict(
                dict((k, batch_spaces([s[k] for s in b_spaces])) for k in b_spaces[0])
            )

        else:
            raise ValueError(
                f"{[type(s) for s in b_spaces]} Unsupported space type or mixed types in the list of observation spaces"
            )

    config = habitat.get_config(
        str(Path(__file__).parent.absolute() / f"{config_yaml_name}.yaml"),
        overrides=[
            f"+habitat_baselines.num_environments={int(args.num_envs)}",
            f"++habitat.seed={args.seed}",
            f"++habitat.simulator.concur_render={concur_render}",
            f"++habitat.simulator.auto_sleep={auto_sleep}",
        ],
    )
    vec_env = HabitatVectorEnvFactory().construct_envs(config)

    vec_env.reset()
    vec_env.single_observation_space = vec_env.observation_spaces[0]
    vec_env.observation_space = batch_spaces(vec_env.observation_spaces)
    vec_env.single_action_space = vec_env.action_spaces[0]
    vec_env.action_space = batch_spaces(vec_env.action_spaces)

    return vec_env


def make_interact_env():
    return gym.make(
        "SceneManipulation-v1",
        num_envs=args.num_envs,
        obs_mode="rgbd",
        reward_mode="normalized_dense",
        control_mode="pd_joint_delta_pos_stiff_body",
        render_mode="all",
        shader_dir="minimal",
        robot_uids="fetch",
        sim_backend="gpu",
        sim_config=MS_SIM_CONFIG,
        # time limit
        max_episode_steps=100_000,
        # SceneManipulationEnv args
        scene_builder_cls=ReplicaCADInteractSceneBuilder,
    )


def make_env() -> Tuple[gym.vector.VectorEnv, bool]:
    if "Habitat" in args.bench_preset:
        config_yaml_name = "interact"
        concur_render, auto_sleep = True, True
        if "NoConcur" in args.bench_preset or "NoOpts" in args.bench_preset:
            concur_render = False
        if "NoSleep" in args.bench_preset or "NoOpts" in args.bench_preset:
            auto_sleep = False

        env = make_habitat_env(config_yaml_name, concur_render, auto_sleep)

        env.reset()
        return env, False

    if args.bench_preset == "MSInteract":
        env = make_interact_env()
    else:
        raise NotImplementedError(args.bench_preset)

    env = ManiSkillVectorEnv(env, auto_reset=False, ignore_terminations=True)
    env.reset(seed=args.seed)
    return env, True


@dataclass
class BenchArgs:
    seed: int
    num_envs: int
    bench_preset: str
    bench_type: str = "interact"

    # not allowed to be changed for now
    result_dir: Path = Path("bench_results")

    def __post_init__(self):
        assert self.bench_preset in [
            "MSInteract",
            "HabitatInteract",
            "HabitatInteractNoConcur",
            "HabitatInteractNoSleep",
            "HabitatInteractNoOpts",
            "Habitat",
            "HabitatNoConcur",
            "HabitatNoSleep",
            "HabitatNoOpts",
            "PickSubtaskTrain",
            "PickSubtaskTrainSingleTaskPlan",
            "RCADSceneManipulation",
            "RCADSceneManipulationSingleBuildConfig",
        ]
        if "Interact" in self.bench_preset:
            assert self.bench_type == "interact"
        assert self.bench_type in ["realistic", "simple", "interact"]
        os.makedirs(self.result_dir, exist_ok=True)


def parse_args(args=None) -> BenchArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed.",
        default=2024,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Num parallel envs.",
        default=1,
    )
    parser.add_argument(
        "--bench-preset",
        type=str,
        help="Benchmark name.",
        default="MSInteract",
    )
    parser.add_argument(
        "--bench-type",
        type=str,
        help="Benchmark type: interact, simple, or realistic.",
        default="interact",
    )
    return BenchArgs(**parser.parse_args(args).__dict__)


def save_stats(stats: Dict[str, Any], name="simple_bench"):
    results: Dict[str, List] = dict()
    all_results_fp = args.result_dir / f"{name}.json"
    if Path(all_results_fp).exists():
        with open(all_results_fp, "r") as f:
            all_results = json.load(f)
    else:
        all_results = dict()

    bench_preset_results: Dict = all_results.get(args.bench_preset, dict())
    results = bench_preset_results.get(str(args.num_envs), dict())

    for k, v in stats.items():
        if k not in results:
            results[k] = []
        results[k].append(v)

    bench_preset_results[str(args.num_envs)] = results
    all_results[args.bench_preset] = bench_preset_results

    def recurisve_noindent_lists(x):
        if isinstance(x, dict):
            return dict((k, recurisve_noindent_lists(v)) for k, v in x.items())
        if isinstance(x, list):
            return NoIndent(x)
        return x

    all_results = recurisve_noindent_lists(all_results)

    io_utils.dump_json(
        all_results_fp,
        all_results,
        encoder_cls=NoIndentSupportingJSONEncoder,
        indent=2,
    )
    return json.dumps(
        all_results[args.bench_preset][str(args.num_envs)],
        cls=NoIndentSupportingJSONEncoder,
        indent=2,
    )


def interact_bench():
    envs, use_torch = make_env()

    # we use timer to get the SPS for specific function calls
    # we use profiler to get SPS and resource usage for a block of code
    timer = NonOverlappingTimeProfiler()
    profiler = Profiler()

    preloaded_actions_fp = (
        ASSET_DIR
        / "scene_datasets/replica_cad_dataset/rearrange/hab2_bench_assets/interact_bench_actions.txt"
    )
    with open(preloaded_actions_fp, "rb") as f:
        preloaded_actions = np.load(f)
    if use_torch:
        preloaded_actions = torch.from_numpy(preloaded_actions).to(
            envs.unwrapped.device
        )

    print("=" * 100)
    print(f"interact_bench {args.seed=} {args.num_envs=} horizon=200 total_steps=200")

    timer.end("other")
    envs.reset()
    timer.end("reset")

    with profiler.profile(
        name="interact_bench", total_steps=200, num_envs=args.num_envs
    ):
        for act in preloaded_actions:
            if "Habitat" in args.bench_preset:
                action = np.full(envs.action_space.shape, -1, dtype=np.float32)

                # no magical grasp
                action[..., :-1] = act[:-1]

                # arm actions
                # Habitat RL env throws errors if actions not clipped to [-1, 1]
                # we clip in MS3 as well, so the actions are the same regardless
                action = np.clip(action, -1, 1)
            else:
                action = torch.zeros(
                    envs.action_space.shape, device=envs.unwrapped.device
                )

                # no torso movement
                action[..., -3] = 0.15

                # arm actions
                action[..., : act.numel() - 1] = act[:-1]

                # ms3 envs clip action for us

            timer.end("other")
            obs = envs.step(action)
            timer.end("step")

    timer_stats = timer.read()
    profiler_stats = profiler.stats["interact_bench"]
    results_str = save_stats(
        dict(
            seed=args.seed,
            num_envs=args.num_envs,
            reset_and_step_SPS=(
                (len(preloaded_actions) * args.num_envs)
                / (timer_stats["reset"] + timer_stats["step"])
            ),
            step_SPS=(len(preloaded_actions) * args.num_envs) / timer_stats["step"],
            **profiler_stats,
            cpu_mem_use_GB=profiler_stats["cpu_mem_use"] / (10**9),
            gpu_mem_use_GB=profiler_stats["gpu_mem_use"] / (10**9),
        ),
        "interact_bench",
    )
    print("=" * 100)
    print(results_str)
    print("=" * 100)

    envs.close()


if __name__ == "__main__":
    args = parse_args()
    if args.bench_type == "interact":
        interact_bench()
    else:
        raise NotImplementedError(args.bench_type)
