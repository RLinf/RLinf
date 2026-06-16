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

"""Smoke test and validation program for the RLinf RoboCasa365 environment.

This follows the "testing and validation" pattern from the RLinf new-env
tutorial: instantiate the registered env class, validate reset output, step
with no-op actions, validate chunk stepping, and verify action preparation.

Usage:

    cd /path/to/RLinf_robocasa
    MUJOCO_GL=egl python toolkits/robocasa365_check/test_robocasa365_env.py

For a faster local sanity run:

    MUJOCO_GL=egl python toolkits/robocasa365_check/test_robocasa365_env.py \
        --num-envs 1 --num-steps 1 --chunk-size 2 --camera-size 128
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class CheckSuite:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  PASS  {name}")
            return
        self.failed += 1
        suffix = f"  {detail}" if detail else ""
        print(f"  FAIL  {name}{suffix}")

    def require(self, name: str, condition: bool, detail: str = "") -> None:
        self.check(name, condition, detail)
        if not condition:
            raise AssertionError(f"{name} failed. {detail}")

    def summary(self) -> None:
        total = self.passed + self.failed
        print(f"\nResults: {self.passed}/{total} passed, {self.failed}/{total} failed")
        if self.failed:
            raise SystemExit(1)


def _shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    return None


def _dtype(value: Any) -> str:
    return str(getattr(value, "dtype", type(value).__name__))


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_cfg(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config)

    cfg.seed = args.seed
    cfg.total_num_envs = args.num_envs
    cfg.group_size = args.group_size
    cfg.split = args.split
    cfg.task_soup = args.task_soup
    cfg.task_sampling_strategy = args.task_sampling_strategy
    cfg.auto_reset = args.auto_reset
    cfg.ignore_terminations = args.ignore_terminations
    cfg.max_episode_steps = args.max_episode_steps
    cfg.max_steps_per_rollout_epoch = args.max_episode_steps
    cfg.init_params.camera_heights = args.camera_size
    cfg.init_params.camera_widths = args.camera_size

    if args.task_filter:
        cfg.task_filter = {"include": args.task_filter, "exclude": []}

    cfg.debug_env_init.enabled = args.debug_env_init
    cfg.debug_env_init.log_dir = args.debug_log_dir
    return cfg


def _make_noop_actions(num_envs: int, action_dim: int) -> np.ndarray:
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)
    if action_dim >= 7:
        actions[:, 6] = -1.0
    if action_dim >= 12:
        actions[:, 11] = -1.0
    return actions


def _validate_obs(
    suite: CheckSuite,
    obs: dict[str, Any],
    *,
    num_envs: int,
    camera_size: int,
    expected_state_dim: int,
    label: str,
) -> None:
    print(f"\n[{label}] Observation keys: {sorted(obs.keys())}")

    suite.require(f"{label}: obs is dict", isinstance(obs, dict))
    for key in ["main_images", "states", "task_descriptions", "task_metadata"]:
        suite.require(f"{label}: has {key}", key in obs, f"keys={sorted(obs.keys())}")

    main_images = obs["main_images"]
    states = obs["states"]
    suite.check(
        f"{label}: main_images shape",
        _shape(main_images) == (num_envs, camera_size, camera_size, 3),
        f"got shape={_shape(main_images)}, dtype={_dtype(main_images)}",
    )
    suite.check(
        f"{label}: main_images uint8",
        str(getattr(main_images, "dtype", "")) in {"torch.uint8", "uint8"},
        f"got dtype={_dtype(main_images)}",
    )
    suite.check(
        f"{label}: states shape",
        _shape(states) == (num_envs, expected_state_dim),
        f"got shape={_shape(states)}, dtype={_dtype(states)}",
    )
    suite.check(
        f"{label}: states finite",
        np.isfinite(_as_numpy(states)).all(),
        "state vector contains NaN or Inf",
    )
    suite.check(
        f"{label}: task_descriptions length",
        len(obs["task_descriptions"]) == num_envs,
        f"got {len(obs['task_descriptions'])}",
    )
    suite.check(
        f"{label}: task_descriptions non-empty",
        all(isinstance(text, str) and text for text in obs["task_descriptions"]),
        f"got {obs['task_descriptions']}",
    )

    wrist_images = obs.get("wrist_images")
    suite.check(
        f"{label}: wrist_images shape",
        _shape(wrist_images) == (num_envs, camera_size, camera_size, 3),
        f"got shape={_shape(wrist_images)}, dtype={_dtype(wrist_images)}",
    )

    extra_view_images = obs.get("extra_view_images")
    suite.check(
        f"{label}: extra_view_images shape",
        _shape(extra_view_images) == (num_envs, 1, camera_size, camera_size, 3),
        f"got shape={_shape(extra_view_images)}, dtype={_dtype(extra_view_images)}",
    )


def _validate_step_output(
    suite: CheckSuite,
    output: tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]],
    *,
    num_envs: int,
    camera_size: int,
    expected_state_dim: int,
    label: str,
) -> dict[str, Any]:
    obs, rewards, terminations, truncations, infos = output
    _validate_obs(
        suite,
        obs,
        num_envs=num_envs,
        camera_size=camera_size,
        expected_state_dim=expected_state_dim,
        label=f"{label}: obs",
    )
    suite.check(
        f"{label}: rewards shape",
        _shape(rewards) == (num_envs,),
        f"got shape={_shape(rewards)}, dtype={_dtype(rewards)}",
    )
    suite.check(
        f"{label}: terminations shape",
        _shape(terminations) == (num_envs,),
        f"got shape={_shape(terminations)}, dtype={_dtype(terminations)}",
    )
    suite.check(
        f"{label}: truncations shape",
        _shape(truncations) == (num_envs,),
        f"got shape={_shape(truncations)}, dtype={_dtype(truncations)}",
    )
    suite.check(f"{label}: infos is dict", isinstance(infos, dict))
    suite.check(
        f"{label}: infos has episode metrics",
        isinstance(infos, dict) and "episode" in infos,
        f"keys={sorted(infos.keys()) if isinstance(infos, dict) else type(infos)}",
    )
    return obs


def _validate_action_prepare(
    suite: CheckSuite,
    cfg: Any,
    *,
    num_envs: int,
    chunk_size: int,
    action_dim: int,
) -> None:
    from rlinf.envs.action_utils import prepare_actions

    raw_actions = np.zeros((num_envs, chunk_size, action_dim), dtype=np.float32)
    prepared = prepare_actions(
        raw_actions,
        env_type="robocasa365",
        model_type="openpi",
        num_action_chunks=chunk_size,
        action_dim=action_dim,
        env_cfg=cfg,
    )
    suite.check(
        "prepare_actions: shape",
        _shape(prepared) == (num_envs, chunk_size, action_dim),
        f"got shape={_shape(prepared)}",
    )
    suite.check(
        "prepare_actions: dtype",
        np.asarray(prepared).dtype == np.float32,
        f"got dtype={np.asarray(prepared).dtype}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RLinf RoboCasa365 reset/step/chunk_step behavior."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=REPO_ROOT / "examples/embodiment/config/env/robocasa365.yaml",
        help="RoboCasa365 env yaml.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument("--task-soup", type=str, default="atomic_seen")
    parser.add_argument("--task-filter", action="append", default=[])
    parser.add_argument(
        "--task-sampling-strategy",
        type=str,
        default="ordered",
        choices=["ordered", "random"],
    )
    parser.add_argument("--camera-size", type=int, default=224)
    parser.add_argument("--max-episode-steps", type=int, default=20)
    parser.add_argument("--auto-reset", action="store_true")
    parser.add_argument("--ignore-terminations", action="store_true")
    parser.add_argument("--debug-env-init", action="store_true")
    parser.add_argument(
        "--debug-log-dir",
        type=str,
        default="../results/robocasa365_env_debug",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default="egl",
        help="Default MUJOCO_GL value if the environment variable is unset.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    cfg = _load_cfg(args)
    action_dim = int(cfg.action_space.env_action_dim)
    expected_state_dim = sum(
        int(part.split(":", 1)[1]) if str(part).startswith("zeros:") else {
            "eef_pos": 3,
            "eef_quat": 4,
            "gripper_qpos": 2,
            "gripper_qvel": 2,
            "base_to_eef_pos": 3,
            "base_to_eef_quat": 4,
            "base_pos": 3,
            "base_quat": 4,
        }[str(part)]
        for part in cfg.observation.state_layout
    )

    print("RoboCasa365 validation config:")
    print(f"  config: {args.config}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  task_soup: {cfg.task_soup}")
    print(f"  split: {cfg.split}")
    print(f"  camera_size: {args.camera_size}")
    print(f"  action_dim: {action_dim}")
    print(f"  state_dim: {expected_state_dim}")
    print(f"  MUJOCO_GL: {os.environ.get('MUJOCO_GL')}")

    suite = CheckSuite()
    env = None
    try:
        from rlinf.envs import get_env_cls
        from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env

        env_cls = get_env_cls("robocasa365", cfg)
        suite.require(
            "env registry returns Robocasa365Env",
            env_cls is Robocasa365Env,
            f"got {env_cls}",
        )

        print("\n[1] Instantiating RoboCasa365Env ...")
        env = env_cls(
            cfg=cfg,
            num_envs=args.num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        suite.check("num_envs set", env.num_envs == args.num_envs)
        suite.check("task_specs non-empty", len(env.task_specs) > 0)
        print(f"  selected tasks: {[spec['task_name'] for spec in env.task_specs[:5]]}")

        print("\n[2] reset() ...")
        reset_obs, reset_infos = env.reset()
        suite.check("reset infos is dict", isinstance(reset_infos, dict))
        _validate_obs(
            suite,
            reset_obs,
            num_envs=args.num_envs,
            camera_size=args.camera_size,
            expected_state_dim=expected_state_dim,
            label="reset",
        )

        actions = _make_noop_actions(args.num_envs, action_dim)
        print(f"\n[3] step() with no-op action shape={actions.shape} ...")
        for step_idx in range(args.num_steps):
            output = env.step(actions, auto_reset=False)
            _validate_step_output(
                suite,
                output,
                num_envs=args.num_envs,
                camera_size=args.camera_size,
                expected_state_dim=expected_state_dim,
                label=f"step[{step_idx}]",
            )

        print("\n[4] chunk_step() ...")
        chunk_actions = np.repeat(actions[:, None, :], args.chunk_size, axis=1)
        obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
            chunk_actions
        )
        suite.check("chunk_step obs_list length", len(obs_list) == args.chunk_size)
        suite.check("chunk_step infos_list length", len(infos_list) == args.chunk_size)
        suite.check(
            "chunk_step rewards shape",
            _shape(rewards) == (args.num_envs, args.chunk_size),
            f"got shape={_shape(rewards)}",
        )
        suite.check(
            "chunk_step terminations shape",
            _shape(terminations) == (args.num_envs, args.chunk_size),
            f"got shape={_shape(terminations)}",
        )
        suite.check(
            "chunk_step truncations shape",
            _shape(truncations) == (args.num_envs, args.chunk_size),
            f"got shape={_shape(truncations)}",
        )
        if obs_list:
            _validate_obs(
                suite,
                obs_list[-1],
                num_envs=args.num_envs,
                camera_size=args.camera_size,
                expected_state_dim=expected_state_dim,
                label="chunk_step[-1]",
            )

        print("\n[5] prepare_actions() ...")
        _validate_action_prepare(
            suite,
            cfg,
            num_envs=args.num_envs,
            chunk_size=args.chunk_size,
            action_dim=action_dim,
        )

    finally:
        if env is not None:
            print("\n[cleanup] Closing environment ...")
            env.close()

    suite.summary()


if __name__ == "__main__":
    main()
