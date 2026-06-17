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

"""Validate RoboCasa365 task assignment and reset rotation logic.

This is a lightweight companion to ``test_robocasa365_env.py``. It reuses the
real ``Robocasa365Env`` task sampling/reset code but replaces the subprocess
RoboCasa environments with a fake vector env, so it can be run without MuJoCo or
RoboCasa assets.

Usage:

    cd /path/to/RLinf_robocasa
    python toolkits/robocasa365_check/test_robocasa365_task_assignment.py

To make the current ``use_fixed_reset_state_ids`` behavior a hard failure:

    python toolkits/robocasa365_check/test_robocasa365_task_assignment.py \
        --strict-fixed-reset
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import types
from typing import Any

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class AttrDict(dict):
    """Dict with attribute access, used by the optional OmegaConf fallback."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _to_attr_dict(value)


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, AttrDict):
        return AttrDict({key: _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def _default_cfg() -> AttrDict:
    return _to_attr_dict(
        {
            "seed": 0,
            "total_num_envs": None,
            "group_size": 1,
            "task_source": "mock",
            "dataset_source": "human",
            "split": "pretrain",
            "task_soup": ["mock"],
            "task_mode": "atomic",
            "task_filter": [],
            "task_sampling_strategy": "random",
            "rotate_tasks_on_auto_reset": True,
            "auto_reset": True,
            "ignore_terminations": False,
            "max_episode_steps": 8,
            "max_steps_per_rollout_epoch": 8,
            "use_fixed_reset_state_ids": False,
            "use_ordered_reset_state_ids": False,
            "use_rel_reward": False,
            "reward_coef": 1.0,
            "is_eval": False,
            "seed_strategy": "worker_offset",
            "camera_names": [
                "robot0_agentview_left",
                "robot0_eye_in_hand",
                "robot0_agentview_right",
            ],
            "init_params": {
                "camera_heights": 1,
                "camera_widths": 1,
            },
            "debug_env_init": {
                "enabled": False,
                "log_dir": "../results/robocasa365_env_debug",
                "include_full_ep_meta": True,
            },
            "video_cfg": {
                "save_video": False,
                "info_on_video": True,
                "video_base_dir": "../results/video/train",
            },
            "observation": {},
            "action_space": {
                "env_action_dim": 12,
            },
        }
    )


def _install_gym_stub(module_name: str) -> None:
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    class Env:
        pass

    class Space:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            self.dtype = kwargs.get("dtype", np.float32)
            self.shape = kwargs.get("shape", ())

    class Dict(Space):
        pass

    class Tuple(Space):
        pass

    stub = types.ModuleType(module_name)
    stub.Env = Env
    stub.Space = Space
    stub.spaces = types.SimpleNamespace(Dict=Dict, Tuple=Tuple)
    sys.modules[module_name] = stub


try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    class _OmegaConfFallback:
        @staticmethod
        def load(path: Any) -> AttrDict:
            del path
            return _default_cfg()

        @staticmethod
        def is_config(value: Any) -> bool:
            del value
            return False

        @staticmethod
        def to_container(value: Any, resolve: bool = True) -> Any:
            del resolve
            return value

        @staticmethod
        def register_new_resolver(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            return None

    OmegaConf = _OmegaConfFallback
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = omegaconf_stub


_install_gym_stub("gymnasium")


from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env


class CheckSuite:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.warned = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  PASS  {name}")
            return
        self.failed += 1
        suffix = f"  {detail}" if detail else ""
        print(f"  FAIL  {name}{suffix}")

    def warn(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  PASS  {name}")
            return
        self.warned += 1
        suffix = f"  {detail}" if detail else ""
        print(f"  WARN  {name}{suffix}")

    def note(self, message: str) -> None:
        print(f"  NOTE  {message}")

    def summary(self) -> None:
        total = self.passed + self.failed + self.warned
        print(
            f"\nResults: {self.passed}/{total} passed, "
            f"{self.warned} warned, {self.failed} failed"
        )
        if self.failed:
            raise SystemExit(1)


class FakeVectorEnv:
    """Small vector-env stub used to exercise wrapper reset/step logic."""

    def __init__(self, owner: "ProbeRobocasa365Env") -> None:
        self.owner = owner
        self.next_success_mask = np.zeros(owner.num_envs, dtype=bool)
        self.reconfigure_calls: list[dict[str, Any]] = []

    def _obs_for_env(self, env_id: int) -> dict[str, np.ndarray]:
        task_id = int(self.owner.task_ids[env_id])
        return {
            "mock_env_id": np.asarray([env_id], dtype=np.int32),
            "mock_task_id": np.asarray([task_id], dtype=np.int32),
        }

    def _info_for_env(self, env_id: int, success: bool = False) -> dict[str, Any]:
        task_id = int(self.owner.task_ids[env_id])
        task_spec = self.owner.task_specs[task_id]
        return {
            "success": bool(success),
            "mock_env_id": env_id,
            "mock_task_id": task_id,
            "ep_meta": {
                "lang": task_spec["task_description"],
                "task": task_spec["task_name"],
            },
        }

    def reset(self, id: Any) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
        env_ids = np.asarray(id, dtype=np.int32).reshape(-1)
        obs = [self._obs_for_env(int(env_id)) for env_id in env_ids]
        infos = [self._info_for_env(int(env_id)) for env_id in env_ids]
        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        list[dict[str, np.ndarray]], np.ndarray, np.ndarray, list[dict[str, Any]]
    ]:
        del actions
        env_ids = np.arange(self.owner.num_envs, dtype=np.int32)
        success_mask = self.next_success_mask.astype(bool, copy=True)
        self.next_success_mask[:] = False
        obs = [self._obs_for_env(int(env_id)) for env_id in env_ids]
        rewards = success_mask.astype(np.float32)
        dones = success_mask.copy()
        infos = [
            self._info_for_env(int(env_id), bool(success_mask[env_id]))
            for env_id in env_ids
        ]
        return obs, rewards, dones, infos

    def reconfigure_env_fns(self, env_fns: list[dict[str, int]], id: Any) -> None:
        self.reconfigure_calls.append(
            {
                "env_ids": np.asarray(id, dtype=np.int32).tolist(),
                "env_fns": copy.deepcopy(env_fns),
            }
        )

    def close(self) -> None:
        return None


class ProbeRobocasa365Env(Robocasa365Env):
    """Robocasa365Env with real task logic and fake simulator IO."""

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        *,
        mock_num_tasks: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        self._mock_num_tasks = mock_num_tasks
        super().__init__(
            cfg,
            num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=None,
        )

    def _load_task_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "task_name": f"MockTask{i:02d}",
                "task_description": f"mock task {i:02d}",
                "task_mode": "atomic",
                "benchmark_selection": "mock",
                "horizon": int(self.cfg.get("max_episode_steps", 8)),
                "metadata_view": {
                    "task_name": f"MockTask{i:02d}",
                    "task_id": i,
                    "task_description": f"mock task {i:02d}",
                },
            }
            for i in range(self._mock_num_tasks)
        ]

    def _init_env(self) -> None:
        self._refresh_task_context()
        self.env = FakeVectorEnv(self)

    def get_env_fns(self, env_idx: Any = None) -> list[dict[str, int]]:
        if env_idx is None:
            env_ids = np.arange(self.num_envs, dtype=np.int32)
        else:
            env_ids = np.asarray(env_idx, dtype=np.int32).reshape(-1)
        return [
            {
                "env_id": int(env_id),
                "task_id": int(self.task_ids[int(env_id)]),
                "seed": int(self.env_seeds[int(env_id)]),
            }
            for env_id in env_ids
        ]

    def _wrap_obs(
        self, obs_list: list[Any], info_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        del obs_list, info_list
        self._refresh_task_context()
        num_envs = self.num_envs
        return {
            "main_images": torch.zeros((num_envs, 1, 1, 3), dtype=torch.uint8),
            "wrist_images": torch.zeros((num_envs, 1, 1, 3), dtype=torch.uint8),
            "extra_view_images": torch.zeros((num_envs, 1, 1, 1, 3), dtype=torch.uint8),
            "states": torch.as_tensor(self.task_ids[:, None], dtype=torch.float32),
            "task_descriptions": list(self.task_descriptions),
            "task_metadata": copy.deepcopy(self.task_metadata),
        }


def _load_cfg(args: argparse.Namespace, *, num_envs: int, group_size: int) -> Any:
    cfg = OmegaConf.load(args.config)
    cfg.seed = args.seed
    cfg.total_num_envs = num_envs
    cfg.group_size = group_size
    cfg.task_source = "mock"
    cfg.task_soup = ["mock"]
    cfg.task_mode = "atomic"
    cfg.task_filter = []
    cfg.auto_reset = True
    cfg.ignore_terminations = False
    cfg.max_episode_steps = 8
    cfg.max_steps_per_rollout_epoch = 8
    cfg.use_rel_reward = False
    cfg.reward_coef = 1.0
    cfg.init_params.camera_heights = 1
    cfg.init_params.camera_widths = 1
    cfg.debug_env_init.enabled = False
    return cfg


def _make_env(
    args: argparse.Namespace,
    *,
    num_envs: int,
    group_size: int = 1,
    strategy: str = "ordered",
    ordered_flag: bool = False,
    fixed_flag: bool = False,
    rotate_on_auto_reset: bool = True,
    seed: int | None = None,
    is_eval: bool = False,
    total_num_processes: int = 1,
) -> ProbeRobocasa365Env:
    cfg = _load_cfg(args, num_envs=num_envs, group_size=group_size)
    cfg.seed = args.seed if seed is None else seed
    cfg.task_sampling_strategy = strategy
    cfg.use_ordered_reset_state_ids = ordered_flag
    cfg.use_fixed_reset_state_ids = fixed_flag
    cfg.rotate_tasks_on_auto_reset = rotate_on_auto_reset
    cfg.is_eval = is_eval
    return ProbeRobocasa365Env(
        cfg,
        num_envs,
        mock_num_tasks=args.num_tasks,
        total_num_processes=total_num_processes,
    )


def _expected_ordered_ids(
    num_envs: int,
    group_size: int,
    num_tasks: int,
    *,
    start: int = 0,
) -> np.ndarray:
    num_groups = num_envs // group_size
    group_ids = (np.arange(start, start + num_groups, dtype=np.int32) % num_tasks)
    return np.repeat(group_ids, group_size).astype(np.int32)


def _noop_actions(num_envs: int) -> np.ndarray:
    return np.zeros((num_envs, 12), dtype=np.float32)


def _run_success_step(
    env: ProbeRobocasa365Env, success_mask: np.ndarray
) -> np.ndarray:
    env.reset()
    env.env.next_success_mask = np.asarray(success_mask, dtype=bool)
    env.step(_noop_actions(env.num_envs), auto_reset=True)
    return env.task_ids.copy()


def _counts(task_ids: np.ndarray, num_tasks: int) -> np.ndarray:
    return np.bincount(task_ids.astype(np.int64), minlength=num_tasks)


def _check_ordered_and_group_assignment(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[ordered assignment]")
    env = _make_env(args, num_envs=6, strategy="ordered")
    expected = _expected_ordered_ids(6, 1, args.num_tasks)
    suite.check(
        "task_sampling_strategy=ordered assigns envs sequentially",
        np.array_equal(env.task_ids, expected),
        f"got={env.task_ids.tolist()}, expected={expected.tolist()}",
    )

    grouped_env = _make_env(args, num_envs=8, group_size=2, strategy="ordered")
    grouped_expected = _expected_ordered_ids(8, 2, args.num_tasks)
    suite.check(
        "ordered assignment is per group, not per env",
        np.array_equal(grouped_env.task_ids, grouped_expected),
        f"got={grouped_env.task_ids.tolist()}, expected={grouped_expected.tolist()}",
    )

    flag_env = _make_env(
        args,
        num_envs=6,
        strategy="random",
        ordered_flag=True,
    )
    suite.check(
        "use_ordered_reset_state_ids overrides random sampling",
        np.array_equal(flag_env.task_ids, expected),
        f"got={flag_env.task_ids.tolist()}, expected={expected.tolist()}",
    )

    eval_env = _make_env(args, num_envs=6, strategy="random", is_eval=True)
    suite.check(
        "is_eval=True forces ordered sampling",
        np.array_equal(eval_env.task_ids, expected),
        f"got={eval_env.task_ids.tolist()}, expected={expected.tolist()}",
    )


def _check_random_assignment(suite: CheckSuite, args: argparse.Namespace) -> None:
    print("\n[random assignment]")
    env_a = _make_env(args, num_envs=12, strategy="random", seed=123)
    env_b = _make_env(args, num_envs=12, strategy="random", seed=123)
    env_c = _make_env(args, num_envs=12, strategy="random", seed=124)
    ordered = _expected_ordered_ids(12, 1, args.num_tasks)

    suite.check(
        "random sampling is reproducible for the same seed",
        np.array_equal(env_a.task_ids, env_b.task_ids),
        f"a={env_a.task_ids.tolist()}, b={env_b.task_ids.tolist()}",
    )
    suite.check(
        "random sampling keeps task ids in range",
        bool(np.all((0 <= env_a.task_ids) & (env_a.task_ids < args.num_tasks))),
        f"task_ids={env_a.task_ids.tolist()}, num_tasks={args.num_tasks}",
    )
    suite.warn(
        "different seeds usually produce different random assignments",
        not np.array_equal(env_a.task_ids, env_c.task_ids),
        f"seed123={env_a.task_ids.tolist()}, seed124={env_c.task_ids.tolist()}",
    )
    suite.warn(
        "random assignment should not usually equal ordered assignment",
        not np.array_equal(env_a.task_ids, ordered),
        f"random={env_a.task_ids.tolist()}, ordered={ordered.tolist()}",
    )


def _check_auto_reset_rotation(suite: CheckSuite, args: argparse.Namespace) -> None:
    print("\n[auto reset rotation]")
    env = _make_env(
        args,
        num_envs=4,
        strategy="ordered",
        rotate_on_auto_reset=True,
    )
    before = env.task_ids.copy()
    after = _run_success_step(env, np.asarray([False, True, False, True]))
    expected = before.copy()
    expected[[1, 3]] = np.asarray([4, 5], dtype=np.int32) % args.num_tasks
    suite.check(
        "rotate_tasks_on_auto_reset=True rotates only done envs when group_size=1",
        np.array_equal(after, expected),
        f"before={before.tolist()}, after={after.tolist()}, expected={expected.tolist()}",
    )
    suite.check(
        "task reconfiguration is requested for rotated envs only",
        env.env.reconfigure_calls[-1]["env_ids"] == [1, 3],
        f"reconfigure_calls={env.env.reconfigure_calls}",
    )

    no_rotate_env = _make_env(
        args,
        num_envs=4,
        strategy="ordered",
        rotate_on_auto_reset=False,
    )
    no_rotate_before = no_rotate_env.task_ids.copy()
    no_rotate_after = _run_success_step(
        no_rotate_env, np.asarray([False, True, False, True])
    )
    suite.check(
        "rotate_tasks_on_auto_reset=False keeps done env tasks unchanged",
        np.array_equal(no_rotate_after, no_rotate_before),
        f"before={no_rotate_before.tolist()}, after={no_rotate_after.tolist()}",
    )

    grouped_env = _make_env(
        args,
        num_envs=4,
        group_size=2,
        strategy="ordered",
        rotate_on_auto_reset=True,
    )
    grouped_before = grouped_env.task_ids.copy()
    grouped_after = _run_success_step(
        grouped_env, np.asarray([False, True, False, True])
    )
    suite.check(
        "auto-reset rotation is disabled for group_size>1",
        np.array_equal(grouped_after, grouped_before),
        f"before={grouped_before.tolist()}, after={grouped_after.tolist()}",
    )


def _check_fixed_reset_state_ids(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[fixed reset state ids]")
    env = _make_env(
        args,
        num_envs=6,
        strategy="ordered",
        fixed_flag=True,
        rotate_on_auto_reset=True,
    )
    before = env.task_ids.copy()
    env.update_reset_state_ids()
    after_update = env.task_ids.copy()

    condition = np.array_equal(after_update, before)
    detail = (
        "Current implementation changes task_ids on update_reset_state_ids even "
        f"when use_fixed_reset_state_ids=True: before={before.tolist()}, "
        f"after={after_update.tolist()}."
    )
    if args.strict_fixed_reset:
        suite.check("use_fixed_reset_state_ids keeps task ids fixed", condition, detail)
    else:
        suite.warn("use_fixed_reset_state_ids keeps task ids fixed", condition, detail)


def _check_env_count_distribution(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[env count distribution]")
    for num_envs in args.env_counts:
        env = _make_env(
            args,
            num_envs=num_envs,
            strategy="ordered",
            rotate_on_auto_reset=True,
        )
        before = env.task_ids.copy()
        expected_before = _expected_ordered_ids(num_envs, 1, args.num_tasks)
        counts_before = _counts(before, args.num_tasks)
        suite.check(
            f"{num_envs} envs: initial ordered assignment is deterministic",
            np.array_equal(before, expected_before),
            f"got={before.tolist()}, expected={expected_before.tolist()}",
        )
        suite.check(
            f"{num_envs} envs: initial task distribution is balanced",
            int(counts_before.max() - counts_before.min()) <= 1,
            f"counts={counts_before.tolist()}",
        )
        suite.check(
            f"{num_envs} envs: initial assignment is not collapsed to one task",
            len(set(before.tolist())) > 1 or num_envs == 1,
            f"task_ids={before.tolist()}",
        )

        after = _run_success_step(env, np.ones(num_envs, dtype=bool))
        expected_after = _expected_ordered_ids(
            num_envs,
            1,
            args.num_tasks,
            start=num_envs % args.num_tasks,
        )
        counts_after = _counts(after, args.num_tasks)
        suite.check(
            f"{num_envs} envs: all-env auto reset follows ordered cursor",
            np.array_equal(after, expected_after),
            f"after={after.tolist()}, expected={expected_after.tolist()}",
        )
        suite.check(
            f"{num_envs} envs: post-reset task distribution remains balanced",
            int(counts_after.max() - counts_after.min()) <= 1,
            f"counts={counts_after.tolist()}",
        )
        same_count = int(np.sum(before == after))
        suite.note(
            f"{num_envs} envs: {same_count}/{num_envs} envs keep the same task "
            "after a synchronized all-env reset"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RoboCasa365 task sampling and reset rotation logic."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=REPO_ROOT / "examples" / "embodiment" / "config" / "env" / "robocasa365.yaml",
        help="Base RoboCasa365 env config.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=18,
        help="Mock task count. atomic_seen has 18 tasks in the common debug setup.",
    )
    parser.add_argument(
        "--env-counts",
        type=int,
        nargs="+",
        default=[17, 18, 19],
        help="Env counts used to check <18, =18, and >18 task assignment behavior.",
    )
    parser.add_argument(
        "--strict-fixed-reset",
        action="store_true",
        help=(
            "Treat use_fixed_reset_state_ids not keeping task ids fixed as a hard "
            "failure instead of a warning."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_tasks <= 0:
        raise ValueError(f"--num-tasks must be positive, got {args.num_tasks}")
    if any(num_envs <= 0 for num_envs in args.env_counts):
        raise ValueError(f"--env-counts must all be positive, got {args.env_counts}")

    suite = CheckSuite()
    print(f"Config: {args.config}")
    print(f"Mock tasks: {args.num_tasks}")

    _check_ordered_and_group_assignment(suite, args)
    _check_random_assignment(suite, args)
    _check_auto_reset_rotation(suite, args)
    _check_fixed_reset_state_ids(suite, args)
    _check_env_count_distribution(suite, args)
    suite.summary()


if __name__ == "__main__":
    main()
