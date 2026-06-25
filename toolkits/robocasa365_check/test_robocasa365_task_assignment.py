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
import yaml

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


from rlinf.envs.robocasa365.eval_schedule import (  # noqa: E402
    resolve_robocasa365_episode_horizons,
    resolve_robocasa365_eval_schedule,
    resolve_robocasa365_rollout_budget,
    validate_robocasa365_eval_horizons,
)
from rlinf.envs.robocasa365.robocasa365_env import (  # noqa: E402
    Robocasa365Env,
    _task_matches_filter,
    configure_robocasa365_eval_horizon,
)


class CheckSuite:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.warned = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  通过  {name}")
            return
        self.failed += 1
        suffix = f"  {detail}" if detail else ""
        print(f"  失败  {name}{suffix}")

    def warn(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  通过  {name}")
            return
        self.warned += 1
        suffix = f"  {detail}" if detail else ""
        print(f"  警告  {name}{suffix}")

    def note(self, message: str) -> None:
        print(f"  说明  {message}")

    def summary(self) -> None:
        total = self.passed + self.failed + self.warned
        print(
            f"\n结果：{self.passed}/{total} 通过，"
            f"{self.warned} 个警告，{self.failed} 个失败"
        )
        if self.failed:
            raise SystemExit(1)


class FakeVectorEnv:
    """Small vector-env stub used to exercise wrapper reset/step logic."""

    def __init__(self, owner: "ProbeRobocasa365Env") -> None:
        self.owner = owner
        self.workers: list[Any] = []
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

    def reset(
        self, id: Any
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
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
        mock_task_horizons: list[int] | None = None,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        self._mock_num_tasks = mock_num_tasks
        self._mock_task_horizons = mock_task_horizons
        super().__init__(
            cfg,
            num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=None,
        )

    def _load_task_specs(self) -> list[dict[str, Any]]:
        horizons = self._mock_task_horizons or [
            int(self.cfg.get("max_episode_steps", 8))
        ] * self._mock_num_tasks
        if len(horizons) != self._mock_num_tasks:
            raise ValueError(
                "mock_task_horizons length must equal mock_num_tasks, got "
                f"{len(horizons)} and {self._mock_num_tasks}."
            )
        task_specs = [
            {
                "task_name": f"MockTask{i:02d}",
                "task_description": f"mock task {i:02d}",
                "task_mode": "atomic",
                "benchmark_selection": "mock",
                "horizon": int(horizons[i]),
                "metadata_view": {
                    "task_name": f"MockTask{i:02d}",
                    "task_id": i,
                    "task_description": f"mock task {i:02d}",
                },
            }
            for i in range(self._mock_num_tasks)
        ]
        task_specs = [
            task_spec
            for task_spec in task_specs
            if self._task_mode_matches(task_spec)
            and _task_matches_filter(task_spec, self.task_filter)
        ]
        if not task_specs:
            raise ValueError(
                f"No mock tasks selected by task_filter={self.task_filter}."
            )
        return task_specs

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
    seed_offset: int = 0,
    seed_strategy: str | None = None,
    task_filter: Any = None,
    mock_num_tasks: int | None = None,
    mock_task_horizons: list[int] | None = None,
    episode_horizon_source: str = "task_horizon",
    max_episode_steps: int = 8,
    max_steps_per_rollout_epoch: int | None = None,
) -> ProbeRobocasa365Env:
    cfg = _load_cfg(args, num_envs=num_envs, group_size=group_size)
    cfg.seed = args.seed if seed is None else seed
    cfg.task_sampling_strategy = strategy
    cfg.use_ordered_reset_state_ids = ordered_flag
    cfg.use_fixed_reset_state_ids = fixed_flag
    cfg.rotate_tasks_on_auto_reset = rotate_on_auto_reset
    cfg.is_eval = is_eval
    cfg.episode_horizon_source = episode_horizon_source
    cfg.max_episode_steps = max_episode_steps
    cfg.max_steps_per_rollout_epoch = (
        max_episode_steps
        if max_steps_per_rollout_epoch is None
        else max_steps_per_rollout_epoch
    )
    if seed_strategy is not None:
        cfg.seed_strategy = seed_strategy
    if task_filter is not None:
        cfg.task_filter = task_filter
    return ProbeRobocasa365Env(
        cfg,
        num_envs,
        mock_num_tasks=(args.num_tasks if mock_num_tasks is None else mock_num_tasks),
        mock_task_horizons=mock_task_horizons,
        seed_offset=seed_offset,
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
    group_ids = np.arange(start, start + num_groups, dtype=np.int32) % num_tasks
    return np.repeat(group_ids, group_size).astype(np.int32)


def _noop_actions(num_envs: int) -> np.ndarray:
    return np.zeros((num_envs, 12), dtype=np.float32)


def _run_success_step(env: ProbeRobocasa365Env, success_mask: np.ndarray) -> np.ndarray:
    env.reset()
    env.env.next_success_mask = np.asarray(success_mask, dtype=bool)
    env.step(_noop_actions(env.num_envs), auto_reset=True)
    return env.task_ids.copy()


def _counts(task_ids: np.ndarray, num_tasks: int) -> np.ndarray:
    return np.bincount(task_ids.astype(np.int64), minlength=num_tasks)


def _task_names(env: ProbeRobocasa365Env) -> list[str]:
    return [
        env.task_specs[int(task_id)]["task_name"]
        for task_id in env.task_ids.astype(np.int64)
    ]


def _groups_are_uniform(task_ids: np.ndarray, group_size: int) -> bool:
    grouped = task_ids.reshape(-1, group_size)
    return bool(np.all(grouped == grouped[:, :1]))


def _group_task_ids(task_ids: np.ndarray, group_size: int) -> np.ndarray:
    return task_ids.reshape(-1, group_size)[:, 0]


def _check_ordered_and_group_assignment(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[顺序任务分配]")
    env = _make_env(args, num_envs=6, strategy="ordered")
    expected = _expected_ordered_ids(6, 1, args.num_tasks)
    suite.check(
        "task_sampling_strategy=ordered 时按顺序给 env 分配任务",
        np.array_equal(env.task_ids, expected),
        f"实际={env.task_ids.tolist()}，期望={expected.tolist()}",
    )

    grouped_env = _make_env(args, num_envs=8, group_size=2, strategy="ordered")
    grouped_expected = _expected_ordered_ids(8, 2, args.num_tasks)
    suite.check(
        "顺序分配的单位是 group，不是单个 env",
        np.array_equal(grouped_env.task_ids, grouped_expected),
        f"实际={grouped_env.task_ids.tolist()}，期望={grouped_expected.tolist()}",
    )

    flag_env = _make_env(
        args,
        num_envs=6,
        strategy="random",
        ordered_flag=True,
    )
    suite.check(
        "use_ordered_reset_state_ids 会覆盖 random 采样并强制顺序分配",
        np.array_equal(flag_env.task_ids, expected),
        f"实际={flag_env.task_ids.tolist()}，期望={expected.tolist()}",
    )

    eval_env = _make_env(args, num_envs=6, strategy="random", is_eval=True)
    suite.check(
        "is_eval=True 会强制顺序采样",
        np.array_equal(eval_env.task_ids, expected),
        f"实际={eval_env.task_ids.tolist()}，期望={expected.tolist()}",
    )


def _check_random_assignment(suite: CheckSuite, args: argparse.Namespace) -> None:
    print("\n[随机任务分配]")
    env_a = _make_env(args, num_envs=12, strategy="random", seed=123)
    env_b = _make_env(args, num_envs=12, strategy="random", seed=123)
    env_c = _make_env(args, num_envs=12, strategy="random", seed=124)
    ordered = _expected_ordered_ids(12, 1, args.num_tasks)

    suite.check(
        "random 采样在相同 seed 下可复现",
        np.array_equal(env_a.task_ids, env_b.task_ids),
        f"a={env_a.task_ids.tolist()}, b={env_b.task_ids.tolist()}",
    )
    suite.check(
        "random 采样得到的 task id 在合法范围内",
        bool(np.all((0 <= env_a.task_ids) & (env_a.task_ids < args.num_tasks))),
        f"task_ids={env_a.task_ids.tolist()}, num_tasks={args.num_tasks}",
    )
    suite.warn(
        "不同 seed 通常会产生不同的随机分配",
        not np.array_equal(env_a.task_ids, env_c.task_ids),
        f"seed123={env_a.task_ids.tolist()}, seed124={env_c.task_ids.tolist()}",
    )
    suite.warn(
        "random 分配通常不应刚好等于 ordered 分配",
        not np.array_equal(env_a.task_ids, ordered),
        f"random={env_a.task_ids.tolist()}, ordered={ordered.tolist()}",
    )


def _check_auto_reset_rotation(suite: CheckSuite, args: argparse.Namespace) -> None:
    print("\n[auto reset 任务轮换]")
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
        "group_size=1 且 rotate_tasks_on_auto_reset=True 时只轮换 done 的 env",
        np.array_equal(after, expected),
        f"reset 前={before.tolist()}，reset 后={after.tolist()}，期望={expected.tolist()}",
    )
    suite.check(
        "只对发生任务轮换的 env 请求重建环境",
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
        "rotate_tasks_on_auto_reset=False 时 done env 的任务不变",
        np.array_equal(no_rotate_after, no_rotate_before),
        f"reset 前={no_rotate_before.tolist()}，reset 后={no_rotate_after.tolist()}",
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
        "group_size>1 时 auto reset 不会触发任务轮换",
        np.array_equal(grouped_after, grouped_before),
        f"reset 前={grouped_before.tolist()}，reset 后={grouped_after.tolist()}",
    )


def _check_single_task_filter_reset_behavior(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[单任务 task_filter 的 reset 行为]")
    task_filter = {"include": ["MockTask03"], "exclude": []}
    scenarios = [
        ("ordered+rotate", "ordered", False, True),
        ("random+rotate", "random", False, True),
        ("random+ordered_flag+rotate", "random", True, True),
        ("ordered+no_rotate", "ordered", False, False),
    ]

    for num_envs in args.env_counts:
        for label, strategy, ordered_flag, rotate in scenarios:
            env = _make_env(
                args,
                num_envs=num_envs,
                strategy=strategy,
                ordered_flag=ordered_flag,
                rotate_on_auto_reset=rotate,
                task_filter=task_filter,
            )
            before_names = _task_names(env)
            after = _run_success_step(env, np.ones(num_envs, dtype=bool))
            after_names = _task_names(env)
            expected_names = ["MockTask03"] * num_envs
            suite.check(
                f"{num_envs} 个 env，{label}：task_filter 只选中一个任务",
                len(env.task_specs) == 1
                and env.task_specs[0]["task_name"] == "MockTask03",
                f"选中任务={[task['task_name'] for task in env.task_specs]}",
            )
            suite.check(
                f"{num_envs} 个 env，{label}：reset 不能改变被过滤出的单任务",
                before_names == expected_names and after_names == expected_names,
                (
                    f"reset 前任务名={before_names}，reset 后任务名={after_names}，"
                    f"after_task_ids={after.tolist()}"
                ),
            )


def _check_unfiltered_reset_parameter_matrix(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[未开启 task_filter 的 reset 参数矩阵]")
    for num_envs in args.env_counts:
        ordered_rotate = _make_env(
            args,
            num_envs=num_envs,
            strategy="ordered",
            rotate_on_auto_reset=True,
        )
        ordered_before = ordered_rotate.task_ids.copy()
        ordered_after = _run_success_step(ordered_rotate, np.ones(num_envs, dtype=bool))
        ordered_expected_after = _expected_ordered_ids(
            num_envs,
            1,
            args.num_tasks,
            start=num_envs % args.num_tasks,
        )
        suite.check(
            f"{num_envs} 个 env，ordered+rotate：全量 env reset 会推进顺序游标",
            np.array_equal(ordered_after, ordered_expected_after),
            (
                f"reset 前={ordered_before.tolist()}，reset 后={ordered_after.tolist()}，"
                f"期望={ordered_expected_after.tolist()}"
            ),
        )

        ordered_no_rotate = _make_env(
            args,
            num_envs=num_envs,
            strategy="ordered",
            rotate_on_auto_reset=False,
        )
        no_rotate_before = ordered_no_rotate.task_ids.copy()
        no_rotate_after = _run_success_step(
            ordered_no_rotate, np.ones(num_envs, dtype=bool)
        )
        suite.check(
            f"{num_envs} 个 env，ordered+no_rotate：全量 env reset 保持任务不变",
            np.array_equal(no_rotate_after, no_rotate_before),
            (
                f"reset 前={no_rotate_before.tolist()}，"
                f"reset 后={no_rotate_after.tolist()}"
            ),
        )

        ordered_flag_env = _make_env(
            args,
            num_envs=num_envs,
            strategy="random",
            ordered_flag=True,
            rotate_on_auto_reset=True,
        )
        ordered_flag_before = ordered_flag_env.task_ids.copy()
        ordered_flag_after = _run_success_step(
            ordered_flag_env, np.ones(num_envs, dtype=bool)
        )
        suite.check(
            (
                f"{num_envs} 个 env，random+use_ordered_reset_state_ids："
                "reset 行为等价于 ordered"
            ),
            np.array_equal(
                ordered_flag_before, _expected_ordered_ids(num_envs, 1, args.num_tasks)
            )
            and np.array_equal(ordered_flag_after, ordered_expected_after),
            (
                f"reset 前={ordered_flag_before.tolist()}，"
                f"reset 后={ordered_flag_after.tolist()}，"
                f"期望 reset 后={ordered_expected_after.tolist()}"
            ),
        )

        random_rotate = _make_env(
            args,
            num_envs=num_envs,
            strategy="random",
            rotate_on_auto_reset=True,
            seed=777 + num_envs,
        )
        random_before = random_rotate.task_ids.copy()
        random_after = _run_success_step(random_rotate, np.ones(num_envs, dtype=bool))
        in_range = bool(np.all((0 <= random_after) & (random_after < args.num_tasks)))
        suite.check(
            f"{num_envs} 个 env，random+rotate：reset 采样到合法 task id",
            in_range,
            f"reset 前={random_before.tolist()}，reset 后={random_after.tolist()}",
        )
        suite.warn(
            f"{num_envs} 个 env，random+rotate：reset 通常会改变一部分任务",
            bool(np.any(random_before != random_after)),
            f"reset 前={random_before.tolist()}，reset 后={random_after.tolist()}",
        )


def _check_fixed_reset_state_ids(suite: CheckSuite, args: argparse.Namespace) -> None:
    print("\n[固定 reset state id]")
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
        "当前实现中，即使 use_fixed_reset_state_ids=True，"
        "update_reset_state_ids 仍会改变 task_ids："
        f"更新前={before.tolist()}，更新后={after_update.tolist()}。"
    )
    if args.strict_fixed_reset:
        suite.check("use_fixed_reset_state_ids 应保持 task id 固定", condition, detail)
    else:
        suite.warn("use_fixed_reset_state_ids 应保持 task id 固定", condition, detail)


def _check_env_count_distribution(suite: CheckSuite, args: argparse.Namespace) -> None:
    print("\n[不同 env 数量下的任务分布]")
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
            f"{num_envs} 个 env：初始 ordered 分配是确定的",
            np.array_equal(before, expected_before),
            f"实际={before.tolist()}，期望={expected_before.tolist()}",
        )
        suite.check(
            f"{num_envs} 个 env：初始任务分布是均衡的",
            int(counts_before.max() - counts_before.min()) <= 1,
            f"counts={counts_before.tolist()}",
        )
        suite.check(
            f"{num_envs} 个 env：初始分配没有坍缩到同一个任务",
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
            f"{num_envs} 个 env：全量 env auto reset 符合 ordered 游标推进",
            np.array_equal(after, expected_after),
            f"reset 后={after.tolist()}，期望={expected_after.tolist()}",
        )
        suite.check(
            f"{num_envs} 个 env：reset 后任务分布仍然均衡",
            int(counts_after.max() - counts_after.min()) <= 1,
            f"counts={counts_after.tolist()}",
        )
        same_count = int(np.sum(before == after))
        suite.note(
            f"{num_envs} 个 env：同步全量 reset 后，"
            f"{same_count}/{num_envs} 个 env 保持同一个任务"
        )


def _check_ppo_grpo_group_semantics(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[PPO/GRPO 的 group 语义]")
    ppo_env = _make_env(
        args,
        num_envs=8,
        group_size=1,
        strategy="ordered",
        rotate_on_auto_reset=True,
    )
    suite.check(
        "PPO 风格 group_size=1 时，每个 env 都是独立任务组",
        ppo_env.group_size == 1 and ppo_env.num_group == ppo_env.num_envs,
        f"group_size={ppo_env.group_size}, num_group={ppo_env.num_group}",
    )
    ppo_before = ppo_env.task_ids.copy()
    ppo_after = _run_success_step(
        ppo_env, np.asarray([False, True, False, True, False, False, True, False])
    )
    ppo_expected = ppo_before.copy()
    ppo_expected[[1, 3, 6]] = np.asarray([8, 9, 10], dtype=np.int32) % args.num_tasks
    suite.check(
        "PPO 风格 group_size=1 支持 done env 独立轮换任务",
        np.array_equal(ppo_after, ppo_expected),
        f"reset 前={ppo_before.tolist()}，reset 后={ppo_after.tolist()}",
    )

    grpo_env = _make_env(
        args,
        num_envs=64,
        group_size=8,
        strategy="ordered",
        rotate_on_auto_reset=True,
    )
    grpo_before = grpo_env.task_ids.copy()
    suite.check(
        "GRPO 风格 group_size=8 时，64 个 env 会形成 8 个本地任务组",
        grpo_env.group_size == 8 and grpo_env.num_group == 8,
        f"group_size={grpo_env.group_size}, num_group={grpo_env.num_group}",
    )
    suite.check(
        "GRPO 风格初始分配会保证每个 group 内是同一个任务",
        _groups_are_uniform(grpo_before, grpo_env.group_size),
        f"task_ids={grpo_before.tolist()}",
    )
    suite.check(
        "GRPO 风格初始 group task id 按 group 顺序分配",
        np.array_equal(_group_task_ids(grpo_before, 8), np.arange(8, dtype=np.int32)),
        f"group_task_ids={_group_task_ids(grpo_before, 8).tolist()}",
    )

    grpo_after_auto = _run_success_step(grpo_env, np.ones(64, dtype=bool))
    suite.check(
        "GRPO 风格 auto reset 会保持 group 任务稳定",
        np.array_equal(grpo_after_auto, grpo_before),
        f"reset 前={grpo_before.tolist()}，reset 后={grpo_after_auto.tolist()}",
    )
    grpo_env.update_reset_state_ids()
    grpo_after_update = grpo_env.task_ids.copy()
    grpo_expected_update = _expected_ordered_ids(64, 8, args.num_tasks, start=8)
    suite.check(
        "GRPO 风格 update_reset_state_ids 会按完整 group 轮换任务",
        np.array_equal(grpo_after_update, grpo_expected_update)
        and _groups_are_uniform(grpo_after_update, 8),
        (f"更新后={grpo_after_update.tolist()}，期望={grpo_expected_update.tolist()}"),
    )


def _check_parallel_eval_episode_schedule(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[每个任务 50 个 episode 的并行评估调度]")
    scenarios = [
        ("2 个任务、100 个 env、1 个 epoch", 2, 100, 1),
        ("18 个任务、18 个 env、50 个 epoch", 18, 18, 50),
        ("18 个任务、90 个 env、10 个 epoch", 18, 90, 10),
        ("18 个任务、180 个 env、5 个 epoch", 18, 180, 5),
        ("18 个任务、900 个 env、1 个 epoch", 18, 900, 1),
    ]

    for label, num_tasks, total_num_envs, configured_epochs in scenarios:
        schedule = resolve_robocasa365_eval_schedule(
            num_tasks=num_tasks,
            total_num_envs=total_num_envs,
            eval_rollout_epoch=configured_epochs,
            expected_episodes_per_task=50,
        )
        env = _make_env(
            args,
            num_envs=total_num_envs,
            strategy="ordered",
            fixed_flag=True,
            rotate_on_auto_reset=False,
            is_eval=True,
            mock_num_tasks=num_tasks,
        )
        episode_counts = np.zeros(num_tasks, dtype=np.int32)
        initial_task_ids = env.task_ids.copy()
        for _ in range(schedule.eval_rollout_epoch):
            episode_counts += np.bincount(
                env.task_ids.astype(np.int64),
                minlength=num_tasks,
            )
            env.update_reset_state_ids()

        suite.check(
            f"{label}：保留显式配置的 eval_rollout_epoch",
            schedule.eval_rollout_epoch == configured_epochs,
            f"实际={schedule.eval_rollout_epoch}，配置={configured_epochs}",
        )
        suite.check(
            f"{label}：每轮每个任务获得相同数量的并行 env",
            bool(
                np.all(
                    np.bincount(initial_task_ids, minlength=num_tasks)
                    == schedule.envs_per_task
                )
            ),
            (
                f"每任务 env 数="
                f"{np.bincount(initial_task_ids, minlength=num_tasks).tolist()}"
            ),
        )
        suite.check(
            f"{label}：所有轮次结束后每个任务恰好有 50 个 episode",
            bool(np.all(episode_counts == 50)),
            f"每任务 episode 数={episode_counts.tolist()}",
        )
        suite.check(
            f"{label}：固定任务模式下各 env 的任务不会跨轮次改变",
            np.array_equal(env.task_ids, initial_task_ids),
            (f"初始={initial_task_ids.tolist()}，最终={env.task_ids.tolist()}"),
        )

    filtered_env = _make_env(
        args,
        num_envs=10,
        strategy="ordered",
        fixed_flag=True,
        rotate_on_auto_reset=False,
        is_eval=True,
        task_filter={"include": ["MockTask03"], "exclude": []},
    )
    filtered_schedule = resolve_robocasa365_eval_schedule(
        num_tasks=len(filtered_env.task_specs),
        total_num_envs=filtered_env.num_envs,
        eval_rollout_epoch=5,
        expected_episodes_per_task=50,
    )
    filtered_episode_count = 0
    for _ in range(filtered_schedule.eval_rollout_epoch):
        filtered_episode_count += filtered_env.num_envs
        filtered_env.update_reset_state_ids()
    suite.check(
        "单任务 task_filter、10 个 env、显式 5 个 epoch：完成 50 个 episode",
        filtered_schedule.eval_rollout_epoch == 5
        and filtered_episode_count == 50
        and set(_task_names(filtered_env)) == {"MockTask03"},
        (
            f"轮数={filtered_schedule.eval_rollout_epoch}，"
            f"episode 数={filtered_episode_count}，"
            f"任务={sorted(set(_task_names(filtered_env)))}"
        ),
    )

    for num_processes in (2, 5):
        local_num_envs = 90 // num_processes
        process_envs = [
            _make_env(
                args,
                num_envs=local_num_envs,
                strategy="ordered",
                fixed_flag=True,
                rotate_on_auto_reset=False,
                is_eval=True,
                total_num_processes=num_processes,
                seed_offset=process_id,
                seed_strategy="global_unique",
            )
            for process_id in range(num_processes)
        ]
        global_task_ids = np.concatenate([env.task_ids for env in process_envs])
        global_seeds = [seed for env in process_envs for seed in env.env_seeds]
        suite.check(
            f"90 个 env 分布到 {num_processes} 个进程后仍是每任务 5 个 env",
            bool(np.all(np.bincount(global_task_ids, minlength=args.num_tasks) == 5)),
            (
                f"全局每任务 env 数="
                f"{np.bincount(global_task_ids, minlength=args.num_tasks).tolist()}"
            ),
        )
        suite.check(
            f"{num_processes} 个进程使用 global_unique 时 env seed 全局不重复",
            len(global_seeds) == len(set(global_seeds)),
            f"seed 数={len(global_seeds)}，去重后={len(set(global_seeds))}",
        )

    invalid_cases = [
        (18, 20, 10, 50),
        (18, 72, 10, 50),
        (1, 100, 1, 50),
    ]
    for (
        num_tasks,
        total_num_envs,
        eval_rollout_epoch,
        expected_episodes_per_task,
    ) in invalid_cases:
        try:
            resolve_robocasa365_eval_schedule(
                num_tasks=num_tasks,
                total_num_envs=total_num_envs,
                eval_rollout_epoch=eval_rollout_epoch,
                expected_episodes_per_task=expected_episodes_per_task,
            )
        except ValueError:
            rejected = True
        else:
            rejected = False
        suite.check(
            (
                f"非法调度会被拒绝：tasks={num_tasks}, "
                f"envs={total_num_envs}, epochs={eval_rollout_epoch}, "
                f"expected episodes/task={expected_episodes_per_task}"
            ),
            rejected,
        )


def _first_truncation_steps(
    env: ProbeRobocasa365Env, max_steps: int
) -> np.ndarray:
    env.reset()
    first_steps = np.zeros(env.num_envs, dtype=np.int32)
    seen = np.zeros(env.num_envs, dtype=bool)
    for step in range(1, max_steps + 1):
        _, _, _, truncations, _ = env.step(
            _noop_actions(env.num_envs), auto_reset=False
        )
        current = truncations.cpu().numpy().astype(bool)
        newly_truncated = current & ~seen
        first_steps[newly_truncated] = step
        seen |= current
    return first_steps


def _check_episode_horizon_sources(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[episode horizon 来源]")
    registry_horizons = [3, 5]

    task_env = _make_env(
        args,
        num_envs=4,
        strategy="ordered",
        is_eval=True,
        mock_num_tasks=2,
        mock_task_horizons=registry_horizons,
        episode_horizon_source="task_horizon",
        max_episode_steps=4,
        max_steps_per_rollout_epoch=5,
    )
    suite.check(
        "task_horizon 模式按每个 env 对应任务的 horizon 截断",
        np.array_equal(task_env.task_horizons, np.asarray([3, 5, 3, 5])),
        f"task_horizons={task_env.task_horizons.tolist()}",
    )
    task_first_steps = _first_truncation_steps(task_env, max_steps=5)
    suite.check(
        "task_horizon 模式下多任务失败 episode 在各自 horizon 完成",
        np.array_equal(task_first_steps, np.asarray([3, 5, 3, 5])),
        f"first_truncation_steps={task_first_steps.tolist()}",
    )
    task_env.reset()
    task_env.env.next_success_mask = np.asarray([True, False, False, False])
    _, _, chunk_terminations, _, _ = task_env.chunk_step(
        np.zeros((task_env.num_envs, 2, 12), dtype=np.float32)
    )
    suite.check(
        "eval action chunk 中间成功会折叠到 chunk 最后一步供统计",
        bool(chunk_terminations[0, -1])
        and int(chunk_terminations[0].sum().item()) == 1,
        f"chunk_terminations={chunk_terminations.tolist()}",
    )

    fixed_env = _make_env(
        args,
        num_envs=4,
        strategy="ordered",
        is_eval=True,
        mock_num_tasks=2,
        mock_task_horizons=registry_horizons,
        episode_horizon_source="max_episode_steps",
        max_episode_steps=4,
    )
    suite.check(
        "max_episode_steps 模式为所有任务使用统一 horizon",
        np.array_equal(fixed_env.task_horizons, np.asarray([4, 4, 4, 4])),
        f"task_horizons={fixed_env.task_horizons.tolist()}",
    )
    fixed_first_steps = _first_truncation_steps(fixed_env, max_steps=4)
    suite.check(
        "max_episode_steps 模式下所有失败 episode 在固定长度完成",
        np.array_equal(fixed_first_steps, np.asarray([4, 4, 4, 4])),
        f"first_truncation_steps={fixed_first_steps.tolist()}",
    )

    resolved = resolve_robocasa365_episode_horizons(
        task_horizons=registry_horizons,
        max_episode_steps=4,
        episode_horizon_source="task_horizon",
    )
    suite.check(
        "task_horizon 模式保留 registry horizon",
        resolved == (3, 5),
        f"resolved={resolved}",
    )
    suite.check(
        "共享 rollout budget 自动取最大 horizon 并对齐 action chunk",
        resolve_robocasa365_rollout_budget(
            episode_horizons=resolved,
            num_action_chunks=2,
        )
        == 6,
    )

    auto_cfg = _to_attr_dict(
        {
            "episode_horizon_source": "task_horizon",
            "max_episode_steps": 700,
            "max_steps_per_rollout_epoch": 700,
        }
    )
    original_loader = configure_robocasa365_eval_horizon.__globals__[
        "load_robocasa365_task_specs"
    ]
    configure_robocasa365_eval_horizon.__globals__[
        "load_robocasa365_task_specs"
    ] = lambda cfg: [{"task_name": "MockTask", "horizon": 900}]
    try:
        auto_budget = configure_robocasa365_eval_horizon(
            auto_cfg,
            num_action_chunks=5,
        )
    finally:
        configure_robocasa365_eval_horizon.__globals__[
            "load_robocasa365_task_specs"
        ] = original_loader
    suite.check(
        "task_horizon 模式自动覆盖 YAML 中手写的 rollout budget",
        auto_budget == 900 and auto_cfg.max_steps_per_rollout_epoch == 900,
        (
            f"auto_budget={auto_budget}, "
            f"cfg_budget={auto_cfg.max_steps_per_rollout_epoch}"
        ),
    )
    try:
        validate_robocasa365_eval_horizons(
            episode_horizons=resolved,
            max_steps_per_rollout_epoch=4,
        )
    except ValueError:
        rejected_short_budget = True
    else:
        rejected_short_budget = False
    suite.check(
        "共享 rollout budget 小于最大任务 horizon 时拒绝启动",
        rejected_short_budget,
    )


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 没有被解析为 YAML mapping。")
    return data


def _selected_task_count_from_yaml(task_filter: Any, default_count: int) -> int:
    """Infer the selected task count for lightweight YAML consistency checks."""

    if not task_filter:
        return default_count
    if isinstance(task_filter, str):
        return 1
    if isinstance(task_filter, list):
        return len(set(str(task) for task in task_filter))
    if isinstance(task_filter, dict):
        include = task_filter.get("include", [])
        exclude = task_filter.get("exclude", [])
        include = [include] if isinstance(include, str) else list(include or [])
        exclude = [exclude] if isinstance(exclude, str) else list(exclude or [])
        include_set = {str(task) for task in include}
        exclude_set = {str(task) for task in exclude}
        if include_set:
            return len(include_set - exclude_set)
        return default_count - len(exclude_set)
    raise TypeError(f"Unsupported task_filter type: {type(task_filter)}")


def _check_yaml_group_size_consistency(
    suite: CheckSuite, args: argparse.Namespace
) -> None:
    print("\n[YAML group_size 一致性]")
    for label, path, expect_grpo in [
        ("GRPO", args.grpo_config, True),
        ("PPO/GAE", args.ppo_config, False),
    ]:
        data = _load_yaml(path)
        algorithm = data.get("algorithm", {})
        env = data.get("env", {})
        train_env = env.get("train", {})
        eval_env = env.get("eval", {})
        alg_group_size = int(algorithm.get("group_size", -1))
        train_group_size = int(train_env.get("group_size", -1))
        eval_group_size = int(eval_env.get("group_size", -1))
        adv_type = str(algorithm.get("adv_type", ""))

        if expect_grpo:
            suite.check(
                f"{label} 配置使用 GRPO，且 algorithm.group_size > 1",
                adv_type in {"grpo", "grpo_dynamic", "reinpp_baseline"}
                and alg_group_size > 1,
                f"adv_type={adv_type}, algorithm.group_size={alg_group_size}",
            )
            suite.check(
                f"{label} 配置中 env.train.group_size 与 algorithm.group_size 一致",
                train_group_size == alg_group_size,
                (
                    f"env.train.group_size={train_group_size}, "
                    f"algorithm.group_size={alg_group_size}"
                ),
            )
            suite.check(
                f"{label} 配置中 eval 保持 group_size=1",
                eval_group_size == 1,
                f"env.eval.group_size={eval_group_size}",
            )
        else:
            suite.check(
                f"{label} 配置使用非 GRPO，且 group_size=1",
                adv_type not in {"grpo", "grpo_dynamic", "reinpp_baseline"}
                and alg_group_size == 1
                and train_group_size == 1,
                (
                    f"adv_type={adv_type}, algorithm.group_size={alg_group_size}, "
                    f"env.train.group_size={train_group_size}"
                ),
            )
            parallel_eval = eval_env.get("parallel_eval", {})
            if isinstance(parallel_eval, bool):
                parallel_enabled = parallel_eval
                episodes_per_task = 50
            else:
                parallel_enabled = bool(parallel_eval.get("enabled", False))
                episodes_per_task = int(parallel_eval.get("episodes_per_task", 50))

            if parallel_enabled:
                selected_task_count = _selected_task_count_from_yaml(
                    eval_env.get("task_filter", []),
                    args.num_tasks,
                )
                schedule = resolve_robocasa365_eval_schedule(
                    num_tasks=selected_task_count,
                    total_num_envs=int(eval_env.get("total_num_envs", 0)),
                    eval_rollout_epoch=int(algorithm.get("eval_rollout_epoch", 0)),
                    expected_episodes_per_task=episodes_per_task,
                )
                suite.check(
                    f"{label} 配置开启每任务 {episodes_per_task} 个 episode 的并行评估",
                    schedule.episodes_per_task == episodes_per_task,
                    (
                        f"parallel_eval={parallel_eval}, "
                        f"selected_task_count={selected_task_count}"
                    ),
                )
                suite.check(
                    f"{label} 配置显式保留 eval_rollout_epoch",
                    schedule.eval_rollout_epoch
                    == int(algorithm.get("eval_rollout_epoch", -1)),
                    (
                        f"调度轮数={schedule.eval_rollout_epoch}，"
                        f"配置轮数={algorithm.get('eval_rollout_epoch')}"
                    ),
                )
            else:
                suite.note(
                    f"{label} 配置未开启 parallel_eval，跳过每任务 episode 调度检查"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 RoboCasa365 的任务采样、reset 轮换和 PPO/GRPO group 语义。"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=REPO_ROOT
        / "examples"
        / "embodiment"
        / "config"
        / "env"
        / "robocasa365.yaml",
        help="基础 RoboCasa365 env 配置。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=18,
        help="mock 任务数量。常见 debug 设置中 atomic_seen 有 18 个任务。",
    )
    parser.add_argument(
        "--env-counts",
        type=int,
        nargs="+",
        default=[17, 18, 19],
        help="用于检查少于、等于、多于 18 个任务时行为的 env 数量。",
    )
    parser.add_argument(
        "--strict-fixed-reset",
        action="store_true",
        help=(
            "如果 use_fixed_reset_state_ids 没有固定 task id，则记为失败；"
            "默认只记为警告。"
        ),
    )
    parser.add_argument(
        "--grpo-config",
        type=pathlib.Path,
        default=REPO_ROOT
        / "examples"
        / "embodiment"
        / "config"
        / "robocasa365_grpo_openpi.yaml",
        help="用于检查 group_size 一致性的 RoboCasa365 GRPO 顶层配置。",
    )
    parser.add_argument(
        "--ppo-config",
        type=pathlib.Path,
        default=REPO_ROOT
        / "examples"
        / "embodiment"
        / "config"
        / "robocasa365_eval_openpi.yaml",
        help="用于检查 group_size 的 RoboCasa365 PPO/GAE 风格顶层配置。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_tasks <= 0:
        raise ValueError(f"--num-tasks 必须为正数，当前为 {args.num_tasks}")
    if any(num_envs <= 0 for num_envs in args.env_counts):
        raise ValueError(f"--env-counts 必须全部为正数，当前为 {args.env_counts}")

    suite = CheckSuite()
    print(f"配置文件：{args.config}")
    print(f"mock 任务数量：{args.num_tasks}")

    _check_ordered_and_group_assignment(suite, args)
    _check_random_assignment(suite, args)
    _check_auto_reset_rotation(suite, args)
    _check_single_task_filter_reset_behavior(suite, args)
    _check_fixed_reset_state_ids(suite, args)
    _check_env_count_distribution(suite, args)
    _check_unfiltered_reset_parameter_matrix(suite, args)
    _check_ppo_grpo_group_semantics(suite, args)
    _check_parallel_eval_episode_schedule(suite, args)
    _check_episode_horizon_sources(suite, args)
    _check_yaml_group_size_consistency(suite, args)
    suite.summary()


if __name__ == "__main__":
    main()
