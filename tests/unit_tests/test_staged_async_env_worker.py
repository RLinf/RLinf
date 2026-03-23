from types import SimpleNamespace

import pytest
import torch

try:
    import ray  # noqa: F401
except Exception:
    import sys
    import types

    ray_stub = types.ModuleType("ray")
    ray_actor_stub = types.ModuleType("ray.actor")
    ray_stub.get = lambda ref: ref
    ray_stub.wait = lambda refs, num_returns=1, timeout=0: (refs[:num_returns], [])
    ray_stub.actor = ray_actor_stub
    ray_actor_stub.ActorHandle = object
    sys.modules.setdefault("ray", ray_stub)
    sys.modules.setdefault("ray.actor", ray_actor_stub)

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker


def test_top_reward_can_use_initial_task_description():
    worker = EnvWorker.__new__(EnvWorker)
    worker._top_reward_instruction_source = "initial_task"
    worker._initial_task_descriptions = ["fold the towel"]
    worker.env_list = [
        SimpleNamespace(unwrapped=SimpleNamespace(task_description="pick the corner"))
    ]

    assert worker._get_top_reward_instruction(0) == "fold the towel"


def test_subtask_update_does_not_reset_top_reward_for_initial_task_anchor():
    inner_env = SimpleNamespace(task_description="fold the towel")
    worker = EnvWorker.__new__(EnvWorker)
    worker.env_list = [SimpleNamespace(unwrapped=inner_env)]
    worker._top_reward_enabled = True
    worker._top_reward_instruction_source = "initial_task"
    reset_calls = []
    worker._reset_top_reward_state = lambda: reset_calls.append(True)
    worker.log_info = lambda *_args, **_kwargs: None

    assert worker._apply_subtask_update(0, "grasp the left corner")
    assert inner_env.task_description == "grasp the left corner"
    assert reset_calls == []


def test_async_pending_top_reward_is_resolved_into_env_output(monkeypatch):
    env_output = EnvOutput(
        obs={},
        rewards=torch.zeros((1, 2), dtype=torch.float32),
        dones=torch.zeros((1, 2), dtype=torch.bool),
        terminations=torch.zeros((1, 2), dtype=torch.bool),
        truncations=torch.zeros((1, 2), dtype=torch.bool),
    )
    worker = AsyncEnvWorker.__new__(AsyncEnvWorker)
    worker._top_reward_enabled = True
    worker._vlm_planner = object()
    worker._pending_top_reward_refs = ["score-ref"]
    worker._pending_top_reward_outputs = [env_output]
    worker._pending_top_reward_done_flags = [False]
    worker._async_prev_top_scores = [1.2]
    worker._async_episode_frames = [[]]
    worker.log_info = lambda *_args, **_kwargs: None
    worker._reset_top_reward_state = lambda stage_id=None: None

    monkeypatch.setattr(
        "rlinf.workers.env.async_env_worker.ray.get",
        lambda ref: 1.7 if ref == "score-ref" else None,
    )

    resolved = worker._resolve_pending_top_reward(0, env_output, wait=True)

    assert resolved is env_output
    assert resolved.rewards[0, -1].item() == pytest.approx(0.5)
    assert worker._async_prev_top_scores[0] == pytest.approx(1.7)
    assert worker._pending_top_reward_refs[0] is None
