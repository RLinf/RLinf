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

"""Reproducibility of the embodied FSDP path (issue #1685).

``actor.seed`` must reach more than the minibatch shuffle: the rollout, actor
and env workers have to seed their process-global RNGs (mirroring the Megatron
path), and the world-model env must stop opening every backend session with
the constant session seed ``0``.
"""

import ast
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.utils.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every ``init_worker`` that a worker class on the embodied FSDP path owns.
# Subclasses that override ``init_worker`` without calling ``super()`` (SAC,
# DAGGER, IQL) are listed explicitly; everything else inherits one of these.
EMBODIED_INIT_SITES = [
    ("rlinf/workers/actor/embodied_fsdp_actor_worker.py", "EmbodiedFSDPActor"),
    ("rlinf/workers/actor/fsdp_sac_policy_worker.py", "EmbodiedSACFSDPPolicy"),
    ("rlinf/workers/actor/fsdp_dagger_policy_worker.py", "EmbodiedDAGGERFSDPPolicy"),
    ("rlinf/workers/actor/fsdp_iql_policy_worker.py", "EmbodiedIQLFSDPPolicy"),
    ("rlinf/workers/rollout/hf/huggingface_worker.py", "MultiStepRolloutWorker"),
    ("rlinf/workers/env/env_worker.py", "EnvWorker"),
]


def _calls_in_function(source: str, class_name: str, function_name: str) -> list[str]:
    """Names called inside ``class_name.function_name``, via AST (no import)."""
    tree = ast.parse(source)
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == function_name:
                for sub in ast.walk(item):
                    if isinstance(sub, ast.Call):
                        func = sub.func
                        if isinstance(func, ast.Name):
                            calls.append(func.id)
                        elif isinstance(func, ast.Attribute):
                            calls.append(func.attr)
    return calls


def test_seed_everything_pins_all_three_rng_stacks():
    """The helper the fix relies on pins Python, NumPy and torch draws."""
    seed_everything(1234)
    first = (random.random(), np.random.random(4), torch.rand(4))

    seed_everything(1234)
    second = (random.random(), np.random.random(4), torch.rand(4))

    seed_everything(4321)
    third = (random.random(), np.random.random(4), torch.rand(4))

    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert first[0] != third[0]
    assert not np.array_equal(first[1], third[1])
    assert not torch.equal(first[2], third[2])
    assert seed_everything(1234) == 1234


def test_seed_everything_varies_by_rank():
    """Per-rank seeds (``actor.seed + rank``) must give per-rank streams."""
    seed_everything(1234 + 0)
    rank0 = torch.rand(4)

    seed_everything(1234 + 1)
    rank1 = torch.rand(4)

    assert not torch.equal(rank0, rank1)


@pytest.mark.parametrize(
    "module_path,class_name",
    EMBODIED_INIT_SITES,
    ids=[class_name for _, class_name in EMBODIED_INIT_SITES],
)
def test_embodied_worker_init_seeds_everything(module_path, class_name):
    """Each embodied-path worker seeds its global RNGs in ``init_worker``.

    Inspected via AST rather than instantiated: building the real workers
    needs models, checkpoints and distributed groups that a CPU-only run
    does not have.
    """
    source = (REPO_ROOT / module_path).read_text()

    assert "seed_everything" in _calls_in_function(source, class_name, "init_worker"), (
        f"{class_name}.init_worker in {module_path} does not call seed_everything; "
        "actor.seed never reaches this worker's global RNGs."
    )
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert "seed_everything" in imports, (
        f"{module_path} must import seed_everything from rlinf.utils.utils"
    )


class _RecordingBackend:
    """Stands in for a Wan/OpenSora backend and records open_session calls."""

    def __init__(self):
        self.open_calls = []
        self.closed_calls = []

    def close_session(self, env_ids):
        self.closed_calls.append(list(env_ids))

    def open_session(self, env_ids, init_frames, init_actions, seeds):
        self.open_calls.append(
            {
                "env_ids": list(env_ids),
                "seeds": [int(seed) for seed in seeds],
            }
        )


def _make_world_model_env(cfg_seed: int, seed_offset: int, num_envs: int = 2):
    """Build a ``WorldModelEnv`` without backends, datasets or checkpoints.

    Only the attributes that ``reset`` touches are injected; the backend is a
    recorder so the test can observe the session seeds ``reset`` passes.
    """
    from rlinf.envs.sim.world_model.env import WorldModelEnv

    env = object.__new__(WorldModelEnv)
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.seed = cfg_seed + seed_offset
    env._is_start = True
    env._is_offloaded = False
    env.use_fixed_reset_state_ids = False
    env.record_metrics = False
    env.prev_step_reward = torch.zeros(num_envs)
    env._elapsed_steps = torch.zeros(num_envs, dtype=torch.long)
    env.condition_frame_length = 2
    env.image_size = [8, 8]
    env.dataset = [object() for _ in range(num_envs)]
    env.backend = _RecordingBackend()

    frames = torch.randn(num_envs, 3, env.condition_frame_length, 8, 8)

    def _condition_window(_episode_idx):
        # (obs window, action chunk, task description, init ee pose)
        return frames[0], torch.zeros(4, 7), "task", None

    env._build_condition_window = _condition_window
    return env


def test_world_model_env_reset_derives_session_seeds_from_env_seed():
    """Backend sessions must not all open with the hardcoded seed 0.

    The Wan pipeline draws its generation noise from the session seed, so a
    constant ``[0] * num_slots`` makes the world-model rollout identical in
    every run regardless of the configured seed.
    """
    cfg_seed, seed_offset = 1234, 3
    env = _make_world_model_env(cfg_seed, seed_offset)

    env.reset(seed=cfg_seed)

    assert env.seed == cfg_seed + seed_offset
    assert len(env.backend.open_calls) == 1
    call = env.backend.open_calls[0]
    assert call["env_ids"] == [0, 1]
    assert call["seeds"] == [env.seed] * env.num_envs
    assert set(call["seeds"]) != {0}


def test_world_model_env_session_seed_follows_the_configured_seed():
    """Two runs that differ only in the configured seed must not share noise."""
    first = _make_world_model_env(cfg_seed=1234, seed_offset=0)
    first.reset(seed=1234)

    second = _make_world_model_env(cfg_seed=4321, seed_offset=0)
    second.reset(seed=4321)

    first_seeds = first.backend.open_calls[0]["seeds"]
    second_seeds = second.backend.open_calls[0]["seeds"]
    assert first_seeds != second_seeds

    # Same config, same process offset => reproducible session seed.
    repeat = _make_world_model_env(cfg_seed=1234, seed_offset=0)
    repeat.reset(seed=1234)
    assert repeat.backend.open_calls[0]["seeds"] == first_seeds
