# Copyright 2025 The RLinf Authors.
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

import numpy as np
import pytest
import torch

libero_env = pytest.importorskip("rlinf.envs.libero.libero_env")
LiberoEnv = libero_env.LiberoEnv


class _Recorder:
    """Captures what ``chunk_step`` asked the inner env and ``step`` to do."""

    def __init__(self):
        self.render_calls = []
        self.skip_obs_wrap_flags = []


def _make_env(skip: bool, fail_at_substep: int | None = None):
    env = object.__new__(LiberoEnv)
    env.skip_intermediate_renders = skip
    env.num_envs = 1
    env.auto_reset = False
    env.ignore_terminations = True

    rec = _Recorder()

    class _InnerEnv:
        def set_camera_rendering(self, enabled, id=None):
            rec.render_calls.append(bool(enabled))

    env.env = _InnerEnv()

    def _step(action, auto_reset=False, _skip_obs_wrap=False):
        idx = len(rec.skip_obs_wrap_flags)
        rec.skip_obs_wrap_flags.append(bool(_skip_obs_wrap))
        if fail_at_substep is not None and idx == fail_at_substep:
            raise RuntimeError("simulated env failure")
        z = torch.zeros(env.num_envs)
        obs = None if _skip_obs_wrap else {"obs": idx}
        return obs, z, z.bool(), z.bool(), [{}]

    env.step = _step
    return env, rec


def _actions(chunk_size: int) -> np.ndarray:
    return np.zeros((1, chunk_size, 7), dtype=np.float32)


@pytest.mark.parametrize("chunk_size", [2, 4, 8])
def test_rendering_is_toggled_once_off_and_once_on(chunk_size):
    """Only two IPC round trips per chunk, regardless of chunk size."""
    env, rec = _make_env(skip=True)
    env.chunk_step(_actions(chunk_size))
    assert rec.render_calls == [False, True]


@pytest.mark.parametrize("chunk_size", [2, 4, 8])
def test_only_the_final_substep_builds_an_observation(chunk_size):
    env, rec = _make_env(skip=True)
    obs_list, _rewards, _terms, _truncs, infos_list = env.chunk_step(
        _actions(chunk_size)
    )

    assert rec.skip_obs_wrap_flags == [True] * (chunk_size - 1) + [False]
    assert obs_list[:-1] == [None] * (chunk_size - 1)
    assert obs_list[-1] == {"obs": chunk_size - 1}
    # infos are still collected for every substep, only observations are skipped.
    assert len(infos_list) == chunk_size


def test_single_substep_chunk_still_renders():
    """chunk_size == 1 has no intermediate substep to skip."""
    env, rec = _make_env(skip=True)
    obs_list, *_ = env.chunk_step(_actions(1))

    assert rec.render_calls == [True]
    assert rec.skip_obs_wrap_flags == [False]
    assert obs_list == [{"obs": 0}]


def test_rendering_is_restored_when_a_substep_raises():
    """The env must not be left with cameras disabled after a mid-chunk failure."""
    env, rec = _make_env(skip=True, fail_at_substep=1)
    with pytest.raises(RuntimeError, match="simulated env failure"):
        env.chunk_step(_actions(8))

    assert rec.render_calls == [False, True]


@pytest.mark.parametrize("chunk_size", [1, 8])
def test_no_toggling_or_skipping_when_disabled(chunk_size):
    env, rec = _make_env(skip=False)
    obs_list, *_ = env.chunk_step(_actions(chunk_size))

    assert rec.render_calls == []
    assert rec.skip_obs_wrap_flags == [False] * chunk_size
    assert all(obs is not None for obs in obs_list)
