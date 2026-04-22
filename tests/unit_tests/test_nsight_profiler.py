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

"""Tests for rlinf.utils.nsight_profiler module.

All tests are CPU-only and mock ``nvtx`` / ``torch.cuda.profiler`` so they
can run on any machine without an NVIDIA GPU or nsys installation.
"""

import asyncio
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fixtures — import module and patch nvtx availability
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_channel_flag():
    """Ensure the module-level channel flag is reset between tests."""
    from rlinf.utils import nsight_profiler as mod
    original = mod._channel_nvtx_enabled
    yield
    mod._channel_nvtx_enabled = original


@pytest.fixture()
def mod():
    """Return the nsight_profiler module."""
    from rlinf.utils import nsight_profiler as mod
    return mod


@pytest.fixture()
def mock_nvtx(mod):
    """Patch ``_NVTX_AVAILABLE=True`` and mock ``nvtx`` calls."""
    fake_nvtx = mock.MagicMock()
    fake_nvtx.start_range.return_value = 42
    with mock.patch.object(mod, "_NVTX_AVAILABLE", True), \
         mock.patch.object(mod, "nvtx", fake_nvtx, create=True):
        yield fake_nvtx


@pytest.fixture()
def mock_cuda_profiler():
    """Mock ``torch.cuda.profiler.start/stop``."""
    with mock.patch("torch.cuda.profiler.start") as start, \
         mock.patch("torch.cuda.profiler.stop") as stop:
        yield start, stop


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_none_config_returns_disabled(self, mod):
        p = mod.NsightProfiler.from_config(None, "actor", rank=0)
        assert p.enable is False
        assert p.is_active is False

    def test_steps_none_returns_disabled(self, mod):
        cfg = {"steps": None, "actor": {"enable": True}}
        p = mod.NsightProfiler.from_config(cfg, "actor", rank=0)
        assert p.enable is False

    def test_valid_config_returns_enabled(self, mod):
        cfg = {
            "steps": [2, 3],
            "discrete": False,
            "actor": {"enable": True, "all_ranks": True, "ranks": []},
        }
        p = mod.NsightProfiler.from_config(cfg, "actor", rank=0)
        assert p.enable is True
        assert p.all_ranks is True
        assert p.discrete is False

    def test_role_not_in_config_uses_defaults(self, mod):
        cfg = {"steps": [1], "actor": {"enable": True}}
        p = mod.NsightProfiler.from_config(cfg, "rollout", rank=0)
        assert p.enable is False

    def test_discrete_flag_forwarded(self, mod):
        cfg = {
            "steps": [1],
            "discrete": True,
            "env": {"enable": True},
        }
        p = mod.NsightProfiler.from_config(cfg, "env", rank=0)
        assert p.discrete is True


# ---------------------------------------------------------------------------
# Rank filtering
# ---------------------------------------------------------------------------

class TestRankFiltering:
    def test_all_ranks(self, mod):
        p = mod.NsightProfiler(enable=True, rank=5, all_ranks=True)
        assert p._this_rank is True

    def test_specific_ranks_included(self, mod):
        p = mod.NsightProfiler(enable=True, rank=2, all_ranks=False, ranks=[0, 2])
        assert p._this_rank is True

    def test_specific_ranks_excluded(self, mod):
        p = mod.NsightProfiler(enable=True, rank=3, all_ranks=False, ranks=[0, 2])
        assert p._this_rank is False

    def test_disabled_profiler_rank_is_false(self, mod):
        p = mod.NsightProfiler(enable=False, rank=0, all_ranks=True)
        assert p._this_rank is False


# ---------------------------------------------------------------------------
# start / stop / is_active
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_start_activates_profiler(self, mod, mock_nvtx, mock_cuda_profiler):
        p = mod.NsightProfiler(enable=True, rank=0)
        assert p.is_active is False
        p.start(step=1)
        assert p.is_active is True

    def test_stop_deactivates_profiler(self, mod, mock_nvtx, mock_cuda_profiler):
        p = mod.NsightProfiler(enable=True, rank=0)
        p.start(step=1)
        p.stop()
        assert p.is_active is False

    def test_start_calls_cuda_profiler(self, mod, mock_nvtx, mock_cuda_profiler):
        cuda_start, _ = mock_cuda_profiler
        p = mod.NsightProfiler(enable=True, rank=0, discrete=False)
        p.start(step=0)
        cuda_start.assert_called_once()

    def test_stop_calls_cuda_profiler(self, mod, mock_nvtx, mock_cuda_profiler):
        _, cuda_stop = mock_cuda_profiler
        p = mod.NsightProfiler(enable=True, rank=0, discrete=False)
        p.start(step=0)
        p.stop()
        cuda_stop.assert_called_once()

    def test_discrete_mode_skips_cuda_profiler(self, mod, mock_nvtx, mock_cuda_profiler):
        cuda_start, cuda_stop = mock_cuda_profiler
        p = mod.NsightProfiler(enable=True, rank=0, discrete=True)
        p.start(step=0)
        p.stop()
        cuda_start.assert_not_called()
        cuda_stop.assert_not_called()

    def test_disabled_profiler_noop(self, mod, mock_cuda_profiler):
        cuda_start, cuda_stop = mock_cuda_profiler
        p = mod.NsightProfiler(enable=False, rank=0)
        p.start(step=0)
        p.stop()
        cuda_start.assert_not_called()
        cuda_stop.assert_not_called()
        assert p.is_active is False


# ---------------------------------------------------------------------------
# Channel NVTX flag side-effect
# ---------------------------------------------------------------------------

class TestChannelNvtxFlag:
    def test_start_enables_channel_flag(self, mod, mock_nvtx, mock_cuda_profiler):
        assert mod.is_channel_nvtx_enabled() is False
        p = mod.NsightProfiler(enable=True, rank=0)
        p.start(step=0)
        assert mod.is_channel_nvtx_enabled() is True

    def test_stop_disables_channel_flag(self, mod, mock_nvtx, mock_cuda_profiler):
        p = mod.NsightProfiler(enable=True, rank=0)
        p.start(step=0)
        p.stop()
        assert mod.is_channel_nvtx_enabled() is False

    def test_set_and_get(self, mod):
        mod.set_channel_nvtx_enabled(True)
        assert mod.is_channel_nvtx_enabled() is True
        mod.set_channel_nvtx_enabled(False)
        assert mod.is_channel_nvtx_enabled() is False


# ---------------------------------------------------------------------------
# nvtx_range
# ---------------------------------------------------------------------------

class TestNvtxRange:
    def test_noop_when_disabled(self, mod, mock_nvtx):
        with mod.nvtx_range("test", enabled=False):
            pass
        mock_nvtx.start_range.assert_not_called()

    def test_fires_when_enabled(self, mod, mock_nvtx):
        with mod.nvtx_range("test_label", color="red", enabled=True):
            pass
        mock_nvtx.start_range.assert_called_once_with(
            message="test_label", color="red", domain=None,
        )
        mock_nvtx.end_range.assert_called_once_with(42)

    def test_noop_when_nvtx_unavailable(self, mod):
        with mock.patch.object(mod, "_NVTX_AVAILABLE", False):
            with mod.nvtx_range("test", enabled=True):
                pass


# ---------------------------------------------------------------------------
# @annotate decorator — sync
# ---------------------------------------------------------------------------

class TestAnnotateSync:
    def test_sync_noop_when_profiler_inactive(self, mod):
        profiler = mod.NsightProfiler(enable=False, rank=0)

        class Stub:
            nsight_profiler = profiler

            @mod.NsightProfiler.annotate("label")
            def method(self):
                return "ok"

        assert Stub().method() == "ok"

    def test_sync_fires_nvtx_when_active(self, mod, mock_nvtx, mock_cuda_profiler):
        profiler = mod.NsightProfiler(enable=True, rank=0)
        profiler.start(step=0)

        class Stub:
            nsight_profiler = profiler

            @mod.NsightProfiler.annotate("test/sync")
            def method(self):
                return 42

        result = Stub().method()
        assert result == 42
        mock_nvtx.start_range.assert_called()
        mock_nvtx.end_range.assert_called()

    def test_sync_uses_func_name_as_default_label(self, mod, mock_nvtx, mock_cuda_profiler):
        profiler = mod.NsightProfiler(enable=True, rank=0)
        profiler.start(step=0)

        class Stub:
            nsight_profiler = profiler

            @mod.NsightProfiler.annotate()
            def my_custom_method(self):
                return "val"

        Stub().my_custom_method()
        call_args = mock_nvtx.start_range.call_args
        assert call_args.kwargs.get("message") == "my_custom_method" or \
               call_args[1].get("message") == "my_custom_method"


# ---------------------------------------------------------------------------
# @annotate decorator — async
# ---------------------------------------------------------------------------

class TestAnnotateAsync:
    def test_async_noop_when_profiler_inactive(self, mod):
        profiler = mod.NsightProfiler(enable=False, rank=0)

        class Stub:
            nsight_profiler = profiler

            @mod.NsightProfiler.annotate("label")
            async def method(self):
                return "async_ok"

        result = asyncio.get_event_loop().run_until_complete(Stub().method())
        assert result == "async_ok"

    def test_async_fires_nvtx_when_active(self, mod, mock_nvtx, mock_cuda_profiler):
        profiler = mod.NsightProfiler(enable=True, rank=0)
        profiler.start(step=0)

        class Stub:
            nsight_profiler = profiler

            @mod.NsightProfiler.annotate("test/async")
            async def method(self):
                return 99

        result = asyncio.get_event_loop().run_until_complete(Stub().method())
        assert result == 99
        mock_nvtx.start_range.assert_called()
        mock_nvtx.end_range.assert_called()

    def test_async_discrete_mode(self, mod, mock_nvtx, mock_cuda_profiler):
        cuda_start, cuda_stop = mock_cuda_profiler
        profiler = mod.NsightProfiler(enable=True, rank=0, discrete=True)
        profiler._this_step = True

        class Stub:
            nsight_profiler = profiler

            @mod.NsightProfiler.annotate("test/discrete")
            async def method(self):
                return "d"

        asyncio.get_event_loop().run_until_complete(Stub().method())
        cuda_start.assert_called()
        cuda_stop.assert_called()


# ---------------------------------------------------------------------------
# @annotate — no nsight_profiler attr on instance
# ---------------------------------------------------------------------------

class TestAnnotateNoProfilerAttr:
    def test_sync_no_profiler_attr(self, mod):
        class Stub:
            @mod.NsightProfiler.annotate("label")
            def method(self):
                return "no_profiler"

        assert Stub().method() == "no_profiler"

    def test_async_no_profiler_attr(self, mod):
        class Stub:
            @mod.NsightProfiler.annotate("label")
            async def method(self):
                return "no_profiler_async"

        result = asyncio.get_event_loop().run_until_complete(Stub().method())
        assert result == "no_profiler_async"
