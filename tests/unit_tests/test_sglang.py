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

import copy
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class _FinishAbort:
    pass


class _LegacyReq:
    def __init__(self):
        self.to_abort = False


class _ModernReq:
    def __init__(self):
        self.to_finish = None


class _DualProtocolReq(_ModernReq):
    def __init__(self):
        super().__init__()
        self.to_abort = False


@pytest.fixture
def sglang_scheduler(monkeypatch):
    """Load RLinf's scheduler with only the external SGLang imports replaced."""
    modules = {
        "sglang": {},
        "sglang.srt": {},
        "sglang.srt.managers": {},
        "sglang.srt.constants": {
            "GPU_MEMORY_ALL_TYPES": ["weights", "kv_cache", "cuda_graph"],
        },
        "sglang.srt.managers.schedule_batch": {
            "Req": _ModernReq,
            "FINISH_ABORT": _FinishAbort,
        },
        "sglang.srt.managers.io_struct": {
            "ReleaseMemoryOccupationReqInput": object,
            "ResumeMemoryOccupationReqInput": object,
        },
        "sglang.srt.managers.scheduler": {
            "Scheduler": object,
            "logger": logging.getLogger("test.sglang"),
            "run_scheduler_process": object,
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    source_dir = (
        Path(__file__).resolve().parents[2] / "rlinf/hybrid_engines/sglang/common"
    )
    loaded = {}
    for name in ("io_struct", "sgl_scheduler"):
        qualified_name = f"rlinf.hybrid_engines.sglang.common.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name, source_dir / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified_name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["sgl_scheduler"], loaded["io_struct"]


def _scheduler_with_requests(modules, waiting, running):
    scheduler_module, _ = modules
    scheduler = object.__new__(scheduler_module.Scheduler)
    scheduler.waiting_queue = waiting
    scheduler.running_batch = SimpleNamespace(reqs=running)
    return scheduler


@pytest.mark.parametrize("request_type", [_LegacyReq, _ModernReq])
@pytest.mark.parametrize(
    "waiting_count,running_count", [(0, 0), (2, 0), (0, 2), (1, 1)]
)
def test_abort_generation_marks_waiting_and_running_requests(
    sglang_scheduler, request_type, waiting_count, running_count
):
    waiting = [request_type() for _ in range(waiting_count)]
    running = [request_type() for _ in range(running_count)]
    requests = waiting + running
    for req in requests:
        req.origin_input_ids = [7, 8]
        req.output_ids = [101, 102]
        req.output_token_logprobs_val = [-0.1, -0.2]
        req.output_token_logprobs_idx = [101, 102]
    before = [copy.deepcopy(vars(req)) for req in requests]
    scheduler = _scheduler_with_requests(sglang_scheduler, waiting, running)
    _, io_struct = sglang_scheduler

    result = scheduler.abort_generation(io_struct.AbortGenerationInput())

    assert isinstance(result, io_struct.AbortGenerationOutput)
    assert scheduler.waiting_queue is waiting
    assert scheduler.running_batch.reqs is running
    for req, original in zip(requests, before):
        if request_type is _LegacyReq:
            assert req.to_abort is True
            assert not hasattr(req, "to_finish")
            original.pop("to_abort")
            unchanged = {k: v for k, v in vars(req).items() if k != "to_abort"}
        else:
            assert isinstance(req.to_finish, _FinishAbort)
            assert not hasattr(req, "to_abort")
            original.pop("to_finish")
            unchanged = {k: v for k, v in vars(req).items() if k != "to_finish"}
        assert unchanged == original


@pytest.mark.parametrize("queue", ["waiting", "running"])
def test_abort_generation_prefers_pending_finish_protocol(sglang_scheduler, queue):
    req = _DualProtocolReq()
    scheduler = _scheduler_with_requests(
        sglang_scheduler,
        [req] if queue == "waiting" else [],
        [req] if queue == "running" else [],
    )
    _, io_struct = sglang_scheduler

    scheduler.abort_generation(io_struct.AbortGenerationInput())

    assert isinstance(req.to_finish, _FinishAbort)
    assert req.to_abort is False


@pytest.mark.parametrize("queue", ["waiting", "running"])
@pytest.mark.parametrize("already_finishing", [False, True])
def test_abort_generation_preserves_pending_finish_on_repeated_calls(
    sglang_scheduler, queue, already_finishing
):
    req = _ModernReq()
    pending_reason = object() if already_finishing else None
    req.to_finish = pending_reason
    scheduler = _scheduler_with_requests(
        sglang_scheduler,
        [req] if queue == "waiting" else [],
        [req] if queue == "running" else [],
    )
    _, io_struct = sglang_scheduler

    scheduler.abort_generation(io_struct.AbortGenerationInput())
    first_reason = req.to_finish
    if already_finishing:
        assert first_reason is pending_reason
    else:
        assert isinstance(first_reason, _FinishAbort)
    scheduler.abort_generation(io_struct.AbortGenerationInput())

    assert req.to_finish is first_reason
    assert not hasattr(req, "to_abort")
