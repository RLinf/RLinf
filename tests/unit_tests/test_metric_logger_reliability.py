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

"""Metric backend ownership and teardown regression tests."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from omegaconf import OmegaConf

from rlinf.utils.metric_logger import MetricLogger, _RunLogger


class _FakeRun:
    def __init__(self, name: str):
        self.name = name
        self.logged = []
        self.finish_count = 0

    def log(self, data, step=None):
        self.logged.append((data, step))

    def finish(self):
        self.finish_count += 1


class _FailingRun(_FakeRun):
    def finish(self):
        super().finish()
        raise RuntimeError("finish failed")


class _FakeWandb(ModuleType):
    class Table:
        def __init__(self, dataframe=None):
            self.dataframe = dataframe

    def __init__(self):
        super().__init__("wandb")
        self.init_calls = []
        self.runs = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run = _FakeRun(kwargs["name"])
        self.runs.append(run)
        return run


class _FakeSwanlab(ModuleType):
    def __init__(self, *, init_error: BaseException | None = None):
        super().__init__("swanlab")
        self.init_error = init_error
        self.init_calls = []
        self.runs = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.init_error is not None:
            raise self.init_error
        run = _FakeRun(kwargs["experiment_name"])
        self.runs.append(run)
        return run


def _cfg(tmp_path, backends, *, per_worker: bool = False):
    return OmegaConf.create(
        {
            "runner": {
                "per_worker_log": per_worker,
                "logger": {
                    "log_path": str(tmp_path),
                    "project_name": "test-project",
                    "experiment_name": "aggregate",
                    "logger_backends": backends,
                },
            }
        }
    )


def test_per_worker_wandb_logs_stay_with_their_owning_runs(tmp_path, monkeypatch):
    wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    logger = MetricLogger(_cfg(tmp_path, ["wandb"], per_worker=True))

    logger.log({"loss": 3.0}, step=1)
    logger.log({"loss": 2.0}, step=1, worker_group_name="actor", rank=0)
    logger.log({"loss": 1.0}, step=1, worker_group_name="actor", rank=1)

    assert [call["reinit"] for call in wandb.init_calls] == ["create_new"] * 3
    assert [run.logged for run in wandb.runs] == [
        [({"loss": 3.0}, 1)],
        [({"loss": 2.0}, 1)],
        [({"loss": 1.0}, 1)],
    ]

    logger.log_table([[1]], "samples", step=2)
    table_data, table_step = wandb.runs[0].logged[-1]
    assert table_step == 2
    assert table_data["samples"].dataframe == [[1]]

    logger.finish()
    logger.finish()
    logger.__del__()
    assert [run.finish_count for run in wandb.runs] == [1, 1, 1]


def test_single_wandb_run_preserves_existing_reinit_behavior(tmp_path, monkeypatch):
    wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    MetricLogger(_cfg(tmp_path, ["wandb"]))

    assert wandb.init_calls[0]["reinit"] is True


def test_per_worker_swanlab_is_rejected_before_initialization(tmp_path, monkeypatch):
    swanlab = _FakeSwanlab()
    monkeypatch.setitem(sys.modules, "swanlab", swanlab)

    with pytest.raises(ValueError, match="one active run per process"):
        MetricLogger(_cfg(tmp_path, ["swanlab"], per_worker=True))

    assert swanlab.init_calls == []


def test_single_swanlab_run_uses_its_returned_handle(tmp_path, monkeypatch):
    swanlab = _FakeSwanlab()
    monkeypatch.setitem(sys.modules, "swanlab", swanlab)
    logger = MetricLogger(_cfg(tmp_path, ["swanlab"]))

    logger.log({"reward": 0.5}, step=4)
    logger.finish()
    logger.finish()

    assert swanlab.runs[0].logged == [({"reward": 0.5}, 4)]
    assert swanlab.runs[0].finish_count == 1


@pytest.mark.parametrize(
    "init_error", [RuntimeError("swanlab init failed"), KeyboardInterrupt()]
)
def test_failed_backend_construction_closes_initialized_runs(
    tmp_path, monkeypatch, init_error
):
    wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setitem(sys.modules, "swanlab", _FakeSwanlab(init_error=init_error))

    with pytest.raises(type(init_error)):
        MetricLogger(_cfg(tmp_path, ["wandb", "swanlab"]))

    assert wandb.runs[0].finish_count == 1


def test_backend_teardown_runs_once_but_retries_after_a_failure():
    healthy = _RunLogger(_FakeRun("healthy"))
    healthy.finish()
    healthy.finish()
    assert healthy.run.finish_count == 1

    failing = _RunLogger(_FailingRun("failing"))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="finish failed"):
            failing.finish()
    assert failing.run.finish_count == 2


def test_finish_closes_every_backend_before_reporting_a_failure(tmp_path):
    logger = MetricLogger(_cfg(tmp_path, []))
    failing = _RunLogger(_FailingRun("failing"))
    healthy = _RunLogger(_FakeRun("healthy"))
    logger.logger.update(failing=failing, healthy=healthy)

    with pytest.raises(RuntimeError, match="finish failed"):
        logger.finish()

    assert healthy.run.finish_count == 1

    logger.__del__()  # swallows the still-failing backend
    assert healthy.run.finish_count == 1


def test_destructor_is_safe_for_an_uninitialized_metric_logger():
    MetricLogger.__new__(MetricLogger).__del__()
