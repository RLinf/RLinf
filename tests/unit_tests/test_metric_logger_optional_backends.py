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

import importlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _load_metric_logger_cls():
    module_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "metric_logger.py"
    )
    spec = importlib.util.spec_from_file_location(
        "metric_logger_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MetricLogger


MetricLogger = _load_metric_logger_cls()


def _logger_cfg(tmp_path, backend: str):
    return OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "project_name": "rlinf",
                    "experiment_name": "optional-logger-test",
                    "logger_backends": [backend],
                },
                "per_worker_log": False,
            }
        }
    )


@pytest.mark.parametrize("backend", ["wandb", "swanlab"])
def test_missing_optional_logger_backend_has_extra_hint(monkeypatch, tmp_path, backend):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == backend:
            raise ImportError(f"No module named {backend!r}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=rf"rlinf\[{backend}\]"):
        MetricLogger(_logger_cfg(tmp_path, backend))
