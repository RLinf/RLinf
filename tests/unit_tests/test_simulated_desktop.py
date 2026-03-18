# Copyright 2026 Shirui Chen
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

"""Tests for local simulated desktop support used by staged YAM training."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.envs.remote.simulated_desktop import (
    build_simulated_desktop_server_spec,
    parse_host_port,
)


def _make_cfg(remote_server_url: str = "localhost:50051", enable_eval: bool = False):
    return OmegaConf.create(
        {
            "runner": {
                "only_eval": False,
                "val_check_interval": 10 if enable_eval else -1,
            },
            "env": {
                "remote_desktop_simulation": {
                    "enabled": True,
                    "dummy": True,
                    "startup_timeout": 12.5,
                },
                "train": {
                    "env_type": "remote",
                    "remote_server_url": remote_server_url,
                },
                "eval": {
                    "env_type": "remote",
                    "remote_server_url": remote_server_url,
                },
            },
        }
    )


def test_parse_host_port_requires_host_and_port():
    assert parse_host_port("localhost:50051") == ("localhost", 50051)

    with pytest.raises(ValueError):
        parse_host_port("localhost")


def test_build_simulated_desktop_server_spec_uses_default_local_yam_config():
    cfg = _make_cfg()

    spec = build_simulated_desktop_server_spec(cfg)

    assert spec is not None
    assert spec.host == "localhost"
    assert spec.port == 50051
    assert spec.dummy is True
    assert spec.startup_timeout == pytest.approx(12.5)
    assert spec.env_config_path.endswith("examples/embodiment/config/env/yam.yaml")
    assert Path(spec.env_config_path).is_absolute()


def test_build_simulated_desktop_server_spec_rejects_non_local_remote_url():
    cfg = _make_cfg(remote_server_url="10.0.0.8:50051")

    with pytest.raises(RuntimeError, match="only supports local RemoteEnv URLs"):
        build_simulated_desktop_server_spec(cfg)


def test_build_simulated_desktop_server_spec_requires_matching_active_urls():
    cfg = _make_cfg(enable_eval=True)
    cfg.env.eval.remote_server_url = "localhost:50052"

    with pytest.raises(RuntimeError, match="share the same remote_server_url"):
        build_simulated_desktop_server_spec(cfg)
