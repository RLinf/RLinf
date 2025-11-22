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

import os
import sys
from pathlib import Path

import pytest
import ray
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.scheduler.cluster.config import ClusterConfig
from rlinf.scheduler.hardware.robots.franka import FrankaConfig


def test_cluster_config_parses_node_group_hardware():
    config = DictConfig(
        {
            "num_nodes": 4,
            "component_placement": {"actor": "0-3"},
            "node_groups": [
                {
                    "label": "a800",
                    "node_ranks": "0-1",
                    "env_configs": [
                        {
                            "node_ranks": "0-1",
                            "python_interpreter_path": "/opt/a800/bin/python3",
                            "env_vars": [{"GLOO_SOCKET_IFNAME": "eth0"}],
                        }
                    ],
                },
                {
                    "label": "franka",
                    "node_ranks": "2,3",
                    "hardware": {
                        "type": "Franka",
                        "configs": [
                            {
                                "node_rank": 2,
                                "robot_ip": "10.10.10.1",
                                "camera_serials": ["322142001230"],
                                "disable_validate": True,
                            },
                            {
                                "node_rank": 3,
                                "robot_ip": "10.10.10.2",
                                "camera_serials": ["322142001231"],
                                "disable_validate": True,
                            },
                        ],
                    },
                },
            ],
        }
    )

    cluster_cfg = ClusterConfig.from_dict_cfg(config)

    assert cluster_cfg.num_nodes == 4
    assert len(cluster_cfg.node_groups) == 2

    a800_group = next(
        group for group in cluster_cfg.node_groups if group.label == "a800"
    )
    assert a800_group.node_ranks == [0, 1]
    assert a800_group.hardware is None
    assert a800_group.env_configs is not None
    assert len(a800_group.env_configs) == 1
    env_cfg = a800_group.env_configs[0]
    assert env_cfg.node_ranks == [0, 1]
    assert env_cfg.python_interpreter_path == "/opt/a800/bin/python3"
    assert env_cfg.env_vars == [{"GLOO_SOCKET_IFNAME": "eth0"}]

    assert cluster_cfg.get_node_labels_by_rank(0) == ["a800"]
    assert (
        cluster_cfg.get_node_python_interpreter_path_by_rank(0)[0]
        == "/opt/a800/bin/python3"
    )
    assert cluster_cfg.get_node_hw_configs_by_rank(0) == []

    franka_group = next(
        group for group in cluster_cfg.node_groups if group.label == "franka"
    )
    assert franka_group.node_ranks == [2, 3]
    assert franka_group.hardware_type == "Franka"

    node2_hw = cluster_cfg.get_node_hw_configs_by_rank(2)
    assert len(node2_hw) == 1
    assert isinstance(node2_hw[0], FrankaConfig)
    assert node2_hw[0].robot_ip == "10.10.10.1"
    assert node2_hw[0].camera_serials == ["322142001230"]

    node3_hw = cluster_cfg.get_node_hw_configs_by_rank(3)
    assert len(node3_hw) == 1
    assert isinstance(node3_hw[0], FrankaConfig)
    assert node3_hw[0].robot_ip == "10.10.10.2"

    assert cluster_cfg.get_node_labels_by_rank(2) == ["franka"]
    assert cluster_cfg.get_node_labels_by_rank(3) == ["franka"]
    assert cluster_cfg.get_node_labels_by_rank(1) == ["a800"]
    assert cluster_cfg.get_node_labels_by_rank(99) == []


def test_cluster_config_rejects_reserved_node_group_label():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [{"label": "node", "node_ranks": "0"}],
        }
    )

    with pytest.raises(AssertionError, match="reserved"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_env_vars_must_be_single_kv():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "train",
                    "node_ranks": "0",
                    "env_configs": [
                        {
                            "node_ranks": "0",
                            "env_vars": [{"A": "1", "B": "2"}],
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(AssertionError, match="exactly one key-value pair"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_rejects_node_rank_out_of_range():
    config = DictConfig(
        {
            "num_nodes": 2,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "a",
                    "node_ranks": "0-3",
                }
            ],
        }
    )

    with pytest.raises(AssertionError, match="Error parsing node_ranks"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_duplicate_hardware_type_same_node():
    config = DictConfig(
        {
            "num_nodes": 2,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "franka_a",
                    "node_ranks": "0",
                    "hardware": {
                        "type": "Franka",
                        "configs": [
                            {
                                "node_rank": 0,
                                "robot_ip": "10.0.0.1",
                                "disable_validate": True,
                            }
                        ],
                    },
                },
                {
                    "label": "franka_b",
                    "node_ranks": "0,1",
                    "hardware": {
                        "type": "Franka",
                        "configs": [
                            {
                                "node_rank": 0,
                                "robot_ip": "10.0.0.2",
                                "disable_validate": True,
                            }
                        ],
                    },
                },
            ],
        }
    )

    with pytest.raises(AssertionError, match="Cannot have multiple hardware configs"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_duplicate_hardware_entries_disallowed():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "robots",
                    "node_ranks": "0",
                    "hardware": {
                        "type": "Franka",
                        "configs": [
                            {
                                "node_rank": 0,
                                "robot_ip": "10.0.0.1",
                                "disable_validate": True,
                            },
                            {
                                "node_rank": 0,
                                "robot_ip": "10.0.0.1",
                                "disable_validate": True,
                            },
                        ],
                    },
                }
            ],
        }
    )

    with pytest.raises(AssertionError, match="Duplicate hardware configs"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_node_group_entry_must_be_mapping():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": ["train"],
        }
    )

    with pytest.raises(
        AssertionError, match="Each node yaml config must be a dictionary"
    ):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_env_var_entry_must_be_mapping():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "train",
                    "node_ranks": "0",
                    "env_configs": [
                        {
                            "node_ranks": "0",
                            "env_vars": ["BAD"],
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(AssertionError, match="Each node env_var must be a dict"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_hardware_configs_must_be_mapping():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "robot",
                    "node_ranks": "0",
                    "hardware": {
                        "type": "Franka",
                        "configs": ["BAD"],
                    },
                }
            ],
        }
    )

    with pytest.raises(
        AssertionError, match="Each hardware config must be a dictionary"
    ):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_hardware_node_rank_must_be_in_group():
    config = DictConfig(
        {
            "num_nodes": 2,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "franka",
                    "node_ranks": "0",
                    "hardware": {
                        "type": "Franka",
                        "configs": [
                            {
                                "node_rank": 1,
                                "robot_ip": "10.0.0.1",
                                "disable_validate": True,
                            }
                        ],
                    },
                }
            ],
        }
    )

    with pytest.raises(AssertionError, match="must be within node_ranks"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_unsupported_hardware_type():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "robot",
                    "node_ranks": "0",
                    "hardware": {
                        "type": "UnknownHardware",
                        "configs": [],
                    },
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="Unsupported hardware type"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_invalid_node_rank_format():
    config = DictConfig(
        {
            "num_nodes": 2,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "train",
                    "node_ranks": "bad-format",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="Invalid rank format"):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_missing_required_hardware_field():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "robot",
                    "node_ranks": "0",
                    "hardware": {
                        "type": "Franka",
                        "configs": [
                            {
                                "node_rank": 0,
                                # Missing robot_ip
                                "disable_validate": True,
                            }
                        ],
                    },
                }
            ],
        }
    )

    with pytest.raises(
        AssertionError, match=r"Missing fields '\['robot_ip'\]' detected"
    ):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_unknown_hardware_field_rejected():
    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "robot",
                    "node_ranks": "0",
                    "hardware": {
                        "type": "Franka",
                        "configs": [
                            {
                                "node_rank": 0,
                                "robot_ip": "10.0.0.1",
                                "disable_validate": True,
                                "unknown_field": 1,
                            }
                        ],
                    },
                }
            ],
        }
    )

    with pytest.raises(
        AssertionError, match="Unknown fields '{'unknown_field'}' detected"
    ):
        ClusterConfig.from_dict_cfg(config)


def test_cluster_config_num_nodes_must_be_positive():
    config = DictConfig(
        {
            "num_nodes": 0,
            "component_placement": {},
        }
    )

    with pytest.raises(AssertionError, match="'num_nodes' must be a positive integer"):
        ClusterConfig.from_dict_cfg(config)


class EnvConfigCheckWorker(Worker):
    def __init__(self):
        super().__init__()

    def get_env_marker(self, key: str):
        return os.environ.get(key)


def _reset_cluster_singleton():
    if ray.is_initialized():
        ray.shutdown()
    if hasattr(Cluster, "_instance"):
        instance = getattr(Cluster, "_instance")
        if instance is not None:
            instance._has_initialized = False
        delattr(Cluster, "_instance")
    Cluster.NAMESPACE = Cluster.SYS_NAME


def test_cluster_env_configs_applied_in_worker_launch():
    env_key = "RLINF_TEST_ENV_CONFIG_MARKER"
    env_value = "marker-value"
    assert env_key not in os.environ

    _reset_cluster_singleton()

    tests_root = Path(__file__).resolve().parent
    python_path_entries = [str(tests_root)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    python_path_value = os.pathsep.join(python_path_entries)

    cluster_cfg = OmegaConf.create(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "train",
                    "node_ranks": "0",
                    "env_configs": [
                        {
                            "node_ranks": "0",
                            "python_interpreter_path": sys.executable,
                            "env_vars": [
                                {env_key: env_value},
                                {"PYTHONPATH": python_path_value},
                            ],
                        }
                    ],
                }
            ],
        }
    )

    cluster = Cluster(cluster_cfg=cluster_cfg)
    placement = NodePlacementStrategy([0], node_group_label="train")
    worker_group = EnvConfigCheckWorker.create_group().launch(
        cluster=cluster,
        placement_strategy=placement,
        name="env_config_launch",
    )

    try:
        env_values = worker_group.get_env_marker(env_key).wait()
        pythonpath_values = worker_group.get_env_marker("PYTHONPATH").wait()
    finally:
        worker_group._close()
        _reset_cluster_singleton()

    assert env_values == [env_value]
    assert pythonpath_values[0] is not None
    assert pythonpath_values[0].split(os.pathsep)[0] == str(tests_root)
