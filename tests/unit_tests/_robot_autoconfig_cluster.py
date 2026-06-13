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

"""Real cluster + worker scenarios for robot env-var auto-config.

Each scenario boots a real single-node cluster from a cluster config and
launches real workers (like ``test.py`` at the repo root). It is invoked as a
subprocess by ``test_robot_autoconfig.py`` with a mode argument so every
scenario gets a fresh process: ``Cluster`` is a process-wide singleton and the
enumeration actor must inherit the env vars set before Ray starts.

Usage:  python _robot_autoconfig_cluster.py <mode>
Prints ``<mode>:OK`` and exits 0 on success.
"""

import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "create_multi"

# Env vars must be set before Ray (and its enumeration actor) starts.
_ENV = {
    # No configs in YAML -> two robots created, one comma value each.
    "create_multi": {
        "ROBOT_IP": "10.20.30.40,10.20.30.41",
        "CAMERA_SERIALS": "serialA,serialB",
        "GRIPPER_CONNECTION": "/dev/ttyUSB0,/dev/ttyUSB1",
        "DISABLE_VALIDATE": "true,true",
    },
    # No configs in YAML, a single value -> one robot that keeps the full
    # comma-separated camera list (single-robot uses the whole value).
    "create_single": {
        "ROBOT_IP": "10.20.30.40",
        "CAMERA_SERIALS": "serialA,serialB",
        "GRIPPER_CONNECTION": "/dev/ttyUSB0",
        "DISABLE_VALIDATE": "true",
    },
    # Explicit configs: one robot_ip set in YAML (kept), one omitted (filled).
    "explicit_fill": {"ROBOT_IP": "10.20.30.40,10.20.30.41"},
    # A second robot type is created from its own identifier env var.
    "gim_create": {
        "CAN_INTERFACE": "can3",
        "CAMERA_SERIALS": "gimcam",
        "DISABLE_VALIDATE": "true",
    },
    # A shared field without the identifier (ROBOT_IP) -> no robot created.
    "gating": {"CAMERA_SERIALS": "serialA,serialB"},
    # Too few values for the two explicit configs.
    "mismatch_low": {"ROBOT_IP": "10.20.30.40"},
    # Too many values for the two explicit configs.
    "mismatch_high": {"ROBOT_IP": "a,b,c"},
    # Identifier count ok, but a secondary field disagrees.
    "mismatch_secondary": {
        "ROBOT_IP": "10.20.30.40,10.20.30.41",
        "CAMERA_SERIALS": "x,y,z",
    },
    # Legacy path: everything specified in YAML, no env vars at all.
    "yaml_single": {},
    "yaml_multi": {},
    "yaml_dosw1": {},
}
if MODE not in _ENV:
    raise SystemExit(f"unknown mode: {MODE}")

# Clear any ambient managed vars first so each mode's environment is hermetic
# (the ``yaml_*`` modes in particular must see no robot env vars at all).
_MANAGED = (
    "ROBOT_IP",
    "CAMERA_SERIALS",
    "GRIPPER_CONNECTION",
    "DISABLE_VALIDATE",
    "CAN_INTERFACE",
    "ROBOT_URL",
    "LEFT_ROBOT_IP",
    "RIGHT_ROBOT_IP",
    "CAMERA_TYPE",
    "GRIPPER_TYPE",
    "ARM_VARIANT",
    "LEFT_ARM_PORT",
)
for _name in _MANAGED:
    os.environ.pop(_name, None)
os.environ.update(_ENV[MODE])

import dataclasses  # noqa: E402

from omegaconf import DictConfig  # noqa: E402

from rlinf.scheduler import (  # noqa: E402
    Cluster,
    FlexiblePlacementStrategy,
    Worker,
)


class RobotEnvWorker(Worker):
    """A minimal env-worker stand-in that reports its robot hardware info."""

    def __init__(self):
        super().__init__()

    def configs(self) -> list[dict]:
        """Return each assigned hardware info's config as a plain dict."""
        return [dataclasses.asdict(info.config) for info in self.hardware_infos]


def cluster_cfg(label: str, hw_type: str, configs: list) -> DictConfig:
    """A single-node cluster with one hardware node group."""
    return DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": label,
                    "node_ranks": "0",
                    "hardware": {"type": hw_type, "configs": configs},
                }
            ],
        }
    )


def enumerated_configs(cluster: Cluster, hw_type: str) -> list:
    """The configs of all ``hw_type`` hardware infos enumerated on node 0."""
    return [
        info.config
        for resource in cluster.get_node_info(0).hardware_resources
        if resource.type == hw_type
        for info in resource.infos
    ]


def worker_configs(cluster: Cluster, label: str, ranks_list: list) -> list:
    """Launch a worker group over ``ranks_list`` and collect each worker's view."""
    group = RobotEnvWorker.create_group().launch(
        cluster=cluster,
        placement_strategy=FlexiblePlacementStrategy(
            hardware_ranks_list=ranks_list,
            node_group_label=label,
        ),
        name=f"RobotEnvWorker-{label}-{len(ranks_list)}",
    )
    return group.configs().wait()


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def run_create_multi() -> None:
    """No configs in YAML: two Franka robots created, one comma value each."""
    cluster = Cluster(cluster_cfg=cluster_cfg("franka", "Franka", []))

    configs = enumerated_configs(cluster, "Franka")
    assert [c.robot_ip for c in configs] == ["10.20.30.40", "10.20.30.41"]
    assert [c.camera_serials for c in configs] == [["serialA"], ["serialB"]]
    assert [c.gripper_connection for c in configs] == [
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
    ]
    assert all(c.disable_validate is True for c in configs)

    # One worker per robot: each sees exactly its own robot, in rank order.
    per_robot = worker_configs(cluster, "franka", [[0], [1]])
    assert [len(w) for w in per_robot] == [1, 1]
    assert [w[0]["robot_ip"] for w in per_robot] == ["10.20.30.40", "10.20.30.41"]

    # One worker owning both robots: it sees both hardware infos.
    both = worker_configs(cluster, "franka", [[0, 1]])
    assert len(both) == 1 and len(both[0]) == 2
    assert [c["robot_ip"] for c in both[0]] == ["10.20.30.40", "10.20.30.41"]


def run_create_single() -> None:
    """A single robot keeps the whole comma-separated camera list."""
    cluster = Cluster(cluster_cfg=cluster_cfg("franka", "Franka", []))

    configs = enumerated_configs(cluster, "Franka")
    assert len(configs) == 1
    assert configs[0].robot_ip == "10.20.30.40"
    assert configs[0].camera_serials == ["serialA", "serialB"]

    [worker] = worker_configs(cluster, "franka", [[0]])
    assert worker[0]["robot_ip"] == "10.20.30.40"
    assert worker[0]["camera_serials"] == ["serialA", "serialB"]


def run_explicit_fill() -> None:
    """Env fills omitted robot_ip in order; YAML values win and are kept."""
    configs = [
        # robot_ip set in YAML -> must be kept; env value for slot 0 ignored.
        {
            "node_rank": 0,
            "robot_ip": "192.168.0.5",
            "camera_serials": ["camL"],
            "disable_validate": True,
        },
        # robot_ip omitted -> filled from the env value for slot 1.
        {"node_rank": 0, "camera_serials": ["camR"], "disable_validate": True},
    ]
    cluster = Cluster(cluster_cfg=cluster_cfg("franka", "Franka", configs))

    enum = enumerated_configs(cluster, "Franka")
    assert [c.robot_ip for c in enum] == ["192.168.0.5", "10.20.30.41"]
    assert [c.camera_serials for c in enum] == [["camL"], ["camR"]]

    per_robot = worker_configs(cluster, "franka", [[0], [1]])
    assert [w[0]["robot_ip"] for w in per_robot] == ["192.168.0.5", "10.20.30.41"]
    assert [w[0]["camera_serials"] for w in per_robot] == [["camL"], ["camR"]]


def run_gim_create() -> None:
    """A different robot type is created from its own identifier env var."""
    cluster = Cluster(cluster_cfg=cluster_cfg("gimarm", "GimArm", []))

    configs = enumerated_configs(cluster, "GimArm")
    assert len(configs) == 1
    assert configs[0].can_interface == "can3"
    assert configs[0].camera_serials == ["gimcam"]

    [worker] = worker_configs(cluster, "gimarm", [[0]])
    assert worker[0]["can_interface"] == "can3"
    assert worker[0]["camera_serials"] == ["gimcam"]


def run_gating() -> None:
    """Only a shared field is set, so no Franka robot is created."""
    cluster = Cluster(cluster_cfg=cluster_cfg("franka", "Franka", []))
    assert enumerated_configs(cluster, "Franka") == []


def run_yaml_single() -> None:
    """Legacy path: a fully specified YAML config passes through unchanged."""
    configs = [
        {
            "node_rank": 0,
            "robot_ip": "10.1.1.1",
            "camera_serials": ["camYAML"],
            "gripper_connection": "/dev/ttyYAML",
            "disable_validate": True,
        }
    ]
    cluster = Cluster(cluster_cfg=cluster_cfg("franka", "Franka", configs))

    enum = enumerated_configs(cluster, "Franka")
    assert len(enum) == 1
    assert enum[0].robot_ip == "10.1.1.1"
    assert enum[0].camera_serials == ["camYAML"]
    assert enum[0].gripper_connection == "/dev/ttyYAML"

    [worker] = worker_configs(cluster, "franka", [[0]])
    assert worker[0]["robot_ip"] == "10.1.1.1"
    assert worker[0]["camera_serials"] == ["camYAML"]
    assert worker[0]["gripper_connection"] == "/dev/ttyYAML"


def run_yaml_multi() -> None:
    """Legacy path: several fully specified YAML configs are kept verbatim."""
    configs = [
        {
            "node_rank": 0,
            "robot_ip": "10.1.1.1",
            "camera_serials": ["camA"],
            "disable_validate": True,
        },
        {
            "node_rank": 0,
            "robot_ip": "10.1.1.2",
            "camera_serials": ["camB"],
            "disable_validate": True,
        },
    ]
    cluster = Cluster(cluster_cfg=cluster_cfg("franka", "Franka", configs))

    enum = enumerated_configs(cluster, "Franka")
    assert [c.robot_ip for c in enum] == ["10.1.1.1", "10.1.1.2"]
    assert [c.camera_serials for c in enum] == [["camA"], ["camB"]]

    per_robot = worker_configs(cluster, "franka", [[0], [1]])
    assert [w[0]["robot_ip"] for w in per_robot] == ["10.1.1.1", "10.1.1.2"]
    assert [w[0]["camera_serials"] for w in per_robot] == [["camA"], ["camB"]]


def run_yaml_dosw1() -> None:
    """Legacy path for another robot type: DOSW1 YAML config passes through."""
    configs = [
        {
            "node_rank": 0,
            "robot_url": "192.168.5.5",
            "left_arm_port": 50061,
            "camera_serials": ["dosCam"],
        }
    ]
    cluster = Cluster(cluster_cfg=cluster_cfg("dosw1", "DOSW1", configs))

    enum = enumerated_configs(cluster, "DOSW1")
    assert len(enum) == 1
    assert enum[0].robot_url == "192.168.5.5"
    assert enum[0].left_arm_port == 50061
    assert enum[0].camera_serials == ["dosCam"]

    [worker] = worker_configs(cluster, "dosw1", [[0]])
    assert worker[0]["robot_url"] == "192.168.5.5"
    assert worker[0]["left_arm_port"] == 50061
    assert worker[0]["camera_serials"] == ["dosCam"]


def _expect_enumeration_error(configs: list) -> None:
    """Boot a Franka group expecting the comma-count error during enumeration."""
    try:
        Cluster(cluster_cfg=cluster_cfg("franka", "Franka", configs))
    except Exception as exc:
        assert "comma-separated" in str(exc), str(exc)
        return
    raise AssertionError("cluster was built despite a comma-count mismatch")


def run_mismatch_ip() -> None:
    """The identifier (ROBOT_IP) value count disagrees with the config count."""
    _expect_enumeration_error(
        [
            {"node_rank": 0, "camera_serials": ["camL"], "disable_validate": True},
            {"node_rank": 0, "camera_serials": ["camR"], "disable_validate": True},
        ]
    )


def run_mismatch_secondary() -> None:
    """ROBOT_IP count is fine, but a secondary field (cameras) disagrees."""
    # Cameras are omitted so the (mismatched) CAMERA_SERIALS env is read;
    # the configs are kept distinct via their gripper connection.
    _expect_enumeration_error(
        [
            {"node_rank": 0, "gripper_connection": "/dev/a", "disable_validate": True},
            {"node_rank": 0, "gripper_connection": "/dev/b", "disable_validate": True},
        ]
    )


RUNNERS = {
    "create_multi": run_create_multi,
    "create_single": run_create_single,
    "explicit_fill": run_explicit_fill,
    "gim_create": run_gim_create,
    "gating": run_gating,
    "mismatch_low": run_mismatch_ip,
    "mismatch_high": run_mismatch_ip,
    "mismatch_secondary": run_mismatch_secondary,
    "yaml_single": run_yaml_single,
    "yaml_multi": run_yaml_multi,
    "yaml_dosw1": run_yaml_dosw1,
}


if __name__ == "__main__":
    RUNNERS[MODE]()
    print(f"{MODE}:OK")
