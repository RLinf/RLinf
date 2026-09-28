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

"""Static checks for the portable YAM PICO collection recipe."""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_ENV = _ROOT / "examples/embodiment/config/env/realworld_dual_yam_joint.yaml"
_COLLECT = (
    _ROOT / "examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml"
)


def _read(path):
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def test_yam_pico_example_requires_site_resources_and_exposes_three_views():
    env = _read(_ENV)
    collect = _read(_COLLECT)
    station = collect["cluster"]["node_groups"][0]["hardware"]["configs"][0]

    assert env["init_params"]["id"] == "DualYamJointEnv-v1"
    assert env["main_image_key"] == "top_rgb"
    assert env["override_cfg"]["reset"]["enabled"] is False
    assert [camera["name"] for camera in station["cameras"]] == [
        "top_rgb",
        "left_rgb",
        "right_rgb",
    ]
    recording_fps = collect["env"]["eval"]["data_collection"]["fps"]
    assert env["override_cfg"]["step_frequency"] == recording_fps
    assert all(camera["fps"] == recording_fps for camera in station["cameras"])
    assert station["left_follower"]["channel"] == "${oc.env:YAM_LEFT_FOLLOWER_CAN}"
    assert station["right_follower"]["channel"] == "${oc.env:YAM_RIGHT_FOLLOWER_CAN}"
    for camera, side in zip(station["cameras"], ("TOP", "LEFT", "RIGHT"), strict=True):
        assert camera["serial"] == f"${{oc.env:YAM_{side}_CAMERA_SERIAL}}"
    assert "left_leader" not in station and "right_leader" not in station


def test_yam_pico_recipe_records_accepted_actions_without_automatic_park():
    config = _read(_COLLECT)
    eval_config = config["env"]["eval"]
    pico = eval_config["pico"]
    data = eval_config["data_collection"]

    assert eval_config["teleop"] == "yam_pico"
    assert pico["zmq_addr"] == "${oc.env:YAM_PICO_ZMQ_ADDR}"
    assert pico["left"]["operator_to_robot_yaw"] == (
        "${oc.env:YAM_LEFT_OPERATOR_TO_ROBOT_YAW}"
    )
    assert pico["right"]["operator_to_robot_yaw"] == (
        "${oc.env:YAM_RIGHT_OPERATOR_TO_ROBOT_YAW}"
    )
    assert pico["record_button"] == "right_menu_button"
    assert pico["discard_button"] == "left_menu_button"
    assert not eval_config["override_cfg"]["park_on_close"]["enabled"]
    assert not eval_config["override_cfg"]["reset"]["enabled"]
    assert data["enabled"] and data["streaming"]
    assert data["export_format"] == "lerobot"
    assert not data["only_success"]


def test_yam_pico_recipe_hydra_composes_with_deployment_values(monkeypatch):
    """The public entry config must resolve with synthetic site values."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("EMBODIED_PATH", str(_ROOT / "examples/embodiment"))
    for key in (
        "YAM_LEFT_FOLLOWER_CAN",
        "YAM_RIGHT_FOLLOWER_CAN",
        "YAM_TOP_CAMERA_SERIAL",
        "YAM_LEFT_CAMERA_SERIAL",
        "YAM_RIGHT_CAMERA_SERIAL",
        "YAM_PICO_ZMQ_ADDR",
        "YAM_LEFT_OPERATOR_TO_ROBOT_YAW",
        "YAM_RIGHT_OPERATOR_TO_ROBOT_YAW",
    ):
        monkeypatch.setenv(key, "0" if key.endswith("YAW") else f"test-{key}")

    with initialize_config_dir(
        config_dir=str(_ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        config = compose(config_name="realworld_dual_yam_collect_data_pico")
    resolved = OmegaConf.to_container(config, resolve=True)

    assert resolved["env"]["eval"]["teleop"] == "yam_pico"
    assert resolved["env"]["eval"]["init_params"]["id"] == "DualYamJointEnv-v1"
    assert resolved["cluster"]["num_nodes"] == 1
