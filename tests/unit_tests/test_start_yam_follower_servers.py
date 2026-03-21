# Copyright 2026 Ying-Chun Lee
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

"""Unit tests for the YAM follower server launcher helpers."""

from pathlib import Path

from scripts import start_yam_follower_servers as follower_servers


def test_build_server_command_includes_gripper_limits_when_supported():
    command = follower_servers.build_server_command(
        script_path=Path("/tmp/follower.py"),
        can_channel="can_left",
        server_port=1235,
        gripper_open=0.1,
        gripper_close=-0.2,
        supports_gripper_limits=True,
    )

    assert "--can_channel" in command
    assert "--gripper_open_limit" in command
    assert "--gripper_close_limit" in command


def test_build_server_command_omits_gripper_limits_when_not_supported():
    command = follower_servers.build_server_command(
        script_path=Path("/tmp/follower.py"),
        can_channel="can_right",
        server_port=1234,
        gripper_open=0.1,
        gripper_close=-0.2,
        supports_gripper_limits=False,
    )

    assert "--gripper_open_limit" not in command
    assert "--gripper_close_limit" not in command


def test_find_follower_server_script_accepts_explicit_existing_path(tmp_path):
    script_path = tmp_path / "follower.py"
    script_path.write_text("#!/usr/bin/env python3\n")

    found = follower_servers.find_follower_server_script(str(script_path))

    assert found == script_path.resolve()
