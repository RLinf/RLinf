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

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_qwen_vlm_reward_sglang_requirements_are_pinned():
    requirements = (
        REPO_ROOT / "requirements/embodied/models/qwen_vlm_reward_sglang.txt"
    ).read_text()

    assert "qwen-vl-utils" in requirements
    assert "transformers==4.57.1" in requirements
    assert "tokenizers>=0.22,<0.23" in requirements
    assert "sglang==0.5.4" in requirements


def test_qwen_reward_runtime_is_installed_in_qwen3_vl_env():
    install_script = (REPO_ROOT / "requirements/install.sh").read_text()

    assert "install_qwen_vlm_reward_deps" not in install_script
    assert "install_qwen_vlm_reward_sglang_deps" in install_script
    assert "--vlm-reward" not in install_script
    assert "VLM_REWARD" not in install_script
    assert "apply_openpi_transformers_4_57_1_patch" not in install_script
    assert "openpi_transformers_4_57_1/transformers" not in install_script
    assert "sglang=={expected}" in install_script


def test_qwen3_vl_model_can_install_sglang_reward_runtime():
    install_script = (REPO_ROOT / "requirements/install.sh").read_text()

    match = re.search(
        r"install_qwen3_vl_model\(\) \{\n(?P<body>.*?)\n\}\n\ninstall_franka_realworld_env",
        install_script,
        flags=re.S,
    )
    assert match is not None

    body = match.group("body")
    assert "create_and_sync_venv" in body
    assert "install_common_embodied_deps" in body
    assert "install_qwen_vlm_reward_sglang_deps" in body
    assert "install_qwen_vlm_reward_deps" not in body
