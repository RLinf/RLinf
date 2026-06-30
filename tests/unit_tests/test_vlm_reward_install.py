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
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_qwen_vlm_reward_sglang_requirements_are_pinned():
    requirements = (
        REPO_ROOT / "requirements/embodied/models/qwen_vlm_reward_sglang.txt"
    ).read_text()
    requirement_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "sglang==0.5.5.post3" in requirement_lines
    assert "torch==2.8.0" in requirement_lines
    assert "torchvision==0.23.0" in requirement_lines
    assert "torchaudio==2.8.0" in requirement_lines
    assert "qwen-vl-utils" in requirement_lines
    assert "transformers==4.57.1" in requirement_lines
    assert "tokenizers>=0.22,<0.23" in requirement_lines
    assert "xgrammar==0.1.25" in requirement_lines
    assert "sglang==0.5.4" not in requirement_lines
    assert not any(line.startswith("sglang-router") for line in requirement_lines)
    assert not any(line.startswith("sgl-kernel") for line in requirement_lines)


def test_qwen_reward_runtime_is_installed_in_qwen3_vl_env():
    install_script = (REPO_ROOT / "requirements/install.sh").read_text()

    assert "install_qwen_vlm_reward_deps" not in install_script
    assert "install_qwen_vlm_reward_sglang_deps" in install_script
    assert "uv pip install --no-config" in install_script
    assert "install_qwen_vlm_reward_client_deps" not in install_script
    assert "--vlm-reward" not in install_script
    assert "VLM_REWARD" not in install_script
    assert "apply_openpi_transformers_4_57_1_patch" not in install_script
    assert "openpi_transformers_4_57_1/transformers" not in install_script
    assert "Expected sglang=={expected}" in install_script


def test_qwen3_vl_model_installs_sglang_reward_runtime():
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
    assert "install_qwen_vlm_reward_client_deps" not in body
    assert "install_qwen_vlm_reward_deps" not in body


def test_flash_attn_install_does_not_resolve_torch_dependencies():
    install_script = (REPO_ROOT / "requirements/install.sh").read_text()

    flash_attn_body_match = re.search(
        r"install_flash_attn\(\) \{\n(?P<body>.*?)\n\}\n\ninstall_apex",
        install_script,
        flags=re.S,
    )
    assert flash_attn_body_match is not None

    body = flash_attn_body_match.group("body")
    flash_attn_installs = [
        line.strip()
        for line in body.splitlines()
        if "uv pip install" in line and "flash-attn" in line
    ]

    assert flash_attn_installs
    assert all("--no-deps" in line for line in flash_attn_installs)
    assert 'uv pip install --no-deps "${base_url}/${wheel_name}"' in body


def test_agentic_sglang_extra_keeps_base_runtime():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    agentic_sglang = set(
        pyproject["project"]["optional-dependencies"]["agentic-sglang"]
    )
    overrides = set(pyproject["tool"]["uv"]["override-dependencies"])

    assert "sglang[all]==0.4.6.post5" in agentic_sglang
    assert "sglang[all]==0.5.5.post3" not in agentic_sglang
    assert "torch==2.6.0" in overrides
    assert "torchvision==0.21.0" in overrides
    assert "torchaudio==2.6.0" in overrides
    assert "xgrammar==0.1.19" in overrides
