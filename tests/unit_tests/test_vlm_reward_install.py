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


def test_openpi_sglang_install_flag_reapplies_transformers_patch():
    install_script = (REPO_ROOT / "requirements/install.sh").read_text()

    assert "--vlm-reward-sglang" in install_script
    assert "VLM_REWARD_SGLANG=1" in install_script
    assert "install_qwen_vlm_reward_sglang_deps" in install_script
    assert "apply_openpi_transformers_patch" in install_script
    assert "sglang=={expected}" in install_script
    assert (
        "--vlm-reward-sglang is supported only with --model openpi or --model qwen_vlm_reward"
        in install_script
    )

    match = re.search(
        r"install_openpi_model\(\) \{\n(?P<body>.*?)\n\}\n\ninstall_starvla_model",
        install_script,
        flags=re.S,
    )
    assert match is not None

    body = match.group("body")
    assert "install_qwen_vlm_reward_sglang_deps" in body
    assert "apply_openpi_transformers_patch" in body
    assert "install_qwen_vlm_reward_model" not in body

    sglang_branch = re.search(
        r'if \[ "\$VLM_REWARD_SGLANG" -eq 1 \]; then\n(?P<body>.*?)\n\s*fi',
        body,
        flags=re.S,
    )
    assert sglang_branch is not None

    sglang_branch_body = sglang_branch.group("body")
    assert "install_qwen_vlm_reward_sglang_deps" in sglang_branch_body
    assert sglang_branch_body.count("apply_openpi_transformers_patch") == 1
    assert "create_and_sync_venv" not in sglang_branch_body


def test_common_embodied_deps_skip_hf_reward_when_sglang_reward_is_selected():
    install_script = (REPO_ROOT / "requirements/install.sh").read_text()
    match = re.search(
        r"install_common_embodied_deps\(\) \{\n(?P<body>.*?)\n\}\n\ninstall_openvla_model",
        install_script,
        flags=re.S,
    )
    assert match is not None

    body = match.group("body")
    assert (
        'if [ "$VLM_REWARD" -eq 1 ] && [ "$VLM_REWARD_SGLANG" -eq 0 ]; then'
        in body
    )
    assert "install_qwen_vlm_reward_deps" in body
    assert "install_qwen_vlm_reward_sglang_deps" not in body


def test_qwen_vlm_reward_can_install_sglang_without_env():
    install_script = (REPO_ROOT / "requirements/install.sh").read_text()

    assert (
        '[ "$MODEL" != "dreamzero" ] && [ "$MODEL" != "qwen_vlm_reward" ]'
        in install_script
    )

    match = re.search(
        r"install_qwen_vlm_reward_model\(\) \{\n(?P<body>.*?)\n\}\n\ninstall_gr00t_model",
        install_script,
        flags=re.S,
    )
    assert match is not None

    body = match.group("body")
    light_branch = re.search(
        r'\n\s*""\)\n(?P<body>.*?)\n\s*;;',
        body,
        flags=re.S,
    )
    assert light_branch is not None

    light_branch_body = light_branch.group("body")
    assert "create_and_sync_venv" in light_branch_body
    assert (
        "uv sync --extra embodied --active $NO_INSTALL_RLINF_CMD" in light_branch_body
    )
    assert "install_common_embodied_deps" not in light_branch_body
    assert "install_maniskill_libero_env" not in light_branch_body


def test_vlm_reward_docs_describe_openpi_same_venv_install():
    doc_paths = [
        REPO_ROOT
        / "docs/source-en/rst_resources/examples/embodied/maniskill_vlm_reward.rst",
        REPO_ROOT
        / "docs/source-zh/rst_resources/examples/embodied/maniskill_vlm_reward.rst",
        REPO_ROOT / "docs/source-en/rst_source/start/installation.rst",
        REPO_ROOT / "docs/source-zh/rst_source/start/installation.rst",
    ]

    stale_phrases = [
        "keep it separate from the default Hugging Face reward install",
        "请使用默认 Hugging Face reward\n安装路径",
    ]

    for doc_path in doc_paths:
        text = doc_path.read_text()
        assert "--model openpi --env maniskill_libero --vlm-reward" in text
        assert "--model openpi --env maniskill_libero --vlm-reward-sglang" in text
        assert "--model qwen_vlm_reward --vlm-reward" in text
        assert "--model qwen_vlm_reward --vlm-reward-sglang" in text
        assert "transformers==4.57.1" in text
        for stale_phrase in stale_phrases:
            assert stale_phrase not in text
