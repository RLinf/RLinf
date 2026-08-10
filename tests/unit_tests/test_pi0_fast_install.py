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

import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPO_ROOT / "requirements" / "install.sh"
PI0_FAST_REQUIREMENTS = (
    REPO_ROOT / "requirements" / "embodied" / "models" / "pi0_fast.txt"
)


def _source_install_script(command: str) -> str:
    source = INSTALL_SCRIPT.read_text().rsplit('main "$@"', maxsplit=1)[0]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as source_file:
        source_file.write(source)
        source_file.flush()
        result = subprocess.run(
            ["bash", "-c", f"source {source_file.name}; {command}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


def test_install_script_is_valid_bash():
    subprocess.run(["bash", "-n", INSTALL_SCRIPT], check=True)


def test_pi0_fast_uses_isolated_validated_runtime_by_default():
    output = _source_install_script(
        "parse_args embodied --model pi0_fast --env libero; "
        "configure_pi0_fast_runtime; "
        'printf \'%s|%s|%s\' "$VENV_DIR" "$PYTHON_VERSION" "$TORCH_VERSION"'
    )

    assert output.endswith(".venv-pi0-fast|3.12.12|2.11.0")


def test_pi0_fast_respects_explicit_runtime_overrides():
    output = _source_install_script(
        "parse_args embodied --model pi0_fast --env libero "
        "--venv custom-venv --python 3.12.9 --torch 2.11.1; "
        "configure_pi0_fast_runtime; "
        'printf \'%s|%s|%s\' "$VENV_DIR" "$PYTHON_VERSION" "$TORCH_VERSION"'
    )

    assert output.endswith("custom-venv|3.12.9|2.11.1")


def test_pi0_fast_requirement_pins_validated_lerobot_commit():
    requirements = PI0_FAST_REQUIREMENTS.read_text()

    assert (
        "huggingface/lerobot.git@8a74e0ac6d01706d67fddfed682a09d694d9c8c0"
        in requirements
    )
    install_script = INSTALL_SCRIPT.read_text()
    assert (
        'PI0_FAST_TRANSFORMERS_VERSION="${PI0_FAST_TRANSFORMERS_VERSION:-5.5.4}"'
        in install_script
    )
