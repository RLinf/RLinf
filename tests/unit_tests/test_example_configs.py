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

"""Every reasoning example config must compose and resolve.

``examples/reasoning/main_grpo.py`` calls
``OmegaConf.to_container(cfg, resolve=True)`` right after ``validate_cfg``, so
an interpolation that points at a missing key crashes the run at startup.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import rlinf.utils.omega_resolver  # noqa: F401  (registers custom resolvers)

REPO_ROOT = Path(__file__).resolve().parents[2]
REASONING_MATH_CONFIG_DIR = REPO_ROOT / "examples" / "reasoning" / "config" / "math"
REASONING_MATH_CONFIGS = sorted(
    path.stem for path in REASONING_MATH_CONFIG_DIR.glob("*.yaml")
)


@pytest.mark.parametrize("config_name", REASONING_MATH_CONFIGS)
def test_reasoning_math_example_config_resolves(config_name):
    with initialize_config_dir(
        config_dir=str(REASONING_MATH_CONFIG_DIR), version_base="1.1"
    ):
        cfg = compose(config_name=config_name)
    OmegaConf.to_container(cfg, resolve=True)
