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

import pytest

from rlinf.algorithms.sac_policy_utils import resolve_sac_q_head_type


def test_resolve_sac_q_head_type_defaults_to_standard_head():
    assert resolve_sac_q_head_type({}, {}) == "default"


@pytest.mark.parametrize(
    ("algorithm_cfg", "model_cfg"),
    [({"q_head_type": "default"}, {}), ({}, {"q_head_type": "default"})],
)
def test_resolve_sac_q_head_type_preserves_one_sided_standard_configs(
    algorithm_cfg, model_cfg
):
    assert resolve_sac_q_head_type(algorithm_cfg, model_cfg) == "default"


@pytest.mark.parametrize("model_type", ["mlp_policy", "cnn_policy"])
def test_resolve_sac_q_head_type_accepts_supported_crossq_models(model_type):
    assert (
        resolve_sac_q_head_type(
            {"q_head_type": "crossq"},
            {"q_head_type": "crossq", "model_type": model_type},
        )
        == "crossq"
    )


@pytest.mark.parametrize(
    ("algorithm_cfg", "model_cfg"),
    [
        ({"q_head_type": "crossq"}, {"model_type": "mlp_policy"}),
        ({}, {"q_head_type": "crossq", "model_type": "mlp_policy"}),
        (
            {"q_head_type": "crossq"},
            {"q_head_type": "default", "model_type": "mlp_policy"},
        ),
    ],
)
def test_resolve_sac_q_head_type_rejects_incomplete_or_divergent_crossq(
    algorithm_cfg, model_cfg
):
    with pytest.raises(ValueError, match="q_head_type"):
        resolve_sac_q_head_type(algorithm_cfg, model_cfg)


@pytest.mark.parametrize(
    ("algorithm_cfg", "model_cfg"),
    [
        ({"q_head_type": "invalid"}, {}),
        ({}, {"q_head_type": "invalid"}),
        ({"q_head_type": []}, {}),
    ],
)
def test_resolve_sac_q_head_type_rejects_invalid_values(algorithm_cfg, model_cfg):
    with pytest.raises(ValueError, match="q_head_type"):
        resolve_sac_q_head_type(algorithm_cfg, model_cfg)


@pytest.mark.parametrize("model_type", [None, "flow_policy"])
def test_resolve_sac_q_head_type_rejects_unsupported_crossq_model(model_type):
    with pytest.raises(ValueError, match="CrossQ.*model_type"):
        resolve_sac_q_head_type(
            {"q_head_type": "crossq"},
            {"q_head_type": "crossq", "model_type": model_type},
        )
