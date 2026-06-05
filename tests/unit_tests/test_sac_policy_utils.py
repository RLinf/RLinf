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

from rlinf.algorithms.sac_policy_utils import (
    resolve_sac_q_head_type,
    validate_fsdpsac_edac_support,
    validate_sac_crossq_support,
)


def test_resolve_sac_q_head_type_defaults_to_standard_head():
    assert resolve_sac_q_head_type({}, {}) == "default"


def test_resolve_sac_q_head_type_accepts_matching_algorithm_and_model_values():
    assert (
        resolve_sac_q_head_type(
            {"q_head_type": "crossq"},
            {"q_head_type": "crossq"},
        )
        == "crossq"
    )


def test_resolve_sac_q_head_type_falls_back_to_model_default_value():
    assert resolve_sac_q_head_type({}, {"q_head_type": "default"}) == "default"


def test_resolve_sac_q_head_type_rejects_model_only_crossq():
    with pytest.raises(ValueError, match="algorithm.q_head_type.*crossq"):
        resolve_sac_q_head_type({}, {"q_head_type": "crossq"})


def test_resolve_sac_q_head_type_allows_algorithm_only_default_value():
    assert resolve_sac_q_head_type({"q_head_type": "default"}, {}) == "default"


def test_resolve_sac_q_head_type_rejects_algorithm_only_crossq():
    with pytest.raises(ValueError, match="actor.model.q_head_type.*crossq"):
        resolve_sac_q_head_type({"q_head_type": "crossq"}, {})


def test_resolve_sac_q_head_type_rejects_divergent_values():
    with pytest.raises(
        ValueError, match="algorithm.q_head_type.*actor.model.q_head_type"
    ):
        resolve_sac_q_head_type(
            {"q_head_type": "crossq"},
            {"q_head_type": "default"},
        )


@pytest.mark.parametrize(
    ("algorithm_cfg", "model_cfg"),
    [
        ({"q_head_type": "bogus"}, {}),
        ({}, {"q_head_type": "bogus"}),
        ({"q_head_type": "bogus"}, {"q_head_type": "bogus"}),
    ],
)
def test_resolve_sac_q_head_type_rejects_invalid_values(algorithm_cfg, model_cfg):
    with pytest.raises(ValueError, match="q_head_type.*bogus"):
        resolve_sac_q_head_type(algorithm_cfg, model_cfg)


@pytest.mark.parametrize("model_type", ["mlp_policy", "cnn_policy"])
def test_crossq_support_guard_allows_supported_model_types(model_type):
    validate_sac_crossq_support(
        {"q_head_type": "crossq"},
        {"q_head_type": "crossq", "model_type": model_type},
    )


def test_crossq_support_guard_rejects_unsupported_model_type():
    with pytest.raises(ValueError, match="CrossQ.*flow_policy|flow_policy.*CrossQ"):
        validate_sac_crossq_support(
            {"q_head_type": "crossq"},
            {"q_head_type": "crossq", "model_type": "flow_policy"},
        )


def test_crossq_support_guard_rejects_missing_model_type():
    with pytest.raises(ValueError, match="CrossQ.*model_type"):
        validate_sac_crossq_support(
            {"q_head_type": "crossq"},
            {"q_head_type": "crossq"},
        )


def test_fsdpsac_edac_guard_allows_default_disabled_config():
    validate_fsdpsac_edac_support({"edac_eta": 0.0})
    validate_fsdpsac_edac_support({})


def test_fsdpsac_edac_guard_rejects_positive_eta():
    with pytest.raises(NotImplementedError, match="FSDP.*edac_eta|edac_eta.*FSDP"):
        validate_fsdpsac_edac_support({"edac_eta": 0.1})
