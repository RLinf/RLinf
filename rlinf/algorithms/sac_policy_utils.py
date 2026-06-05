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

from collections.abc import Mapping

SAC_CROSSQ_MODEL_TYPES = frozenset({"cnn_policy", "mlp_policy"})
SAC_Q_HEAD_TYPES = frozenset({"crossq", "default"})


def _validate_sac_q_head_type(field_name: str, q_head_type: str | None) -> None:
    if q_head_type is None:
        return
    if q_head_type not in SAC_Q_HEAD_TYPES:
        supported_types = ", ".join(sorted(SAC_Q_HEAD_TYPES))
        raise ValueError(
            f"{field_name} must be one of {supported_types}, got {q_head_type!r}."
        )


def resolve_sac_q_head_type(
    algorithm_cfg: Mapping,
    model_cfg: Mapping,
) -> str:
    """Return the SAC Q-head type after checking config consistency."""
    algorithm_q_head_type = algorithm_cfg.get("q_head_type", None)
    model_q_head_type = model_cfg.get("q_head_type", None)
    _validate_sac_q_head_type("algorithm.q_head_type", algorithm_q_head_type)
    _validate_sac_q_head_type("actor.model.q_head_type", model_q_head_type)

    if algorithm_q_head_type is not None and model_q_head_type is not None:
        if algorithm_q_head_type != model_q_head_type:
            raise ValueError(
                "algorithm.q_head_type and actor.model.q_head_type must match "
                f"when both are set, got {algorithm_q_head_type!r} and "
                f"{model_q_head_type!r}."
            )
        return algorithm_q_head_type

    if model_q_head_type is not None:
        if model_q_head_type == "crossq":
            raise ValueError(
                "algorithm.q_head_type must be set to 'crossq' when "
                "actor.model.q_head_type is 'crossq', because CrossQ requires "
                "both the training path and critic model architecture."
            )
        return model_q_head_type
    if algorithm_q_head_type is not None:
        if algorithm_q_head_type == "crossq":
            raise ValueError(
                "actor.model.q_head_type must be set to 'crossq' when "
                "algorithm.q_head_type is 'crossq', because q_head_type controls "
                "the critic model architecture."
            )
        return algorithm_q_head_type
    return "default"


def validate_sac_crossq_support(
    algorithm_cfg: Mapping,
    model_cfg: Mapping,
) -> None:
    """Reject CrossQ configs whose actor model cannot build a CrossQ critic."""
    q_head_type = resolve_sac_q_head_type(algorithm_cfg, model_cfg)
    if q_head_type != "crossq":
        return

    model_type = model_cfg.get("model_type", None)
    if model_type not in SAC_CROSSQ_MODEL_TYPES:
        supported_models = ", ".join(sorted(SAC_CROSSQ_MODEL_TYPES))
        raise ValueError(
            f"CrossQ is not supported for actor.model.model_type={model_type!r}. "
            f"Supported CrossQ model types: {supported_models}."
        )


def validate_fsdpsac_edac_support(algorithm_cfg: Mapping) -> None:
    """Reject EDAC in the current FSDP SAC/RLPD worker."""
    edac_eta = float(algorithm_cfg.get("edac_eta", 0.0))
    if edac_eta <= 0.0:
        return

    raise NotImplementedError(
        "EDAC critic diversity regularization is not supported in the current "
        "FSDP SAC/RLPD worker because it requires second-order autograd through "
        "an FSDP-wrapped critic. Set algorithm.edac_eta to 0.0."
    )
