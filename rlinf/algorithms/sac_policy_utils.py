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


def _validate_sac_q_head_type(field_name: str, q_head_type: object) -> None:
    if q_head_type is None:
        return
    if not isinstance(q_head_type, str) or q_head_type not in SAC_Q_HEAD_TYPES:
        supported_types = ", ".join(sorted(SAC_Q_HEAD_TYPES))
        raise ValueError(
            f"{field_name} must be one of {supported_types}, got {q_head_type!r}."
        )


def resolve_sac_q_head_type(
    algorithm_cfg: Mapping[str, object],
    model_cfg: Mapping[str, object],
) -> str:
    """Resolve the SAC Q-head type after validating its duplicated config.

    CrossQ changes both the training forward path and the model architecture,
    which are configured in separate sections. Both fields must therefore be
    explicitly set to ``crossq`` before CrossQ can be enabled.

    Args:
        algorithm_cfg: SAC algorithm configuration.
        model_cfg: Actor model configuration.

    Returns:
        The resolved Q-head type, either ``default`` or ``crossq``.

    Raises:
        ValueError: If either field is invalid, the fields disagree, or the
            selected model cannot construct a CrossQ critic.
    """
    algorithm_q_head_type = algorithm_cfg.get("q_head_type")
    model_q_head_type = model_cfg.get("q_head_type")
    _validate_sac_q_head_type("algorithm.q_head_type", algorithm_q_head_type)
    _validate_sac_q_head_type("actor.model.q_head_type", model_q_head_type)

    if algorithm_q_head_type is not None and model_q_head_type is not None:
        if algorithm_q_head_type != model_q_head_type:
            raise ValueError(
                "algorithm.q_head_type and actor.model.q_head_type must match "
                f"when both are set, got {algorithm_q_head_type!r} and "
                f"{model_q_head_type!r}."
            )
        q_head_type = algorithm_q_head_type
    elif algorithm_q_head_type is not None:
        if algorithm_q_head_type == "crossq":
            raise ValueError(
                "actor.model.q_head_type must be set to 'crossq' when "
                "algorithm.q_head_type is 'crossq', because CrossQ changes "
                "the critic model architecture."
            )
        q_head_type = algorithm_q_head_type
    elif model_q_head_type is not None:
        if model_q_head_type == "crossq":
            raise ValueError(
                "algorithm.q_head_type must be set to 'crossq' when "
                "actor.model.q_head_type is 'crossq', because CrossQ changes "
                "the SAC training path."
            )
        q_head_type = model_q_head_type
    else:
        q_head_type = "default"

    if q_head_type == "crossq":
        model_type = model_cfg.get("model_type")
        if model_type not in SAC_CROSSQ_MODEL_TYPES:
            supported_models = ", ".join(sorted(SAC_CROSSQ_MODEL_TYPES))
            raise ValueError(
                f"CrossQ is not supported for actor.model.model_type={model_type!r}. "
                f"Supported CrossQ model types: {supported_models}."
            )

    return q_head_type
