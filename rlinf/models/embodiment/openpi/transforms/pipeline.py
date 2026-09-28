# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from rlinf.utils.logging import get_logger

logger = get_logger()


def norm_stats_path_from_data_kwargs(
    data_kwargs: dict[str, Any] | None,
) -> str | None:
    """Return ``openpi_data.norm_stats_path``, or ``None`` if it is unset."""
    if not data_kwargs:
        return None
    raw = data_kwargs.get("norm_stats_path")
    if raw is None:
        return None
    path = str(raw).strip()
    return path or None


def select_openpi_norm_stats(
    norm_stats: Any,
    *,
    norm_stats_path: str | None,
) -> Any:
    """Choose stats already loaded by OpenPI ``DataConfigFactory.create``.

    An explicit ``norm_stats_path`` must resolve; otherwise warn and keep the
    OpenPI default (``None`` skips ``Normalize``).
    """
    if not norm_stats_path:
        logger.warning(
            "openpi: actor.model.openpi_data.norm_stats_path is empty; "
            "loading norm stats with the OpenPI default "
            "({assets_dir}/{asset_id}/norm_stats.json from the TrainConfig)."
        )
        return norm_stats
    if norm_stats is None:
        raise FileNotFoundError(
            "openpi: norm_stats not found at "
            f"{norm_stats_path}. Set openpi_data.norm_stats_path to a valid "
            "norm_stats.json file or directory."
        )
    return norm_stats


def select_so101_norm_stats(norm_stats: Any) -> Any:
    """Keep only the keys used by the SO-101 OpenPI transform pipeline.

    Some exported stats files contain both singular/plural action aliases and
    both ``state``/``observation.state`` aliases. OpenPI's tree transforms
    interpret every stats key as a selector, so retaining aliases that are not
    present in the transformed sample makes normalization fail at runtime.
    SO-101's transforms use exactly ``state`` and ``actions``.
    """
    if not isinstance(norm_stats, Mapping):
        return norm_stats
    selected = {
        key: value
        for key, value in norm_stats.items()
        if key in {"state", "actions"}
    }
    # SO-101's RLinf/OpenPI contract is degree-like LeRobot values scaled by
    # 0.01 (including the gripper) before normalization.  Accepting a raw
    # degree/0-100 stats file here silently produces unsafe actions at eval:
    # the policy sees a different coordinate system from the one used by the
    # runtime adapter.  Fail before a hardware worker can be started.
    for key in ("state", "actions"):
        stats = selected.get(key)
        if not isinstance(stats, Mapping):
            continue
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            continue
        values = list(mean)[:6] + list(std)[:6]
        if any(abs(float(value)) > 3.0 for value in values):
            raise ValueError(
                "SO-101 norm stats are in raw degree/0-100 units; expected "
                "the canonical degree-like*0.01 contract for state/actions. "
                "Regenerate norm_stats.json after the SO-101 loader scaling "
                "and retrain the checkpoint before real-robot evaluation."
            )
    return selected


def build_openpi_transforms(
    model_path: str,
    config_name: str,
    data_kwargs: dict[str, Any] | None = None,
    discrete_state_input: bool | None = None,
) -> tuple[Sequence, Sequence]:
    """Build ``(input_transforms, output_transforms)`` for ``config_name``.

    Returns two lists ready for :func:`openpi.transforms.compose`:

    * input:  ``[InjectDefaultPrompt(None), *data.inputs, Normalize, *model.inputs]``
    * output: ``[*model.outputs, Unnormalize, *data.outputs]``

    Set ``data_kwargs["norm_stats_path"]`` (YAML ``openpi_data.norm_stats_path``)
    to pin the stats file. When it is empty, OpenPI loads
    ``{assets_dir}/{asset_id}/norm_stats.json`` from the TrainConfig (after
    ``model_path`` is applied) and missing files skip normalization.
    """
    import openpi.transforms as transforms

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    train_config = get_openpi_config(
        config_name, model_path=str(model_path), data_kwargs=data_kwargs
    )
    upstream_model_config = train_config.model
    if discrete_state_input is not None:
        import dataclasses

        upstream_model_config = dataclasses.replace(
            upstream_model_config,
            discrete_state_input=bool(discrete_state_input),
        )
    data_config = train_config.data.create(
        train_config.assets_dirs, upstream_model_config
    )
    norm_stats = select_openpi_norm_stats(
        data_config.norm_stats,
        norm_stats_path=norm_stats_path_from_data_kwargs(data_kwargs),
    )
    if config_name == "pi05_so101_joint":
        norm_stats = select_so101_norm_stats(norm_stats)

    input_transforms = [
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    return input_transforms, output_transforms
