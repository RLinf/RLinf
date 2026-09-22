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


def build_openpi_transforms(
    model_path: str,
    config_name: str,
    data_kwargs: dict[str, Any] | None = None,
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
    data_config = train_config.data.create(
        train_config.assets_dirs, upstream_model_config
    )
    norm_stats = select_openpi_norm_stats(
        data_config.norm_stats,
        norm_stats_path=norm_stats_path_from_data_kwargs(data_kwargs),
    )

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
