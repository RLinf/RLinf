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


def build_openpi_transforms(
    model_path: str,
    config_name: str,
    data_kwargs: dict[str, Any] | None = None,
    *,
    norm_stats_dir: str | None = None,
    norm_stats_asset_id: str | None = None,
) -> tuple[Sequence, Sequence]:
    """Build ``(input_transforms, output_transforms)`` for ``config_name``.

    Returns two lists ready for :func:`openpi.transforms.compose`, matching
    ``rlinf/models/embodiment/openpi/__init__.py`` exactly:

    * input:  ``[InjectDefaultPrompt(None), *data.inputs, Normalize, *model.inputs]``
    * output: ``[*model.outputs, Unnormalize, *data.outputs]``

    Norm stats resolve from ``{norm_stats_dir}/{asset_id}/norm_stats.json`` when
    ``norm_stats_dir`` is given, else from the checkpoint dir via
    ``data_config.asset_id`` (``{model_path}/{asset_id}/norm_stats.json`` — the
    same canonical stats the original openpi path resolves). Eval / RL leave
    ``norm_stats_dir`` unset (their checkpoint bundles the stats); the BEHAVIOR
    SFT loader passes the experiment's ``assets_dir`` + ``asset_id`` so it reads
    the exact same ``norm_stats.json`` the old SFT path did (the SFT *base*
    checkpoint bundles no stats).
    """
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    train_config = get_openpi_config(
        config_name, model_path=str(model_path), data_kwargs=data_kwargs
    )
    upstream_model_config = train_config.model

    data_config = train_config.data.create(
        train_config.assets_dirs, upstream_model_config
    )

    asset_id = norm_stats_asset_id or data_config.asset_id
    if asset_id is None:
        raise ValueError("asset_id is required to load norm_stats.")
    stats_dir = (
        norm_stats_dir
        if norm_stats_dir is not None
        else download.maybe_download(str(model_path))
    )
    norm_stats = _checkpoints.load_norm_stats(stats_dir, asset_id)
    if norm_stats is None:
        raise FileNotFoundError(
            f"openpi_pytorch: norm_stats not found at {stats_dir}/{asset_id}/"
            "norm_stats.json. For eval/RL the checkpoint dir must bundle them; "
            "for SFT set actor.model.openpi.assets_dir/asset_id to the stats dir."
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
