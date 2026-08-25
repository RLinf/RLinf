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

"""Cosmos3 SFT dataloader builder.

The batch structure ``OmniMoTModel.training_step`` consumes (per-sample grouped
lists for video / text_token_ids / action, packed under a token budget) is
produced by cosmos's own ``PackingDataLoader``.
"""

from __future__ import annotations

import contextlib
import os

from rlinf.utils.logging import get_logger

logger = get_logger()

# Now the cosmos3 model just support libero in RLinf.
_DATA_TYPE_TO_EXPERIMENT = {
    "libero_10": "action_policy_libero_nano",  # libero_10 task, Default Settings
    "libero_all": "action_policy_libero_all_nano",  # libero_all task
}

_VALID_DATA_TYPES = frozenset(_DATA_TYPE_TO_EXPERIMENT.keys())


def _detect_data_type(data_paths: str) -> str:
    """Auto-detect data_type from the dataset directory structure."""
    import os as _os

    _LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    if all(_os.path.isdir(_os.path.join(data_paths, s)) for s in _LIBERO_SUITES):
        return "libero_all"
    return "libero_10"


def build_cosmos3_sft_dataloader(
    cfg,
    data_paths,
    eval_dataset: bool = False,
):
    """Build the Cosmos3 action-policy SFT dataloader."""
    from cosmos_framework.configs.base.config import make_config
    from cosmos_framework.utils.config_helper import override
    from cosmos_framework.utils.lazy_config import instantiate

    from rlinf.models.embodiment.cosmos3 import _cosmos_framework_root

    if eval_dataset:
        raise NotImplementedError("Cosmos3 SFT eval dataloader is not supported yet.")

    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    data_type = data_cfg.get("data_type", None)
    if data_type is None:
        # Auto-detect the data type from the dataset directory structure.
        data_type = _detect_data_type(str(data_paths))
        logger.info(
            f"Auto-detected data_type='{data_type}' from dataset at {data_paths}"
        )

    assert data_type in _VALID_DATA_TYPES, (
        f"Invalid data_type='{data_type}'. Valid: {sorted(_VALID_DATA_TYPES)}"
    )

    experiment_name = _DATA_TYPE_TO_EXPERIMENT[data_type]

    # Set the LIBERO_ROOT environment variable to the dataset directory.
    os.environ["LIBERO_ROOT"] = str(data_paths)

    max_samples_per_batch = model_cfg.get("max_samples_per_batch", None)
    num_workers = int(data_cfg.get("num_workers", 4))

    with contextlib.chdir(_cosmos_framework_root()):
        cosmos_data_cfg = make_config()
        cosmos_data_cfg = override(
            cosmos_data_cfg, ["--", f"experiment={experiment_name}"]
        )

        dl_cfg = cosmos_data_cfg.dataloader_train
        # Optional smoke/memory knob: shrink the packed batch (recipe default 128).
        if max_samples_per_batch is not None:
            dl_cfg.max_samples_per_batch = int(max_samples_per_batch)
        try:
            dl_cfg.dataloader.num_workers = num_workers
        except Exception:  # noqa: BLE001
            pass

        loader = instantiate(dl_cfg)

    try:
        num_batches = len(loader)
    except Exception:  # noqa: BLE001 - iterable loader may not expose len
        num_batches = -1

    logger.info(
        "Cosmos3 SFT dataloader (cosmos PackingDataLoader): "
        "data_type='%s' -> experiment='%s' dataset=%s "
        "len=%s max_samples_per_batch=%s",
        data_type,
        experiment_name,
        data_paths,
        num_batches,
        max_samples_per_batch,
    )
    return loader, {"num_samples": num_batches if num_batches > 0 else 1}
