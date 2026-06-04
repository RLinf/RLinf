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

import ast
import json
from pathlib import Path
from typing import Any

from groot.vla.data.schema import DatasetMetadata
from groot.vla.data.transform.base import ComposedModalityTransform

from rlinf.data.datasets.dreamzero.data_transforms.base import (
    DreamZeroEmbodimentTransform,
    RolloutObsLayout,
    convert_rollout_env_obs_with_layout,
)
from rlinf.data.datasets.dreamzero.data_transforms.embodiment_tag import EmbodimentTag
from rlinf.data.datasets.dreamzero.data_transforms.franka_pnp import (
    FrankaPnpDataTransform,
)
from rlinf.data.datasets.dreamzero.data_transforms.libero_sim import (
    LiberoSimDataTransform,
)
from rlinf.data.datasets.dreamzero.data_transforms.oxe_droid import (
    OxeDroidDataTransform,
)

_EMBODIMENT_REGISTRY: dict[str, type[DreamZeroEmbodimentTransform]] = {
    LiberoSimDataTransform.TAG: LiberoSimDataTransform,
    OxeDroidDataTransform.TAG: OxeDroidDataTransform,
    FrankaPnpDataTransform.TAG: FrankaPnpDataTransform,
}

DEFAULT_EMBODIMENT_TAG_MAPPING: dict[str, dict[str, int]] = {
    tag: dict(cls.DEFAULT_TAG_MAPPING) for tag, cls in _EMBODIMENT_REGISTRY.items()
}

__all__ = [
    "RolloutObsLayout",
    "build_dreamzero_composed_transform",
    "collect_dreamzero_dataset_keys",
    "convert_rollout_env_obs",
    "embodiment_tag_mapping_for_embodiment",
    "load_dreamzero_dataset_metadata",
    "format_training_prompt",
    "normalize_instruction_text",
    "rollout_obs_layout_for_embodiment",
]


def _require_embodiment(tag: str) -> type[DreamZeroEmbodimentTransform]:
    try:
        return _EMBODIMENT_REGISTRY[tag]
    except KeyError:
        raise ValueError(
            f"Unsupported embodiment_tag {tag!r}. "
            f"Built-in tags: {sorted(_EMBODIMENT_REGISTRY)}. "
            "Register the class in _EMBODIMENT_REGISTRY."
        ) from None


def _language_keys_for_tag(tag: str) -> list[str]:
    modality = _require_embodiment(tag).get_modality_config()
    language_cfg = modality.get("language")
    if language_cfg is None:
        raise KeyError(f"Missing language ModalityConfig for {tag!r}")
    return [str(k) for k in language_cfg.modality_keys]


def normalize_instruction_text(raw: Any) -> str:
    """Decode a dataset ``text`` field into a lowercase instruction string."""
    if not isinstance(raw, str):
        return str(raw).lower()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return str(parsed[0]).lower()
        return str(parsed).lower()
    except (ValueError, SyntaxError, TypeError):
        return raw.lower()


def format_training_prompt(
    instruction: str,
    embodiment_id: int,
    embodiment_tag_mapping: dict[str, int],
) -> str:
    """Wrap a task instruction with embodiment-specific multi-view layout text for T5."""
    id_to_tag = {v: k for k, v in embodiment_tag_mapping.items()}
    tag = id_to_tag.get(embodiment_id)
    if tag is None:
        raise ValueError(
            f"Embodiment ID {embodiment_id} not found in embodiment_tag_mapping "
            f"{embodiment_tag_mapping!r}."
        )
    return _require_embodiment(tag).format_training_prompt(instruction)


def embodiment_tag_mapping_for_embodiment(
    tag: str,
    override: dict[str, int] | None = None,
) -> dict[str, int]:
    """Return embodiment tag -> projector id mapping for collate / DreamTransform."""
    if override is not None:
        return dict(override)
    return dict(_require_embodiment(tag).DEFAULT_TAG_MAPPING)


def rollout_obs_layout_for_embodiment(tag: str) -> RolloutObsLayout:
    """Return rollout observation layout for ``embodiment_tag``."""
    return _require_embodiment(tag).ROLLOUT_OBS_LAYOUT


def convert_rollout_env_obs(
    embodiment_tag: str, env_obs: dict[str, Any]
) -> dict[str, Any]:
    """Convert RLinf rollout ``env_obs`` to DreamZero modality keys for inference."""
    tag = str(embodiment_tag)
    cls = _require_embodiment(tag)
    language_key = _language_keys_for_tag(tag)[0]
    return convert_rollout_env_obs_with_layout(
        env_obs, cls.ROLLOUT_OBS_LAYOUT, language_key
    )


def collect_dreamzero_dataset_keys(
    data_transform: Any,
    embodiment_tag: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Collect video/state/action keys from the transform chain and language keys from embodiment config."""
    video_keys: list[str] = []
    state_keys: list[str] = []
    action_keys: list[str] = []
    for transform in getattr(data_transform, "transforms", []):
        video_keys.extend(getattr(transform, "video_concat_order", []) or [])
        state_keys.extend(getattr(transform, "state_concat_order", []) or [])
        action_keys.extend(getattr(transform, "action_concat_order", []) or [])
    language_keys = _language_keys_for_tag(embodiment_tag)
    return video_keys, state_keys, action_keys, language_keys


def load_dreamzero_dataset_metadata(cfg: Any) -> DatasetMetadata:
    """Load :class:`DatasetMetadata` for ``embodiment_tag``."""
    tag = cfg.embodiment_tag
    if cfg.get("metadata_json_path", None):
        path = Path(str(cfg["metadata_json_path"])).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"metadata_json_path is not a file: {path}")
    else:
        model_path = cfg.get("model_path", None)
        path = (
            Path(model_path) / "experiment_cfg" / "metadata.json"
            if model_path is not None
            else None
        )
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "DreamZero metadata.json not found. This file is generated from "
                "the dataset and its path must be specified.\n"
                "Set metadata_json_path in your config to the path of the "
                "metadata.json file, or ensure it exists at "
                "model_path/experiment_cfg/metadata.json."
            )

    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if tag not in blob:
        raise KeyError(
            f"embodiment_tag {tag!r} not found in {path} (keys: {list(blob.keys())})."
        )
    # ``DatasetMetadata``'s top-level validator still uses the pre-patch enum schema.
    # Validate nested fields explicitly, then construct with RLinf ``EmbodimentTag``.
    from groot.vla.data.schema.lerobot import DatasetModalities, DatasetStatistics

    raw = blob[tag]
    return DatasetMetadata.model_construct(
        statistics=DatasetStatistics.model_validate(raw["statistics"]),
        modalities=DatasetModalities.model_validate(raw["modalities"]),
        embodiment_tag=EmbodimentTag(str(tag)),
    )


def build_dreamzero_composed_transform(
    cfg: Any,
    tokenizer_path: str,
) -> ComposedModalityTransform:
    """Construct ``ComposedModalityTransform`` for the current ``embodiment_tag``."""
    tag = cfg.embodiment_tag
    cls = _require_embodiment(tag)
    embodiment_tag_mapping = embodiment_tag_mapping_for_embodiment(
        tag, cfg.get("embodiment_tag_mapping")
    )
    return cls.get_transform(
        tokenizer_path=tokenizer_path,
        cfg=cfg,
        embodiment_tag_mapping=embodiment_tag_mapping,
    )
