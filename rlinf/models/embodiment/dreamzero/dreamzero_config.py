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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.dreamzero.base_vla import VLAConfig
from omegaconf import DictConfig, OmegaConf
from transformers.configuration_utils import PretrainedConfig

from rlinf.utils.logging import get_logger

logger = get_logger()

_PATH_OVERRIDE_KEYS = frozenset(
    {
        "tokenizer_path",
        "diffusion_model_pretrained_path",
        "image_encoder_pretrained_path",
        "text_encoder_pretrained_path",
        "vae_pretrained_path",
    }
)


def _yaml_value_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _yaml_takes_precedence(key: str, yaml_val: Any) -> bool:
    if key in _PATH_OVERRIDE_KEYS:
        return _yaml_value_set(yaml_val)
    return yaml_val is not None


def _merge_checkpoint_with_yaml(loaded: Any, yaml: Any) -> Any:
    """Merge checkpoint ``config.json`` into YAML-shaped config.

    YAML wins on conflict except ``_PATH_OVERRIDE_KEYS``: null/empty yaml paths
    keep the checkpoint value.
    """
    if isinstance(loaded, dict) and isinstance(yaml, dict):
        merged: dict[str, Any] = {}
        for key in set(loaded) | set(yaml):
            loaded_val = loaded.get(key)
            yaml_val = yaml.get(key)
            if key not in yaml:
                merged[key] = loaded_val
            elif key not in loaded:
                if _yaml_takes_precedence(key, yaml_val):
                    merged[key] = (
                        _merge_checkpoint_with_yaml({}, yaml_val)
                        if isinstance(yaml_val, dict)
                        else yaml_val
                    )
            elif isinstance(loaded_val, dict) and isinstance(yaml_val, dict):
                merged[key] = _merge_checkpoint_with_yaml(loaded_val, yaml_val)
            elif _yaml_takes_precedence(key, yaml_val):
                merged[key] = yaml_val
            else:
                merged[key] = loaded_val
        return merged
    return yaml if _yaml_value_set(yaml) else loaded


def _values_equal(a: Any, b: Any) -> bool:
    try:
        return OmegaConf.create({"a": a}).a == OmegaConf.create({"b": b}).b
    except Exception:
        return a == b


def _log_loaded_vs_yaml_diff(
    loaded: Any,
    yaml_cfg: Any,
    *,
    field: str | None = None,
) -> None:
    """Log fields where checkpoint and YAML disagree and YAML will override on merge.

    Null/empty ``_PATH_OVERRIDE_KEYS`` in YAML are ignored (checkpoint kept).
    """

    def _warn_mismatch(field_name: str, checkpoint_val: Any, yaml_val: Any) -> None:
        logger.warning(
            "DreamZero field model.%s: checkpoint=%r yaml=%r differ (using yaml)",
            field_name,
            checkpoint_val,
            yaml_val,
        )

    if isinstance(loaded, dict) and isinstance(yaml_cfg, dict):
        for key in loaded:
            child_field = key if field is None else f"{field}.{key}"
            yaml_val = yaml_cfg.get(key) if isinstance(yaml_cfg, dict) else None
            loaded_val = loaded[key]
            if key in yaml_cfg and _yaml_takes_precedence(key, yaml_val):
                if not _values_equal(loaded_val, yaml_val):
                    _warn_mismatch(child_field, loaded_val, yaml_val)
                if isinstance(loaded_val, dict) and isinstance(yaml_val, dict):
                    _log_loaded_vs_yaml_diff(loaded_val, yaml_val, field=child_field)
            elif isinstance(loaded_val, dict):
                _log_loaded_vs_yaml_diff(loaded_val, yaml_cfg, field=child_field)
        return

    if (
        field is not None
        and _yaml_value_set(yaml_cfg)
        and not _values_equal(loaded, yaml_cfg)
    ):
        _warn_mismatch(field, loaded, yaml_cfg)


def load_dreamzero_config_dict(cfg: Any) -> dict[str, Any]:
    """Load architecture from ``model_path/config.json`` or Hydra ``actor.model``."""
    model_path = cfg.get("model_path", None)

    if model_path is not None:
        json_path = Path(model_path) / "config.json"
        if not json_path.is_file():
            raise FileNotFoundError(
                f"DreamZero model_path is set but config.json is missing: {json_path}"
            )
        return json.loads(json_path.read_text(encoding="utf-8"))

    yaml_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(yaml_dict, dict):
        raise ValueError("DreamZero actor.model must resolve to a mapping.")
    for key in (
        "tokenizer_path",
        "diffusion_model_pretrained_path",
        "image_encoder_pretrained_path",
        "text_encoder_pretrained_path",
        "vae_pretrained_path",
    ):
        if cfg.get(key) is None:
            raise ValueError(
                f"DreamZero: model_path unset; actor.model.{key} must be set."
            )
    return yaml_dict


def validate_dreamzero_sft_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Validate DreamZero SFT ``actor.model`` and merge checkpoint config when set.

    When ``model_path`` is set, ``config.json`` is merged into ``actor.model``.
    YAML overrides overlapping keys; null pretrained paths keep checkpoint paths.
    """

    yaml_snapshot = OmegaConf.to_container(model_cfg, resolve=True)
    if not isinstance(yaml_snapshot, dict):
        raise ValueError("DreamZero actor.model must resolve to a mapping.")

    loaded_dict = load_dreamzero_config_dict(model_cfg)
    model_path = model_cfg.get("model_path", None)

    if model_path is not None:
        logger.info(
            "DreamZero: merging checkpoint config from %s/config.json into actor.model "
            "(yaml overrides on conflict; null pretrained paths use checkpoint)",
            model_path,
        )
        _log_loaded_vs_yaml_diff(loaded_dict, yaml_snapshot)
        model_cfg = OmegaConf.create(
            _merge_checkpoint_with_yaml(loaded_dict, yaml_snapshot)
        )

    assert model_cfg.get("embodiment_tag"), (
        "DreamZero SFT requires actor.model.embodiment_tag"
    )
    assert model_cfg.get("tokenizer_path"), (
        "DreamZero SFT requires actor.model.tokenizer_path"
    )

    max_chunk_size = model_cfg.action_head_cfg.config.diffusion_model_cfg.max_chunk_size
    num_frames = model_cfg.action_head_cfg.config.num_frames
    action_horizon = model_cfg.action_horizon

    assert max_chunk_size is not None, (
        "DreamZero requires action_head_cfg.config.diffusion_model_cfg.max_chunk_size"
    )
    assert int(max_chunk_size) > 0, (
        f"diffusion_model_cfg.max_chunk_size must be positive, got {max_chunk_size!r}"
    )
    assert num_frames is not None, (
        "DreamZero requires action_head_cfg.config.num_frames"
    )
    assert int(num_frames) > 0, f"num_frames must be positive, got {num_frames!r}"
    assert action_horizon is not None, "DreamZero SFT requires action_horizon"
    assert int(action_horizon) > 0, (
        f"action_horizon must be positive, got {action_horizon!r}"
    )
    return model_cfg


@dataclass
class DreamZeroConfig(VLAConfig):
    model_type = "dreamzero"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})

    num_action_chunks: int = field(
        default=16, metadata={"help": "Number of action chunks."}
    )

    relative_action: bool = field(default=False, metadata={"help": "Relative action."})
    relative_action_per_horizon: bool = field(
        default=False, metadata={"help": "Relative action per horizon."}
    )
    relative_action_keys: list = field(
        default_factory=list, metadata={"help": "Relative action keys."}
    )

    data_transforms: ComposedModalityTransform = field(
        default=None,
        metadata={
            "help": "Transforming data modalities, e.g. video frame augmentation or action normalization."
        },
    )

    embodiment_tag: str = field(
        default=None, metadata={"help": "Embodiment tag for rollout obs mapping."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
