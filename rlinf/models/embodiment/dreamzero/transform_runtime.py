from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.dreamzero.schema import DatasetMetadata
from rlinf.models.embodiment.dreamzero.transform import ComposedModalityTransform


EMBODIMENT_TAG_MAPPING = {
    "libero_sim": 21,
    "oxe_droid": 17,
    # Compatibility keys required by DreamTransform.collate() branching.
    # These are not enabled as supported runtime environments in RLinf here;
    # they only prevent KeyError when collate checks fixed key names.
    "agibot": 26,
    "gr1_unified": 24,
    "mecka_hands": 27,
    "xdof": 22,
    "yam": 32,
    "dream": 31,
    "lapa": 27,
}


def _default_stats(dim: int) -> dict[str, list[float]]:
    zeros = [0.0] * dim
    ones = [1.0] * dim
    minus_ones = [-1.0] * dim
    return {
        "max": ones,
        "min": minus_ones,
        "mean": zeros,
        "std": ones,
        "q01": minus_ones,
        "q99": ones,
    }


def _builtin_metadata(embodiment_tag: str) -> DatasetMetadata:
    if embodiment_tag in {"droid", "oxe_droid"}:
        metadata = {
            "statistics": {
                "state": {
                    "joint_position": _default_stats(7),
                    "gripper_position": _default_stats(1),
                },
                "action": {
                    "joint_position": _default_stats(7),
                    "gripper_position": _default_stats(1),
                },
            },
            "modalities": {
                "video": {
                    "exterior_image_1_left": {"resolution": [256, 256], "channels": 3, "fps": 10.0},
                    "exterior_image_2_left": {"resolution": [256, 256], "channels": 3, "fps": 10.0},
                    "wrist_image_left": {"resolution": [256, 256], "channels": 3, "fps": 10.0},
                },
                "state": {
                    "joint_position": {"absolute": True, "rotation_type": None, "shape": [7], "continuous": True},
                    "gripper_position": {"absolute": True, "rotation_type": None, "shape": [1], "continuous": True},
                },
                "action": {
                    "joint_position": {"absolute": True, "rotation_type": None, "shape": [7], "continuous": True},
                    "gripper_position": {"absolute": True, "rotation_type": None, "shape": [1], "continuous": True},
                },
            },
            "embodiment_tag": "oxe_droid",
        }
    elif embodiment_tag == "libero_sim":
        metadata = {
            "statistics": {
                "state": {"state": _default_stats(8)},
                "action": {"actions": _default_stats(7)},
            },
            "modalities": {
                "video": {
                    "image": {"resolution": [256, 256], "channels": 3, "fps": 10.0},
                    "wrist_image": {"resolution": [256, 256], "channels": 3, "fps": 10.0},
                },
                "state": {
                    "state": {"absolute": True, "rotation_type": None, "shape": [8], "continuous": True},
                },
                "action": {
                    "actions": {"absolute": True, "rotation_type": None, "shape": [7], "continuous": True},
                },
            },
            "embodiment_tag": "libero_sim",
        }
    else:
        raise KeyError(
            f"Unsupported embodiment_tag: {embodiment_tag}. "
            "Only 'libero_sim' and 'droid/oxe_droid' are supported."
        )
    return DatasetMetadata.model_validate(metadata)


def _preset_libero_sim(tokenizer_path: str) -> dict[str, Any]:
    return {
        "_target_": "rlinf.models.embodiment.dreamzero.transform.ComposedModalityTransform",
        "transforms": [
            {"_target_": "groot.vla.data.transform.VideoToTensor", "apply_to": ["video.image", "video.wrist_image"]},
            {"_target_": "groot.vla.data.transform.VideoCrop", "apply_to": ["video.image", "video.wrist_image"], "scale": 0.95, "mode": "random"},
            {
                "_target_": "groot.vla.data.transform.VideoResize",
                "apply_to": ["video.image", "video.wrist_image"],
                "height": 256,
                "width": 256,
                "interpolation": "linear",
            },
            {"_target_": "groot.vla.data.transform.VideoToNumpy", "apply_to": ["video.image", "video.wrist_image"]},
            {"_target_": "groot.vla.data.transform.StateActionToTensor", "apply_to": ["state.state"]},
            {
                "_target_": "groot.vla.data.transform.StateActionTransform",
                "apply_to": ["state.state"],
                "normalization_modes": {"state.state": "q99"},
            },
            {"_target_": "groot.vla.data.transform.StateActionToTensor", "apply_to": ["action.actions"]},
            {
                "_target_": "groot.vla.data.transform.StateActionTransform",
                "apply_to": ["action.actions"],
                "normalization_modes": {"action.actions": "q99"},
            },
            {
                "_target_": "groot.vla.data.transform.ConcatTransform",
                "video_concat_order": ["video.image", "video.wrist_image"],
                "state_concat_order": ["state.state"],
                "action_concat_order": ["action.actions"],
            },
            {
                "_target_": "rlinf.models.embodiment.dreamzero.transform_wrapper.RlinfDreamTransform",
                "default_instruction": "Perform the default behavior.",
                "language_dropout_prob": 0.0,
                "always_use_default_instruction": False,
                "max_state_dim": 64,
                "max_action_dim": 32,
                "max_length": 512,
                "state_horizon": 1,
                "action_horizon": 16,
                "embodiment_tag_mapping": EMBODIMENT_TAG_MAPPING,
                "tokenizer_path": tokenizer_path,
            },
        ],
    }


def _preset_oxe_droid(tokenizer_path: str) -> dict[str, Any]:
    return {
        "_target_": "rlinf.models.embodiment.dreamzero.transform.ComposedModalityTransform",
        "transforms": [
            {
                "_target_": "groot.vla.data.transform.VideoToTensor",
                "apply_to": ["video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"],
            },
            {
                "_target_": "groot.vla.data.transform.VideoCrop",
                "apply_to": ["video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"],
                "scale": 0.95,
                "mode": "random",
            },
            {
                "_target_": "groot.vla.data.transform.VideoResize",
                "apply_to": ["video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"],
                "height": 256,
                "width": 256,
                "interpolation": "linear",
            },
            {
                "_target_": "groot.vla.data.transform.VideoToNumpy",
                "apply_to": ["video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"],
            },
            {
                "_target_": "groot.vla.data.transform.StateActionToTensor",
                "apply_to": ["state.joint_position", "state.gripper_position"],
            },
            {
                "_target_": "groot.vla.data.transform.StateActionTransform",
                "apply_to": ["state.joint_position", "state.gripper_position"],
                "normalization_modes": {"state.joint_position": "q99", "state.gripper_position": "q99"},
            },
            {
                "_target_": "groot.vla.data.transform.StateActionToTensor",
                "apply_to": ["action.joint_position", "action.gripper_position"],
            },
            {
                "_target_": "groot.vla.data.transform.StateActionTransform",
                "apply_to": ["action.joint_position", "action.gripper_position"],
                "normalization_modes": {"action.joint_position": "q99", "action.gripper_position": "q99"},
            },
            {
                "_target_": "groot.vla.data.transform.ConcatTransform",
                "video_concat_order": ["video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"],
                "state_concat_order": ["state.joint_position", "state.gripper_position"],
                "action_concat_order": ["action.joint_position", "action.gripper_position"],
            },
            {
                "_target_": "rlinf.models.embodiment.dreamzero.transform_wrapper.RlinfDreamTransform",
                "default_instruction": "Perform the default behavior.",
                "language_dropout_prob": 0.0,
                "always_use_default_instruction": False,
                "max_state_dim": 64,
                "max_action_dim": 32,
                "max_length": 512,
                "state_horizon": 1,
                "action_horizon": 16,
                "embodiment_tag_mapping": EMBODIMENT_TAG_MAPPING,
                "tokenizer_path": tokenizer_path,
            },
        ],
    }


PRESET_BUILDERS = {
    "libero_sim": _preset_libero_sim,
    "droid": _preset_oxe_droid,
    "oxe_droid": _preset_oxe_droid,
}


@dataclass
class DreamZeroTransformConfig:
    source: str = "experiment_cfg"
    embodiment_tag: str = "libero_sim"
    preset: str = "libero_sim"
    transform_yaml_path: str | None = None
    metadata_json_path: str | None = None
    inline_transform_cfg: dict[str, Any] | None = None
    inline_metadata: dict[str, Any] | None = None
    eval_mode: bool = True
    relative_action: bool = False
    relative_action_per_horizon: bool = False
    relative_action_keys: list[str] | None = None

    @classmethod
    def from_model_cfg(cls, cfg: DictConfig) -> "DreamZeroTransformConfig":
        embodiment_tag = str(cfg.get("embodiment_tag", "libero_sim"))
        if embodiment_tag == "droid":
            embodiment_tag = "oxe_droid"
        return cls(
            source=str(cfg.get("dreamzero_transform_source", "experiment_cfg")),
            embodiment_tag=embodiment_tag,
            preset=str(cfg.get("dreamzero_transform_preset", "libero_sim")),
            transform_yaml_path=cfg.get("dreamzero_transform_yaml_path", None),
            metadata_json_path=cfg.get("dreamzero_metadata_json_path", None),
            inline_transform_cfg=cfg.get("dreamzero_transform_inline_cfg", None),
            inline_metadata=cfg.get("dreamzero_metadata_inline", None),
            eval_mode=bool(cfg.get("dreamzero_transform_eval_mode", True)),
            relative_action=bool(cfg.get("relative_action", False)),
            relative_action_per_horizon=bool(cfg.get("relative_action_per_horizon", False)),
            relative_action_keys=list(cfg.get("relative_action_keys", [])),
        )


def _load_from_experiment_cfg(
    model_path: Path,
    tokenizer_path: str,
    embodiment_tag: str,
    transform_yaml_path: str | None = None,
    metadata_json_path: str | None = None,
):
    if embodiment_tag not in {"libero_sim", "oxe_droid"}:
        raise KeyError(
            f"Unsupported embodiment_tag={embodiment_tag} for experiment_cfg loading. "
            "Only 'libero_sim' and 'oxe_droid' are supported."
        )
    exp_cfg_dir = model_path / "experiment_cfg"
    metadata_path = Path(metadata_json_path) if metadata_json_path else exp_cfg_dir / "metadata.json"
    conf_path = Path(transform_yaml_path) if transform_yaml_path else exp_cfg_dir / "conf.yaml"

    with open(metadata_path, "r") as f:
        metadatas = json.load(f)
    metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])

    train_cfg = OmegaConf.load(conf_path)
    node = copy.deepcopy(train_cfg.transforms[embodiment_tag])
    node._target_ = "rlinf.models.embodiment.dreamzero.transform.ComposedModalityTransform"
    for i, tf in enumerate(node.transforms):
        if tf.get("_target_") == "groot.vla.model.dreamzero.transform.dreamzero_cotrain.DreamTransform":
            node.transforms[i]["_target_"] = (
                "rlinf.models.embodiment.dreamzero.transform_wrapper.RlinfDreamTransform"
            )
    node.transforms[-1].tokenizer_path = tokenizer_path
    transforms = instantiate(node)
    relative_action = bool(train_cfg.get("relative_action", False))
    relative_action_per_horizon = bool(train_cfg.get("relative_action_per_horizon", False))
    relative_action_keys = list(train_cfg.get("relative_action_keys", []))
    return transforms, metadata, relative_action, relative_action_per_horizon, relative_action_keys


def resolve_dreamzero_transforms(
    cfg: DictConfig,
    model_path: Path,
    tokenizer_path: str,
) -> tuple[ComposedModalityTransform, bool, bool, list[str]]:
    runtime_cfg = DreamZeroTransformConfig.from_model_cfg(cfg)
    embodiment_tag = runtime_cfg.embodiment_tag
    if embodiment_tag not in {"libero_sim", "oxe_droid"}:
        raise KeyError(
            f"Unsupported embodiment_tag={embodiment_tag}. "
            "Only 'libero_sim' and 'droid/oxe_droid' are supported."
        )

    if runtime_cfg.source == "experiment_cfg":
        transforms, metadata, rel, rel_per_h, rel_keys = _load_from_experiment_cfg(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            embodiment_tag=embodiment_tag,
            transform_yaml_path=runtime_cfg.transform_yaml_path,
            metadata_json_path=runtime_cfg.metadata_json_path,
        )
    else:
        if runtime_cfg.source == "inline" and runtime_cfg.inline_transform_cfg is not None:
            node = copy.deepcopy(runtime_cfg.inline_transform_cfg)
        else:
            builder = PRESET_BUILDERS.get(runtime_cfg.preset, PRESET_BUILDERS.get(embodiment_tag))
            if builder is None:
                raise KeyError(
                    f"Unsupported DreamZero transform preset: {runtime_cfg.preset}. "
                    f"Supported presets: {sorted(PRESET_BUILDERS.keys())}"
                )
            node = builder(tokenizer_path)
        transforms = instantiate(node)
        if runtime_cfg.inline_metadata is not None:
            metadata = DatasetMetadata.model_validate(runtime_cfg.inline_metadata)
        else:
            metadata = _builtin_metadata(embodiment_tag)
        rel = runtime_cfg.relative_action
        rel_per_h = runtime_cfg.relative_action_per_horizon
        rel_keys = runtime_cfg.relative_action_keys or []

    assert isinstance(transforms, ComposedModalityTransform), f"{transforms=}"
    transforms.set_metadata(metadata)
    if runtime_cfg.eval_mode:
        transforms.eval()
    else:
        transforms.train()
    return transforms, rel, rel_per_h, rel_keys

