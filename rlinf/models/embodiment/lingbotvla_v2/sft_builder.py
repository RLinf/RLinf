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

import os
from contextlib import chdir
from types import SimpleNamespace
from typing import Any

import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig


def _require_lingbot_v2_data() -> SimpleNamespace:
    try:
        from lingbotvla.data import build_vla_dataset
        from lingbotvla.models import build_processor
        from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
            LingbotVLAV2Config,
        )
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
            apply_lingbot_qwen3_vl_patch,
        )
    except ImportError as exc:
        raise ImportError(
            "LingBot-VLA V2 SFT requires the upstream "
            "`robbyant/lingbot-vla-v2` package. Install it with "
            "`bash requirements/install.sh embodied --model lingbotvla_v2 "
            "--env robotwin`."
        ) from exc

    return SimpleNamespace(
        LingbotVLAV2Config=LingbotVLAV2Config,
        apply_lingbot_qwen3_vl_patch=apply_lingbot_qwen3_vl_patch,
        build_processor=build_processor,
        build_vla_dataset=build_vla_dataset,
    )


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _as_literal_list(entries: list[Any] | None) -> list[str]:
    return [entry if isinstance(entry, str) else repr(entry) for entry in entries or []]


def _load_training_config(config_path: str) -> dict[str, Any]:
    candidate_paths = [
        os.path.join(config_path, "lingbotvla_cli.yaml"),
        os.path.join(config_path, "configs/vla/robotwin/robotwin.yaml"),
    ]
    training_config_path = next(
        (path for path in candidate_paths if os.path.exists(path)), None
    )
    if training_config_path is None:
        raise FileNotFoundError(
            "LingBot-VLA V2 SFT requires lingbotvla_cli.yaml or the upstream "
            f"RoboTwin config. Checked: {candidate_paths}"
        )

    with open(training_config_path) as config_file:
        training_config = yaml.safe_load(config_file)
    for section in ("model", "train", "data"):
        if section not in training_config:
            raise KeyError(
                f"Missing `{section}` in LingBot-VLA V2 training config: "
                f"{training_config_path}"
            )
    return training_config


def _merge_qwen_config(policy_config: Any, qwen_config: Any) -> Any:
    config_dict = (
        qwen_config.to_dict() if hasattr(qwen_config, "to_dict") else qwen_config
    )
    text_config = config_dict.get("text_config", {})
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
        "tokenizer_path",
    ):
        if key in text_config:
            setattr(policy_config, key, text_config[key])
        elif key in config_dict:
            setattr(policy_config, key, config_dict[key])
    policy_config.vision_config = qwen_config.vision_config
    return policy_config


def _resolve_data_path(data_paths: Any) -> str:
    if isinstance(data_paths, str):
        return data_paths
    if len(data_paths) != 1:
        raise ValueError(
            "LingBot-VLA V2 SFT expects one LeRobot dataset path or one "
            "upstream multi-dataset manifest."
        )
    return str(data_paths[0])


def build_lingbot_v2_sft_dataloader(
    cfg: Any,
    world_size: int,
    global_rank: int,
    data_paths: Any,
) -> tuple[DataLoader, dict[str, int]]:
    """Build the upstream LingBot-VLA V2 dataset for RLinf SFT."""

    v2 = _require_lingbot_v2_data()
    v2.apply_lingbot_qwen3_vl_patch()

    model_cfg = cfg.actor.model
    lingbotvla_cfg = getattr(model_cfg, "lingbotvla_v2", model_cfg)
    config_path = getattr(
        lingbotvla_cfg,
        "config_path",
        os.environ.get("LINGBOT_VLA_V2_PATH", ""),
    )
    training_config = _load_training_config(config_path)

    training_model_config = dict(training_config["model"])
    training_model_config.update(training_config["train"])
    for key in (
        "bias_update_speed",
        "sequence_wise_loss_coeff",
        "router_z_loss_coeff",
        "routed_scaling_factor",
    ):
        if key in training_model_config:
            training_model_config[key] = float(training_model_config[key])
    use_auxiliary_alignment = bool(
        getattr(lingbotvla_cfg, "use_auxiliary_alignment", False)
    )
    if not use_auxiliary_alignment:
        training_model_config["align_params"] = {}
    training_model_config["_moe_implementation"] = getattr(
        model_cfg,
        "moe_implementation",
        training_model_config.get("moe_implementation", "fused"),
    )
    training_model_config["moe_implementation"] = training_model_config[
        "_moe_implementation"
    ]
    training_model_config["attn_implementation"] = getattr(
        model_cfg, "attn_implementation", "flash_attention_2"
    )
    training_model_config["attention_implementation"] = getattr(
        model_cfg, "attention_implementation", "flex_cached"
    )
    training_model_config["tokenizer_path"] = str(model_cfg.tokenizer_path)
    training_model_config["return_image_grid_thw"] = True

    policy_config = v2.LingbotVLAV2Config(**training_model_config)
    policy_config.n_action_steps = int(model_cfg.num_action_chunks)
    policy_config.num_steps = int(
        getattr(model_cfg, "num_steps", policy_config.num_steps)
    )
    policy_config = _merge_qwen_config(
        policy_config,
        AutoConfig.from_pretrained(training_model_config["tokenizer_path"]),
    )

    processor = v2.build_processor(training_model_config["tokenizer_path"])
    data_config = dict(training_config["data"])
    data_config["train_path"] = _resolve_data_path(data_paths)
    data_config["robot_config_root"] = getattr(
        lingbotvla_cfg,
        "robot_config_root",
        os.path.join(config_path, "configs/robot_configs"),
    )
    data_config["data_name"] = getattr(
        lingbotvla_cfg, "data_name", data_config.get("data_name", "robotwin")
    )
    data_config["chunk_size"] = int(model_cfg.num_action_chunks)
    data_config["prompt_type"] = data_config.get("prompt_type", "global")
    data_config["img_size"] = int(data_config.get("img_size", 256))
    data_config["joints"] = _as_literal_list(data_config.get("joints"))
    data_config["norm_type"] = _as_literal_list(data_config.get("norm_type"))
    data_config["use_future_image"] = use_auxiliary_alignment and bool(
        data_config.get("use_future_image", False)
    )
    dataset_config = _namespace(data_config)

    # Upstream robot configs resolve norm_stats relative to the project root.
    with chdir(config_path):
        dataset = v2.build_vla_dataset(
            dataset_config=dataset_config,
            model_config=_namespace(training_model_config),
            config=policy_config,
            processor=processor,
            use_depth_align=use_auxiliary_alignment,
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
    )
    num_workers = int(cfg.data.get("num_workers", data_config.get("num_workers", 4)))
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return data_loader, {"num_samples": len(dataset)}
