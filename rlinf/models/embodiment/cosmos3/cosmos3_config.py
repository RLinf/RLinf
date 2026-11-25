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

"""Cosmos3 model validation and shared native LIBERO action recipes."""

import contextlib
import copy
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, open_dict

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DATA_EXPERIMENTS = {
    "libero_10": "action_policy_libero_nano",
    "libero_all": "action_policy_libero_all_nano",
}


def cosmos_framework_root() -> Path:
    """Return the root used to resolve native architecture configuration files."""
    import cosmos_framework

    return Path(cosmos_framework.__file__).resolve().parent.parent


def _configure_tokenizer(native_cfg: Any, model_cfg: DictConfig, variant: str) -> None:
    tokenizer_path = model_cfg.get("tokenizer_path") or None
    if tokenizer_path is None and model_cfg.get("checkpoint_format") == "diffusers":
        tokenizer_path = Path(str(model_cfg.model_path))
        if variant == "nano":
            tokenizer_path /= "text_tokenizer"
    if tokenizer_path is None:
        return

    tokenizer_path = Path(str(tokenizer_path)).expanduser().resolve()
    if variant == "edge":
        from cosmos_framework.data.generator.processors import build_processor
        from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
            is_cosmos3_edge_native_snapshot,
        )
        from cosmos_framework.utils.lazy_config import LazyCall

        required_files = (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        )
        missing = [
            name for name in required_files if not (tokenizer_path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Cosmos3 Edge processor directory {tokenizer_path} is missing {missing}"
            )
        if not is_cosmos3_edge_native_snapshot(str(tokenizer_path)):
            raise ValueError(
                "Cosmos3 Edge tokenizer_path must point to an Edge snapshot with "
                f"config.json or processor_config.json identifying Edge: {tokenizer_path}"
            )
        native_cfg.model.config.vlm_config.tokenizer = LazyCall(build_processor)(
            tokenizer_type=str(tokenizer_path)
        )
    else:
        if not all(
            (tokenizer_path / name).is_file() for name in ("vocab.json", "merges.txt")
        ):
            raise FileNotFoundError(
                f"Cosmos3 Nano tokenizer requires vocab.json and merges.txt: {tokenizer_path}"
            )
        native_cfg.model.config.vlm_config.tokenizer.pretrained_model_name = str(
            tokenizer_path
        )


def build_cosmos3_config(
    model_cfg: DictConfig,
    data_type: str = "libero_all",
    data_root: str | Path | None = None,
) -> Any:
    """Compose the native Nano action recipe with the requested backbone.

    Edge replaces the VLM architecture and processor. Data transformations,
    action heads, flow-matching losses, VAE encoding and packing retain the
    native LIBERO recipe. Both model and dataset consume the same processor.
    """
    from cosmos_framework.configs.base.config import make_config
    from cosmos_framework.utils.config_helper import override

    variant = str(model_cfg.get("model_variant", "nano")).lower()
    if variant not in ("nano", "edge"):
        raise ValueError(
            f"Expected Cosmos3 model_variant nano or edge, got {variant!r}"
        )
    if data_type not in DATA_EXPERIMENTS:
        raise ValueError(
            f"Expected Cosmos3 data_type in {tuple(DATA_EXPERIMENTS)}, got {data_type!r}"
        )

    # Model-only construction resolves the unused dataset against an existing root.
    root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else cosmos_framework_root()
    )
    roots = (
        {suite: root / suite for suite in LIBERO_SUITES}
        if data_type == "libero_all"
        else {"libero": root}
    )
    overrides = ["--", f"experiment={DATA_EXPERIMENTS[data_type]}"]
    overrides.extend(
        f"dataloader_train.dataloader.datasets.{name}.dataset.root={json.dumps(str(path))}"
        for name, path in roots.items()
    )
    with contextlib.chdir(cosmos_framework_root()):
        native_cfg = override(make_config(), overrides)

    if variant == "edge":
        from cosmos_framework.configs.base.defaults.reasoner import (
            PretrainedWeightsConfig,
            VLMConfig,
        )
        from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import (
            EDGE_MODEL_CONFIG,
        )

        vlm_config = copy.deepcopy(EDGE_MODEL_CONFIG["vlm_config"])
        vlm_config["pretrained_weights"] = PretrainedWeightsConfig(
            **vlm_config["pretrained_weights"]
        )
        native_cfg.model.config.vlm_config = VLMConfig(**vlm_config)
        native_cfg.model.config.resolution = EDGE_MODEL_CONFIG["resolution"]

    _configure_tokenizer(native_cfg, model_cfg, variant)
    # Native override() resolves interpolations before the backbone is replaced.
    for entry in native_cfg.dataloader_train.dataloader.datasets.values():
        entry.dataset.tokenizer_config = copy.deepcopy(
            native_cfg.model.config.vlm_config.tokenizer
        )
    return native_cfg


def validate_cosmos3_sft_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Validate a Cosmos3 SFT cfg."""

    with open_dict(model_cfg):
        model_path = model_cfg.get("model_path", None)
        assert model_path, "Cosmos3 SFT requires actor.model.model_path"

    return model_cfg
