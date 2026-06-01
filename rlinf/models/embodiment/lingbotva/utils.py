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

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

_TRANSFORMER_STATE_PREFIXES = (
    "_sft_adapter.transformer.",
    "_training_adapter.transformer.",
    "_sft_core.transformer.",
    "transformer.",
)


def _extend_import_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _extract_transformer_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    for prefix in _TRANSFORMER_STATE_PREFIXES:
        extracted = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if extracted:
            return extracted
    return state_dict


def export_official_transformer_checkpoint(
    *,
    model_path: str | Path,
    state_dict_path: str | Path,
    output_dir: str | Path,
) -> Path:
    model_path = Path(model_path)
    state_dict_path = Path(state_dict_path)
    output_dir = Path(output_dir)

    if not state_dict_path.exists():
        raise FileNotFoundError(
            f"LingBot-VA state dict path does not exist: {state_dict_path}"
        )

    config_src = model_path / "transformer" / "config.json"
    if not config_src.exists():
        raise FileNotFoundError(
            f"LingBot-VA transformer config not found at {config_src}"
        )

    raw_state = torch.load(
        state_dict_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(raw_state, dict):
        raise TypeError(
            "LingBot-VA full_weights checkpoint must deserialize to a dict, got "
            f"{type(raw_state)!r}."
        )

    transformer_state = _extract_transformer_state_dict(raw_state)
    tensor_state = {}
    for key, value in transformer_state.items():
        if not torch.is_tensor(value):
            raise TypeError(
                "LingBot-VA transformer checkpoint values must be tensors, got "
                f"{type(value)!r} for key {key!r}."
            )
        tensor_state[key] = value.detach().cpu().contiguous().to(torch.bfloat16)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_src, output_dir / "config.json")

    config_data = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    config_data.pop("_name_or_path", None)
    (output_dir / "config.json").write_text(
        json.dumps(config_data, indent=2) + "\n",
        encoding="utf-8",
    )
    save_file(tensor_state, output_dir / "diffusion_pytorch_model.safetensors")
    return output_dir
