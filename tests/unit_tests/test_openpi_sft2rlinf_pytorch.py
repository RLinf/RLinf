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

import json
from types import SimpleNamespace

import pytest
import torch

from rlinf.utils.ckpt_convertor.openpi import convert as openpi_convert
from rlinf.utils.ckpt_convertor.openpi import sft2rlinf_pytorch
from rlinf.utils.ckpt_convertor.openpi._core import load_safetensors


def _train_config(
    *, pi05: bool, action_horizon: int, max_token_len: int
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            pi05=pi05,
            action_dim=32,
            action_horizon=action_horizon,
            max_token_len=max_token_len,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            discrete_state_input=pi05,
            pcd=False,
        )
    )


def _write_checkpoint(tmp_path, required_keys: tuple[str, ...]):
    checkpoint_dir = tmp_path / "global_step_10" / "actor" / "model_state_dict"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "full_weights.pt"
    state_dict = {
        f"model._fsdp_wrapped_module.{key}": torch.ones(2, 2, dtype=torch.float64)
        for key in required_keys
    }
    torch.save(state_dict, checkpoint_path)
    input_norm_stats = tmp_path / "norm_stats.json"
    input_norm_stats.write_text('{"state": {}}')
    return checkpoint_dir, input_norm_stats


def test_sft2rlinf_pytorch_uses_robotwin_pi0_dataconfig_and_explicit_fp32(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sft2rlinf_pytorch,
        "_get_openpi_train_config",
        lambda config_name: _train_config(
            pi05=False, action_horizon=50, max_token_len=48
        ),
    )
    required_keys = (
        "img.stem.weight",
        "llm.embedder.embedding.weight",
        "action_in_proj.weight",
        "action_out_proj.weight",
        "state_proj.weight",
        "action_time_mlp_in.weight",
        "action_time_mlp_out.weight",
    )
    checkpoint_dir, input_norm_stats = _write_checkpoint(tmp_path, required_keys)
    output_model = tmp_path / "pi0_robotwin_sft"
    output_norm_stats = (
        output_model / "physical-intelligence" / "robotwin" / "norm_stats.json"
    )

    sft2rlinf_pytorch.convert(
        checkpoint_dir.parent.parent,
        input_norm_stats,
        output_model,
        output_norm_stats,
        config_name="pi0_aloha_robotwin",
        dtype="fp32",
    )

    converted = load_safetensors(output_model / "model.safetensors")
    assert set(converted) == set(required_keys)
    assert all(tensor.dtype == torch.float32 for tensor in converted.values())
    config = json.loads((output_model / "config.json").read_text())
    assert config == {
        "action_dim": 32,
        "action_expert_variant": "gemma_300m",
        "action_horizon": 50,
        "discrete_state_input": False,
        "dtype": "float32",
        "max_token_len": 48,
        "paligemma_variant": "gemma_2b",
        "pcd": False,
        "pi05": False,
    }
    assert output_norm_stats.read_text() == '{"state": {}}'


def test_sft2rlinf_pytorch_uses_behavior_dataconfig_and_explicit_bf16(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sft2rlinf_pytorch,
        "_get_openpi_train_config",
        lambda config_name: _train_config(
            pi05=True, action_horizon=32, max_token_len=200
        ),
    )
    required_keys = (
        "img.stem.weight",
        "llm.embedder.embedding.weight",
        "action_in_proj.weight",
        "action_out_proj.weight",
        "time_mlp_in.weight",
        "time_mlp_out.weight",
    )
    checkpoint_dir, input_norm_stats = _write_checkpoint(tmp_path, required_keys)
    output_model = tmp_path / "pi05_behavior_sft"
    output_norm_stats = (
        output_model / "physical-intelligence" / "behavior" / "norm_stats.json"
    )

    sft2rlinf_pytorch.convert(
        checkpoint_dir.parent.parent,
        input_norm_stats,
        output_model,
        output_norm_stats,
        config_name="pi05_behavior",
        dtype="bf16",
    )

    converted = load_safetensors(output_model / "model.safetensors")
    assert set(converted) == set(required_keys)
    assert all(tensor.dtype == torch.bfloat16 for tensor in converted.values())
    config = json.loads((output_model / "config.json").read_text())
    assert config["pi05"] is True
    assert config["action_horizon"] == 32
    assert config["max_token_len"] == 200
    assert config["dtype"] == "bfloat16"


def test_sft2rlinf_pytorch_cli_requires_shared_config_name_and_dtype():
    args = openpi_convert.build_parser().parse_args(
        [
            "sft2rlinf_pytorch",
            "--config-name",
            "pi0_aloha_robotwin",
            "--dtype",
            "fp32",
            "--ckpt",
            "checkpoint.pt",
            "--input-norm-stats",
            "input.json",
            "--output-model",
            "output",
            "--output-norm-stats",
            "output/norm_stats.json",
        ]
    )
    assert args.config_name == "pi0_aloha_robotwin"
    assert args.dtype == "fp32"
    assert args._run is sft2rlinf_pytorch.run


def test_sft2rlinf_pytorch_cli_rejects_an_implicit_storage_dtype():
    with pytest.raises(SystemExit):
        openpi_convert.build_parser().parse_args(
            [
                "sft2rlinf_pytorch",
                "--config-name",
                "pi0_aloha_robotwin",
                "--ckpt",
                "checkpoint.pt",
                "--input-norm-stats",
                "input.json",
                "--output-model",
                "output",
                "--output-norm-stats",
                "output/norm_stats.json",
            ]
        )
