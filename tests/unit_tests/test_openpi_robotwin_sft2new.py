import json

import torch

from rlinf.utils.ckpt_convertor.openpi import robotwin_sft2new
from rlinf.utils.ckpt_convertor.openpi._core import load_safetensors


def test_robotwin_sft2new_writes_pi0_hf_style_layout(tmp_path):
    checkpoint_dir = tmp_path / "global_step_10" / "actor" / "model_state_dict"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "full_weights.pt"

    required_keys = (
        "img.stem.weight",
        "llm.embedder.embedding.weight",
        "action_in_proj.weight",
        "action_out_proj.weight",
        "state_proj.weight",
        "action_time_mlp_in.weight",
        "action_time_mlp_out.weight",
    )
    state_dict = {
        f"model._fsdp_wrapped_module.{key}": torch.ones(2, 2, dtype=torch.float64)
        for key in required_keys
    }
    torch.save(state_dict, checkpoint_path)

    input_norm_stats = tmp_path / "norm_stats.json"
    input_norm_stats.write_text('{"state": {}}')
    output_model = tmp_path / "robotwin_pi0_sft"
    output_norm_stats = (
        output_model / "physical-intelligence" / "robotwin" / "norm_stats.json"
    )

    robotwin_sft2new.convert(
        checkpoint_dir.parent.parent,
        input_norm_stats,
        output_model,
        output_norm_stats,
    )

    converted = load_safetensors(output_model / "model.safetensors")
    assert set(converted) == set(required_keys)
    assert all(tensor.dtype == torch.float32 for tensor in converted.values())

    config = json.loads((output_model / "config.json").read_text())
    assert config["pi05"] is False
    assert config["action_horizon"] == 50
    assert config["action_dim"] == 32
    assert config["max_token_len"] == 48
    assert output_norm_stats.read_text() == '{"state": {}}'


def test_robotwin_sft2new_writes_pi05_hf_style_layout(tmp_path):
    checkpoint_dir = tmp_path / "global_step_10" / "actor" / "model_state_dict"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "full_weights.pt"

    required_keys = (
        "img.stem.weight",
        "llm.embedder.embedding.weight",
        "action_in_proj.weight",
        "action_out_proj.weight",
        "time_mlp_in.weight",
        "time_mlp_out.weight",
    )
    state_dict = {
        f"model._fsdp_wrapped_module.{key}": torch.ones(2, 2, dtype=torch.float64)
        for key in required_keys
    }
    torch.save(state_dict, checkpoint_path)

    input_norm_stats = tmp_path / "norm_stats.json"
    input_norm_stats.write_text('{"state": {}}')
    output_model = tmp_path / "robotwin_pi05_sft"
    output_norm_stats = (
        output_model / "physical-intelligence" / "robotwin" / "norm_stats.json"
    )

    robotwin_sft2new.convert(
        checkpoint_dir.parent.parent,
        input_norm_stats,
        output_model,
        output_norm_stats,
        pi05=True,
    )

    converted = load_safetensors(output_model / "model.safetensors")
    assert set(converted) == set(required_keys)
    assert all(tensor.dtype == torch.float32 for tensor in converted.values())

    config = json.loads((output_model / "config.json").read_text())
    assert config["pi05"] is True
    assert config["discrete_state_input"] is True
    assert config["action_horizon"] == 50
    assert config["action_dim"] == 32
    assert config["max_token_len"] == 200
    assert output_norm_stats.read_text() == '{"state": {}}'
