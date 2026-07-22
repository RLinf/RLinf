#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert raw DreamZero safetensors into SGLang's diffusers-like layout."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors.torch import safe_open, save_file

P = {
    "transformer": "action_head.model.",
    "text_encoder": "action_head.text_encoder.",
    "image_encoder": "action_head.image_encoder.",
    "vae": "action_head.vae.model.",
}
TXT0 = {
    "token_embedding.weight": "shared.weight",
    "norm.weight": "encoder.final_layer_norm.weight",
}
TXT = {
    "norm1.weight": "layer.0.layer_norm.weight",
    "attn.q.weight": "layer.0.SelfAttention.q.weight",
    "attn.k.weight": "layer.0.SelfAttention.k.weight",
    "attn.v.weight": "layer.0.SelfAttention.v.weight",
    "attn.o.weight": "layer.0.SelfAttention.o.weight",
    "pos_embedding.embedding.weight": "layer.0.SelfAttention.relative_attention_bias.weight",
    "norm2.weight": "layer.1.layer_norm.weight",
    "ffn.gate.0.weight": "layer.1.DenseReluDense.wi_0.weight",
    "ffn.fc1.weight": "layer.1.DenseReluDense.wi_1.weight",
    "ffn.fc2.weight": "layer.1.DenseReluDense.wo.weight",
}
IMG0 = {
    "model.visual.cls_embedding": "vision_model.embeddings.class_embedding",
    "model.visual.patch_embedding.weight": "vision_model.embeddings.patch_embedding.weight",
    "model.visual.pos_embedding": "vision_model.embeddings.position_embedding.weight",
    "model.visual.pre_norm.weight": "vision_model.pre_layrnorm.weight",
    "model.visual.pre_norm.bias": "vision_model.pre_layrnorm.bias",
}
IMG = {
    "norm1.weight": "layer_norm1.weight",
    "norm1.bias": "layer_norm1.bias",
    "attn.to_qkv.weight": "self_attn.qkv_proj.weight",
    "attn.to_qkv.bias": "self_attn.qkv_proj.bias",
    "attn.proj.weight": "self_attn.out_proj.weight",
    "attn.proj.bias": "self_attn.out_proj.bias",
    "norm2.weight": "layer_norm2.weight",
    "norm2.bias": "layer_norm2.bias",
    "mlp.0.weight": "mlp.fc1.weight",
    "mlp.0.bias": "mlp.fc1.bias",
    "mlp.2.weight": "mlp.fc2.weight",
    "mlp.2.bias": "mlp.fc2.bias",
}
VAE_RB = (
    ("residual.0.", "norm1."),
    ("residual.2.", "conv1."),
    ("residual.3.", "norm2."),
    ("residual.6.", "conv2."),
    ("shortcut.", "conv_shortcut."),
)

# Wan2.2 TI2V VAE normalization constants are not present in raw DreamZero config.
VAE_MEAN = [
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.157,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.123,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.052,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
]
VAE_STD = [
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.499,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.06,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
]


def jload(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def jdump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def n5(xs: list[float]) -> list:
    return [[[[[f32(x)]]] for x in xs]]


def tensor_shape(src: Path, wm: dict[str, str], key: str) -> tuple[int, ...]:
    with safe_open(src / wm[key], framework="pt", device="cpu") as f:
        return tuple(f.get_tensor(key).shape)


def write_configs(src: Path, out: Path) -> None:
    raw = jload(src / "config.json")
    wm = jload(src / "model.safetensors.index.json")["weight_map"]
    dit = dict(raw["action_head_cfg"]["config"]["diffusion_model_cfg"])
    for key in ("_convert_", "_target_", "diffusion_model_pretrained_path"):
        dit.pop(key, None)
    dit.update(
        {
            "_class_name": "DreamZeroCausalWanModel",
            "action_dim": raw.get("max_action_dim", raw["action_dim"]),
            "hidden_size": tensor_shape(
                src, wm, P["transformer"] + "state_encoder.layer1.W"
            )[-1],
            "max_state_dim": raw["max_state_dim"],
            "patch_size": [1, 2, 2],
        }
    )

    # Fixed DreamZero 5B encoder/VAE configs consumed by SGLang generic loaders.
    text = {
        "_class_name": "UMT5EncoderModel",
        "architectures": ["UMT5EncoderModel"],
        "classifier_dropout": 0.0,
        "d_ff": 10240,
        "d_kv": 64,
        "d_model": 4096,
        "decoder_start_token_id": 0,
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "layer_norm_epsilon": 1e-6,
        "model_type": "umt5",
        "num_decoder_layers": None,
        "num_heads": 64,
        "num_layers": 24,
        "output_hidden_states": False,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "stacked_params_mapping": [
            [".qkv_proj", ".q", "q"],
            [".qkv_proj", ".k", "k"],
            [".qkv_proj", ".v", "v"],
        ],
        "use_cache": True,
        "vocab_size": 256384,
    }
    image = {
        "_class_name": "CLIPVisionModel",
        "architectures": ["CLIPVisionModel"],
        "attention_dropout": 0.0,
        "dropout": 0.0,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "image_size": 224,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 5120,
        "layer_norm_eps": 1e-5,
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 32,
        "patch_size": 14,
        "projection_dim": 1024,
        "stacked_params_mapping": [],
    }
    raw_vae = raw["action_head_cfg"]["config"]["vae_cfg"]
    scale = n5([f32(f32(1.0) / f32(x)) for x in VAE_STD])
    vae = {
        "_class_name": "AutoencoderKLWan",
        "attn_scales": [],
        "base_dim": raw_vae["dim"],
        "clip_output": True,
        "decoder_base_dim": 256,
        "dim_mult": [1, 2, 4, 4],
        "dropout": 0.0,
        "extra_attrs": {"shift_factor": n5(VAE_MEAN)},
        "in_channels": 12,
        "is_residual": True,
        "latents_mean": VAE_MEAN,
        "latents_std": VAE_STD,
        "num_res_blocks": 2,
        "out_channels": 12,
        "patch_size": 2,
        "scale_factor_spatial": 16,
        "scale_factor_temporal": 4,
        "scaling_factor": scale,
        "spatial_compression_ratio": 16,
        "stacked_params_mapping": [],
        "temperal_downsample": [False, True, True],
        "temporal_compression_ratio": 4,
        "z_dim": raw_vae["z_dim"],
    }

    jdump(
        out / "model_index.json",
        {
            "_class_name": "DreamZeroPipeline",
            "_diffusers_version": "0.0.0",
            "image_encoder": ["transformers", "CLIPVisionModel"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "transformer": ["diffusers", "DreamZeroCausalWanModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        },
    )
    for name, cfg in {
        "transformer": dit,
        "text_encoder": text,
        "image_encoder": image,
        "vae": vae,
    }.items():
        jdump(out / name / "config.json", cfg)


def remap_txt(key: str) -> str | None:
    key = key.removeprefix(P["text_encoder"])
    parts = key.split(".", 2)
    if key in TXT0:
        return TXT0[key]
    if len(parts) == 3 and parts[0] == "blocks" and parts[2] in TXT:
        return f"encoder.block.{parts[1]}.{TXT[parts[2]]}"
    return None


def remap_img(key: str) -> str | None:
    key = key.removeprefix(P["image_encoder"])
    if key in IMG0:
        return IMG0[key]
    if key in ("model.log_scale", "model.visual.head") or key.startswith(
        ("model.visual.post_norm.", "model.visual.transformer.31.")
    ):
        return None
    parts = key.split(".")
    tail = ".".join(parts[4:])
    if (
        len(parts) >= 6
        and parts[:3] == ["model", "visual", "transformer"]
        and tail in IMG
    ):
        return f"vision_model.encoder.layers.{parts[3]}.{IMG[tail]}"
    return None


def rb(suffix: str) -> str | None:
    return next((b + suffix[len(a) :] for a, b in VAE_RB if suffix.startswith(a)), None)


def mid(prefix: str, suffix: str) -> str | None:
    i, _, rest = suffix.partition(".")
    r = rb(rest)
    if i in ("0", "2") and r:
        return f"{prefix}.mid_block.resnets.{0 if i == '0' else 1}.{r}"
    return f"{prefix}.mid_block.attentions.0.{rest}" if i == "1" and rest else None


def down(suffix: str) -> str | None:
    i, _, rest = suffix.partition(".")
    if rest.startswith("downsamples."):
        j, _, inner = rest[12:].partition(".")
        r = rb(inner)
        return (
            f"encoder.down_blocks.{i}.resnets.{j}.{r}"
            if r
            else f"encoder.down_blocks.{i}.downsampler.{inner}"
            if inner.startswith(("resample.", "time_conv."))
            else None
        )
    r = rb(rest)
    return (
        f"encoder.down_blocks.{i}.{r}"
        if r
        else f"encoder.down_blocks.{i}.{rest}"
        if rest.startswith(("resample.", "time_conv."))
        else None
    )


def remap_vae(key: str) -> str | None:
    key = key.removeprefix(P["vae"])
    for a, b in (
        ("encoder.conv1.", "encoder.conv_in."),
        ("encoder.head.0.", "encoder.norm_out."),
        ("encoder.head.2.", "encoder.conv_out."),
        ("conv1.", "quant_conv."),
    ):
        if key.startswith(a):
            return b + key[len(a) :]
    # Action inference only encodes observations; decoder/post_quant_conv are omitted.
    if key.startswith("encoder.downsamples."):
        return down(key[20:])
    return mid("encoder", key[15:]) if key.startswith("encoder.middle.") else None


def remap(component: str, key: str) -> str | None:
    return (
        key.removeprefix(P[component])
        if component == "transformer"
        else {"text_encoder": remap_txt, "image_encoder": remap_img, "vae": remap_vae}[
            component
        ](key)
    )


def repack_component(src: Path, out: Path, component: str) -> dict[str, Any]:
    raw_wm = jload(src / "model.safetensors.index.json")["weight_map"]
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    source = ignored = written = 0
    for key, shard in raw_wm.items():
        if not key.startswith(P[component]):
            continue
        source += 1
        target = remap(component, key)
        ignored += int(target is None)
        if target is not None:
            grouped[shard].append((key, target))

    out_wm, files = {}, []
    for i, shard in enumerate(sorted(grouped), 1):
        name = (
            f"diffusion_pytorch_model-{i:05d}.safetensors"
            if component in ("transformer", "vae")
            else f"model-{i:05d}.safetensors"
        )
        tensors, wanted = {}, dict(grouped[shard])
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in wanted:
                    continue
                target, tensor = wanted[key], f.get_tensor(key)
                tensor = (
                    tensor.reshape(-1)
                    if target == "vision_model.embeddings.class_embedding"
                    else tensor.squeeze(0)
                    if target == "vision_model.embeddings.position_embedding.weight"
                    else tensor
                )
                tensors[target], out_wm[target] = tensor, name
        if tensors:
            save_file(tensors, out / component / name)
            files.append(shard)
            written += len(tensors)
    if not out_wm:
        raise RuntimeError(f"{component} wrote no tensors")

    idx = (
        "diffusion_pytorch_model.safetensors.index.json"
        if component in ("transformer", "vae")
        else "model.safetensors.index.json"
    )
    jdump(
        out / component / idx,
        {"metadata": {"format": "pt"}, "weight_map": dict(sorted(out_wm.items()))},
    )
    return {"source": source, "written": written, "ignored": ignored, "files": files}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--path", type=Path, help="Raw DreamZero checkpoint directory.")
    g.add_argument("--source", type=Path, help="Alias of --path.")
    p.add_argument("--output", type=Path, help="Defaults to '<source>-repacked'.")
    args = p.parse_args()
    src = args.source or args.path
    out = args.output or src.with_name(src.name + "-repacked")

    out.mkdir(parents=True)
    write_configs(src, out)
    shutil.copy2(src / "config.json", out / "source_config.json")
    if (src / "experiment_cfg").is_dir():
        shutil.copytree(src / "experiment_cfg", out / "experiment_cfg")
    report = {c: repack_component(src, out, c) for c in P}
    jdump(
        out / "dreamzero_repack_report.json",
        {
            "format": "sglang-dreamzero-repacked-v1",
            "source": str(src),
            "report": report,
        },
    )


if __name__ == "__main__":
    main()
