#!/usr/bin/env python3
"""Generate a resolved DreamZero Wan2.2 5B train config for RLinf.

This script composes DreamZero's native Hydra config with the same overrides used
by `scripts/train/droid_training_wan22.sh`, then writes a single resolved
`conf.yaml` that RLinf can consume via `actor.model.train_cfg_path`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a resolved DreamZero Wan2.2 5B conf.yaml for RLinf."
    )
    parser.add_argument(
        "--droid-data-root",
        required=True,
        help="Path to the DROID LeRobot dataset root.",
    )
    parser.add_argument(
        "--wan22-ckpt-dir",
        required=True,
        help="Path to the Wan2.2-TI2V-5B checkpoint directory.",
    )
    parser.add_argument(
        "--image-encoder-path",
        required=True,
        help="Path to models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth.",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to the umt5-xxl tokenizer directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/dreamzero_droid_wan22_lora",
        help="Native DreamZero output_dir value baked into the generated config.",
    )
    parser.add_argument(
        "--output-conf",
        default=(
            "RLinf/examples/sft/generated/"
            "dreamzero_droid_wan22_component/conf.yaml"
        ),
        help="Where to write the resolved conf.yaml.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    config_dir = repo_root / "groot" / "vla" / "configs"
    droid_data_root = Path(args.droid_data_root).resolve()
    wan22_ckpt_dir = Path(args.wan22_ckpt_dir).resolve()
    image_encoder_path = Path(args.image_encoder_path).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_conf = Path(args.output_conf)
    if not output_conf.is_absolute():
        output_conf = (repo_root / output_conf).resolve()

    overrides = [
        "report_to=wandb",
        "data=dreamzero/droid_relative_wan22",
        "wandb_project=dreamzero",
        "train_architecture=lora",
        "num_frames=33",
        "action_horizon=24",
        "num_views=3",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22",
        "model/dreamzero/transform=dreamzero_cotrain",
        "num_frame_per_block=2",
        "num_action_per_block=24",
        "num_state_per_block=1",
        "seed=42",
        "training_args.learning_rate=1e-5",
        "training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json",
        "save_steps=1000",
        "training_args.warmup_ratio=0.05",
        f"output_dir={output_dir}",
        "per_device_train_batch_size=1",
        "max_steps=100",
        "weight_decay=1e-5",
        "save_total_limit=10",
        "upload_checkpoints=false",
        "bf16=true",
        "tf32=true",
        "eval_bf16=true",
        "dataloader_pin_memory=false",
        "dataloader_num_workers=1",
        "save_lora_only=true",
        "max_chunk_size=4",
        'save_strategy="no"',
        f"droid_data_root={droid_data_root}",
        f"dit_version={wan22_ckpt_dir}",
        (
            "text_encoder_pretrained_path="
            f"{wan22_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth"
        ),
        f"image_encoder_pretrained_path={image_encoder_path}",
        f"vae_pretrained_path={wan22_ckpt_dir}/Wan2.2_VAE.pth",
        f"tokenizer_path={tokenizer_path}",
    ]

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="conf", overrides=overrides)

    output_conf.parent.mkdir(parents=True, exist_ok=True)
    output_conf.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    print(f"Wrote resolved DreamZero Wan2.2 config to: {output_conf}")
    print()
    print("Use this path in RLinf:")
    print(f"  actor.model.train_cfg_path={output_conf}")
    print()
    print("Suggested RLinf overrides:")
    print(f"  actor.model.model_path={wan22_ckpt_dir}")
    print(f"  actor.model.pretrained_model_path={wan22_ckpt_dir}")
    print(f"  actor.model.dit_version={wan22_ckpt_dir}")
    print(
        "  actor.model.text_encoder_pretrained_path="
        f"{wan22_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth"
    )
    print(f"  actor.model.vae_pretrained_path={wan22_ckpt_dir}/Wan2.2_VAE.pth")
    print(f"  actor.model.image_encoder_pretrained_path={image_encoder_path}")


if __name__ == "__main__":
    main()
