#!/usr/bin/env python
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
"""
Pre-extract Wan2.2 VAE latents + UMT5 text embeddings for a LeRobot v2.1
dataset, in the on-disk format LingBot-VA's `LatentLeRobotDataset` expects.

For each episode E and each camera key K listed in `obs_cam_keys`:
    <dataset>/latents/chunk-000/<K>/episode_<E:06d>_<start>_<end>.pth

The .pth payload is a dict with these keys (LingBot-VA contract — see
README §"Custom Dataset Preparation" and `lerobot_latent_dataset.py:209`):

    latent              torch.bfloat16, shape (latent_num_frames * latent_height * latent_width, C)
                        — flat over (T,H,W), channel-last. The loader does
                          `rearrange(latent, '(f h w) c -> f h w c', f=..., h=..., w=...)`.
    latent_num_frames   int   — temporal extent in latent space (Wan VAE: (T-1)//4 + 1)
    latent_height       int   — H // 8
    latent_width        int   — W // 8
    video_num_frames    int   — number of frames actually fed to the VAE (after fps subsampling)
    video_height        int   — input H after resize
    video_width         int   — input W after resize
    text_emb            torch.bfloat16, shape (512, D) — same shape the server produces in `_get_t5_prompt_embeds`
                          with max_sequence_length=512 (real tokens then zero-padded).
    text                str   — raw action_text from episodes.jsonl
    frame_ids           list[int] — source-video frame indices actually sampled
    start_frame         int   — start_frame from action_config (in original-fps frame indexing)
    end_frame           int   — end_frame from action_config
    fps                 int   — target sampling fps
    ori_fps             int   — original-video fps

Side outputs:
    <dataset>/empty_emb.pt    Tensor (512, D) bfloat16 — UMT5 embedding of "",
                              used by the loader for classifier-free guidance dropout.

Replicates the server's exact preprocessing so latents extracted here match
the streaming-server protocol bit-for-bit:
    * Pixel norm:  `frames/255.0 * 2.0 - 1.0`            (videos in [-1, +1])
    * Latent norm: `(mu - latents_mean) * (1/latents_std)` using
                   vae.config.latents_mean / latents_std
    * Text:        tokenize with padding='max_length', max_length=512;
                   encode; truncate to attn-mask length; re-pad with zeros.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from einops import rearrange
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel

CHUNK_DIR = "chunk-000"
DEFAULT_CAM_KEYS = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]
MAX_TEXT_LEN = 512  # matches wan_va_server.py:_reset (encode_prompt max_sequence_length=512)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_vae(ckpt_dir: Path, device: str, dtype: torch.dtype) -> AutoencoderKLWan:
    vae = AutoencoderKLWan.from_pretrained(ckpt_dir / "vae", torch_dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)
    return vae.to(device)


def load_text_stack(
    ckpt_dir: Path, device: str, dtype: torch.dtype
) -> tuple[T5TokenizerFast, UMT5EncoderModel]:
    tokenizer = T5TokenizerFast.from_pretrained(ckpt_dir / "tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        ckpt_dir / "text_encoder", torch_dtype=dtype
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return tokenizer, text_encoder.to(device)


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def prompt_clean(text: str) -> str:
    """Match diffusers.pipelines.wan.pipeline_wan.prompt_clean — collapse
    consecutive whitespace and strip."""
    return " ".join(text.split()).strip()


@torch.no_grad()
def encode_text(
    text: str,
    tokenizer: T5TokenizerFast,
    text_encoder: UMT5EncoderModel,
    device: str,
    dtype: torch.dtype,
    max_len: int = MAX_TEXT_LEN,
) -> torch.Tensor:
    """Return a CPU tensor (max_len, D) in `dtype`. Mirrors
    wan_va_server.py:_get_t5_prompt_embeds (single prompt)."""
    cleaned = prompt_clean(text)
    inputs = tokenizer(
        [cleaned],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    seq_len = int(mask.gt(0).sum(dim=1).item())

    emb = text_encoder(ids, mask).last_hidden_state[0]  # (max_len, D)
    emb = emb.to(dtype)
    truncated = emb[:seq_len]
    padded = torch.cat(
        [truncated, truncated.new_zeros(max_len - seq_len, truncated.size(-1))],
        dim=0,
    )
    return padded.detach().cpu()


@torch.no_grad()
def encode_video(
    frames_uint8: np.ndarray,
    vae: AutoencoderKLWan,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int, int]:
    """Return (latent_flat, latent_num_frames, latent_height, latent_width).

    latent_flat is bfloat16 on CPU, shape `(T_lat * H_lat * W_lat, C)`,
    laid out f-major (matches the loader's `(f h w) c -> f h w c` rearrange).
    """
    # (T, H, W, 3) uint8 -> (1, 3, T, H, W) float in [-1, +1]
    x = torch.from_numpy(frames_uint8).float().permute(3, 0, 1, 2).unsqueeze(0)
    x = x / 255.0 * 2.0 - 1.0
    x = x.to(device=device, dtype=dtype)

    mu = vae.encode(x).latent_dist.mode()  # (1, C, T_lat, H_lat, W_lat)

    latents_mean = torch.tensor(
        vae.config.latents_mean, device=device, dtype=torch.float32
    ).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std, device=device, dtype=torch.float32
    ).view(1, -1, 1, 1, 1)
    mu_norm = ((mu.float() - latents_mean) / latents_std).to(dtype)

    _, _, T_lat, H_lat, W_lat = mu_norm.shape
    flat = rearrange(mu_norm[0], "c f h w -> (f h w) c").detach().cpu()
    return flat, int(T_lat), int(H_lat), int(W_lat)


# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #

def read_video_frames(
    mp4_path: Path,
    start_frame: int,
    end_frame: int,
    fps: int,
    ori_fps: int,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, list[int]]:
    """Read [start_frame, end_frame) from mp4 (indexed at ori_fps), subsample to
    target fps, resize to (target_h, target_w), return (T, H, W, 3) uint8 and
    the list of source-video frame ids actually sampled."""
    frames_all = iio.imread(mp4_path, plugin="pyav")  # (T_total, H, W, 3) uint8
    if frames_all.ndim != 4 or frames_all.shape[-1] != 3:
        raise ValueError(f"Unexpected video shape {frames_all.shape} for {mp4_path}")

    segment = frames_all[start_frame:end_frame]
    if fps == ori_fps:
        frame_ids = list(range(start_frame, end_frame))
        sampled = segment
    else:
        if ori_fps % fps != 0:
            raise ValueError(
                f"ori_fps ({ori_fps}) must be a multiple of fps ({fps}) "
                f"for integer subsampling"
            )
        stride = ori_fps // fps
        local_ids = list(range(0, segment.shape[0], stride))
        frame_ids = [start_frame + i for i in local_ids]
        sampled = segment[local_ids]

    if sampled.shape[1] != target_h or sampled.shape[2] != target_w:
        # Bilinear resize via torch (matches server's F.interpolate path).
        t = torch.from_numpy(sampled).float().permute(0, 3, 1, 2)  # (T,3,H,W)
        t = torch.nn.functional.interpolate(
            t, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        sampled = (
            t.clamp(0, 255).permute(0, 2, 3, 1).contiguous().byte().numpy()
        )

    return np.ascontiguousarray(sampled), frame_ids


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    device = args.device
    dtype = torch.bfloat16

    dataset = args.dataset
    meta = dataset / "meta"
    if not (meta / "episodes.jsonl").exists():
        raise SystemExit(f"Not a LeRobot dataset: {dataset}")

    info = json.loads((meta / "info.json").read_text())
    ori_fps_dataset = int(info.get("fps", args.ori_fps))
    if ori_fps_dataset != args.ori_fps:
        print(
            f"[warn] --ori-fps={args.ori_fps} differs from info.json fps="
            f"{ori_fps_dataset}; using --ori-fps."
        )

    episodes = [
        json.loads(line)
        for line in (meta / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    cam_keys = args.cam_keys or DEFAULT_CAM_KEYS

    print(f"[extract] loading VAE + text encoder from {args.ckpt_dir}")
    vae = load_vae(args.ckpt_dir, device, dtype)
    tokenizer, text_encoder = load_text_stack(args.ckpt_dir, device, dtype)

    # empty_emb.pt — written once per dataset, regenerate-on-demand.
    empty_emb_path = dataset / "empty_emb.pt"
    if not empty_emb_path.exists() or not args.skip_existing:
        print(f"[extract] writing {empty_emb_path}")
        empty = encode_text("", tokenizer, text_encoder, device, dtype)
        torch.save(empty, empty_emb_path)

    # Cache text embeddings per unique action_text to avoid re-encoding.
    text_cache: dict[str, torch.Tensor] = {}

    def get_text_emb(text: str) -> torch.Tensor:
        if text not in text_cache:
            text_cache[text] = encode_text(
                text, tokenizer, text_encoder, device, dtype
            )
        return text_cache[text]

    total_jobs = sum(
        len(ep.get("action_config", [])) * len(cam_keys) for ep in episodes
    )
    pbar = tqdm(total=total_jobs, desc="extract")
    n_skipped = 0
    n_written = 0
    for ep in episodes:
        ep_idx = ep["episode_index"]
        for acfg in ep.get("action_config", []):
            start_frame = acfg["start_frame"]
            end_frame = acfg["end_frame"]
            text = acfg["action_text"]
            text_emb = get_text_emb(text)

            for cam in cam_keys:
                mp4 = (
                    dataset / "videos" / CHUNK_DIR / cam
                    / f"episode_{ep_idx:06d}.mp4"
                )
                if not mp4.exists():
                    raise FileNotFoundError(mp4)

                out_path = (
                    dataset / "latents" / CHUNK_DIR / cam
                    / f"episode_{ep_idx:06d}_{start_frame}_{end_frame}.pth"
                )

                # `out_path` may be a dangling symlink into the parent dataset
                # (select_subset.py sets these up). Resolve to the symlink
                # target for existence checks; torch.save will follow it.
                target = out_path.resolve()
                if args.skip_existing and target.exists():
                    n_skipped += 1
                    pbar.update(1)
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                target.parent.mkdir(parents=True, exist_ok=True)

                frames, frame_ids = read_video_frames(
                    mp4,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=args.fps,
                    ori_fps=args.ori_fps,
                    target_h=args.height,
                    target_w=args.width,
                )
                latent_flat, T_lat, H_lat, W_lat = encode_video(
                    frames, vae, device, dtype
                )

                payload = {
                    "latent": latent_flat,
                    "latent_num_frames": T_lat,
                    "latent_height": H_lat,
                    "latent_width": W_lat,
                    "video_num_frames": int(frames.shape[0]),
                    "video_height": int(frames.shape[1]),
                    "video_width": int(frames.shape[2]),
                    "text_emb": text_emb,
                    "text": text,
                    "frame_ids": frame_ids,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "fps": int(args.fps),
                    "ori_fps": int(args.ori_fps),
                }
                torch.save(payload, out_path)
                n_written += 1
                pbar.update(1)
    pbar.close()
    print(
        f"[extract] done. written={n_written} skipped={n_skipped} "
        f"unique_texts={len(text_cache)}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="LeRobot v2.1 dataset directory (post-conversion / subset)")
    p.add_argument("--ckpt-dir", type=Path, required=True,
                   help="Base ckpt dir holding vae/, text_encoder/, tokenizer/")
    p.add_argument("--height", type=int, default=128,
                   help="Target frame height fed to the VAE")
    p.add_argument("--width", type=int, default=128,
                   help="Target frame width fed to the VAE")
    p.add_argument("--fps", type=int, default=20,
                   help="Target sampling fps")
    p.add_argument("--ori-fps", type=int, default=20,
                   help="Source video fps (must match the dataset's recording fps)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip episodes whose .pth target already exists.")
    p.add_argument("--cam-keys", nargs="*", default=None,
                   help=f"Override camera keys (default: {DEFAULT_CAM_KEYS})")
    return p.parse_args()


if __name__ == "__main__":
    main()
