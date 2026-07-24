# FastWAM in RLinf

[FastWAM](https://github.com/yuantianyuan01/FastWAM) (*Fast-WAM: Do World Action
Models Need Test-time Future Imagination?*) is a Wan2.2 video-diffusion world model
with a flow-matching action expert. This package adapts FastWAM's own implementation
to RLinf so it can be **evaluated** (LIBERO / LIBERO-Plus) and **SFT-trained** through
RLinf's standard workers, by wrapping the upstream package rather than reimplementing it.

## Layout

| File | Purpose |
|------|---------|
| `fastwam_policy.py` | `FastWAMPolicy(BasePolicy)` — `predict_action_batch`→`infer_action`, `sft_forward`→`training_loss` |
| `__init__.py` | `get_model` — composes FastWAM's Hydra configs, builds model + processor, loads checkpoint |
| `../../../data/datasets/fastwam/` | SFT dataloader wrapping FastWAM `RobotVideoDataset` |

Registered in `rlinf/config.py` (`SupportedModel.FASTWAM`, `EMBODIED_MODEL`),
`rlinf/models/__init__.py` (`_build_fastwam`), and dispatched in
`rlinf/workers/sft/fsdp_vla_sft_worker.py`.

## Prerequisites

- Install FastWAM as a package (`pip install -e /path/to/FastWAM`) so `import fastwam` works.
- Wan2.2 VAE + T5 are fetched by DiffSynth on first model build. The default converted
  -safetensors mirror is **ModelScope-only**; configs set `model.redirect_common_files=false`
  to use the original `Wan-AI/Wan2.2-TI2V-5B` `.pth` files on HuggingFace.
- Eval uses `skip_dit_load_from_pretrain=true`: the released FastWAM checkpoint's `mot`
  provides the trained video+action experts, so the 20GB Wan DiT is **not** downloaded.

## Evaluation (LIBERO + LIBERO-Plus)

```bash
# weights
huggingface-cli download yuanty/fastwam libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json --local-dir /workspace/checkpoints/fastwam

# LIBERO
MUJOCO_GL=egl bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

# LIBERO-Plus  (needs the liberoplus package + its assets.zip, and ImageMagick)
LIBERO_TYPE=plus LIBERO_SUFFIX=all MUJOCO_GL=egl \
  bash evaluations/run_eval.sh libero libero_spatial_fastwam_plus_eval
```

Set `rollout.model.checkpoint_path` / `dataset_stats_path` (configs default to
`/workspace/checkpoints/fastwam/...`). `num_action_chunks` is the executed replan
length; FastWAM predicts `action_horizon` (=`num_frames-1`) per inference.

## SFT

```bash
# 1) LeRobot LIBERO data + 2) precomputed T5 text embeddings
python FastWAM/scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 \
  'data.train.dataset_dirs=[/path/to/libero_spatial_no_noops_lerobot]' \
  data.train.text_embedding_cache_dir=/workspace/data/text_embeds_cache/libero \
  model.redirect_common_files=false

bash examples/sft/run_vla_sft.sh libero_sft_fastwam
```

SFT initialises the MoT from the released checkpoint (`skip_dit_load_from_pretrain`),
so only VAE+T5 + the 12GB checkpoint are needed (no 20GB DiT / ActionDiT backbone).
Only the MoT experts (+ proprio encoder) are trained (`freeze_non_dit`).

**Multi-GPU note:** the full MoT is ~6B trainable params, and the MoT performs *manual*
cross-expert attention (accessing DiT-block internals rather than `block.forward`),
which is incompatible with per-block FSDP2 auto-wrap. Full-MoT SFT therefore needs
multi-GPU FSDP with a whole-model wrap policy; on a single GPU it OOMs. See
`smoke_sft.py` for a single-GPU action-expert smoke that exercises the full
dataloader → `sft_forward` → `training_loss` → backward path.
