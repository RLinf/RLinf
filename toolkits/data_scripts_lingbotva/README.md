# LingBot-VA Data Preparation Scripts

End-to-end recipe that turns raw LIBERO-Object HDF5 demonstrations into
the LeRobot v2.1 + pre-extracted Wan 2.2 VAE latents bundle that the
LingBot-VA SFT loader (`rlinf/data/datasets/lingbotva.py`) expects.

These three scripts must be run in order — each consumes the output of
the previous one.

| Step | Script | Input | Output |
|---|---|---|---|
| (a) | [`convert_libero_object_to_lerobot.py`](convert_libero_object_to_lerobot.py) | `${LIBERO_RAW_DIR}/libero_object/*.hdf5` (10 files, 50 demos each, from `yifengzhu-hf/LIBERO-datasets`) | `${LINGBOT_VA_DATASET_PATH}_full/` — LeRobot v2.1 dataset with LingBot-VA's `action_config` annotation injected into `episodes.jsonl` and both cameras vertically flipped to display orientation. |
| (b) | [`select_subset.py`](select_subset.py) | `${LINGBOT_VA_DATASET_PATH}_full/` | `${LINGBOT_VA_DATASET_PATH}/` — deterministic per-task subset (default 10 demos × 10 tasks via `--per-task 10 --seed 42`). Symlinks parquet + video files from the parent dataset so the subset doesn't duplicate ~7 GB of mp4s. |
| (c) | [`extract_latents.py`](extract_latents.py) | `${LINGBOT_VA_DATASET_PATH}/` and the LingBot-VA base checkpoint (`${LINGBOT_VA_MODEL_PATH}`) | Pre-extracted Wan 2.2 VAE latents at `latents/chunk-000/observation.images.*/episode_NNNNNN_<s>_<e>.pth` plus a cached UMT5 empty-prompt embedding at `empty_emb.pt`. |

## Usage

```bash
# Set paths
export LIBERO_RAW_DIR=<your-libero-raw-dir>
export LINGBOT_VA_DATASET_PATH=<your-dataset-dir>
export LINGBOT_VA_MODEL_PATH=<your-model-dir>

# (a) HDF5 -> LeRobot v2.1
python toolkits/data_scripts_lingbotva/convert_libero_object_to_lerobot.py \
  --hdf5-root ${LIBERO_RAW_DIR}/libero_object \
  --output ${LINGBOT_VA_DATASET_PATH}_full

# (b) Subsample 10 demos x 10 tasks (deterministic via seed 42)
python toolkits/data_scripts_lingbotva/select_subset.py \
  --src ${LINGBOT_VA_DATASET_PATH}_full \
  --dst ${LINGBOT_VA_DATASET_PATH} \
  --per-task 10 --seed 42

# (c) Pre-extract Wan 2.2 VAE latents + cache UMT5 empty embedding
python toolkits/data_scripts_lingbotva/extract_latents.py \
  --dataset ${LINGBOT_VA_DATASET_PATH} \
  --model-path ${LINGBOT_VA_MODEL_PATH}
```

## Prerequisites

- `LINGBOT_VA_REPO_PATH` and `PYTHONPATH` set so `wan_va` and its
  dependencies (`AutoencoderKLWan`, `UMT5EncoderModel`, etc.) import. See
  `examples/embodiment/config/model/lingbotva.yaml` and
  `docs/source-en/rst_source/examples/embodied/sft_lingbotva.rst` for the
  full environment setup.
- Raw HDF5 demos from **`yifengzhu-hf/LIBERO-datasets`** specifically — the
  IPEC-COMMUNITY mirror uses a different wrist-camera mount and produces
  0 % eval SR even though training looks healthy.

## Expected layout under `${LINGBOT_VA_DATASET_PATH}` after step (c)

```
meta/{info.json,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl}
data/chunk-000/episode_NNNNNN.parquet                                       (100 episodes)
videos/chunk-000/observation.images.{agentview_rgb,eye_in_hand_rgb}/episode_NNNNNN.mp4
latents/chunk-000/observation.images.{agentview_rgb,eye_in_hand_rgb}/episode_NNNNNN_<s>_<e>.pth
empty_emb.pt
```

## Notes

- **`(b) → (c)` symlink trick.** `select_subset.py` pre-creates dangling
  symlinks under the subset's `latents/...` tree pointing at the parent
  dataset's `latents/...` slot. When `extract_latents.py` then runs on
  the subset, `torch.save` follows the symlink and writes the actual
  `.pth` into the parent dataset — so additional subsets that share
  episodes don't have to re-encode the same frames.
- If `lerobot>=0.3.3` rejects the dataset because `episodes_stats.jsonl`
  has image stats stored as `(3,)` instead of `(3, 1, 1)`, reshape the
  per-channel stats in place (the `_full` copy from step (a) preserves
  the original).
