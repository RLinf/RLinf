# ALOHA Sandwich RECAP Design

Date: 2026-07-05

## Goal

Add an RLinf-native path to run RECAP off-policy fine-tuning on the ALOHA
sandwich human-in-the-loop dataset and an existing pi0.5 SFT checkpoint.

The target data and checkpoint are:

- Raw HDF5 data:
  `/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_rl`
- SFT checkpoint:
  `/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all/pi05_sandwich_new_all_20260628_193430/49999`
- Reference converter:
  `/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/.worktrees/offline-recap-design/examples/aloha_real/convert_aloha_data_to_lerobot_v21.py`

The result should use RLinf's merged RECAP pipeline from PR
https://github.com/RLinf/RLinf/pull/1015, not a one-off OpenPI worktree
pipeline.

## Current Context

PR 1015 is merged and local RLinf already contains the RECAP stages under:

- `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`
- `examples/offline_rl/advantage_labeling/recap/train_value.py`
- `examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py`
- `examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py`

The local code already has OpenPI ALOHA policy/data config support through
`pi05_aloha_robotwin`, `LeRobotAlohaDataConfig`, and
`aloha_policy.AlohaInputs`. The missing pieces are in the RECAP value and
advantage data paths, which currently recognize `libero`, `libero_v2`,
`franka`, `franka_co_train`, and related variants, but not `aloha`.

The raw sandwich dataset contains 26 HDF5 episodes and 60162 frames. The
episode-level reward split is 17 successful episodes (`reward == 1.0`) and
9 failed episodes (`reward == 0.0`). Each file contains:

- `action` with shape `(T, 14)`
- `observations/qpos` with shape `(T, 14)`
- `observations/qvel` with shape `(T, 14)`
- `observations/effort` with shape `(T, 14)`
- `observations/images/cam_high`
- `observations/images/cam_left_wrist`
- `observations/images/cam_right_wrist`
- scalar `reward`
- `teleop_segments` with shape `(N, 2)`

The SFT checkpoint is an OpenPI/JAX Orbax checkpoint with `params/` and
`assets/pi05_sandwich_new_all/norm_stats.json`. RLinf CFG training expects a
PyTorch-style checkpoint directory with `model_state_dict/full_weights.pt` or
safetensors, so a conversion step is required.

The value-model backbones are available locally under:

- `/inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/siglip2-so400m-patch14-224`
- `/inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/gemma-3-270m`

Both directories contain Hugging Face files such as `config.json`,
`model.safetensors`, tokenizer files, and tokenizer configs.

## Decisions

Use the RLinf-native approach:

1. Add ALOHA as a first-class RECAP robot type.
2. Convert sandwich HDF5 to LeRobot v2.1 without changing ALOHA semantics.
3. Treat the data as `type: rollout`, because it contains both successes and
   failures.
4. Let Step 1 compute returns from `is_success`, with HITL-aware splitting
   enabled for successful episodes that contain `teleop_mask`.
5. Let Step 3 compute continuous RECAP advantages with the value model, then
   force frames inside `teleop_segments` to positive labels for CFG training.
6. Initialize CFG policy training from a PyTorch conversion of the provided
   pi0.5 SFT checkpoint.

## Non-Goals

- Do not replace RLinf's RECAP implementation with OpenPI worktree scripts.
- Do not disguise ALOHA data as LIBERO or Franka data.
- Do not change the mathematical return and advantage definitions outside the
  teleop positive-label override.
- Do not run full long training as part of the first implementation pass unless
  explicitly requested after the pipeline is validated.

## Data Conversion

Add an RLinf converter, proposed path:

`examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py`

The converter will be based on the reference OpenPI script, but scoped to the
fields RLinf needs for RECAP. It will write a LeRobot v2.1 dataset with:

- `observation.images.cam_high`
- `observation.images.cam_left_wrist`
- `observation.images.cam_right_wrist`
- `observation.state` as 14 float32 values from `observations/qpos`
- `action` as 14 float32 values from `action`
- optional `observation.velocity` from `observations/qvel`
- optional `observation.effort` from `observations/effort`
- `task`, using the configured sandwich task prompt
- `is_success`, per frame, derived from scalar episode `reward > 0.0`
- `teleop_mask`, per frame, derived from all `[start, end)` ranges in
  `teleop_segments`

The converter will also write a compact metadata file, for example
`meta/hil_segments.json`, containing each episode's raw reward and teleop
segments. This is for auditability; training code will consume `is_success`
and `teleop_mask`.

The default sandwich task prompt will be "Assemble a sandwich." and can be
overridden by CLI argument.

## HITL-Aware Return Computation

Update `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`:

- Add `data.hitl_aware_returns`.
- Keep the default disabled for existing datasets.
- When enabled, read the optional per-frame `teleop_mask` column.
- For successful episodes with at least one teleop frame, split returns at the
  first intervention frame:
  - Frames before the first intervention use failed returns over the autonomous
    prefix.
  - Frames from the first intervention onward use successful returns over the
    suffix.
- Failed episodes remain failed episodes even if they contain teleop frames.

This prevents the value model from learning that pre-intervention autonomous
failure states are high value just because a human later rescued the episode.
The teleop action preference signal still enters through Step 3's positive
advantage-label override.

## RECAP ALOHA Support

### Value Dataset

Update `rlinf/data/datasets/recap/value_dataset.py`:

- Add `_REPACK_KEYS["aloha"]` mapping:
  - `images.cam_high` from `observation.images.cam_high`
  - `images.cam_left_wrist` from `observation.images.cam_left_wrist`
  - `images.cam_right_wrist` from `observation.images.cam_right_wrist`
  - `state` from `observation.state`
  - `actions` from `action`
  - `prompt` from `prompt`
- Add transform branch for `robot_type == "aloha"` that uses
  `aloha_policy.AlohaInputs`.
- Use `action_dim: 14` and `action_horizon: 10` for sandwich pi0.5 RECAP.

This keeps value-model training aligned with OpenPI ALOHA policy training.

### Advantage Inference

Update
`examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py`:

- Add an ALOHA branch in `build_obs` that constructs the raw observation shape
  expected by `aloha_policy.AlohaInputs`:
  `{"images": {"cam_high": ..., "cam_left_wrist": ..., "cam_right_wrist": ...},
  "state": ..., "prompt": ...}`.
- Include `teleop_mask` in the inference dataset metadata when present.
- Carry `teleop_mask` into the advantages DataFrame.
- During boolean labeling, compute the normal quantile-based label first, then
  set `advantage=True` for every row with `teleop_mask == 1`.
- Store `teleop_positive` in the output parquet so downstream analysis can
  distinguish value-based positive labels from human-correction overrides.
- Log the number and percentage of teleop-overridden positive frames.

The continuous `advantage_continuous` value remains unchanged; only the boolean
training label is overridden.

### Value Checkpoint Inference Transforms

Update `rlinf/models/embodiment/value_model/recap/checkpoint_utils.py`:

- Import `aloha_policy`.
- Add `env_type == "aloha"` support in `build_input_transforms`.
- Use `InjectDefaultPrompt`, `aloha_policy.AlohaInputs`, optional
  normalization, and `PadStatesAndActions(action_dim)`.

This is required because Step 3 loads the trained value model with
`ValueCriticModel.from_checkpoint(..., env_type="aloha")`.

### CFG Policy Training

Use the existing `FSDPCfgWorker` path with:

- `actor.model.model_type: cfg_model`
- `actor.model.openpi.config_name: pi05_aloha_robotwin`
- `actor.model.openpi.positive_only_conditional: true`
- `actor.model.openpi.unconditional_prob: 0.1`
- `actor.model.openpi.cfgrl_guidance_scale: 1.0`
- `actor.model.model_path` pointing to the converted PyTorch checkpoint

No new CFG worker is needed.

## Checkpoint Conversion

Use OpenPI's local converter:

`/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/examples/convert_jax_model_to_pytorch.py`

The design expectation is a converted directory under a writable experiment
area, such as:

`/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all_pytorch/49999`

The converted directory must preserve or copy the checkpoint `assets/`
directory so RLinf can load the sandwich normalization stats.

RLinf should not silently accept the Orbax checkpoint in Step 4. The run
instructions should require conversion first, and the config should point to
the converted PyTorch directory.

## Configuration Shape

Add sandwich-specific configs or documented overrides for these values:

Step 1, returns:

- `data.train_data_paths[0].dataset_path`: converted LeRobot dataset
- `data.train_data_paths[0].type: rollout`
- `data.gamma: 1.0`
- `data.failure_reward: -300.0`
- `data.hitl_aware_returns: true`
- `data.tag: sandwich_fail300`

Step 2, value SFT:

- `data.robot_type: aloha`
- `data.model_type: pi05`
- `data.action_dim: 14`
- `data.action_horizon: 10`
- `data.tag: sandwich_fail300`
- `actor.model.siglip_path`:
  `/inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/siglip2-so400m-patch14-224`
- `actor.model.gemma3_path`:
  `/inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/gemma-3-270m`
- `actor.model.tokenizer_path`:
  `/inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/gemma-3-270m`

If symlinks are preferred for shorter config paths, create them under a repo
root `models/` or `models/hf/google/` directory and keep that directory out of
git. Do not place external model symlinks inside the Python source package
`rlinf/models/`.

Step 3, advantages:

- `advantage.value_checkpoint`: value SFT checkpoint from Step 2
- `advantage.returns_tag: sandwich_fail300`
- `advantage.tag`: e.g. `sandwich_fail300_N10_q30_teleop`
- `advantage.positive_quantile: 0.3`
- `data.model_type: pi05`
- `data.train_data_paths[0].robot_type: aloha`
- `data.advantage_lookahead_step: 10`

Step 4, CFG:

- `data.advantage_tag`: same as Step 3 `advantage.tag`
- `data.train_data_paths[0].dataset_path`: converted LeRobot dataset
- `actor.model.model_path`: converted PyTorch pi0.5 SFT checkpoint
- `actor.model.openpi.config_name: pi05_aloha_robotwin`
- `actor.model.openpi.action_env_dim: 14`

## Validation Plan

Data validation:

- Verify the converter writes 26 episodes and 60162 frames.
- Verify success/failure counts are 17 and 9.
- Verify every `teleop_segments` range maps to `teleop_mask == 1` on the same
  frame interval.
- Verify image columns, state, and action appear in `meta/info.json`.

Unit tests:

- Add a small ALOHA LeRobot fixture or synthetic metadata-backed sample.
- Test `ValueDataset` can build an ALOHA transformed sample.
- Test HITL-aware Step 1 returns split successful intervention episodes at the
  first teleop frame and leave failed intervention episodes failed.
- Test `compute_advantages.build_obs(..., robot_type="aloha")` maps cameras,
  state, and prompt.
- Test teleop label override changes only the boolean `advantage` label and
  preserves `advantage_continuous`.

Smoke tests:

- Run Step 1 on the converted sandwich dataset and verify
  `meta/returns_sandwich_fail300.parquet`.
- Run Step 3 with a very small `advantage.max_samples` once a value checkpoint
  exists, verifying `teleop_positive` appears in the output parquet.
- Run Step 4 until dataloader/model initialization succeeds with a tiny
  training budget.

Full training:

- Value SFT and CFG fine-tuning are GPU-heavy and should be run separately with
  explicit resource allocation.
- Full run success criteria are decreasing value loss, non-degenerate advantage
  label distribution, and CFG training loss producing checkpoints.

## Risks

- The value model depends on local SigLIP2 and Gemma3 paths. Known-good paths
  are listed above. The implementation must still fail early with a clear error
  when those paths are missing, point at incomplete directories, or do not
  contain Hugging Face model/tokenizer files.
- OpenPI checkpoint conversion may need enough CPU memory and disk space for a
  full pi0.5 checkpoint.
- `pi05_aloha_robotwin` uses ALOHA action/state adaptation defaults. If the
  original sandwich SFT used different `adapt_to_pi` or delta-action settings,
  the RLinf config must match that training setup before a long run.
- The dataset contains episode-level rewards, not dense task rewards. This is
  expected for RECAP, but the return signal will mostly encode time-to-success
  and terminal failure penalty.

## Approval State

The user approved the RLinf-native approach and the recommended teleop-positive
label rule before this spec was written.
