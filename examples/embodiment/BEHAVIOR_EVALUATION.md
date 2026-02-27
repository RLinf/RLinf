# BEHAVIOR Model Evaluation in RLinf

Complete guide for evaluating BEHAVIOR models using RLinf's worker-based architecture.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Configuration Guide](#configuration-guide)
- [Multi-GPU Evaluation](#multi-gpu-evaluation)
- [Video Recording](#video-recording)
- [BEHAVIOR Tasks](#behavior-tasks)
- [Metrics and Output](#metrics-and-output)

---

## Quick Start

### Single Command Evaluation

```bash
cd /mnt/public/quanlu/behavior_sft_dev/RLinf

# Run evaluation using the example script
bash examples/embodiment/eval_behavior_example.sh
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `env.eval.task_idx` | BEHAVIOR task index (0-49) | `0` (turning_on_radio) |
| `env.eval.total_num_envs` | Number of parallel environments | `2` |
| `algorithm.eval_rollout_epoch` | Number of evaluation episodes | `10` |
| `env.eval.video_cfg.save_video` | Whether to save videos | `True`/`False` |
| `rollout.model.model_path` | Path to model checkpoint | `/path/to/model/` |

---

## Architecture Overview

### Why Not OpenPI-Comet's Client/Server?

OpenPI-Comet uses a client/server architecture with WebSocket communication:

```
Terminal 1: Policy Server          Terminal 2: Environment Client
┌─────────────────────┐            ┌──────────────────────┐
│  serve_b1k.py       │◄──WebSocket─►│  eval.py            │
│  GPU 1: Policy      │            │  GPU 0: Environment  │
└─────────────────────┘            └──────────────────────┘
```

**Limitations:**
- ❌ Requires two separate processes
- ❌ Manual GPU management via `CUDA_VISIBLE_DEVICES`
- ❌ WebSocket overhead
- ❌ Single environment per client
- ❌ Complex error handling

### RLinf's Worker-Based Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Ray Cluster (Single Command)             │
│                                                                │
│  ┌───────────────────┐                      ┌──────────────┐ │
│  │  Env Workers      │◄───RLinf Channels───►│ Rollout      │ │
│  │  (GPU 0-1)        │                      │ Workers      │ │
│  │                   │                      │ (GPU 2-3)    │ │
│  │ - OmniGibson      │   Observations       │              │ │
│  │ - BEHAVIOR Envs   │─────────────────────►│ - Policy     │ │
│  │ - Metrics         │                      │ - Actions    │ │
│  │ - Video Recording │   Actions            │              │ │
│  │                   │◄─────────────────────│              │ │
│  │ 2 Parallel Envs   │                      │ 2 Workers    │ │
│  └───────────────────┘                      └──────────────┘ │
│                                                                │
│         ▲                                          ▲           │
│         └─────────Coordinated by───────────────────┘           │
│                  EmbodiedEvalRunner                            │
└──────────────────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ Single command execution
- ✅ Automatic GPU placement via Ray
- ✅ Parallel environment execution
- ✅ Efficient RLinf channel communication
- ✅ Scales to multi-GPU and multi-node
- ✅ Built-in fault tolerance
- ✅ Consistent with training pipeline

---

## Configuration Guide

### Main Configuration File

`examples/embodiment/config/behavior_openvlaoft_eval.yaml`:

```yaml
defaults:
  - base_config
  - _self_
  - override embodiment_env: behavior
  - env@env.eval: behavior_r1pro
  - training_backend/fsdp@actor.fsdp_config  # Required for model loading

# GPU Placement
cluster:
  num_nodes: 1
  component_placement:
    actor: 0-0    # Not used in eval, but required by validator
    env: 0-1      # Environment workers on GPU 0 and GPU 1
    rollout: 2-3  # Rollout workers on GPU 2 and GPU 3
    # NOTE: num_rollout_workers must be divisible by num_env_workers!

# Evaluation Mode
runner:
  only_eval: True
  logger:
    log_path: ./logs/behavior_eval

# Environment Settings
env:
  eval:
    task_idx: 0                          # BEHAVIOR task (0-49)
    total_num_envs: 2                    # Number of parallel environments
    max_steps_per_rollout_epoch: 1024    # Must be divisible by num_action_chunks
    video_cfg:
      save_video: True
      video_base_dir: ./logs/behavior_eval/video/eval
      use_high_res_cameras: True         # Enable high-resolution video

# Model Settings
actor:
  model:
    num_action_chunks: 32                # Must match training config
    num_images_in_input: 3               # Head + 2 wrist cameras

rollout:
  model:
    model_path: /mnt/public/quanlu/RLinf-OpenVLAOFT-Behavior/  # Model directory

# Algorithm Settings
algorithm:
  eval_rollout_epoch: 2                  # Number of evaluation episodes
  sampling_params:
    temperature_eval: 0.6
```

### Important Configuration Rules

1. **Action Chunk Divisibility**
   ```yaml
   # max_steps_per_rollout_epoch MUST be divisible by num_action_chunks
   env.eval.max_steps_per_rollout_epoch: 1024  # ✅ 1024 % 32 = 0
   actor.model.num_action_chunks: 32
   ```

2. **Worker Count Relationship**
   ```yaml
   # num_rollout_workers MUST be divisible by num_env_workers
   cluster.component_placement:
     env: 0-1      # 2 env workers     ✅
     rollout: 2-3  # 2 rollout workers ✅ (2 % 2 = 0)
   ```

3. **Model Path**
   ```yaml
   # Use Hugging Face model directory (with .safetensors files)
   rollout.model.model_path: /path/to/model/    # ✅ Directory path
   
   # NOT the old .pt checkpoint format
   # runner.eval_policy_path: /path/model.pt    # ❌ Don't use this
   ```

---

## Multi-GPU Evaluation

### Memory Requirements

| Component | GPU Memory (Approx) |
|-----------|---------------------|
| OmniGibson Env (1 instance) | ~4-6 GB |
| OmniGibson Env (high-res cameras) | ~6-8 GB |
| OpenVLA Policy (7B params) | ~14-16 GB |

| Configuration | Min Memory | Recommended |
|--------------|------------|-------------|
| Option 1 (dedicated) | 16 GB/GPU | 24 GB/GPU |
| Option 2 (shared) | 24 GB/GPU | 40 GB/GPU |
| Option 3 (single) | 24 GB | 40 GB |

### Valid Worker Combinations

Remember: `num_rollout_workers % num_env_workers == 0`

| Env Workers | Rollout Workers | Valid? | Use Case |
|-------------|-----------------|--------|----------|
| 1 | 1 | ✅ (1 % 1 = 0) | Single GPU baseline |
| 2 | 2 | ✅ (2 % 2 = 0) | Standard 4-GPU setup |
| 2 | 4 | ✅ (4 % 2 = 0) | Heavy policy inference |
| 1 | 2 | ✅ (2 % 1 = 0) | Light environment load |
| 2 | 1 | ❌ (1 % 2 = 1) | INVALID! |
| 3 | 2 | ❌ (2 % 3 ≠ 0) | INVALID! |

---

## Video Recording

### Automatic Video Saving

When `save_video: True`, each environment worker records its own video:

```yaml
env:
  eval:
    video_cfg:
      save_video: True
      video_base_dir: ./logs/behavior_eval/video/eval
      use_high_res_cameras: True  # Enable high-resolution cameras
```

### Video File Naming

Each worker creates a uniquely named video file to prevent overwrites:

```
logs/behavior_eval/video/eval/
├── behavior_video_rank0_gpu0.mp4  # From EnvWorker on rank 0, GPU 0
└── behavior_video_rank1_gpu1.mp4  # From EnvWorker on rank 1, GPU 1
```

**Naming format:** `behavior_video_rank{rank}_gpu{accelerator_rank}.mp4`

**Implementation:**
```python
# In rlinf/envs/behavior/behavior_env.py
worker_id = f"rank{worker_info.rank}_gpu{worker_info.accelerator_rank}"
video_name = f"{video_base_dir}/behavior_video_{worker_id}.mp4"
```

**Benefits:**
- ✅ No file conflicts between workers
- ✅ Each environment's rollout recorded separately
- ✅ Easy to identify which worker generated which video
- ✅ Parallel video writing doesn't block

### Video Resolution

#### Standard Resolution (Default)

```yaml
env.eval.video_cfg.use_high_res_cameras: False
```

- Head camera: 128×128
- Wrist cameras: 128×128 each
- Video size: 256×128 (side-by-side layout)

#### High Resolution

```yaml
env.eval.video_cfg.use_high_res_cameras: True
```

- Head camera: 960×960 (zed camera native resolution)
- Wrist cameras: 480×480 each (RealSense native resolutions)
- Video size: 1920×960 (side-by-side layout)

**Implementation:** Uses custom `RGBWrapper` to set native camera resolutions before environment initialization, matching OpenPI-Comet's approach.

**Note:** High-resolution cameras increase GPU memory usage by ~2-4 GB per environment.

---

## BEHAVIOR Tasks

The BEHAVIOR benchmark includes 50 household manipulation tasks:

| Task Index | Task Name | Description |
|------------|-----------|-------------|
| 0 | turning_on_radio | Turn on a radio device |
| 1 | opening_packages | Open sealed packages |
| 2 | packing_grocery_items_into_bags | Pack groceries into bags |
| 3 | cleaning_windows | Clean window surfaces |
| 4 | setting_table | Set up a dining table |
| 5 | putting_away_groceries | Store groceries properly |
| ... | ... | ... |
| 49 | ... | ... |

**Full task list:** See `rlinf/envs/behavior/behavior_task.jsonl`

---

## Metrics and Output

### Evaluation Metrics

Automatically collected by `EnvWorker`:

- **Success Rate**: Percentage of successful task completions
- **Episode Length**: Average number of steps per episode
- **Episode Return**: Cumulative rewards
- **Task-Specific Metrics**: Provided by OmniGibson BEHAVIOR evaluator

### Output Files

```
logs/behavior_eval/
├── video/
│   └── eval/
│       ├── behavior_video_rank0_gpu0.mp4  # Videos from each worker
│       └── behavior_video_rank1_gpu1.mp4
├── tensorboard/                           # TensorBoard logs
│   └── events.out.tfevents.*
└── checkpoints/                           # (Training only)
```

### Viewing Results

```bash
# View TensorBoard logs
tensorboard --logdir logs/behavior_eval/tensorboard

# Check videos
ls -lh logs/behavior_eval/video/eval/
vlc logs/behavior_eval/video/eval/behavior_video_rank0_gpu0.mp4
```

---

## References

- [RLinf Documentation](../../docs/source-en/rst_source/examples/behavior.rst)
- [OmniGibson Documentation](https://behavior.stanford.edu/omnigibson/)
- [BEHAVIOR Benchmark](https://behavior.stanford.edu/)
- [Example Configuration](./config/behavior_openvlaoft_eval.yaml)
- [Example Evaluation Script](./eval_behavior_example.sh)

---
