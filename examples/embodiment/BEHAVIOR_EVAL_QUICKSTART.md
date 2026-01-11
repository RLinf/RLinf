# BEHAVIOR Evaluation Quick Start Guide

This guide helps you quickly get started with evaluating BEHAVIOR models using RLinf's worker-based architecture.

## What Was Implemented

We've created a complete BEHAVIOR evaluation solution that **avoids the client/server architecture** used in OpenPI-Comet, and instead uses **RLinf's worker-based approach** with:

1. **Environment Workers**: Run OmniGibson BEHAVIOR environments
2. **Rollout Workers**: Run the policy model
3. **Ray Channels**: Direct communication (no WebSocket needed)

## Quick Comparison

### ❌ OpenPI-Comet Way (What We're NOT Using)

```bash
# Terminal 1: Start policy server
CUDA_VISIBLE_DEVICES=1 python scripts/serve_b1k.py \
    --task-name=turning_on_radio \
    --policy.dir=/path/to/model

# Terminal 2: Start environment client (connects via WebSocket)
CUDA_VISIBLE_DEVICES=0 python omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio
```

**Issues**: Separate processes, WebSocket overhead, manual GPU management

### ✅ RLinf Way (What We Implemented)

```bash
# Single command - workers automatically distributed
python examples/embodiment/eval_embodied_agent.py \
    --config-name behavior_openvlaoft_eval \
    runner.eval_policy_path=/path/to/model.pt \
    env.eval.task_idx=0
```

**Benefits**: Automatic GPU placement, parallel execution, unified architecture

## Usage Examples

### Example 1: Evaluate OpenVLA-OFT Model on BEHAVIOR

```bash
cd /mnt/public/quanlu/behavior_sft_dev/RLinf

# Set environment
export EMBODIED_PATH="$(pwd)/examples/embodiment"
export PYTHONPATH=$(pwd):$PYTHONPATH

# Evaluate on "turning_on_radio" task (task_idx=0)
python examples/embodiment/eval_embodied_agent.py \
    --config-name behavior_openvlaoft_eval \
    runner.eval_policy_path=/mnt/public/quanlu/pi05-turning_on_radio-sft/model.pt \
    runner.logger.log_path=./logs/behavior_eval \
    env.eval.task_idx=0 \
    env.eval.total_num_envs=2 \
    env.eval.video_cfg.save_video=True \
    algorithm.eval_rollout_epoch=10 \
    actor.model.action_chunk=32
```

### Example 2: Quick Evaluation with Shell Script

```bash
# Use the provided example script
bash examples/embodiment/eval_behavior_example.sh

# Or customize it:
bash examples/embodiment/eval_behavior_example.sh \
    TASK_NAME="opening_packages" \
    MODEL_PATH="/path/to/your/model"
```

### Example 3: Detailed Single-Threaded Evaluation

For debugging or detailed per-instance metrics:

```bash
# Using the convenience script
bash toolkits/eval_scripts_openpi/eval_behavior.sh \
    --task_name turning_on_radio \
    --model_path /mnt/public/quanlu/pi05-turning_on_radio-sft \
    --policy_type rlinf \
    --action_chunk 32 \
    --max_steps 2000 \
    --log_path ./detailed_eval_results
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      RLinf Ray Cluster                         │
│                                                                │
│  ┌───────────────────┐         Ray          ┌──────────────┐ │
│  │  Env Workers      │◄─────Channels───────►│ Rollout      │ │
│  │                   │                      │ Workers      │ │
│  │ - OmniGibson      │   Observations       │              │ │
│  │ - BEHAVIOR Tasks  │◄─────────────────────│ - Policy     │ │
│  │ - Metrics         │                      │ - Actions    │ │
│  │                   │   Actions            │              │ │
│  │ Multiple Envs     │─────────────────────►│ GPU 0-1      │ │
│  │ GPU 0-1           │                      │              │ │
│  └───────────────────┘                      └──────────────┘ │
│                                                                │
│         ▲                                          ▲           │
│         │                                          │           │
│         └──────────Coordinated by──────────────────┘           │
│                  EmbodiedEvalRunner                            │
└──────────────────────────────────────────────────────────────┘
```

## What's Different from OpenPI-Comet?

| Aspect | OpenPI-Comet | RLinf (Our Implementation) |
|--------|--------------|----------------------------|
| **Architecture** | Client/Server + WebSocket | Worker-based + Ray Channels |
| **Processes** | 2 separate processes | Unified Ray cluster |
| **Communication** | WebSocket (network) | Ray channels (optimized) |
| **GPU Management** | Manual (CUDA_VISIBLE_DEVICES) | Automatic via Ray |
| **Scalability** | Single client/server pair | Multi-GPU, multi-node |
| **Setup** | Start server, then client | Single command |
| **Integration** | Separate from training | Same as training pipeline |

## Key Features

### ✅ Worker-Based Architecture
- **Env Workers**: Handle environment stepping, observation collection, metrics
- **Rollout Workers**: Handle policy inference, action generation
- **Communication**: Direct via Ray channels (no WebSocket overhead)

### ✅ Automatic GPU Placement
```yaml
cluster:
  component_placement:
    env: 0      # Env workers on GPU 0
    rollout: 1  # Rollout workers on GPU 1
```

### ✅ Parallel Evaluation
- Multiple environments run simultaneously
- Faster than single-threaded evaluation
- Scales to multi-node clusters

### ✅ Two Evaluation Modes

1. **Parallel (Fast)**: Use `eval_embodied_agent.py`
   - Multiple parallel environments
   - Distributed across GPUs/nodes
   - Production evaluation

2. **Single-Threaded (Detailed)**: Use `toolkits/eval_scripts_openpi/behavior_eval.py`
   - Per-instance metrics
   - Video recording
   - Debugging

## File Structure

```
RLinf/
├── examples/embodiment/
│   ├── eval_embodied_agent.py          # Main parallel evaluation
│   ├── eval_behavior_example.sh         # Example usage script
│   ├── BEHAVIOR_EVALUATION.md           # Detailed documentation
│   ├── BEHAVIOR_EVAL_QUICKSTART.md      # This file
│   └── config/
│       └── behavior_openvlaoft_eval.yaml  # Evaluation config
│
├── toolkits/eval_scripts_openpi/
│   ├── behavior_eval.py                 # Single-threaded evaluation
│   ├── eval_behavior.sh                 # Convenience script
│   └── README.md                        # Updated with BEHAVIOR docs
│
├── rlinf/
│   ├── envs/behavior/
│   │   └── behavior_env.py              # BEHAVIOR environment wrapper
│   ├── workers/
│   │   ├── env/env_worker.py            # Environment worker
│   │   └── rollout/hf/huggingface_worker.py  # Rollout worker
│   └── runners/
│       └── embodied_eval_runner.py      # Evaluation orchestrator
│
└── ray_utils/
    └── start_ray.sh                     # Ray cluster setup
```

## Configuration Files

### Main Evaluation Config: `config/behavior_openvlaoft_eval.yaml`

```yaml
# Key settings for BEHAVIOR evaluation
runner:
  only_eval: True
  eval_policy_path: /path/to/checkpoint.pt

env:
  eval:
    task_idx: 0  # BEHAVIOR task (0-49)
    total_num_envs: 2  # Parallel environments
    max_steps_per_rollout_epoch: 2000

algorithm:
  eval_rollout_epoch: 10  # Number of episodes

actor:
  model:
    action_chunk: 32  # Must match training
```

## Common Tasks

### Evaluate Different BEHAVIOR Tasks

```bash
# Task 0: turning_on_radio
python examples/embodiment/eval_embodied_agent.py \
    --config-name behavior_openvlaoft_eval \
    env.eval.task_idx=0

# Task 1: opening_packages
python examples/embodiment/eval_embodied_agent.py \
    --config-name behavior_openvlaoft_eval \
    env.eval.task_idx=1

# Task 2: packing_grocery_items_into_bags
python examples/embodiment/eval_embodied_agent.py \
    --config-name behavior_openvlaoft_eval \
    env.eval.task_idx=2
```

### Save Evaluation Videos

```yaml
env:
  eval:
    video_cfg:
      save_video: True
      video_base_dir: ./logs/eval_videos
```

### Multi-GPU Evaluation

```yaml
cluster:
  num_nodes: 1
  component_placement:
    env: 0-1      # Env workers on GPU 0 and 1
    rollout: 0-1  # Rollout workers on GPU 0 and 1
```

## Expected Output

### Console Output
```
[INFO] Starting evaluation...
[INFO] Task: turning_on_radio (index: 0)
[INFO] Num instances: 10
[INFO] Episodes per instance: 1
[INFO] Evaluating instance 0... Success! (342 steps)
[INFO] Evaluating instance 1... Success! (298 steps)
...
[INFO] Evaluation complete!
[INFO] Success rate: 70.0%
[INFO] Avg episode length: 315.2
```

### Output Files
```
logs/behavior_eval/
├── metrics/
│   ├── turning_on_radio_summary.json    # Overall metrics
│   ├── turning_on_radio_instances.json  # Per-instance metrics
│   └── turning_on_radio_detailed.json   # Detailed results
├── videos/
│   ├── video_0_0.mp4                    # Evaluation videos
│   ├── video_1_0.mp4
│   └── ...
└── tensorboard/                         # TensorBoard logs
```

## Troubleshooting

### Issue: "No Ray cluster found"
```bash
# Start Ray cluster
cd /mnt/public/quanlu/behavior_sft_dev/RLinf
export RANK=0  # 0 for head node
bash ray_utils/start_ray.sh

# Check Ray status
ray status
```

### Issue: "CUDA out of memory"
```yaml
# Reduce parallel environments
env.eval.total_num_envs: 1

# Enable model offloading
rollout.enable_offload: True
```

### Issue: "Task index out of range"
```python
# Valid task indices: 0-49
# Check available tasks:
from omnigibson.learning.utils.eval_utils import TASK_INDICES_TO_NAMES
print(TASK_INDICES_TO_NAMES)
```

## Next Steps

1. **Read Full Documentation**: See `BEHAVIOR_EVALUATION.md` for detailed architecture explanation
2. **Customize Evaluation**: Modify config files for your needs
3. **Add Custom Metrics**: Extend `BehaviorEnv` class
4. **Scale to Multi-Node**: Configure Ray cluster for distributed evaluation

## Summary

You now have a complete BEHAVIOR evaluation system that:
- ✅ Uses RLinf's worker-based architecture (NOT client/server)
- ✅ Supports parallel, distributed evaluation
- ✅ Provides both fast and detailed evaluation modes
- ✅ Integrates seamlessly with RLinf training pipeline
- ✅ Scales to multi-GPU and multi-node setups

For questions or issues, refer to:
- `BEHAVIOR_EVALUATION.md`: Detailed documentation
- `eval_behavior_example.sh`: Example usage
- `toolkits/eval_scripts_openpi/README.md`: Evaluation script docs

