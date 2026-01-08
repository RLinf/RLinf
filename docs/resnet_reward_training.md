# ResNet Reward Model Training Guide

This guide explains how to train a ManiSkill PickCube task using a ResNet-based reward model.

## Overview

The ResNet reward model is an image-based binary classifier that predicts whether the robot has successfully completed the grasping task. The complete pipeline consists of four stages:

1. **Data Collection**: Collect RGB images with success/failure labels while training a policy
2. **ResNet Training**: Train the ResNet binary classifier on collected data
3. **Policy Pre-training**: Train an initial policy using the environment's dense reward
4. **ResNet Reward Training**: Continue training using ResNet reward to replace the environment reward

## Prerequisites

- ManiSkill environment properly installed
- GPU with sufficient memory for rendering and training

---

## Stage 1: Data Collection

Collect RGB images labeled as success/failure while training a policy with dense reward.

```bash
bash examples/embodiment/run_embodiment.sh maniskill_collect_reward_data
```

This will:
- Train a policy using dense reward (same as `maniskill_ppo_mlp`)
- Render RGB images during training
- Save success/failure images to `examples/embodiment/data/`

### Configuration (`maniskill_collect_reward_data.yaml`)

```yaml
reward_data_collection:
  enabled: True
  save_dir: "${oc.env:EMBODIED_PATH}/data"
  target_success: 5000      # Number of success samples to collect
  target_failure: 5000      # Number of failure samples to collect
  sample_rate_fail: 0.01    # Sample 1% of failure frames
  sample_rate_success: 1.0  # Sample 100% of success frames
```

Data will be saved as:
```
examples/embodiment/data/
├── success/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── failure/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

---

## Stage 2: Train ResNet Reward Model

Train the ResNet binary classifier on collected data.

```bash
python examples/embodiment/train_reward_model.py --config-name maniskill_train_reward_model
```

### Configuration (`maniskill_train_reward_model.yaml`)

```yaml
reward_model_training:
  data_path: "${oc.env:EMBODIED_PATH}/data"
  epochs: 100
  batch_size: 64
  lr: 1.0e-4
  val_split: 0.1
  save_dir: "${oc.env:EMBODIED_PATH}/reward_checkpoints"
  early_stopping_patience: 15
```

The trained model will be saved to `examples/embodiment/reward_checkpoints/best_model.pt`.

---

## Stage 3 & 4: Policy Training with ResNet Reward

### Stage 3: Pre-train Policy with Dense Reward (Optional)

Train an initial policy using the environment's native dense reward (100 steps by default):

```bash
bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp
```

Checkpoints will be saved to `logs/<timestamp>-maniskill_ppo_mlp/maniskill_ppo_mlp/checkpoints/`.

Training will automatically stop at step 100.

### Stage 4: Continue Training with ResNet Reward

Update `resume_dir` in `maniskill_ppo_mlp_resnet_reward.yaml` to point to the Stage 3 checkpoint:

```yaml
runner:
  # TODO: Set to your maniskill_ppo_mlp checkpoint path
  resume_dir: "logs/<timestamp>-maniskill_ppo_mlp/maniskill_ppo_mlp/checkpoints/global_step_100"
```

Then run:

```bash
bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward
```

---

## Configuration

### Key Parameters (`maniskill_ppo_mlp_resnet_reward.yaml`)

```yaml
env:
  train:
    reward_render_mode: "episode_end"  # Must match data collection
    show_goal_site: True               # Show green goal marker
    init_params:
      control_mode: "pd_joint_delta_pos"  # Must match data collection

reward:
  use_reward_model: True
  reward_model_type: "resnet"
  mode: "replace"  # Replace env reward with ResNet reward
  alpha: 1.0
  
  resnet:
    checkpoint_path: "${oc.env:EMBODIED_PATH}/reward_checkpoints/best_model.pt"
    threshold: 0.5
    use_soft_reward: False  # Binary 0/1 reward
```

### Critical Parameter Alignment

The following parameters **must** match those used during data collection:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `control_mode` | `pd_joint_delta_pos` | Control mode (8-dim action space) |
| `reward_render_mode` | `episode_end` | Only render images at episode end |
| `show_goal_site` | `True` | Show green goal marker |
| `image_size` | `[3, 224, 224]` | Image dimensions |

---

## Expected Results

- After ~500-1000 steps, `env/success_once` should approach 100%
- `env/episode_len` should decrease to ~15-20 steps
- `env/reward` will show lower values (expected for sparse binary reward)

---
