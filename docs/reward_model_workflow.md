# Reward Model Workflow Documentation

This document provides a comprehensive guide for the reward model workflow, including data collection, reward model training, and using the trained reward model to replace environment success signals for training RL policies (SAC or PPO).

## Table of Contents

1. [Overview](#overview)
2. [Key Files and Components](#key-files-and-components)
3. [Stage 1: Data Collection](#stage-1-data-collection)
4. [Stage 2: Reward Model Training](#stage-2-reward-model-training)
5. [Stage 3: Training RL Policy with Reward Model](#stage-3-training-rl-policy-with-reward-model)
6. [Complete Workflow Summary](#complete-workflow-summary)
7. [Configuration Examples](#configuration-examples)
8. [Common Issues and Solutions](#common-issues-and-solutions)

## Overview

The reward model workflow consists of three main stages:

1. **Data Collection**: Collect trajectories using an initial RL policy, storing images and success labels
2. **Reward Model Training**: Train a binary classifier to predict success from single frames
3. **RL Policy Training**: Use the trained reward model to replace environment success signals

## Key Files and Components

### Data Collection

#### `rlinf/models/reward_model/reward_data_collector.py`
**Purpose**: Core data collection implementation
- **Main Class**: `RewardDataCollector`
- **Responsibilities**:
  - Maintains separate trajectory buffers for each parallel environment
  - Collects observations (images), `success_frame` labels, and `success_once` flags
  - Saves complete trajectories as `.npy` files
  - Classifies trajectories as positive/negative based on success criteria
  - Ensures data format consistency (`[C, H, W]` images, normalized to `[0, 1]`)

**Key Methods**:
- `add_env_output(env_output, env_info, is_last_step)`: Add environment output to buffers
- `save_trajectory(env_idx)`: Save completed trajectory to disk
- `_extract_image(image, key_offset)`: Extract and format images from observations

#### `rlinf/workers/env/env_worker.py`
**Purpose**: Environment worker that integrates data collection
- **Integration Point**: Initializes `RewardDataCollector` in `__init__` if `cfg.reward.collect_data=True`
- **Integration Method**: Calls `data_collector.add_env_output()` after each environment interaction in `interact()` method
- **Key Logic**: Extracts `success_frame` from `env_output` and `success_once` from `env_info`

### Reward Model Training

#### `rlinf/models/reward_model/train_reward_model.py`
**Purpose**: Training script for reward model
- **Main Function**: `main()`
- **Responsibilities**:
  - Loads trajectory data from `positive_dir` and `negative_dir`
  - Creates `RewardDataset` for frame-level training
  - Trains `BinaryRewardClassifier` with BCE loss
  - Saves best model checkpoint
  - Optionally visualizes positive samples

**Key Arguments**:
- `--positive-dir`: Directory containing positive trajectory `.npy` files
- `--negative-dir`: Directory containing negative trajectory `.npy` files
- `--output-checkpoint`: Path to save trained model (e.g., `./checkpoints/reward_model.pt`)
- `--pretrained-encoder-path`: Path to pretrained ResNet10 encoder weights (e.g., `./resnet10_pretrained.pt`)
- `--image-key`: Image key to use (default: `base_camera`)
- `--image-size`: Image size as `[C H W]` (default: `3 64 64`)
- `--batch-size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--hidden-dim`: Hidden dimension for classifier (default: 256)
- `--num-spatial-blocks`: Number of spatial blocks for pooling (default: 8)
- `--visualize-positive`: Flag to visualize positive samples
- `--vis-output-dir`: Output directory for visualizations (default: `positive_samples_vis`)

#### `rlinf/models/reward_model/reward_classifier.py`
**Purpose**: Reward model architecture definition
- **Main Class**: `BinaryRewardClassifier`
- **Architecture**:
  - **Encoder**: `ResNetEncoderWrapper` (ResNet10 backbone)
    - Can load pretrained weights from `pretrained_encoder_path`
    - Supports freezing encoder weights (`freeze_encoder=True`)
    - Uses `SpatialLearnedEmbeddings` for pooling
  - **Classifier Head**: MLP (Linear → LayerNorm → Tanh → Linear → Sigmoid)
  - **Input**: Single frame image `[B, C, H, W]`
  - **Output**: Binary logit (success probability via sigmoid)

**Key Parameters**:
- `image_keys`: List of image keys to process
- `image_size`: Image dimensions `[C, H, W]`
- `hidden_dim`: Hidden dimension for classifier
- `num_spatial_blocks`: Number of spatial blocks for pooling
- `pretrained_encoder_path`: Path to pretrained ResNet10 weights
- `use_pretrain`: Whether to use pretrained encoder
- `freeze_encoder`: Whether to freeze encoder weights during training

#### `rlinf/models/embodiment/modules/nature_cnn.py`
**Purpose**: ResNet10 backbone implementation
- **Main Class**: `ResNet10`
- **Pretrained Weights**: Expected at `./resnet10_pretrained.pt` (hardcoded path in `ResNetEncoder`)
- **Note**: When using `ResNetEncoderWrapper` in reward classifier, pretrained weights are loaded from `pretrained_encoder_path` parameter instead

### Reward Model Usage

#### `rlinf/workers/reward/embodied_reward_worker.py`
**Purpose**: Reward worker that loads and uses trained reward model
- **Main Class**: `EmbodiedRewardWorker`
- **Responsibilities**:
  - Loads `BinaryRewardClassifier` from checkpoint in `init_worker()`
  - Computes rewards for rollout results in `compute_rewards()`
  - Replaces environment success signals with model predictions
  - Handles batch processing across parallel environments

**Key Methods**:
- `init_worker()`: Initialize and load reward model
  - Creates `BinaryRewardClassifier` with config parameters
  - Loads checkpoint from `cfg.reward.reward_model.checkpoint_path`
  - Automatically infers `use_pretrain` from checkpoint if needed
  - Moves model to device and sets to eval mode
- `_compute_rewards_with_model()`: Compute rewards using model
  - Extracts images from observations
  - Runs forward pass through model
  - Converts logits to rewards based on `reward_type` (binary/continuous)

#### `examples/embodiment/train_embodied_agent.py`
**Purpose**: Main training script for embodied RL agents
- **Integration**: Creates `EmbodiedRewardWorker` if `cfg.reward.use_reward_model=True`
- **Usage**: Can run SAC or PPO training with or without reward model

#### `rlinf/runners/embodied_runner.py`
**Purpose**: Main runner that orchestrates training
- **Integration**: Initializes reward worker if provided
- **Key Method**: `init_workers()` calls `reward.init_worker()` if reward worker exists

## Stage 1: Data Collection

### Configuration (Example: `maniskill_ppo_cnn.yaml`)

```yaml
reward:
  group_name: "RewardGroup"
  use_reward_model: False  # Disabled during data collection
  collect_data: True  # Enable data collection
  reward_model:
    # These settings are not used during data collection, but should match actor settings
    image_keys: ["base_camera"]
    image_size: [3, 64, 64]
  data_collection:
    # Directory to save positive trajectories (success_once=1 or any success_frame>=0.5)
    positive_dir: "./reward_data/positive"
    # Directory to save negative trajectories (success_once=0 and all success_frame<0.5)
    negative_dir: "./reward_data/negative"
    # Image keys to collect (must match actor.model.image_keys)
    image_keys: ["base_camera"]
    # Maximum number of trajectories to save (None = unlimited)
    max_positive_trajectories: 500
    max_negative_trajectories: 500
```

### Startup Command

```bash
# Using the provided script
bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

# Or directly with python
export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config/ \
    --config-name maniskill_ppo_cnn \
    reward.collect_data=True \
    reward.use_reward_model=False
```

### Data Format

Each trajectory is saved as a `.npy` file containing a dictionary:

```python
{
    'images': {
        'base_camera': np.array([T, C, H, W], dtype=np.float32)  # T frames, normalized [0, 1]
    },
    'labels': np.array([T], dtype=np.float32)  # T success_frame labels (0 or 1)
}
```

**File Naming**: `{counter:06d}.npy` (e.g., `000000.npy`, `000001.npy`, ...)

**Save Locations**:
- Positive trajectories: `{positive_dir}/000000.npy`, `{positive_dir}/000001.npy`, ...
- Negative trajectories: `{negative_dir}/000000.npy`, `{negative_dir}/000001.npy`, ...

### Trajectory Classification Logic

A trajectory is saved to `positive_dir` if:
- `success_once=True` **OR**
- Any frame has `success_frame >= 0.5`

Otherwise, it is saved to `negative_dir`.

### Important Notes

1. **Image Format**: Images are automatically converted to `[C, H, W]` format and normalized to `[0, 1]`
2. **Parallel Environments**: Each environment maintains an independent buffer; trajectories are saved when `is_env_done=True`
3. **Data Consistency**: The `image_keys` and `image_size` in `data_collection` must match `actor.model.image_keys` and `actor.model.image_size`

## Stage 2: Reward Model Training

### Prerequisites

1. **Pretrained ResNet10 Encoder** (Optional but recommended):
   - **File**: `resnet10_pretrained.pt`
   - **Location**: Should be placed in the project root or specified via `--pretrained-encoder-path`
   - **Purpose**: Provides pretrained visual features for better initialization
   - **Note**: If not provided, the encoder will be randomly initialized

2. **Collected Data**:
   - Positive trajectories in `positive_dir`
   - Negative trajectories in `negative_dir`

### Configuration

No YAML configuration needed for training. All settings are via command-line arguments.

### Startup Command

```bash
# Using the provided script (modify paths as needed)
bash examples/embodiment/train_reward_model.sh

# Or directly with python (using module syntax)
python -m rlinf.models.reward_model.train_reward_model \
    --positive-dir ./reward_data/positive \
    --negative-dir ./reward_data/negative \
    --output-checkpoint ./checkpoints/reward_model.pt \
    --pretrained-encoder-path ./resnet10_pretrained.pt \
    --backbone resnet10 \
    --image-key base_camera \
    --image-size 3 64 64 \
    --batch-size 128 \
    --epochs 30 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --num-spatial-blocks 8 \
    --visualize-positive \
    --vis-output-dir ./positive_samples_vis
```

### Command Arguments

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--positive-dir` | Required | Directory containing positive `.npy` files | `./reward_data/positive` |
| `--negative-dir` | Required | Directory containing negative `.npy` files | `./reward_data/negative` |
| `--output-checkpoint` | Required | Path to save trained model | `./checkpoints/reward_model.pt` |
| `--pretrained-encoder-path` | None | Path to pretrained ResNet10 encoder | `./resnet10_pretrained.pt` |
| `--backbone` | `resnet10` | Backbone architecture (currently only resnet10) | `resnet10` |
| `--image-key` | `base_camera` | Image key to use | `base_camera` |
| `--image-size` | `3 64 64` | Image size as `[C H W]` | `3 64 64` |
| `--batch-size` | 32 | Training batch size | 128 |
| `--epochs` | 100 | Number of training epochs | 30 |
| `--lr` | 1e-4 | Learning rate | 1e-4 |
| `--hidden-dim` | 256 | Hidden dimension for classifier | 256 |
| `--num-spatial-blocks` | 8 | Number of spatial blocks for pooling | 8 |
| `--visualize-positive` | False | Visualize positive samples before training | (flag) |
| `--vis-output-dir` | `positive_samples_vis` | Output directory for visualizations | `./show` |

### Training Process

1. **Data Loading**:
   - Loads all `.npy` files from `positive_dir` and `negative_dir`
   - Expands trajectory-level data to frame-level samples
   - Each sample = `(trajectory_path, frame_index, label)`
   - Reports statistics:
     - Number of trajectories and frames in each directory
     - Distribution of labels (label=1 vs label=0) in each directory

2. **Model Creation**:
   - Creates `BinaryRewardClassifier` with specified parameters
   - If `--pretrained-encoder-path` is provided, loads pretrained ResNet10 weights
   - Encoder weights are frozen by default (`freeze_encoder=True`)

3. **Training Loop**:
   - Uses BCE loss with logits
   - Adam optimizer
   - Saves best model based on training accuracy

4. **Checkpoint Format**:
   ```python
   {
       'epoch': int,
       'model_state_dict': dict,
       'optimizer_state_dict': dict,
       'train_loss': float,
       'train_acc': float
   }
   ```

### Model Save Location

**Default**: Specified by `--output-checkpoint` argument

**Example Paths**:
- `./checkpoints/reward_model.pt`
- `/path/to/project/toolkits/reward_model/checkpoints/reward_model.pt`

**Note**: The directory will be created automatically if it doesn't exist.

### Visualization (Optional)

If `--visualize-positive` is specified:
- All positive samples (label=1) are visualized
- Images are saved to `--vis-output-dir`
- Filename format: `{folder}_{traj_id}_frame{frame_idx}.png`
  - Example: `positive_000001_frame049.png`
- Useful for verifying data quality and understanding success patterns

## Stage 3: Training RL Policy with Reward Model

### Configuration (Example: `maniskill_ppo_cnn.yaml`)

```yaml
reward:
  group_name: "RewardGroup"
  use_reward_model: True  # Enable reward model
  collect_data: False  # Disable data collection during policy training
  reward_model:
    # Path to trained reward model checkpoint
    checkpoint_path: "./checkpoints/reward_model.pt"
    # Image keys to use (must match actor.model.image_keys)
    image_keys: ["base_camera"]
    # Image size [C, H, W] (must match actor.model.image_size)
    image_size: [3, 64, 64]
    # Hidden dimension for classifier (must match training config)
    hidden_dim: 256
    # Number of spatial blocks for pooling (must match training config)
    num_spatial_blocks: 8
    # Path to pretrained ResNet10 encoder weights (optional, used during model creation)
    pretrained_encoder_path: "./resnet10_pretrained.pt"
    # Whether to use pretrained encoder (must match training config)
    use_pretrain: True
    # Whether to freeze encoder weights (typically True)
    freeze_encoder: True
    # Reward type: "binary" (0 or 1) or "continuous" (probability)
    reward_type: "binary"
```

**Important**: The `image_keys`, `image_size`, `hidden_dim`, `num_spatial_blocks`, `use_pretrain`, and `freeze_encoder` settings **must match** the settings used during reward model training.

### Startup Command

```bash
# Using the provided script
bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

# Or directly with python
export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config/ \
    --config-name maniskill_ppo_cnn \
    reward.use_reward_model=True \
    reward.collect_data=False \
    reward.reward_model.checkpoint_path="./checkpoints/reward_model.pt"
```

### Reward Model Loading Process

1. **Initialization** (`EmbodiedRewardWorker.init_worker()`):
   - Creates `BinaryRewardClassifier` with config parameters
   - Loads checkpoint from `checkpoint_path`
   - Automatically infers `use_pretrain` from checkpoint if there's a mismatch
   - Moves model to device (`torch.cuda.current_device()`)
   - Sets model to eval mode

2. **Checkpoint Compatibility**:
   - If `use_pretrain` in config doesn't match the training config, the code will automatically infer the correct value from the checkpoint's pooling layer kernel shape
   - Warning messages will be printed if adjustments are made

### Reward Computation Process

1. **During Rollout** (`EmbodiedRewardWorker._compute_rewards_with_model()`):
   - Receives `EmbodiedRolloutResult` from rollout worker
   - Extracts observations (images) from `transitions` or `forward_inputs`
   - Processes each observation:
     - Extracts image using `_extract_images_from_obs()`
     - Normalizes to `[0, 1]` and converts to `[C, H, W]` format
     - Runs forward pass through reward model
     - Applies sigmoid to get success probability
     - Converts to reward based on `reward_type`:
       - `"binary"`: `(probs > 0.5).float()` → 0 or 1
       - `"continuous"`: Uses probability directly

2. **Integration with RL Training**:
   - Reward model predictions replace or supplement environment `success_frame` signals
   - For SAC: Used for Q-learning and policy optimization
   - For PPO: Used for advantage computation and policy updates

### Important Notes

1. **Configuration Consistency**: All reward model parameters (`image_keys`, `image_size`, `hidden_dim`, `num_spatial_blocks`, `use_pretrain`, `freeze_encoder`) must match between training and usage configs

2. **Pretrained Encoder**: The `pretrained_encoder_path` in the config is used during model creation but may not be reloaded if the checkpoint already contains encoder weights. The checkpoint's encoder weights take precedence.

3. **Performance**: Reward model inference adds computational overhead. Consider using `freeze_encoder=True` to reduce memory and computation.

4. **Parallel Environments**: The reward model processes all environments in parallel batches for efficiency.

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Data Collection (Using Environment Success Signals)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  RL Policy Training (Initial Policy)│
        │  - Config: reward.collect_data=True │
        │  - Config: reward.use_reward_model=False │
        │  - Command: run_embodiment.sh maniskill_ppo_cnn │
        │  - RewardDataCollector collects data│
        │  - Saves to positive/negative dirs  │
        └─────────────────────────────────────┘
                              │
                              ▼
        Output: ./reward_data/positive/*.npy
                ./reward_data/negative/*.npy
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Reward Model Training                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  train_reward_model.py               │
        │  - Input: positive_dir, negative_dir │
        │  - Optional: pretrained_encoder_path │
        │  - Output: reward_model.pt           │
        │  - Command: train_reward_model.sh    │
        └─────────────────────────────────────┘
                              │
                              ▼
        Output: ./checkpoints/reward_model.pt
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: RL Policy Training with Reward Model                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  RL Policy Training (SAC/PPO)       │
        │  - Config: reward.use_reward_model=True │
        │  - Config: reward.collect_data=False │
        │  - Config: checkpoint_path set      │
        │  - Command: run_embodiment.sh maniskill_ppo_cnn │
        │  - EmbodiedRewardWorker loads model │
        │  - Replaces env success signals     │
        └─────────────────────────────────────┘
```

## Configuration Examples

### Complete Example: `maniskill_ppo_cnn.yaml`

#### Stage 1: Data Collection Config

```yaml
reward:
  group_name: "RewardGroup"
  use_reward_model: False
  collect_data: True
  reward_model:
    image_keys: ["base_camera"]
    image_size: [3, 64, 64]
  data_collection:
    positive_dir: "./reward_data/positive"
    negative_dir: "./reward_data/negative"
    image_keys: ["base_camera"]
    max_positive_trajectories: 500
    max_negative_trajectories: 500
```

#### Stage 2: Reward Model Training (Command-line)

```bash
python -m rlinf.models.reward_model.train_reward_model \
    --positive-dir ./reward_data/positive \
    --negative-dir ./reward_data/negative \
    --output-checkpoint ./checkpoints/reward_model.pt \
    --pretrained-encoder-path ./resnet10_pretrained.pt \
    --image-key base_camera \
    --image-size 3 64 64 \
    --batch-size 128 \
    --epochs 30 \
    --hidden-dim 256 \
    --num-spatial-blocks 8
```

#### Stage 3: Training with Reward Model Config

```yaml
reward:
  group_name: "RewardGroup"
  use_reward_model: True
  collect_data: False
  reward_model:
    checkpoint_path: "./checkpoints/reward_model.pt"
    image_keys: ["base_camera"]
    image_size: [3, 64, 64]
    hidden_dim: 256
    num_spatial_blocks: 8
    pretrained_encoder_path: "./resnet10_pretrained.pt"
    use_pretrain: True
    freeze_encoder: True
    reward_type: "binary"
```

