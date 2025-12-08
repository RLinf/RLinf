# Reward Model Workflow Documentation

This document provides a comprehensive guide for the reward model workflow, including data collection, reward model training, and using the trained reward model to replace environment success signals for training RL policies. 

## 🎯 Algorithm Support

**This workflow supports PPO (Proximal Policy Optimization)** algorithm:

- ✅ **PPO** (Proximal Policy Optimization) - Use config: `maniskill_ppo_cnn.yaml`

**Key Benefits**:
- Simple and unified workflow
- All scripts and configs are located in `examples/embodiment/` directory

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

1. **Data Collection**: Collect trajectories using an initial RL policy (PPO), storing images and success labels
2. **Reward Model Training**: Train a binary classifier to predict success from single frames (algorithm-agnostic)
3. **RL Policy Training**: Use the trained reward model to replace environment success signals for PPO training

**Key Points**:
- The workflow uses PPO algorithm for RL policy training
- Uses unified `RewardWorker` for reward computation

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

#### `examples/embodiment/main_reward.py`
**Purpose**: Training script for reward model using Ray and WorkerGroup
- **Main Function**: `main(cfg: DictConfig)` (uses Hydra decorator)
- **Configuration**: Uses Hydra and OmegaConf, config file at `examples/embodiment/config/train_reward_model.yaml`
- **Architecture**: Uses Ray for distributed execution, `RewardWorker` with `fit()` method for training
- **Responsibilities**:
  - Creates `RewardWorker` group using Ray
  - Initializes workers and datasets
  - Trains `BinaryRewardClassifier` with BCE loss using simple PyTorch training (no FSDP)
  - Saves checkpoints at regular intervals (by step, not epoch)

**Key Configuration Parameters** (in `reward` section):
- `training_backend`: Training backend (set to `simple` for small models, no FSDP)
- `gpu_id`: GPU device ID to use for training (default: `0`)
- `global_batch_size`: Training batch size (default: `32`)
- `num_epochs`: Number of training epochs (default: `100`)
- `save_interval`: Save checkpoint every N steps (default: `1000`)
- `log_interval`: Log metrics every N steps (default: `100`)
- `save_dir`: Directory to save checkpoints (default: `${runner.output_dir}/${runner.experiment_name}/checkpoints`)
- `log_dir`: Directory to save logs (default: `${runner.output_dir}/${runner.experiment_name}/logs`)
- `reward.data.positive_dir`: Directory containing positive trajectory `.npy` files
- `reward.data.negative_dir`: Directory containing negative trajectory `.npy` files
- `reward.model.image_keys`: Image keys to use (default: `["base_camera"]`)
- `reward.model.image_size`: Image size as `[C, H, W]` (default: `[3, 64, 64]`)
- `reward.model.hidden_dim`: Hidden dimension for classifier (default: `256`)
- `reward.model.num_spatial_blocks`: Number of spatial blocks for pooling (default: `8`)
- `reward.model.pretrained_encoder_path`: Path to pretrained ResNet10 encoder weights (default: `null`)
- `reward.optim.lr`: Learning rate (default: `1e-4`)

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

#### `rlinf/workers/reward/reward_worker.py`
**Purpose**: Unified reward worker that supports both text-based reasoning tasks and embodied tasks
- **Main Class**: `RewardWorker`
- **Task Type Detection**: Automatically detects task type (embodied vs text-based) based on config
  - Checks for `cfg.runner.task_type == "embodied"`
  - Checks for presence of `cfg.env` config
  - Checks if reward_model config contains `image_keys` or `image_size`
- **Responsibilities**:
  - For embodied tasks: Loads `BinaryRewardClassifier` from checkpoint and computes frame-based rewards
  - For text-based tasks: Supports rule-based rewards (reward model for text tasks not yet implemented)
  - Handles batch processing across parallel environments
  - Replaces environment success signals with model predictions for embodied tasks

**Key Methods for Embodied Tasks**:
- `init_worker()`: Initialize and load reward model
  - Automatically detects if this is an embodied task
  - Creates `BinaryRewardClassifier` with config parameters
  - Loads checkpoint from `cfg.reward.reward_model.checkpoint_path`
  - Automatically infers `use_pretrain` from checkpoint if needed
  - Moves model to device and sets to eval mode
- `_compute_embodied_rewards_with_model()`: Compute rewards using model for embodied tasks
  - Extracts images from observations in `EmbodiedRolloutResult`
  - Runs forward pass through model
  - Converts logits to rewards based on `reward_type` (binary/continuous)
- `compute_rewards()`: Unified method that handles both `RolloutResult` and `EmbodiedRolloutResult`
  - Automatically detects input type and routes to appropriate processing

**Key Methods for Text-Based Tasks**:
- `init_worker()`: Initialize rule-based reward or text reward model (if implemented)
- `_compute_rule_based_rewards()`: Compute rewards using rule-based methods
- `compute_batch_rewards_with_model()`: Compute rewards using reward model (not yet implemented)

#### `examples/embodiment/train_embodied_agent.py`
**Purpose**: Main training script for embodied RL agents
- **Integration**: Creates `RewardWorker` if `cfg.reward.use_reward_model=True`
- **Usage**: Can run PPO training with or without reward model
- **Note**: The unified `RewardWorker` automatically detects embodied tasks and handles them appropriately

#### `rlinf/runners/embodied_runner.py`
**Purpose**: Main runner that orchestrates training
- **Integration**: Initializes reward worker if provided
- **Key Method**: `init_workers()` calls `reward.init_worker()` if reward worker exists

## Stage 1: Data Collection

### Configuration

The reward model workflow uses PPO algorithm. Use the config file:
- **PPO**: `examples/embodiment/config/maniskill_ppo_cnn.yaml`

Example configuration:

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
# For PPO
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
   - **Location**: Should be placed in the project root or specified via `pretrained_encoder_path` in config
   - **Purpose**: Provides pretrained visual features for better initialization
   - **Note**: If not provided (set to `null`), the encoder will be randomly initialized

2. **Collected Data**:
   - Positive trajectories in `positive_dir`
   - Negative trajectories in `negative_dir`

### Configuration

The training script uses Hydra and OmegaConf for configuration management. The default configuration file is located at `examples/embodiment/config/train_reward_model.yaml`.

#### Default Configuration File Structure

```yaml
defaults:
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null

cluster:
  num_nodes: 1
  component_placement:
    reward: all

runner:
  task_type: embodied
  logger:
    log_path: ${runner.output_dir}/${runner.experiment_name}
    project_name: rlinf
    experiment_name: reward-model-training
    logger_backends: ["tensorboard"]
  output_dir: ../results
  experiment_name: reward-model-training

reward:
  group_name: "RewardGroup"
  training_backend: simple  # Simple training mode (no FSDP)
  gpu_id: 0  # GPU device ID to use for training
  
  global_batch_size: 32
  num_epochs: 100
  log_interval: 100
  save_interval: 1000  # Save checkpoint every N steps
  log_dir: ${runner.output_dir}/${runner.experiment_name}/logs
  save_dir: ${runner.output_dir}/${runner.experiment_name}/checkpoints
  
  model:
    image_keys: ["base_camera"]
    image_size: [3, 64, 64]  # [C, H, W]
    hidden_dim: 256
    num_spatial_blocks: 8
    pretrained_encoder_path: null
    use_pretrain: true
    freeze_encoder: true
  
  optim:
    optimizer: adam
    lr: 1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-05
    weight_decay: 0.0
    clip_grad: 1.0
  
  lr_sched:
    lr_warmup_iters: 0
    max_lr: 1.0e-4
    min_lr: 0.0
    lr_decay_style: constant
  
  data:
    positive_dir: ./reward_data/positive
    negative_dir: ./reward_data/negative
    image_key: base_camera
    image_size: [3, 64, 64]
    shuffle: true
    num_workers: 4
    pin_memory: true
    persistent_workers: false
    prefetch_factor: 2
```

### Startup Command

```bash
# Using the provided script (recommended)
bash examples/embodiment/train_reward_model.sh train_reward_model

# Or directly with python
python examples/embodiment/main_reward.py \
    --config-path examples/embodiment/config \
    --config-name train_reward_model

# Override configuration parameters via command line
bash examples/embodiment/train_reward_model.sh train_reward_model \
    reward.gpu_id=1 \
    reward.global_batch_size=64 \
    reward.num_epochs=50 \
    reward.save_interval=500 \
    reward.data.positive_dir=./reward_data/positive \
    reward.data.negative_dir=./reward_data/negative
```

### Configuration Parameters

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `reward.training_backend` | `simple` | Training backend (use `simple` for small models) | `simple` |
| `reward.gpu_id` | `0` | GPU device ID to use for training | `0`, `1`, `2`, etc. |
| `reward.global_batch_size` | `32` | Training batch size | `64`, `128` |
| `reward.num_epochs` | `100` | Number of training epochs | `50`, `100` |
| `reward.save_interval` | `1000` | Save checkpoint every N steps | `500`, `1000`, `2000` |
| `reward.log_interval` | `100` | Log metrics every N steps | `50`, `100` |
| `reward.save_dir` | `${runner.output_dir}/${runner.experiment_name}/checkpoints` | Directory to save checkpoints | `./checkpoints` |
| `reward.data.positive_dir` | `./reward_data/positive` | Directory containing positive `.npy` files | `./reward_data/positive` |
| `reward.data.negative_dir` | `./reward_data/negative` | Directory containing negative `.npy` files | `./reward_data/negative` |
| `reward.model.image_keys` | `["base_camera"]` | Image keys to use | `["base_camera"]` |
| `reward.model.image_size` | `[3, 64, 64]` | Image size as `[C, H, W]` | `[3, 64, 64]` |
| `reward.model.hidden_dim` | `256` | Hidden dimension for classifier | `256`, `512` |
| `reward.model.num_spatial_blocks` | `8` | Number of spatial blocks for pooling | `8`, `16` |
| `reward.model.pretrained_encoder_path` | `null` | Path to pretrained ResNet10 encoder | `./resnet10_pretrained.pt` |
| `reward.model.use_pretrain` | `true` | Whether to use pretrained encoder | `true`, `false` |
| `reward.model.freeze_encoder` | `true` | Whether to freeze encoder weights | `true`, `false` |
| `reward.optim.lr` | `1e-4` | Learning rate | `1e-4`, `5e-5` |
| `reward.optim.clip_grad` | `1.0` | Gradient clipping threshold | `1.0`, `0.5` |
| `reward.data.num_workers` | `4` | Number of data loading workers | `4`, `8` |

### Training Process

1. **Initialization**:
   - Creates Ray cluster and `RewardWorker` group
   - Initializes dataset from `reward.data.positive_dir` and `reward.data.negative_dir`
   - Loads all `.npy` files and expands trajectory-level data to frame-level samples
   - Reports statistics:
     - Number of trajectories and frames in each directory
     - Distribution of labels (label=1 vs label=0) in each directory

2. **Model Creation**:
   - Creates `BinaryRewardClassifier` with specified parameters in `reward.model`
   - If `pretrained_encoder_path` is provided, loads pretrained ResNet10 weights
   - Encoder weights are frozen by default (`freeze_encoder=True`)
   - Moves model to specified GPU (`gpu_id`)

3. **Training Loop**:
   - Uses BCE loss with logits (`F.binary_cross_entropy_with_logits`)
   - AdamW optimizer (configured in `reward.optim`)
   - Learning rate scheduler (configured in `reward.lr_sched`)
   - Gradient clipping (if `reward.optim.clip_grad > 0`)
   - Logs metrics every `log_interval` steps
   - Saves checkpoint every `save_interval` steps

4. **Checkpoint Format**:
   ```python
   {
       'epoch': int,
       'global_step': int,
       'model_state_dict': dict,
       'optimizer_state_dict': dict,
       'lr_scheduler_state_dict': dict,  # if lr_scheduler exists
       'loss': float,
       'accuracy': float
   }
   ```

### Model Save Location

**Default**: Specified by `reward.save_dir` in configuration

**Save Path Format**: `{save_dir}/step_{global_step}/checkpoint.pt`

**Example Paths**:
- `../results/reward-model-training/checkpoints/step_1000/checkpoint.pt`
- `../results/reward-model-training/checkpoints/step_2000/checkpoint.pt`
- `../results/reward-model-training/checkpoints/step_3000/checkpoint.pt`

**Note**: 
- Checkpoints are saved by **step** (not epoch), controlled by `reward.save_interval`
- The directory structure is created automatically
- Each checkpoint is saved in its own subdirectory named `step_{global_step}`

## Stage 3: Training RL Policy with Reward Model

### Configuration

The reward model workflow uses PPO algorithm. Use the config file:
- **PPO**: `examples/embodiment/config/maniskill_ppo_cnn.yaml`

Example configuration:

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
# For PPO
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

1. **Task Type Detection** (`RewardWorker._is_embodied_task()`):
   - Automatically detects if this is an embodied task based on config
   - For embodied tasks, initializes image-based reward model
   - For text-based tasks, initializes text-based reward handler

2. **Initialization for Embodied Tasks** (`RewardWorker.init_worker()`):
   - Creates `BinaryRewardClassifier` with config parameters
   - Loads checkpoint from `checkpoint_path`
   - Automatically infers `use_pretrain` from checkpoint if there's a mismatch
   - Moves model to device (`torch.cuda.current_device()`)
   - Sets model to eval mode

3. **Checkpoint Compatibility**:
   - If `use_pretrain` in config doesn't match the training config, the code will automatically infer the correct value from the checkpoint's pooling layer kernel shape
   - Warning messages will be printed if adjustments are made

### Reward Computation Process

1. **Unified Reward Computation** (`RewardWorker.compute_rewards()`):
   - Automatically detects input type (`RolloutResult` vs `EmbodiedRolloutResult`)
   - Routes to appropriate processing method based on input type

2. **For Embodied Tasks** (`RewardWorker._compute_embodied_rewards_with_model()`):
   - Receives `EmbodiedRolloutResult` from rollout worker
   - Extracts observations (images) from `transitions` or `forward_inputs`
   - Processes each observation:
     - Extracts image using `_extract_images_from_obs()`
     - Ensures images are in `[B, C, H, W]` format
     - Runs forward pass through reward model
     - Applies sigmoid to get success probability
     - Converts to reward based on `reward_type`:
       - `"binary"`: `(probs > 0.5).float()` → 0 or 1
       - `"continuous"`: Uses probability directly

3. **Integration with RL Training**:
   - Reward model predictions replace or supplement environment `success_frame` signals
   - For PPO: Used for advantage computation and policy updates

### Important Notes

1. **Unified RewardWorker**: The `RewardWorker` class now supports both embodied and text-based reasoning tasks. It automatically detects the task type based on configuration and routes to the appropriate processing methods. This unified design ensures backward compatibility with existing text-based tasks while supporting new embodied tasks.

2. **Configuration Consistency**: All reward model parameters (`image_keys`, `image_size`, `hidden_dim`, `num_spatial_blocks`, `use_pretrain`, `freeze_encoder`) must match between training and usage configs

3. **Pretrained Encoder**: The `pretrained_encoder_path` in the config is used during model creation but may not be reloaded if the checkpoint already contains encoder weights. The checkpoint's encoder weights take precedence.

4. **Performance**: Reward model inference adds computational overhead. Consider using `freeze_encoder=True` to reduce memory and computation.

5. **Parallel Environments**: The reward model processes all environments in parallel batches for efficiency.

6. **Task Type Detection**: The worker automatically determines if it's handling an embodied task by checking:
   - `cfg.runner.task_type == "embodied"` 
   - Presence of `cfg.env` configuration
   - Reward model config containing `image_keys` or `image_size`
   If none of these conditions are met, it treats the task as a text-based reasoning task.

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Data Collection (Using Environment Success Signals)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  RL Policy Training (Initial Policy)│
        │  - Uses PPO algorithm               │
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
        │  main_reward.py                     │
        │  - Location: examples/embodiment/    │
        │  - Uses Hydra config: config/train_reward_model.yaml │
        │  - Uses Ray and RewardWorker        │
        │  - Input: reward.data.positive_dir, reward.data.negative_dir │
        │  - Optional: reward.model.pretrained_encoder_path │
        │  - Output: checkpoints/step_{N}/checkpoint.pt │
        │  - Command: bash examples/embodiment/train_reward_model.sh │
        └─────────────────────────────────────┘
                              │
                              ▼
        Output: {save_dir}/step_{N}/checkpoint.pt
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: RL Policy Training with Reward Model                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  RL Policy Training (PPO)            │
        │  - Uses PPO algorithm                │
        │  - Config: reward.use_reward_model=True │
        │  - Config: reward.collect_data=False │
        │  - Config: checkpoint_path set      │
        │  - Command: run_embodiment.sh maniskill_ppo_cnn │
        │  - RewardWorker automatically detects embodied task │
        │  - RewardWorker loads BinaryRewardClassifier │
        │  - Replaces env success signals with model predictions │
        └─────────────────────────────────────┘
```

## Configuration Examples

### Algorithm Support

The reward model workflow uses **PPO** algorithm:
- **PPO**: `examples/embodiment/config/maniskill_ppo_cnn.yaml`

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

#### Stage 2: Reward Model Training (Hydra Configuration)

**Note**: Reward model training is algorithm-agnostic.

```bash
# Using the provided script (modify paths as needed)
bash examples/embodiment/train_reward_model.sh

# Or directly with python (from examples/embodiment directory)
cd examples/embodiment
python train_reward_model.py

# Or override parameters via command line
python train_reward_model.py \
    positive_dir=./reward_data/positive \
    negative_dir=./reward_data/negative \
    output_checkpoint=./checkpoints/reward_model.pt \
    pretrained_encoder_path=./resnet10_pretrained.pt \
    image_key=base_camera \
    image_size=[3,64,64] \
    batch_size=128 \
    epochs=30 \
    hidden_dim=256 \
    num_spatial_blocks=8
```

#### Stage 3: Training with Reward Model Config

**PPO Example** (`maniskill_ppo_cnn.yaml`):

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

### Complete Workflow Example

Here's a complete example:

```bash
# Step 1: Collect data
bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
    reward.collect_data=True \
    reward.use_reward_model=False

   # Step 2: Train reward model
   bash examples/embodiment/train_reward_model.sh train_reward_model \
       reward.data.positive_dir=./reward_data/positive \
       reward.data.negative_dir=./reward_data/negative

# Step 3: Train with reward model
bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
    reward.use_reward_model=True \
    reward.collect_data=False \
    reward.reward_model.checkpoint_path=./checkpoints/reward_model.pt
```

