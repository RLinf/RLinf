# Embodied Reward Models

This module provides reward model implementations for embodied reinforcement learning tasks, supporting both image-based (single-frame) and video-based (multi-frame) reward models.

## Architecture

```
BaseRewardModel (Abstract Root)
│
├── BaseImageRewardModel (Abstract)    # Single-frame reward
│   └── ResNetRewardModel              # Binary classifier (HIL-SERL style)
│
└── BaseVideoRewardModel (Abstract)    # Multi-frame/video reward
    └── Qwen3VLRewardModel             # VLM-based reward (placeholder)
```

## File Structure

```
rlinf/algorithms/rewards/embodiment/
├── __init__.py                    # Module exports
├── base_reward_model.py           # BaseRewardModel (root abstract)
├── base_image_reward_model.py     # BaseImageRewardModel (single-frame)
├── base_video_reward_model.py     # BaseVideoRewardModel (multi-frame)
├── resnet_reward_model.py         # ResNet binary classifier
├── qwen3_vl_reward_model.py       # Qwen3-VL (reserved implementation)
└── reward_manager.py              # RewardManager with registry pattern

examples/embodiment/config/reward/
├── resnet_binary.yaml             # ResNet configuration
└── qwen3_vl.yaml                  # Qwen3-VL configuration
```

## Quick Start

### 1. Using RewardManager (Recommended)

The `RewardManager` provides a unified interface for all reward models:

```python
from rlinf.algorithms.rewards.embodiment import RewardManager
from omegaconf import OmegaConf

# Load configuration
cfg = OmegaConf.load("examples/embodiment/config/reward/resnet_binary.yaml")
cfg.resnet.checkpoint_path = "/path/to/your/checkpoint.pt"

# Initialize reward manager
reward_manager = RewardManager(cfg)

# Compute rewards
observations = {
    "images": images_tensor,  # [B, C, H, W] or [B, H, W, C]
    "states": states_tensor,  # Optional [B, state_dim]
}
rewards = reward_manager.compute_rewards(observations)
```

### 2. Using Models Directly

```python
from rlinf.algorithms.rewards.embodiment import ResNetRewardModel
from omegaconf import DictConfig

cfg = DictConfig({
    "checkpoint_path": "/path/to/checkpoint.pt",
    "image_size": [3, 224, 224],
    "threshold": 0.5,
    "use_soft_reward": False,
})

model = ResNetRewardModel(cfg)
rewards = model.compute_reward(observations)
```

## Configuration

### ResNet Binary Classifier

```yaml
reward:
  use_reward_model: True
  reward_model_type: "resnet"
  alpha: 1.0  # Reward scaling factor
  
  resnet:
    checkpoint_path: "/path/to/checkpoint.pt"  # Required
    image_size: [3, 224, 224]
    threshold: 0.5        # Classification threshold
    use_soft_reward: False  # True for probability, False for binary 0/1
    freeze_backbone: True
    hidden_dim: 256
```

### Qwen3-VL Video Reward (Reserved)

```yaml
reward:
  use_reward_model: True
  reward_model_type: "qwen3_vl"
  alpha: 1.0
  
  qwen3_vl:
    model_path: "/path/to/Qwen3-VL-2B-Instruct"  # Required
    checkpoint_path: null  # Optional fine-tuned weights
    sample_k: 6
    sample_strategy: "uniform_k"  # uniform_k, last_k, first_last_k, random_k
    task_prompt_template: "Is the task '{task}' completed?"
```

## Integration with RL Algorithms

The reward models are designed to work with any RL algorithm (PPO, SAC, GRPO, etc.). Integration is already added to the rollout workers.

### Automatic Integration

When `reward.use_reward_model: True` is set in your config, the rollout worker will automatically initialize the reward manager:

```yaml
# In your training config (e.g., maniskill_sac_mlp.yaml)
reward:
  use_reward_model: True
  reward_model_type: "resnet"
  resnet:
    checkpoint_path: "/path/to/checkpoint.pt"
```

### Manual Integration

```python
# In your custom code
from rlinf.algorithms.rewards.embodiment import RewardManager

# Initialize
reward_manager = RewardManager(cfg.reward)

# Use in rollout loop
for step in rollout:
    # ... get observations from env ...
    
    # Compute model-based reward
    if reward_manager.is_enabled:
        model_rewards = reward_manager.compute_rewards(observations, task_descriptions)
        # Combine with env rewards if needed
        rewards = env_rewards + model_rewards
```

## Training a ResNet Reward Model

To train a ResNet binary classifier for your task:

1. **Collect Data**: Gather images labeled as success (1) or failure (0)

2. **Train Model**: Use standard PyTorch training loop:

```python
from rlinf.algorithms.rewards.embodiment.resnet_reward_model import (
    ResNetRewardModel,
    ResNet10Backbone,
)

# Create model
model = ResNetRewardModel(cfg)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

for images, labels in dataloader:
    probs = model.forward(images)
    loss = criterion(probs, labels.float())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save checkpoint
torch.save(model.state_dict(), "reward_model.pt")
```

3. **Use in RL Training**: Set checkpoint path in config

## Adding New Reward Models

### 1. Create Model Class

```python
from rlinf.algorithms.rewards.embodiment import BaseImageRewardModel

class MyRewardModel(BaseImageRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Initialize your model
    
    def compute_reward(self, observations, task_descriptions=None):
        # Implement reward computation
        return rewards
    
    def load_checkpoint(self, checkpoint_path):
        # Load weights
        pass
```

### 2. Register Model

```python
from rlinf.algorithms.rewards.embodiment import RewardManager

RewardManager.register_model("my_model", MyRewardModel)
```

### 3. Use in Config

```yaml
reward:
  reward_model_type: "my_model"
  my_model:
    # Your model config
```

## API Reference

### BaseRewardModel

| Method | Description |
|--------|-------------|
| `compute_reward(observations, task_descriptions)` | Compute rewards from observations |
| `load_checkpoint(path)` | Load model weights |
| `scale_reward(reward)` | Apply scaling factor |
| `to_device(device)` | Move model to device |

### BaseImageRewardModel

| Method | Description |
|--------|-------------|
| `preprocess_images(images)` | Normalize and reorder channels |
| `apply_threshold(probabilities)` | Convert to binary rewards |

### BaseVideoRewardModel

| Method | Description |
|--------|-------------|
| `sample_frames(images, strategy, k)` | Sample frames from video |
| `preprocess_video(images)` | Normalize video tensor |
| `format_prompt(task_description)` | Format VLM prompt |

### RewardManager

| Method | Description |
|--------|-------------|
| `compute_rewards(observations, task_descriptions)` | Unified reward computation |
| `register_model(name, cls)` | Register new model type |
| `get_available_models()` | List registered models |
| `to_device(device)` | Move model to device |

