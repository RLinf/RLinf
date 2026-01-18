Reward Model Integration
========================

This guide explains how to integrate a learned reward model into RLinf for embodied RL training.
The reward model can replace or augment environment rewards, enabling learning from visual observations.

Overview
--------

RLinf supports image-based reward models that:

- Take RGB images as input
- Output success/failure probability
- Can be used for terminal reward (end of episode) or per-step reward

Components
----------

The reward model system consists of:

1. **ResNetRewardModel** - A ResNet-based binary classifier
2. **ImageRewardWorker** - Worker for inference during RL training
3. **FSDPRewardWorker** - Worker for training reward models with FSDP

Environment Variables
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``TORCHDYNAMO_DISABLE=1``
     - Disable torch dynamo to avoid jinja2 compatibility issues
   * - ``DEBUG_IMAGE_SAVE_DIR``
     - Directory to save debug images for ResNet classification visualization

Configuration
-------------

Add reward model to your YAML config:

.. code-block:: yaml

    reward:
      use_reward_model: True
      group_name: "RewardGroup"
      
      # Reward mode:
      # - "per_step": Compute reward for every frame
      # - "terminal": Only compute reward at episode end
      reward_mode: "terminal"
      
      # Asymmetric reward mapping
      success_reward: 1.0   # Reward for success (prob > threshold)
      fail_reward: 0.0      # Reward for fail (prob < threshold)
      reward_threshold: 0.5 # Probability threshold
      
      # How to combine with environment rewards
      combine_mode: "replace"  # or "add", "weighted"
      reward_weight: 10.0
      env_reward_weight: 0.0
      
      model:
        model_type: "resnet_reward"
        arch: "resnet18"
        hidden_dim: 256
        dropout: 0.1
        image_size: [3, 128, 128]
        checkpoint_path: "path/to/checkpoint.safetensors"
        debug_save_dir: "${runner.logger.log_path}/${runner.logger.experiment_name}/debug_images"

Training a Reward Model
-----------------------

1. **Collect Data**

   Enable data collection in your config:

   .. code-block:: yaml

       env:
         data_collection:
           enabled: True
           save_dir: "collected_data"
           sample_rate_success: 1.0
           sample_rate_fail: 0.1

2. **Train the Model**

   Use the reward training configuration:

   .. code-block:: bash

       bash examples/reward/run_reward_training.sh

3. **Use in RL Training**

   Point to your trained checkpoint:

   .. code-block:: bash

       bash examples/embodiment/run_embodiment.sh maniskill_ppo_rgb_reward

Reward Modes
------------

**Terminal Mode** (Recommended)

Only provides reward at the end of each episode. Prevents reward hacking where
the agent exploits intermediate states.

.. code-block:: yaml

    reward_mode: "terminal"
    success_reward: 1.0
    fail_reward: 0.0

**Per-Step Mode**

Provides reward for every frame. Can lead to reward hacking but provides
denser learning signal.

.. code-block:: yaml

    reward_mode: "per_step"

Debugging
---------

Enable debug image saving to visualize ResNet classification results:

.. code-block:: yaml

    model:
      debug_save_dir: "logs/debug_images"

Images are saved to:

- ``resnet_success/`` - Images classified as success
- ``resnet_fail/`` - Images classified as failure

Filename format: ``{id:06d}_prob{probability:.4f}.png``

Troubleshooting
---------------

**Reward hacking (success rate stays 0 but rewards increase)**

- Switch to terminal mode: ``reward_mode: "terminal"``
- Increase threshold: ``reward_threshold: 0.6``

**Policy collapse (all negative rewards)**

- Use asymmetric rewards: ``fail_reward: 0.0`` instead of negative
- Don't use ``use_negative_reward: true``

**Checkpoint loading errors**

- Ensure ``hidden_dim`` matches the trained checkpoint
- Check ``image_size`` matches training data

