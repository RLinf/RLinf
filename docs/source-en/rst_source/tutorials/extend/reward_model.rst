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

   You can directly use the ``maniskill_ppo_mlp_collect.yaml`` config, which has
   data collection integrated (see :doc:`data_collection` for details):

   .. code-block:: bash

       bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_collect

   This config simultaneously performs policy warmup training and data collection.
   Collected data is saved to the ``collected_data`` directory.

2. **Train the Model**

   Use the reward training configuration (remember to modify ``data.data_path`` to your actual data path):

   .. code-block:: bash

       # Specify data path
       bash examples/reward/run_reward_training.sh --data /path/to/collected_data

3. **Use in RL Training**

   Point to your trained checkpoint:

   .. code-block:: bash

       bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

Recommended Training Strategy
-----------------------------

For best results, we recommend a **two-stage training approach**:

**Stage 1: Warmup with Environment Reward (100+ steps)**

First, train the policy using the environment's native reward signal:

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_collect

This creates a checkpoint with basic task understanding and collects episode data
for training the reward model.

**Stage 2: Continue with ResNet Reward**

Resume training from the Stage 1 checkpoint, now using the ResNet reward model:

.. code-block:: yaml

    runner:
      resume_dir: "logs/.../checkpoints/global_step_100"
    
    reward:
      use_reward_model: True
      model:
        checkpoint_path: "path/to/trained_resnet.safetensors"

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

**Why Two Stages?**

- Starting from scratch with learned reward often fails (cold start problem)
- Stage 1 provides diverse data for training the reward model
- Stage 1 checkpoint gives the policy basic competence before refinement

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

