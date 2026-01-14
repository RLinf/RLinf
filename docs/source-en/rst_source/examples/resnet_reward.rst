ResNet Reward Model Training Guide
===================================

This guide explains how to train a ManiSkill PickCube task using a ResNet-based reward model.

Overview
--------

The ResNet reward model is an image-based binary classifier that predicts whether the robot has successfully completed the grasping task. The complete pipeline consists of three stages:

1. **Data Collection**: Collect RGB images with success/failure labels during PPO training
2. **ResNet Training**: Train the ResNet binary classifier
3. **Policy Training with ResNet Reward**: Use trained ResNet reward to replace environment reward

Architecture
------------

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                        Training Pipeline                         │
    └─────────────────────────────────────────────────────────────────┘
    
    Stage 1: Data Collection (during PPO training)
    ┌───────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │   EnvWorker   │────▶│ DataCollectorWrapper│────▶│  pkl files   │
    │  (GPU 0)      │     │                     │     │ success/fail │
    └───────────────┘     └─────────────────────┘     └──────────────┘
    
    Stage 2: Reward Model Training
    ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │  pkl files   │────▶│ train_reward_model  │────▶│ best_model.pt│
    │              │     │     .py             │     │              │
    └──────────────┘     └─────────────────────┘     └──────────────┘
    
    Stage 3: Policy Training with Reward Model
    ┌───────────────┐                              ┌──────────────┐
    │   EnvWorker   │──── env_output ────────────▶│   Rollout    │
    │   (GPU 0)     │                             │   Worker     │
    └───────────────┘                             └──────────────┘
                                                         │
    ┌───────────────┐     ┌─────────────────────┐       │
    │ RewardWorker  │◀────│  EmbodiedRunner     │◀──────┘
    │   (GPU 1)     │     │  _apply_reward_model│
    └───────────────┘     └─────────────────────┘
            │                      │
            └──── model_rewards ───┘
                       │
                       ▼
              ┌──────────────┐
              │    Actor     │
              │  (training)  │
              └──────────────┘

Prerequisites
-------------

- ManiSkill environment properly installed
- At least 2 GPUs (one for env/rollout/actor, one for reward model)

Stage 1: Data Collection
------------------------

Collect RGB images labeled as success/failure during PPO training.

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_collect_reward_data

This will:

- Train a policy using dense reward (same as ``maniskill_ppo_mlp``)
- Automatically collect RGB images via ``DataCollectorWrapper``
- Save full episode trajectories to pkl files

Configuration (``maniskill_collect_reward_data.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    env:
      train:
        reward_render_mode: "always"  # Render RGB every frame
        init_params:
          obs_mode: "state"           # Policy uses state
          render_mode: "rgb_array"    # Enable RGB rendering

    reward_data_collection:
      enabled: True
      save_dir: "${oc.env:EMBODIED_PATH}/data"
      target_success: 5000
      target_failure: 5000

Data Format
~~~~~~~~~~~

Data is saved as individual pkl files (one per episode)::

    save_dir/
    ├── success/
    │   ├── episode_000000.pkl
    │   └── ...
    └── failure/
        ├── episode_000001.pkl
        └── ...

Each pkl file contains::

    {
        "frames": [  # List of frames (up to 50)
            {
                "obs": {"main_images": ..., "states": ...},
                "action": ...,
                "reward": ...,
                "done": ...,
                "grasp": ...,
                "success": ...,
            },
            ...
        ],
        "success": True/False,
        "label": 1/0,
    }

Stage 2: Train ResNet Reward Model
----------------------------------

Train the ResNet binary classifier using the integrated ``RewardWorker``.

.. code-block:: bash

    ./run_embodiment.sh maniskill_train_resnet_reward

Configuration (``maniskill_train_resnet_reward.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    reward_training:
      enabled: True
      only: True  # Skip env interaction, train reward model only
      data_path: "${oc.env:EMBODIED_PATH}/data"
      epochs: 100
      micro_batch_size: 64
      global_batch_size: 64
      lr: 1.0e-4
      save_dir: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints"

The trained model will be saved to ``logs/reward_checkpoints/reward_model_best.pt``.

Stage 3: Policy Training with ResNet Reward
-------------------------------------------

Use the trained ResNet reward model to replace environment rewards.

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

Configuration (``maniskill_ppo_mlp_resnet_reward.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    cluster:
      component_placement:
        actor: 0-0
        env: 0-0
        rollout: 0-0
        reward: 0-1  # RewardWorker on GPU 1

    runner:
      resume_dir: null  # Set to pre-trained checkpoint path

    reward:
      use_reward_model: True
      reward_model_type: "resnet"
      mode: "replace"  # Replace env reward with model reward
      resnet:
        checkpoint_path: "logs/reward_checkpoints/best_model.pt"

How It Works
~~~~~~~~~~~~

1. **EmbodiedRunner** receives rollout batch from actor
2. **EmbodiedRunner** calls ``RewardWorker.compute_batch_rewards()`` with observations
3. **RewardWorker** computes model rewards using mini-batches (256 images per batch)
4. **EmbodiedRunner** updates actor's rewards via ``actor.update_rewards()``
5. **Actor** trains with the new rewards

Key Features
~~~~~~~~~~~~

- **Parallel Processing**: All timesteps processed in one batch
- **Mini-batching**: Large batches split into 256-image chunks to avoid OOM
- **Separate GPU**: RewardWorker runs on GPU 1, avoiding memory conflicts

Expected Results
----------------

- After ~500-1000 steps, ``env/success_once`` should approach 90%+
- ``time/compute_model_rewards`` should be ~10-50ms per step
- ``rollout/rewards`` will show model reward values

API Reference
-------------

DataCollectorWrapper
~~~~~~~~~~~~~~~~~~~~

Environment wrapper for automatic data collection.

.. code-block:: python

    from rlinf.envs.wrappers import DataCollectorWrapper

RewardWorker
~~~~~~~~~~~~

Worker for reward model inference (runs on separate GPU).

.. code-block:: python

    from rlinf.workers.reward.reward_worker import RewardWorker

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``init_worker()``
     - Initialize RewardManager with model
   * - ``compute_batch_rewards(observations)``
     - Compute rewards with mini-batching
   * - ``save_checkpoint(path, step)``
     - Save model checkpoint

RewardManager
~~~~~~~~~~~~~

Unified interface for reward computation.

.. code-block:: python

    from rlinf.algorithms.rewards.embodiment import RewardManager

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``compute_rewards(observations)``
     - Compute rewards from images
   * - ``to_device(device)``
     - Move model to device

File Structure
--------------

.. code-block:: text

    rlinf/
    ├── envs/
    │   └── wrappers.py              # DataCollectorWrapper
    │
    ├── workers/
    │   ├── env/
    │   │   └── env_worker.py        # EnvWorker (data collection)
    │   └── reward/
    │       └── reward_worker.py     # RewardWorker (inference)
    │
    ├── runners/
    │   └── embodied_runner.py       # _apply_reward_model()
    │
    └── models/embodiment/reward/
        └── resnet_reward_model.py   # ResNet implementation

    examples/embodiment/
    ├── config/
    │   ├── maniskill_collect_reward_data.yaml
    │   ├── maniskill_train_reward_model.yaml
    │   └── maniskill_ppo_mlp_resnet_reward.yaml
    ├── train_embodied_agent.py      # Main entry
    └── train_reward_model.py        # Standalone reward training
