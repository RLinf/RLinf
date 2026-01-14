ResNet Reward Model Training Guide
===================================

This guide explains how to train a ManiSkill PickCube task using a ResNet-based reward model.

Overview
--------

The ResNet reward model is an image-based binary classifier that predicts whether the robot has successfully completed the grasping task. The complete pipeline consists of four stages:

1. **Data Collection**: Collect RGB images with success/failure labels using ``DataCollectorWrapper``
2. **ResNet Training**: Train the ResNet binary classifier using ``RewardWorker`` with FSDP support
3. **Policy Pre-training**: Train an initial policy using the environment's dense reward
4. **ResNet Reward Training**: Continue training using ResNet reward to replace the environment reward

Architecture
------------

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                        Training Pipeline                         │
    └─────────────────────────────────────────────────────────────────┘
    
    Stage 1: Data Collection
    ┌───────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │   EnvWorker   │────▶│ DataCollectorWrapper│────▶│  pkl files   │
    │               │     │   (Env Wrapper)      │     │ (raw data)   │
    └───────────────┘     └─────────────────────┘     └──────────────┘
    
    Stage 2: Reward Model Training
    ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │  pkl files   │────▶│    RewardWorker     │────▶│ checkpoint   │
    │              │     │   (FSDP support)     │     │   .pt file   │
    └──────────────┘     └─────────────────────┘     └──────────────┘
    
    Stage 3-4: Policy Training with Reward Model
    ┌───────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │EmbodiedRunner │────▶│    RewardWorker     │────▶│   Rewards    │
    │               │     │ compute_rewards()   │     │              │
    └───────────────┘     └─────────────────────┘     └──────────────┘

Prerequisites
-------------

- ManiSkill environment properly installed
- GPU with sufficient memory for rendering and training

Stage 1: Data Collection
------------------------

Collect RGB images labeled as success/failure using the ``DataCollectorWrapper``.

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_collect_reward_data

This will:

- Train a policy using dense reward (same as ``maniskill_ppo_mlp``)
- Automatically collect data via ``DataCollectorWrapper``
- Save success/failure samples to pkl files

Configuration (``maniskill_collect_reward_data.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    reward_data_collection:
      enabled: True
      save_dir: "${oc.env:EMBODIED_PATH}/data"
      target_success: 5000      # Number of success samples to collect
      target_failure: 5000      # Number of failure samples to collect
      sample_rate_fail: 0.1     # Sample 10% of failure frames
      sample_rate_success: 1.0  # Sample 100% of success frames

Using DataCollectorWrapper Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use the ``DataCollectorWrapper`` directly in your code:

.. code-block:: python

    from rlinf.envs.wrappers import DataCollectorWrapper
    
    # Wrap your environment
    env = DataCollectorWrapper(
        env=your_env,
        save_dir="./reward_data",
        target_success=5000,
        target_failure=5000,
        sample_rate_success=1.0,
        sample_rate_failure=0.1,
    )
    
    # Run episodes - data is collected automatically in step()
    obs, info = env.reset()
    while not env.is_full:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Save collected data
    env.save("reward_data.pkl")

Data will be saved as individual pkl files (one per episode)::

    save_dir/
    ├── success/
    │   ├── episode_000000.pkl
    │   ├── episode_000001.pkl
    │   └── ...
    └── failure/
        ├── episode_000002.pkl
        └── ...

Each pkl file contains::

    {
        "frames": [  # List of 50 frames
            {
                "obs": ...,       # Observation (numpy array)
                "action": ...,    # Action taken
                "reward": ...,    # Step reward
                "done": ...,      # Episode done flag
                "grasp": ...,     # Grasp success at this step
                "success": ...,   # Task success at this step
                "info": {...},    # Additional info
            },
            ...
        ],
        "num_frames": 50,
        "grasp": True/False,      # Episode-level grasp success
        "success": True/False,    # Episode-level task success
        "done": True,
        "label": 1/0,             # 1=success, 0=failure
        "metadata": {...},
    }

Stage 2: Train ResNet Reward Model
----------------------------------

Train the ResNet binary classifier using ``RewardWorker`` with FSDP distributed training support.

Option A: Standalone Training Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python examples/embodiment/train_reward_model.py --config-name maniskill_train_reward_model

Option B: Integrated Training with RewardWorker (FSDP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For distributed training, configure ``reward_training`` in your config:

.. code-block:: yaml

    reward_training:
      enabled: True
      data_path: "${oc.env:EMBODIED_PATH}/data"
      epochs: 100
      micro_batch_size: 32
      global_batch_size: 64
      lr: 1.0e-4
      weight_decay: 1.0e-5
      save_dir: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints"

Then run with distributed training:

.. code-block:: bash

    torchrun --nproc_per_node=4 examples/embodiment/train_embodied_agent.py \
        --config-name maniskill_ppo_mlp_resnet_reward \
        reward_training.enabled=True

The ``RewardWorker`` automatically uses:

- **FSDP** for model sharding across GPUs
- **DistributedSampler** for data parallel loading
- **Gradient accumulation** for large effective batch sizes
- **Mixed precision** training

Configuration (``maniskill_train_reward_model.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    reward_model_training:
      data_path: "${oc.env:EMBODIED_PATH}/data"
      epochs: 100
      batch_size: 64
      lr: 1.0e-4
      val_split: 0.1
      save_dir: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints"
      early_stopping_patience: 15

The trained model will be saved to ``logs/reward_checkpoints/best_model.pt``.

Stage 3 & 4: Policy Training with ResNet Reward
-----------------------------------------------

Stage 3: Pre-train Policy with Dense Reward (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train an initial policy using the environment's native dense reward:

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

Checkpoints will be saved to ``logs/<timestamp>-maniskill_ppo_mlp/maniskill_ppo_mlp/checkpoints/``.

Stage 4: Continue Training with ResNet Reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update ``resume_dir`` in ``maniskill_ppo_mlp_resnet_reward.yaml``:

.. code-block:: yaml

    runner:
      # Set to your maniskill_ppo_mlp checkpoint path
      resume_dir: "logs/<timestamp>-maniskill_ppo_mlp/maniskill_ppo_mlp/checkpoints/global_step_100"

Then run:

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

Configuration
-------------

Key Parameters (``maniskill_ppo_mlp_resnet_reward.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

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
        checkpoint_path: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints/best_model.pt"
        threshold: 0.5
        use_soft_reward: False  # Binary 0/1 reward

Critical Parameter Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parameters **must** match those used during data collection:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - ``control_mode``
     - ``pd_joint_delta_pos``
     - Control mode (8-dim action space)
   * - ``reward_render_mode``
     - ``episode_end``
     - Only render images at episode end
   * - ``show_goal_site``
     - ``True``
     - Show green goal marker
   * - ``image_size``
     - ``[3, 224, 224]``
     - Image dimensions

Expected Results
----------------

- After ~500-1000 steps, ``env/success_once`` should approach 100%
- ``env/episode_len`` should decrease to ~15-20 steps
- ``env/reward`` will show lower values (expected for sparse binary reward)

API Reference
-------------

DataCollectorWrapper
~~~~~~~~~~~~~~~~~~~~

Environment wrapper for automatic data collection.

.. code-block:: python

    from rlinf.envs.wrappers import DataCollectorWrapper

.. list-table::
   :header-rows: 1

   * - Method/Property
     - Description
   * - ``step(action)``
     - Execute step and automatically collect data
   * - ``save(filename)``
     - Save collected data to pkl file
   * - ``is_full``
     - Check if collection targets are met
   * - ``success_count``
     - Current number of success samples
   * - ``failure_count``
     - Current number of failure samples
   * - ``get_statistics()``
     - Get collection statistics dict

RewardWorker
~~~~~~~~~~~~

Worker with FSDP support for distributed reward model training and inference.

.. code-block:: python

    from rlinf.workers.reward.reward_worker import RewardWorker

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``init_worker()``
     - Initialize reward model (RewardManager or rule-based)
   * - ``build_dataloader()``
     - Build distributed DataLoader with DistributedSampler
   * - ``run_training()``
     - Run one FSDP training step with gradient accumulation
   * - ``compute_rewards(input_channel, output_channel)``
     - Compute rewards for rollout results
   * - ``compute_batch_rewards_with_model(observations)``
     - Compute rewards using trained model
   * - ``save_checkpoint(path, step)``
     - Save model checkpoint
   * - ``load_checkpoint(path)``
     - Load model checkpoint

RewardManager
~~~~~~~~~~~~~

Unified interface for reward computation with registry pattern.

.. code-block:: python

    from rlinf.algorithms.rewards.embodiment import RewardManager

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``compute_rewards(observations, task_descriptions)``
     - Unified reward computation
   * - ``register_model(name, cls)``
     - Register new model type
   * - ``get_available_models()``
     - List registered models
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
    │   └── reward/
    │       └── reward_worker.py     # RewardWorker (FSDP support)
    │
    ├── algorithms/rewards/embodiment/
    │   ├── reward_manager.py        # RewardManager with registry
    │   ├── reward_data_collector.py # Legacy collector (deprecated)
    │   └── reward_model_trainer.py  # Standalone trainer
    │
    └── models/embodiment/reward/
        ├── base_reward_model.py     # Base class
        ├── resnet_reward_model.py   # ResNet implementation
        └── qwen3_vl_reward_model.py # VLM implementation

    examples/embodiment/
    ├── train_embodied_agent.py      # Main entry (creates RewardWorker)
    ├── train_async.py               # Async entry (creates RewardWorker)
    └── train_reward_model.py        # Standalone reward training
