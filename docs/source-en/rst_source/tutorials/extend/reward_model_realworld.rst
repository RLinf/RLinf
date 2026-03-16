Reward Model Integration (Real-World)
==============================================================

This guide describes how to integrate learned reward models into RLinf for embodied RL training. Reward models can replace or augment environment rewards by learning from visual observations.

Overview
--------

RLinf supports image-based reward models that:

- Take RGB images as input
- Output success/failure probabilities
- Can be used for terminal rewards (at the end of an episode) or per-step rewards

Components
----------

The reward model system includes:

1. **ResNetRewardModel** - A ResNet-based binary classifier.
2. **ImageRewardWorker** - An inference worker used during RL training.
3. **FSDPRewardWorker** - A worker used for training the reward model with FSDP.

Environment Variables
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``TORCHDYNAMO_DISABLE=1``
     - Disables torch dynamo to avoid jinja2 compatibility issues.
   * - ``DEBUG_IMAGE_SAVE_DIR``
     - Directory to save debug visualization images for ResNet classification.

Configuration
-------------

Add the reward model to your YAML configuration:

.. code-block:: yaml

    reward:
      use_reward_model: True
      group_name: "RewardGroup"
      
      # Reward Mode:
      # - "per_step": Calculate reward for every frame
      # - "terminal": Calculate reward only at the end of the episode
      reward_mode: "per_step"
      
      # Asymmetric reward mapping
      success_reward: 1.0   # Reward for success (prob > threshold)
      fail_reward: 0.0      # Reward for failure (prob < threshold)
      reward_threshold: 0.5 # Probability threshold
      
      # Combination method with environment rewards
      combine_mode: "replace"  # Options: "replace", "add", "weighted"
      reward_weight: 10.0
      env_reward_weight: 0.0
      
      model:
        model_type: "resnet_reward"
        arch: "resnet18"
        hidden_dim: 256
        dropout: 0.0
        image_size: [3, 128, 128]
        normalize: true
        precision: "bf16"
        checkpoint_path: "path/to/checkpoint/best_model"
        debug_save_dir: null

Training the Reward Model (Example: Peg Insertion Task)
----------------------------------------------------------------------

1. **Data Collection**
   
   Modify the real-world data collection configuration in ``examples/embodiment/config/realworld_collect_data.yaml``:

   .. code-block:: yaml

      cluster:
        node_groups:
          - hardware:
              type: Franka
              configs:
                - robot_ip: ROBOT_IP # Replace with your robot's IP

      runner:
        num_data_episodes: 20 # Recommend collecting at least 20 episodes

      env:
        eval:
          override_cfg:
            target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0] # Set to the calibrated target pose
            success_hold_steps: 20  # Recommended: hold for 20 steps to count as a success


   Note: It is recommended to set ``success_hold_steps`` to 20 so that a single successful trajectory can capture at least 20 frames of successful states. This helps in quickly collecting a large volume of "success" images for training.
   
   Once configured, launch the collection script:

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh

   Collected data will be saved in ``logs/[running-timestamp]/data.pkl``. Upload ``data.pkl`` to the GPU training server for the next steps.

2. **Data Processing**

   Before training the model, you need to clean and process the collected data into a format suitable for training.
   
   .. code-block:: bash

    # Specify the data path
    python examples/embodiment/process_realworld_data.py /path/to/your/input_data.pkl
   
   The ``process_realworld_data.py`` script will create a ``collected_data`` folder in the same directory as ``input_data.pkl``, which contains the processed data for training.

3. **Model Training**

   Use the reward training configuration (make sure to update ``data_path`` to the actual path). The real-world reward model training command is mapped as follows:

   .. code-block:: bash

       bash examples/reward/run_reward_training.sh \
            --data /path/to/collected_data \
            data.num_samples_per_episode=0 \
            data.fail_success_ratio=3.0 \
            actor.global_batch_size=32 \
            actor.micro_batch_size=16 \
            runner.max_epochs=100 \
            runner.val_check_interval=10 \
            runner.save_interval=10 \
            runner.early_stop.patience=10 \
            data.debug_save_dir="logs/training_data_debug_realworld" \
            runner.logger.log_path="logs/reward_model_realworld"
   
   The trained reward model files will be stored in ``logs/reward_model_realworld``.

4. **Using the Model in RL Training**

   Modify the real-world training configuration ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``:

   .. code-block:: yaml

       reward:
          use_reward_model: True
          reward_mode: "per_step"
          combine_mode: "replace"
          checkpoint_path: "logs/reward_model_realworld/checkpoints/best_model"  # Path to your reward model checkpoint folder
   
   Start training:

   .. code-block:: bash

       bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

Debugging
---------

Enable debug image saving to visualize ResNet classification results:

.. code-block:: yaml

    model:
      debug_save_dir: "logs/debug_images"

Images will be saved to:

- ``resnet_success/`` - Images classified as success.
- ``resnet_fail/`` - Images classified as failure.

Filename format: ``{id:06d}_prob{probability:.4f}.png``

Troubleshooting
------------------------------

**Reward Hacking (High reward but 0% actual success rate)**

- Switch to terminal mode: ``reward_mode: "terminal"``.
- Increase the threshold: ``reward_threshold: 0.6``.

**Policy Collapse (All negative rewards)**

- Use asymmetric rewards: Set ``fail_reward: 0.0`` instead of a negative value.
- Do not set ``use_negative_reward: true``.

**Checkpoint Loading Errors**

- Ensure ``hidden_dim`` matches the trained checkpoint.
- Check that ``image_size`` matches the training data.