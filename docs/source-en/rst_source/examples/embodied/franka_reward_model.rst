Real-World RL with Franka (Reward Model)
=========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document describes how to use a reward model when training on a Franka robotic arm in the real world.
The focus is on training and deploying a ResNet-based reward model from scratch to assist robotic manipulation tasks.

Before getting started, it is strongly recommended to read the following documents:

1. :doc:`franka` — to familiarize yourself with the end-to-end real-world Franka training pipeline.
2. :doc:`../../tutorials/extend/reward_model` — to understand the complete reward model workflow in RLinf's simulated environments.

Prerequisites
-----------------------

Follow all steps in the :doc:`franka` document up to and including **Data Collection** (i.e., everything before the "Running the Experiment" section).

Data Collection
-----------------------

Expert Trajectory Data Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For data collection, expert trajectory data needs to be collected first.
This data will be stored in the demo buffer during training.
Specifically, follow the steps in the **Data Collection** section under **Running the Experiment** in :doc:`franka`.
Make sure that in ``examples/embodiment/config/realworld_collect_data.yaml``, ``data_collection`` under the ``env`` section is enabled:

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: True

After launching the data collection script, episodes are automatically saved to ``save_dir``.
When ``export_format="pickle"``, each episode is written to a separate ``.pkl`` file, which is convenient for subsequent offline preprocessing.

Reward Model Training and Evaluation Data Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To obtain a high-quality reward model, more data needs to be collected for training and evaluation.
Building on the **Data Collection** section in :doc:`franka`, make the following modifications to the collection script.

Increase the ``success_hold_steps`` field to obtain more successful data within a limited number of collection episodes.
When the robot arm end-effector reaches the target pose, success is not declared immediately — the arm must maintain the target pose for a certain number of steps (``success_hold_steps``) before being marked as successful.
If the arm exits the target zone mid-hold, the counter resets.

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

After launching the data collection script, episodes are automatically saved to ``save_dir``.
When ``export_format="pickle"``, each episode is written to a separate ``.pkl`` file, which is convenient for subsequent offline preprocessing.

During collection, move the robot arm slowly to obtain more diverse failure samples.
When reaching the target pose, make small-range movements while maintaining the target pose to obtain more diverse successful samples.


Preprocessing into a Reward Dataset
----------------------------------------------

This step is identical to **Section 1.2 — Preprocessing into a Reward Dataset** in :doc:`../../tutorials/extend/reward_model`.

In particular, it is recommended to increase ``fail-success-ratio`` to ``3``.

.. code-block:: bash

   Example:
       python examples/reward/preprocess_reward_dataset.py \
           --raw-data-path logs/xxx/collected_data \
           --output-dir logs/xxx/processed_reward_data \
           --fail-success-ratio 3

Reward Model Training
-----------------------

This step is identical to **Section 2 — Reward Model Training** in :doc:`../../tutorials/extend/reward_model`.

In particular, for real-world scenarios, it is recommended to lower the ``min_delta`` of ``early_stop``, for example:

.. code-block:: bash

  runner:
    early_stop:
      min_delta: 1e-6

Cluster Configuration
-----------------------

This step is identical to the **Cluster Configuration** section under **Running the Experiment** in :doc:`franka`.

Configuration File
-----------------------

This step is identical to the **Configuration File** section under **Running the Experiment** in :doc:`franka`, applied to ``examples/embodiment/config/realworld_charger_sac_cnn_async_standalone_reward.yaml``.
In addition, enable the reward model parameters under the ``reward`` section:

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.8

     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"

Where:

- ``reward_mode`` controls whether the reward model runs inference at every step or only on terminal frames.
- ``standalone_realworld`` uses the reward model to directly determine task success and trigger environment resets.
- ``reward_threshold`` applies threshold filtering on the success probability output by the reward model; values below the threshold are set to ``0``.
- ``model_path`` points to the reward model checkpoint used for online inference.

Starting the Experiment
-----------------------

Once training begins, the reward model directly judges task success/failure based on image observations and drives environment resets.
The remaining steps follow the **Running the Experiment** section of :doc:`franka`.

Worker Interaction During Rollout
----------------------------------------------

Unlike **Section 3.2 — Worker Interaction During Rollout** and **Section 3.3 — Final Reward Computation** in :doc:`../../tutorials/extend/reward_model`:
in real-world systems with ``standalone_realworld`` enabled, the reward model does **not** combine env rewards with reward model outputs.

In other words, the reward model does **not** act as an additional reward source inside the env worker when constructing the final reward,
because the system bypasses the weighted sum of ``env_reward`` and ``reward_model_output`` entirely.
Therefore, ``reward_mode``, ``reward_weight``, and ``env_reward_weight`` all have no effect.
The final reward is generated directly by FrankaEnv based on the reward model's success/failure determination.

From a system perspective, the actual behavior in the real-world system can be understood as:
directly replacing the ``env_reward`` inside the env worker, re-using the original ``env_reward`` logic to assign rewards and trigger environment resets, thereby fundamentally integrating the reward model.
