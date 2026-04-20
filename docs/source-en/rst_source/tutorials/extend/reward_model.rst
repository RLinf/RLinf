Reward Model Guide
========================

This document describes how to use a reward model in RLinf. It covers three parts:

1. Data collection: collect raw episode data during RL runs.
2. Reward model training: preprocess the raw data and train an image-based reward model.
3. Reward model inference in RL: plug the trained model into online rollout and use it in final reward computation.

1. Data Collection
----------------------------

Reward model training data is typically built from episode-level data collection. RLinf provides
a unified collection wrapper, and the related usage is documented in :doc:`the data collection tutorial <../components/data_collection>`.

For reward model use cases, we recommend saving raw episodes in ``pickle`` format first, then converting
them into processed training splits with the preprocessing script.

1.1 Enable Data Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enable ``data_collection`` under ``env`` in your YAML config:

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: False

After training or evaluation starts, the environment will automatically save episodes into ``save_dir``.
When ``export_format="pickle"``, each episode is written as an individual ``.pkl`` file for later offline preprocessing.

1.2 Preprocess into a Reward Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Raw ``pickle`` files cannot be consumed by reward model training directly. Use
``examples/reward/preprocess_reward_dataset.py`` to convert collected ``.pkl`` episodes into
``.pt`` files that can be loaded by ``RewardBinaryDataset``. In the current implementation,
the script extracts ``main_images`` from observations and builds binary labels from per-step
``info["success"]``.

Example:

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data

By default, this produces:

.. code-block:: text

   logs/xxx/processed_reward_data/
   ├── train.pt
   └── val.pt

The generated ``.pt`` files follow the canonical ``RewardDatasetPayload`` schema:

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],
       "metadata": dict[str, Any],
   }

Where:

- ``images`` stores the training images.
- ``labels`` stores the binary labels.
- ``metadata`` stores source path, sampling arguments, split ratio, and related preprocessing info.

``RewardBinaryDataset`` then loads these ``train.pt`` / ``val.pt`` files directly.

2. Reward Model Training
----------------------------

RLinf provides ``examples/reward/run_reward_training.sh`` as the launch script for reward model training.

2.1 Configure Dataset Paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before training, edit ``examples/reward/config/reward_training.yaml`` so it points to your processed splits:

.. code-block:: yaml

   data:
     train_data_paths: "logs/processed_reward_data/train.pt"
     val_data_paths: "logs/processed_reward_data/val.pt"

.. note::

   At present, ``run_reward_training.sh`` mainly prepares the launch command and log directory.
   The dataset paths are taken from ``reward_training.yaml``, specifically
   ``data.train_data_paths`` and ``data.val_data_paths``.

2.2 Configure the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At present, the reward model registry supports only one implementation: ``ResNetRewardModel``.
Therefore ``actor.model.model_type`` should be set to ``"resnet"``:

.. code-block:: yaml

   actor:
     model:
       model_type: "resnet"
       arch: "resnet18"
       pretrained: False
       image_size: [3, 128, 128]

If you want to continue training from existing weights, set ``model_path`` to a checkpoint.
If you want to train from scratch, keep ``model_path: null``.

2.3 Launch Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the dataset and model are configured, run:

.. code-block:: bash

   bash examples/reward/run_reward_training.sh

Training logs are written to a newly created ``logs/<timestamp>-reward_training`` directory.

3. Reward Model Inference in RL
---------------------------------

RLinf provides two example configs for integrating a reward model into RL:

- ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``
- ``examples/embodiment/config/maniskill_sac_mlp_resnet_reward_async.yaml``

These configs show how to enable a reward worker in RL training while keeping the policy on state observations
and the reward model on image observations.

3.1 Key Config Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reward-model-related settings live under the ``reward`` section:

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     reward_mode: "terminal"   # or "per_step"
     reward_threshold: 0.5
     reward_weight: 1.0
     env_reward_weight: 0.0

     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"

Where:

- ``reward_mode`` controls whether inference happens every step or only on terminal frames.
- ``reward_weight`` and ``env_reward_weight`` control how learned reward and environment reward are combined.
- ``reward_threshold`` filters reward model probabilities; values below the threshold are set to ``0``.
- ``model_path`` points to the reward model checkpoint used for online inference.

3.2 Worker Interaction During Rollout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During online RL, the ``env``, ``rollout``, and ``reward`` workers collaborate as follows:

.. code-block:: text

   Env worker
      | 1. Interacts with the environment and gets obs / env reward / done
      | 2. Sends obs to the Rollout worker to produce actions
      | 3. When reward model is enabled, sends ``main_images`` to the Reward worker
      v
   Reward worker
      | 4. Runs forward inference and returns reward model output
      v
   Env worker
      | 5. Receives bootstrap values from the Rollout worker
      | 6. Combines env reward with reward model output
      v
   Final reward -> stored in rollout results and used by later RL updates

In the implementation, ``EnvWorker`` requests reward model outputs during rollout and then computes the final reward centrally.

3.3 Final Reward Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the reward channel is enabled, ``EnvWorker`` first fetches ``reward_model_output``,
then merges it with the original environment reward inside ``compute_bootstrap_rewards``:

.. code-block:: python

   reward = env_reward_weight * env_reward + reward_weight * reward_model_output

If bootstrap is enabled by the algorithm config, RLinf may also add bootstrap values to the last step reward.

From a system perspective, the reward model does not replace the original bootstrap reward. Instead, it serves as
an additional reward source inside the env worker and participates in final reward construction.

Summary
----------------------------

The full workflow is:

1. Enable ``data_collection`` in the environment config and save raw data in ``pickle`` format.
2. Use ``examples/reward/preprocess_reward_dataset.py`` to convert raw ``.pkl`` files into ``RewardDatasetPayload``-compatible ``train.pt`` / ``val.pt`` splits.
3. Complete the data and model configuration in ``reward_training.yaml``, then use ``examples/reward/run_reward_training.sh`` to start ``ResNetRewardModel`` training.
4. Enable ``reward.use_reward_model=True`` in your RL YAML and plug the trained reward worker into online RL inference.
