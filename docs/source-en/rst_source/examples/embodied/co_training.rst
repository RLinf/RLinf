Co-Training
===========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-co/overview.png
   :align: center
   :width: 90%

   Sim-real co-training overview.

**Sim-real co-training** trains a π₀.₅ policy by combining **PPO in simulation** with **SFT on
real data**, so the policy improves task success in sim while retaining real-world priors and
avoiding sim-only overfitting that hurts sim-to-real transfer. For technical details, see
:doc:`Beyond Imitation: RL-Based Sim-Real Co-Training for VLA Models <../../resources/publications/rlinf_co>`.

Overview
--------

Co-train π₀.₅ on a ManiSkill digital twin — PPO in sim + SFT on 50 real trajectories (~35%→~50% sim success).

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO + SFT (RL-Co)

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Environments / Data
      :text-align: center

      ManiSkill digital twin

   .. grid-item-card:: Training
      :text-align: center

      2-stage sim-real

| **You'll do:** install → download assets + SFT model → SFT (Stage I) → co-train RL (Stage II) → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · ManiSkill assets · an SFT checkpoint and the real dataset (steps below).

Setup
~~~~~

This example ships one demo setup; for your own robot, collect data and build a matching sim scene.

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Part
     - Detail
   * - Task
     - Pick-and-place — move an object from the table into a bowl.
   * - Real
     - Franka Emika Panda + RealSense; third-person RGB (640×480); 7-DoF action (x, y, z, roll, pitch, yaw, gripper).
   * - Sim
     - ManiSkill3 digital twin aligned with the real setup in layout, camera view, task logic, language, and action space; dynamics tuned to approximate real physics.


Setup and Run
-------------

.. include:: _setup_common.rst

See :doc:`ManiSkill <maniskill>` for base ManiSkill asset setup. You can either start directly from the released Stage I SFT checkpoint in **Stage II** (recommended) or train Stage I SFT yourself first.

.. tab-set::

   .. tab-item:: Stage I: SFT Pretraining

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the custom ManiSkill digital-twin assets, the Stage I mix dataset (`RLinf/RLCo-Example-Mix-Data <https://huggingface.co/datasets/RLinf/RLCo-Example-Mix-Data>`__, 50 real + 1499 sim trajectories), and the 50 real trajectories for Stage II:

      .. code-block:: bash

         hf download RLinf/RLCo-maniskill-assets --repo-type dataset --include "custom_assets/*" --local-dir ./rlinf/envs/sim/maniskill/assets
         hf download RLinf/RLCo-Example-Mix-Data --repo-type dataset --local-dir ./RLCo-Example-Mix-Data
         hf download RLinf/RLCo-Example-Real-Data --repo-type dataset --local-dir ./RLCo-Example-Real-Data

      **3. Launch**

      Run Stage I SFT on ``./RLCo-Example-Mix-Data`` using `OpenPi <https://github.com/Physical-Intelligence/openpi>`__ or :doc:`OpenPI SFT <sft_openpi>`, then point ``actor.model.model_path`` to the resulting checkpoint in Stage II.

   .. tab-item:: Stage II: Real-Sim Co-Training

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the custom ManiSkill digital-twin assets, the released Stage I SFT checkpoint, and the 50 real trajectories for co-training:

      .. code-block:: bash

         hf download RLinf/RLCo-maniskill-assets --repo-type dataset --include "custom_assets/*" --local-dir ./rlinf/envs/sim/maniskill/assets
         hf download RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT --local-dir ./RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT
         hf download RLinf/RLCo-Example-Real-Data --repo-type dataset --local-dir ./RLCo-Example-Real-Data

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 24 50 26

         * - Recipe
           - Config
           - Command suffix
         * - OpenPI π₀.₅ + RL-Co
           - ``examples/embodiment/config/maniskill_ppo_co_training_openpi_pi05.yaml``
           - ``maniskill_ppo_co_training_openpi_pi05``

      Point ``actor.model.model_path`` (rollout inherits it automatically) to your SFT checkpoint and ``actor.sft_data_path`` to the real data path:

      .. code-block:: yaml

         actor:
           sft_data_path: ./RLCo-Example-Real-Data
           model:
             model_path: ./RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

      Launch co-training from the repository root (for common path, cluster, and runner fields, see :doc:`Training configuration <../../reference/configuration>`; for general PPO settings, see :doc:`π₀ and π₀.₅ RL training <pi0>`):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_co_training_openpi_pi05

.. dropdown:: Advanced configuration (optional)

   The shipped config already sets co-training options and the ``pi05_maniskill_sim_real_co_training`` dataconfig (keep model architecture and normalization consistent with Stage I):

   .. code-block:: yaml

      actor:
        model:
          openpi:
            config_name: "pi05_maniskill_sim_real_co_training"
        enable_sft_co_train: True
        sft_loss_weight: 0.2

   - ``enable_sft_co_train``: Set to ``True`` to enable co-training; ``False`` for PPO-only.
   - ``sft_loss_weight``: Weight for the SFT loss term in the total loss.

   The config ``batch_size`` is the micro-batch size before gradient accumulation; effective batch size is ``(global_batch_size * input_batch) / (micro_batch_size * num_gpus)``. See :doc:`./pi0` for ``global_batch_size`` and ``micro_batch_size`` settings.

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for standard metrics and logger backends.

Co-training adds these loss metrics:

- ``train/ppo_loss``: PPO (RL) loss.
- ``train/sft_loss``: SFT loss on real data.
- ``actor/total_loss``: Combined objective ``RL_loss + sft_loss_weight * SFT_loss``.
- ``train/loss_ratio``: Magnitude ratio ``(sft_loss_weight * |SFT_loss|) / |RL_loss|``. If this stays very large (for example above ``1e5``), the logger warns; lower ``sft_loss_weight``.

**Example outcomes**

- After loading Stage I: ~35% zero-shot success in sim.
- After 100 co-training steps: ~50% success in sim.

For real-robot deployment and ablations, see the paper: *Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models*.
