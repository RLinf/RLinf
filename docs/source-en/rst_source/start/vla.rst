Quick Start
===========

Train **OpenVLA** or **OpenVLA-OFT** on `ManiSkill3 <https://github.com/haosulab/ManiSkill>`__ with PPO in four steps.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 1. Install
      :text-align: center

      ``install.sh`` or Docker Hub

   .. grid-item-card:: 2. Download
      :text-align: center

      Checkpoint + ManiSkill assets

   .. grid-item-card:: 3. Train
      :text-align: center

      One launch command

   .. grid-item-card:: 4. Watch Results
      :text-align: center

      TensorBoard curves

Prerequisites
-------------

Install the ManiSkill environment first (:doc:`installation`). For faster Hugging Face downloads in mainland China, export ``HF_ENDPOINT=https://hf-mirror.com``.

Download Assets
---------------

Download the ManiSkill3 scene assets once from the repository root:

.. code-block:: bash

   hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets

Download Model and Launch Training
----------------------------------

Select a model tab below. Both recipes use all available accelerators by default; edit ``cluster.component_placement`` to select specific devices (:doc:`../guides/basic_config`).

.. tab-set::

   .. tab-item:: OpenVLA

      Download the warmup checkpoint (`paper <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c0fabe372177d2aded596be2d3b4544-Abstract-Conference.html>`__):

      .. code-block:: bash

         hf download gen-robot/openvla-7b-rlvla-warmup --local-dir ./checkpoints/openvla-7b-rlvla-warmup

      Launch PPO training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
           actor.model.model_path=$PWD/checkpoints/openvla-7b-rlvla-warmup

   .. tab-item:: OpenVLA-OFT

      Download the base model and LoRA adapter:

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir ./checkpoints/openvlaoft-base && \
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir ./checkpoints/openvlaoft-lora

      Launch PPO training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_quickstart \
           actor.model.model_path=$PWD/checkpoints/openvlaoft-base \
           actor.model.lora_path=$PWD/checkpoints/openvlaoft-lora

Rollout inherits ``actor.model.model_path`` automatically, so you pass the checkpoint path once.

Watch Training Results
----------------------

Logs and checkpoints are saved under ``./logs``. Open TensorBoard and track ``rollout/env_info/return`` and ``rollout/env_info/success_once``:

.. code-block:: bash

   tensorboard --host 0.0.0.0 --logdir ./logs

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embody-quickstart-metric.jpg
   :align: center
   :width: 90%

   TensorBoard training curves for the ManiSkill3 quick start.

Next Steps
----------

- :doc:`Browse the Example Gallery <../examples/index>` for LIBERO, RoboTwin, OpenPI, and real-world robots.
- :doc:`Run standalone evaluation <../evaluations/index>` on trained checkpoints.
- :doc:`Tune performance <../guides/performance/index>` (including :doc:`Training Pipeline Mode <../guides/training_pipeline>` and :doc:`Execution Modes <../concepts/execution_modes>`).
