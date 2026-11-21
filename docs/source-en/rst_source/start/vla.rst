VLA Quick Start
===============

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

Setup and Run
-------------

Select a model tab below — each tab contains the full install, download, and launch commands. Both recipes use all available accelerators by default; edit ``cluster.component_placement`` to target specific devices (:doc:`../guides/basic_config`). In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download`` (or pull a prebuilt image from `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__).

.. tab-set::

   .. tab-item:: OpenVLA
      :sync: openvla

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

      **2. Download ManiSkill Assets** (`paper <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c0fabe372177d2aded596be2d3b4544-Abstract-Conference.html>`__)

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets

      **3. Launch PPO Training** (downloads ``gen-robot/openvla-7b-rlvla-warmup`` into your Hugging Face cache automatically on first run)

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. Download ManiSkill Assets**

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets

      **3. Launch PPO Training** (downloads ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` and ``Base-Lora`` automatically on first run)

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_quickstart

Rollout inherits ``actor.model.model_path`` automatically. To use pre-downloaded local checkpoints instead of Hugging Face repo IDs, pass overrides on the command line (``actor.model.model_path=/path/to/checkpoint`` and, for OpenVLA-OFT, ``actor.model.lora_path=/path/to/lora``) or see :doc:`Training Configuration <../reference/configuration>`.

Results and Troubleshooting
---------------------------

Training writes logs and checkpoints under ``./logs``. Open TensorBoard (``tensorboard --host 0.0.0.0 --logdir ./logs``) and track ``rollout/env_info/return`` and ``rollout/env_info/success_once``.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embody-quickstart-metric.jpg
   :align: center
   :width: 90%

   TensorBoard training curves for the ManiSkill3 quick start.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Out of GPU memory (OOM)
     - Lower ``actor.micro_batch_size`` and ``env.train.total_num_envs``; see :ref:`Out of Memory <faq-oom>`.
   * - EGL or Vulkan rendering error
     - Check graphics driver and EGL config in :ref:`Rendering Issues <faq-rendering>`.
   * - Stale Ray / NCCL error on rerun
     - Run ``ray stop`` before relaunching (:ref:`CUDA and Memory Issues <faq-cuda>`).

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Embodied Training
      :link: ../examples/index
      :link-type: doc

      Explore LIBERO, RoboTwin, OpenPI, SFT, and world models.

   .. grid-item-card:: Standalone Evaluation
      :link: ../evaluations/index
      :link-type: doc

      Evaluate trained checkpoints across embodied benchmarks.

   .. grid-item-card:: Performance Tuning
      :link: ../guides/performance/index
      :link-type: doc

      Learn :doc:`Training Pipeline Mode <../guides/training_pipeline>` and :doc:`Execution Modes <../concepts/execution_modes>`.
