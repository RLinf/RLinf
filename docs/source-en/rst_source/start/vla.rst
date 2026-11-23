Quick Start
===========

Start PPO training on `ManiSkill3 <https://github.com/haosulab/ManiSkill>`__ with **MLP**, **OpenVLA**, or **OpenVLA-OFT**.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 1. Install
      :text-align: center

      ``install.sh`` or Docker Hub

   .. grid-item-card:: 2. Pick a Track
      :text-align: center

      1-GPU MLP or 7B VLA

   .. grid-item-card:: 3. Train
      :text-align: center

      Run launch script

   .. grid-item-card:: 4. Watch Results
      :text-align: center

      TensorBoard curves

Setup and Run
-------------

Pick a starting track below and run the commands in order. For regional mirrors, multiple virtual environments, or Docker Hub images, see :doc:`installation`.

.. tab-set::

   .. tab-item:: MLP

      Trains a lightweight MLP policy on ManiSkill3's built-in ``PickCube-v1`` task on GPU 0.

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate

      **2. Launch Training**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

   .. tab-item:: OpenVLA
      :sync: openvla

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

      **2. Download ManiSkill Assets**

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets

      **3. Launch Training**

      First run downloads the default Hugging Face checkpoint automatically:

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

      **3. Launch Training**

      First run downloads the default base and LoRA checkpoints automatically:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_quickstart

Each config name matches a YAML file under ``examples/embodiment/config/`` — see :doc:`../reference/configuration` for all config directories and tuning presets.

.. dropdown:: Common Overrides (Optional)

   - **Use local checkpoints instead of Hugging Face:** append ``actor.model.model_path=/path/to/checkpoint`` (rollout inherits it automatically; for OpenVLA-OFT also pass ``actor.model.lora_path=/path/to/lora``).
   - **Target specific GPUs:** the two VLA quickstarts use all available GPUs by default; set ``cluster.component_placement`` to pin devices (:doc:`../reference/configuration`).

Results and Troubleshooting
---------------------------

Logs, TensorBoard events, and checkpoints are written to ``logs/<timestamp>-<config>/``. Start TensorBoard with ``tensorboard --host 0.0.0.0 --logdir ./logs`` and watch ``rollout/env_info/return`` and ``rollout/env_info/success_once``.

The MLP run prints a metric table at each step:

.. code-block:: text

   ╭───────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step: 10/1000 │ Progress: ████░░░░░░░░░░░░░░ │ 1.0%              │
   │ Elapsed: 00:51       │ ETA: 01:24:09                │ Step Time: 5.122s │
   ├───────────────────────────── Environment ───────────────────────────────┤
   │ episode_len=50.0   │ num_trajectories=1024          │ return=7.187      │
   ├──────────────────────────── Training/Actor ─────────────────────────────┤
   │ actor/policy_loss=-0.013 │ actor/clip_fraction=0.179 │ critic/explained_variance=0.646 │
   ╰─────────────────────────────────────────────────────────────────────────╯

On full runs, MLP ``success_once`` climbs above ``0.90`` within a few hundred epochs, while the VLA quickstarts improve steadily from their SFT baselines:

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embody-quickstart-metric.jpg
   :align: center
   :width: 90%

   TensorBoard training curves for the ManiSkill3 VLA quick start.

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

   .. grid-item-card:: RLinf Walkthrough
      :link: walkthrough
      :link-type: doc

      Visual tour of how RLinf runs, scales across GPUs, and organizes configs and tools.

   .. grid-item-card:: Embodied Training
      :link: ../examples/index
      :link-type: doc

      Explore LIBERO, RoboTwin, OpenPI, SFT, and world models.

   .. grid-item-card:: Standalone Evaluation
      :link: ../evaluations/index
      :link-type: doc

      Evaluate trained checkpoints across embodied benchmarks.
