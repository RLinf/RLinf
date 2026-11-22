Quick Start
===========

Verify your setup in minutes with a **single-GPU MLP** run on `ManiSkill3 <https://github.com/haosulab/ManiSkill>`__ (no downloads), or jump straight into **OpenVLA** / **OpenVLA-OFT** PPO training.

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

      One launch command

   .. grid-item-card:: 4. Watch Results
      :text-align: center

      TensorBoard curves

Setup and Run
-------------

Pick a starting track below and run the commands in order. For regional mirrors, multiple virtual environments, or Docker Hub images, see :doc:`installation`.

.. tab-set::

   .. tab-item:: MLP (1 GPU · ~4 GB VRAM · No Downloads)

      Trains a lightweight MLP policy on ManiSkill3's built-in ``PickCube-v1`` task using GPU 0 — no checkpoints or external assets required (~3.7 GB peak VRAM, ~1.5 min for a 10-epoch check).

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate

      **2. Launch Training**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp runner.max_epochs=10 runner.val_check_interval=-1

      Drop ``runner.max_epochs=10 runner.val_check_interval=-1`` for full 1000-epoch training (``success_once`` climbs above ``0.90``).

   .. tab-item:: OpenVLA (7B VLA)
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

   .. tab-item:: OpenVLA-OFT (7B LoRA VLA)
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

A 10-epoch MLP quick check finishes in about 1.5 minutes on one GPU (~3.7 GB peak VRAM) and prints a metric table at each step:

.. code-block:: text

   ╭───────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step: 10/10 │ Progress: ████████████████████ │ 100.0%            │
   │ Elapsed: 00:51     │ ETA: 00:00                     │ Step Time: 5.122s │
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
