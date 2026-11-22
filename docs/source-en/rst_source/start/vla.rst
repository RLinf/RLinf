Quick Start
===========

Verify your setup in minutes with a **single-GPU MLP** run on `ManiSkill3 <https://github.com/haosulab/ManiSkill>`__ (no model or scene asset downloads), or jump straight into **OpenVLA** / **OpenVLA-OFT** PPO training.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 1. Install
      :text-align: center

      ``install.sh`` or Docker Hub

   .. grid-item-card:: 2. Pick a Track
      :text-align: center

      1-GPU MLP (zero downloads) or 7B VLA

   .. grid-item-card:: 3. Train
      :text-align: center

      One launch command

   .. grid-item-card:: 4. Watch Results
      :text-align: center

      TensorBoard curves

Setup and Run
-------------

Select a tab below — start with **MLP** for the fastest zero-download verification on a single GPU, or choose **OpenVLA** / **OpenVLA-OFT** for full VLA PPO training. If you already installed another model or environment in ``.venv``, pass ``--venv <dir>`` to ``install.sh`` so you do not overwrite it. For mainland China mirrors or prebuilt Docker Hub images, see :doc:`installation`.

.. tab-set::

   .. tab-item:: MLP (1 GPU · No Downloads)

      Trains a lightweight MLP policy from scratch with PPO on ManiSkill3's built-in ``PickCube-v1`` task using GPU 0 — no Hugging Face checkpoint or external scene asset downloads required.

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate

      **2. Launch PPO Training**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

      For a ~1-minute quick check, append ``runner.max_epochs=3 runner.val_check_interval=-1``. More MLP / SAC / GRPO recipes are in :doc:`MLP Policy RL <../examples/embodied/mlp>`.

   .. tab-item:: OpenVLA (7B VLA)
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

   .. tab-item:: OpenVLA-OFT (7B LoRA VLA)
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

The two VLA quickstarts use all available accelerators by default; edit ``cluster.component_placement`` to target specific devices (:doc:`../reference/configuration`). Rollout inherits ``actor.model.model_path`` automatically — to use local checkpoints instead of Hugging Face repo IDs, pass ``actor.model.model_path=/path/to/checkpoint`` (and for OpenVLA-OFT, ``actor.model.lora_path=/path/to/lora``).

Each config name maps directly to a YAML file on disk: embodied RL configs live in ``examples/embodiment/config/<config_name>.yaml`` (for example, ``examples/embodiment/config/maniskill_ppo_mlp.yaml`` and ``examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml``), SFT configs live in ``examples/sft/config/``, reasoning and agentic configs live in ``examples/reasoning/config/`` and ``examples/agent/``, and evaluation configs live in ``evaluations/<benchmark>/<config_name>.yaml``.

Results and Troubleshooting
---------------------------

Each run creates ``logs/<timestamp>-<config>/`` containing ``run_embodiment.log``, ``metrics.log``, TensorBoard events, and saved checkpoints. Open TensorBoard (``tensorboard --host 0.0.0.0 --logdir ./logs``) and track ``rollout/env_info/return`` and ``rollout/env_info/success_once``.

The 3-epoch MLP check (``runner.max_epochs=3 runner.val_check_interval=-1``) finishes in about 1 minute on a single GPU and prints a metric table at each step (also appended to ``metrics.log``):

.. code-block:: text

   ╭────────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step:    3/3 │ Progress: ████████████████████ │ 100.0%            │
   │ Elapsed: 00:52      │ ETA: 00:00                     │ Step Time: 17.3s  │
   ├─────────────────────────────── Rollout ──────────────────────────────────┤
   │ env_info/return=12.400  │ env_info/success_once=0.062 │ rewards=0.248    │
   ╰──────────────────────────────────────────────────────────────────────────╯

On full ``maniskill_ppo_mlp`` runs, ``rollout/env_info/success_once`` climbs above ``0.90`` within a few hundred epochs. On the VLA quickstarts, ``rollout/env_info/success_once`` starts near the SFT baseline and rises steadily as PPO updates progress:

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
