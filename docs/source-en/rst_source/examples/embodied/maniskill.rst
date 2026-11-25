ManiSkill RL
============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/mani-skill/ManiSkill/main/figures/teaser.jpg
   :align: center
   :width: 90%

   Environments rendered in ManiSkill (image: `ManiSkill <https://github.com/haosulab/ManiSkill>`__).

`ManiSkill <https://maniskill.readthedocs.io>`__ is a GPU-parallelized robotics
simulator and benchmark for manipulation. A 7-DoF arm performs language-conditioned
tabletop tasks; RLinf uses ManiSkill3 to RL-fine-tune vision-language-action (VLA)
policies to high success rates, including on out-of-distribution (OOD) variations.

Overview
--------

RL-finetune a VLA on ManiSkill3; OpenVLA and OpenVLA-OFT exceed 90% success on plate-25.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA · OpenVLA-OFT · GR00T N1.5 · π₀ / π₀.₅ · MLP · ResNet

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO · SAC · CrossQ · DAgger

   .. grid-item-card:: Tasks
      :text-align: center

      Tabletop manipulation (plate-25 + OOD)

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA · AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA (model-dependent)

| **You'll do:** install deps → download assets + base model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the ManiSkill assets and a base checkpoint (steps below).

Tasks
~~~~~

The reference recipe trains on the ``PutOnPlateInScene25Main-v3`` (plate-25) task and
evaluates both in-distribution (IND) and on out-of-distribution (OOD) settings:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Setting
     - What it tests
   * - Training (IND)
     - The plate-25 training task.
   * - Vision (OOD)
     - Visual variations of the scene.
   * - Semantic (OOD)
     - Semantic variations (objects, instructions).
   * - Execution (OOD)
     - Execution-time variations.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB from a third-person camera (224×224); language task description.
   * - Action
     - 7-dim continuous: 3D end-effector position, 3D rotation, and 1-D gripper open/close.
   * - Reward
     - Step-level reward based on task progress and success.
   * - Task prompt
     - ``In: What action should the robot take to [task_description]? Out:``

The walkthrough below uses **OpenVLA / OpenVLA-OFT** with **PPO/GRPO**; switch the config to use another supported model.

.. seealso::

   Backend-specific ManiSkill setup is documented with each model: :ref:`OpenVLA-OFT <openvla-oft-hardware>`, :ref:`GR00T N1.5 <gr00t-hardware>`, and :ref:`OpenPI π₀ / π₀.₅ <pi0-hardware>`. AMD, Ascend, and MUSA use CPU PhysX simulation while the model remains on its selected accelerator. GR00T requires a compatible checkpoint with a ``maniskill_widowx`` embodiment head, and MUSA requires the vendor simulator packages.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenVLA
      :sync: openvla

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the ManiSkill scene assets and base checkpoint from the repository root:

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         hf download gen-robot/openvla-7b-rlvla-warmup --local-dir ./openvla-7b-rlvla-warmup

      **3. Launch**

      Available configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 25 75

         * - Algorithm
           - Config
         * - PPO
           - ``maniskill_ppo_openvla.yaml``
         * - GRPO
           - ``maniskill_grpo_openvla.yaml``

      Launch training with ``run_embodiment.sh``, passing ``actor.model.model_path`` on the command line (rollout inherits it automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla \
           actor.model.model_path=./openvla-7b-rlvla-warmup

      .. dropdown:: Advanced configuration (optional)

         When PPO rollout spends time running terminal value bootstrap one environment slice at a time or decoding unused action tokens on OpenVLA, enable batched and value-only bootstrap in the config:

         .. code-block:: yaml

            rollout:
              batch_bootstrap_inference: True
              value_only_bootstrap: True

         ``rollout.batch_bootstrap_inference`` merges pending terminal observations into one rollout batch before splitting values back to their routes. ``rollout.value_only_bootstrap`` estimates values directly from the prompt hidden state instead of autoregressively generating actions when a value head is present (such as OpenVLA PPO). Both options default to ``False``.

      **4. Training Curve**

      Running on a single 8-GPU H100 machine, OpenVLA exceeds 90% success on ManiSkill3's plate-25-main task:

      .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-vla/mani_openvla.png
         :align: center
         :width: 80%

         OpenVLA training curve on ManiSkill3.

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the ManiSkill scene assets, base checkpoint, and LoRA weights from the repository root:

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir ./RLinf-OpenVLAOFT-ManiSkill-Base-Main
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir ./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      **3. Launch**

      Available configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 25 75

         * - Algorithm
           - Config
         * - PPO
           - ``maniskill_ppo_openvlaoft.yaml``
         * - GRPO
           - ``maniskill_grpo_openvlaoft.yaml``

      Launch training with ``run_embodiment.sh``, passing ``actor.model.model_path`` and ``actor.model.lora_path`` on the command line (rollout inherits ``model_path`` automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-ManiSkill-Base-Main \
           actor.model.lora_path=./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      .. dropdown:: Advanced configuration (optional)

         When PPO rollout spends time running terminal value bootstrap one environment slice at a time, enable batched bootstrap in the config:

         .. code-block:: yaml

            rollout:
              batch_bootstrap_inference: True

         ``rollout.batch_bootstrap_inference`` merges pending terminal observations into one rollout batch before splitting values back to their routes (defaults to ``False``).

      **4. Training Curve**

      Running on a single 8-GPU H100 machine, OpenVLA-OFT exceeds 90% success on ManiSkill3's plate-25-main task:

      .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-vla/mani_openvlaoft.png
         :align: center
         :width: 80%

         OpenVLA-OFT training curve on ManiSkill3.

.. admonition:: Configure further
   :class: note

   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - All config keys → :doc:`Configuration <../../guides/index>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`
   - Resuming from a checkpoint → :doc:`Resume <../../guides/resume>`
   - Stuck or hitting OOM? → :doc:`FAQ <../../resources/faq>`

Visualization and Results
-------------------------

Watch ``env/success_once`` in TensorBoard (``tensorboard --logdir ./logs``); set ``env.eval.video_cfg.save_video=True`` to record videos, or see :doc:`Training metrics <../../reference/metrics>` for W&B/SwanLab and all metrics.

ManiSkill3 Results
~~~~~~~~~~~~~~~~~~

We evaluate on both in-distribution (IND) and OOD scenarios (Vision, Semantic, Execution).
The best result per column is in bold.

.. note::

   The same OOD test set as `rl4vla <https://arxiv.org/abs/2505.19789>`_ is used for a fair
   comparison. Base models: OpenVLA uses the pretrained
   `openvla-7b-rlvla-warmup <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_;
   OpenVLA-OFT uses our own LoRA fine-tune on ``PutOnPlateInScene25Main-v3`` data
   (`OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_).

.. list-table:: **OpenVLA and OpenVLA-OFT results on ManiSkill3**
   :header-rows: 1
   :widths: 40 15 15 15 15 15

   * - Model
     - Training Setting(IND)
     - Vision (OOD)
     - Semantic (OOD)
     - Execution (OOD)
     - Average of OOD
   * - |huggingface| `OpenVLA (Base) <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_
     - 53.91%
     - 38.75%
     - 35.75%
     - 42.11%
     - 39.10%
   * - |huggingface| `RL4VLA (PPO) <https://huggingface.co/gen-robot/openvla-7b-rlvla-rl>`_
     - 93.75%
     - 80.47%
     - 75.00%
     - 81.77%
     - 79.15%
   * - |huggingface| `PPO-OpenVLA <https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood>`_
     - 96.09%
     - 82.03%
     - **78.35%**
     - **85.42%**
     - **81.93%**
   * - |huggingface| `GRPO-OpenVLA <https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood>`_
     - 84.38%
     - 74.69%
     - 72.99%
     - 77.86%
     - 75.15%
   * - |huggingface| `OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_
     - 28.13%
     - 27.73%
     - 12.95%
     - 11.72%
     - 18.29%
   * - |huggingface| `PPO-OpenVLA-OFT <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood>`_
     - **97.66%**
     - **92.11%**
     - 64.84%
     - 73.57%
     - 77.05%
   * - |huggingface| `GRPO-OpenVLA-OFT <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood>`_
     - 94.14%
     - 84.69%
     - 45.54%
     - 44.66%
     - 60.64%

.. note::

   The ``rl4vla`` model is PPO + OpenVLA under a **small batch size**, so it should be
   compared only with our PPO+OpenVLA trained under similar conditions. Our PPO+OpenVLA
   uses RLinf's large-scale infrastructure to train with **larger batch sizes**, which we
   found significantly improves performance.

The animation below shows OpenVLA trained on ManiSkill3's multi-task benchmark with PPO in RLinf.

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/embody.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
