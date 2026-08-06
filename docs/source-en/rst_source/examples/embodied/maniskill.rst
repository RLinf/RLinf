RL with ManiSkill Benchmark
===========================

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
policies and reach state-of-the-art success rates, including on out-of-distribution
(OOD) variations.

Overview
--------

RL-finetune a VLA on ManiSkill3; OpenVLA and OpenVLA-OFT exceed 90% success on plate-25.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA · OpenVLA-OFT · π₀ / π₀.₅ · MLP · ResNet

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO · SAC · CrossQ · DAgger

   .. grid-item-card:: Tasks
      :text-align: center

      Tabletop manipulation (plate-25 + OOD)

   .. grid-item-card:: Hardware
      :text-align: center

      1–2 nodes · 8–16 GPUs

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

   To run ManiSkill with **OpenPI** (π\ :sub:`0`\  / π\ :sub:`0.5`\ ), see :doc:`RL on π₀ and π₀.₅ Models <pi0>`.

Installation
------------

.. include:: _setup_common.rst

**Option 1: Docker image** — image tag ``agentic-rlinf0.3-maniskill_libero``:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
      # Mainland China mirror: docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

   # Inside the container, switch to the model's virtual environment:
   source switch_env openvla        # or: source switch_env openvla-oft

**Option 2: Custom environment** — install bundle ``--env maniskill_libero``:

.. code:: bash

   # Add --use-mirror for faster downloads in mainland China.
   # Use --model openvla-oft for the OpenVLA-OFT experiments.
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

Download the Assets
-------------------

Download the ManiSkill assets:

.. code:: bash

   # Set HF_ENDPOINT=https://hf-mirror.com in mainland China.
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./maniskill_assets

.. important::

   The assets **must** be placed under ``rlinf/envs/maniskill/assets`` — this is where the env loads them from. Copy them into the env package directory:

.. code:: bash

   cp -r ./maniskill_assets <path_to_RLinf>/rlinf/envs/maniskill/assets

Download the Model
------------------

Download a pretrained base checkpoint (either method works):

.. code:: bash

   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup

   # Method 2: huggingface-hub (set HF_ENDPOINT=https://hf-mirror.com in mainland China)
   pip install huggingface-hub
   hf download gen-robot/openvla-7b-rlvla-warmup --local-dir openvla-7b-rlvla-warmup

.. include:: _model_path.rst

Run It
------

Each recipe is a YAML config under ``examples/embodiment/config/``:

- **OpenVLA + PPO** — ``maniskill_ppo_openvla.yaml``
- **OpenVLA-OFT + PPO** — ``maniskill_ppo_openvlaoft.yaml``
- **OpenVLA + GRPO** — ``maniskill_grpo_openvla.yaml``
- **OpenVLA-OFT + GRPO** — ``maniskill_grpo_openvlaoft.yaml``

Launch a config with ``run_embodiment.sh``:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla

**What this command does:**

1. Loads ``examples/embodiment/config/maniskill_ppo_openvla.yaml``.
2. Attaches to (or starts) Ray and places the actor, rollout, and env workers per ``cluster.component_placement``.
3. Runs the PPO training loop, writing logs and checkpoints under ``runner.logger.log_path``.

.. admonition:: Configure further
   :class: note

   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - All config keys → :doc:`Configuration <../../guides/index>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`
   - Resuming from a checkpoint → :doc:`Resume <../../guides/resume>`
   - Stuck or hitting OOM? → :doc:`FAQ <../../resources/faq>`

Visualization and Results
-------------------------

Launch TensorBoard to watch training live:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

The key signal to watch is **``env/success_once``** — the unnormalized episodic success
rate. For every logged metric, see :doc:`Training metrics <../../reference/metrics>`.

To save evaluation videos, enable them in the config:

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

ManiSkill3 Results
~~~~~~~~~~~~~~~~~~

Running on a single 8-GPU H100 machine, OpenVLA (left) and OpenVLA-OFT (right) achieve
over 90% success on ManiSkill3's plate-25-main task.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-vla/mani_openvla.png" style="width: 100%;"/>
       <p><em>OpenVLA</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-vla/mani_openvlaoft.png" style="width: 100%;"/>
       <p><em>OpenVLA-OFT</em></p>
     </div>
   </div>

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
