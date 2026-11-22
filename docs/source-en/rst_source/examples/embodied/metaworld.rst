MetaWorld RL
============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png
   :align: center
   :width: 90%

   The Meta-World benchmark (image: `Meta-World <https://metaworld.farama.org>`__).

`Meta-World <https://metaworld.farama.org>`__ is a multi-task manipulation benchmark on
MuJoCo: a 7-DoF arm performs 50 diverse tabletop tasks. RLinf uses it to RL-fine-tune
vision-language-action (VLA) policies, including held-out (OOD) generalization.

Overview
--------

RL-finetune a VLA across Meta-World's 50 tasks; pi0 + PPO reaches ~78% average success.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      MT50 · ML45 (5 OOD)

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install deps → download the SFT model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · an SFT checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Suite
     - Tasks
     - Setting
   * - MT50
     - 50
     - Multi-task training and evaluation across all 50 tasks.
   * - ML45
     - 45 + 5
     - Train on 45 tasks; evaluate on 5 held-out (OOD) tasks.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB (480×480) from off-screen cameras around the workspace.
   * - Action
     - 4-dim continuous: 3D end-effector position (x, y, z) + gripper open/close.
   * - Reward
     - Sparse — based on task completion.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀
      :sync: pi0

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env metaworld && source .venv/bin/activate

      **2. Download**

      Download the OpenPI π₀ SFT checkpoint (also available on `ModelScope <https://www.modelscope.cn/models/RLinf/RLinf-Pi0-MetaWorld>`__):

      .. code-block:: bash

         hf download RLinf/RLinf-Pi0-MetaWorld-SFT --local-dir ./RLinf-Pi0-MetaWorld-SFT

      **3. Launch**

      Available configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 25 25 50

         * - Setting
           - Algorithm
           - Config
         * - MT50
           - PPO
           - ``metaworld_50_ppo_openpi.yaml``
         * - ML45
           - PPO
           - ``metaworld_45_ppo_openpi.yaml``

      Launch training with ``run_embodiment.sh``, passing ``actor.model.model_path`` on the command line (rollout inherits it automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi \
           actor.model.model_path=./RLinf-Pi0-MetaWorld-SFT

   .. tab-item:: OpenPI π₀.₅
      :sync: pi05

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env metaworld && source .venv/bin/activate

      **2. Download**

      Download the OpenPI π₀.₅ SFT checkpoint:

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-MetaWorld-SFT --local-dir ./RLinf-Pi05-MetaWorld-SFT

      **3. Launch**

      Available config under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 25 25 50

         * - Setting
           - Algorithm
           - Config
         * - MT50
           - PPO
           - ``metaworld_50_ppo_openpi_pi05.yaml``

      Launch training with ``run_embodiment.sh``, passing ``actor.model.model_path`` on the command line (rollout inherits it automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi_pi05 \
           actor.model.model_path=./RLinf-Pi05-MetaWorld-SFT

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env metaworld && source .venv/bin/activate

      **2. Download**

      Download the OpenVLA-OFT SFT checkpoint:

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-MetaWorld-SFT --local-dir ./RLinf-OpenVLAOFT-MetaWorld-SFT

      **3. Launch**

      Available config under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 25 25 50

         * - Setting
           - Algorithm
           - Config
         * - MT50
           - GRPO
           - ``metaworld_50_grpo_openvlaoft.yaml``

      Launch training with ``run_embodiment.sh``, passing ``actor.model.model_path`` on the command line (rollout inherits it automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh metaworld_50_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-MetaWorld-SFT

.. admonition:: Configure further
   :class: note

   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - All config keys → :doc:`Configuration <../../guides/index>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`
   - Resuming from a checkpoint → :doc:`Resume <../../guides/resume>`

Visualization and Results
-------------------------

Watch ``env/success_once`` in TensorBoard (``tensorboard --logdir ./logs``); set ``env.eval.video_cfg.save_video=True`` to record videos, or see :doc:`Training metrics <../../reference/metrics>` for W&B/SwanLab and all metrics.

MetaWorld Results
~~~~~~~~~~~~~~~~~

The results for Diffusion Policy, TinyVLA, and SmolVLA in the table below are referenced from the `SmolVLA paper <https://arxiv.org/abs/2403.04880>`_. The SFT results for π\ :sub:`0`\  and π\ :sub:`0.5`\  are obtained by retraining using the official `dataset <https://huggingface.co/datasets/lerobot/metaworld_mt50>`_ provided by LeRobot.

.. list-table:: **MetaWorld-MT50 Performance Comparison (Success Rate, %)**
   :widths: 15 10 10 10 10 10
   :header-rows: 1

   * - **Methods**
     - **Easy**
     - **Medium**
     - **Hard**
     - **Very Hard**
     - **Avg.**
   * - Diffusion Policy
     - 23.1
     - 10.7
     - 1.9
     - 6.1
     - 10.5
   * - TinyVLA
     - 77.6
     - 21.5
     - 11.4
     - 15.8
     - 31.6
   * - SmolVLA
     - 87.1
     - 51.8
     - 70.0
     - 64.0
     - 68.2
   * - π\ :sub:`0`\
     - 77.9
     - 51.8
     - 53.3
     - 20.0
     - 50.8
   * - π\ :sub:`0`\  + PPO
     - **92.1**
     - **74.6**
     - 61.7
     - **84.0**
     - **78.1**
   * - π\ :sub:`0.5`\
     - 68.2
     - 37.3
     - 41.7
     - 28.0
     - 43.8
   * - π\ :sub:`0.5`\  + PPO
     - 86.4
     - 55.5
     - **75.0**
     - 66.0
     - 70.7
