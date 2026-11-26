Evaluation on Isaac Lab-Arena
=============================

Evaluate OpenPI π₀.₅ on a simulated DROID arm and measure how often it places a banana on a plate.

.. figure:: https://media.githubusercontent.com/media/isaac-sim/IsaacLab-Arena/a72262a0f1896f56bb3a1fdf329a1548ba55aad5/docs/images/openpi_droid_get_started.png
   :width: 80%
   :align: center
   :alt: DROID manipulation in Isaac Lab-Arena

   DROID scene from `NVIDIA Isaac Lab-Arena <https://github.com/isaac-sim/IsaacLab-Arena>`__.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀.₅ (DROID)

   .. grid-item-card:: Algorithms
      :text-align: center

      Policy evaluation and :ref:`single-action PPO <isaaclab-arena-single-action-ppo>`

   .. grid-item-card:: Tasks
      :text-align: center

      Tabletop pick and place

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA RTX

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Arena task
     - What the arm does
   * - Banana on plate (default graph)
     - Pick up the banana and place it on the plate.
   * - Registered maple-table task
     - Move a selected object to the destination on the table.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Specification
   * - Observation
     - Exterior and wrist RGB images, seven arm joint positions and gripper closure.
   * - Action
     - Seven absolute joint targets in radians and one gripper command. The default evaluation predicts 15 actions per inference.
   * - Reward
     - Arena task rewards, or a binary success reward for tasks without reward terms.
   * - Prompt
     - Natural-language instruction from the Arena task.

Setup and Run
-------------

Use an x86-64 Linux host with an RTX GPU, a driver compatible with Isaac Sim 6.2, and network access to NVIDIA's simulation assets. Check the `Isaac Sim requirements <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html>`__ before installing.

The OpenPI controller uses PyTorch 2.7.1 with CUDA 12.8, including kernels for Blackwell RTX GPUs. The installer creates a separate Python 3.12 environment for Arena. To reuse an existing installation of the pinned Arena revision (``a72262a0f1``), export ``ISAACLAB_ARENA_PYTHON`` with its absolute interpreter path before installing and keep it set when launching RLinf.

From your RLinf checkout, follow the steps below. See :doc:`the installation guide <../../start/installation>` for host setup and package mirrors.

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ (DROID)

      **1. Install the controller and simulator**

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env isaaclab_arena
         source .venv/bin/activate

      **2. Download the DROID checkpoint and normalization statistics**

      .. code-block:: bash

         export MODEL_DIR="$PWD/checkpoints/RLinf-Pi05-RoboLab-droid-jointpos"
         hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
             --local-dir "$MODEL_DIR"

      **3. Run the evaluation**

      After accepting the Isaac Sim license, launch the banana-on-plate task.

      .. code-block:: bash

         export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
         bash evaluations/run_eval.sh isaaclab_arena_pi05_eval \
             rollout.model.model_path="$MODEL_DIR"

Visualization and Results
-------------------------

The default run evaluates four episodes in parallel. The terminal and TensorBoard report ``eval/success_once``, the fraction of episodes that reached the goal. See :doc:`the evaluation guide <../../evaluations/guides/isaaclab_arena>` to select another task, check a short rollout or record the cameras.
