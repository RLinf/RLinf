IsaacLab RL
===========

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png
   :align: center
   :width: 90%

   IsaacLab (image: `IsaacLab <https://isaac-sim.github.io/IsaacLab/develop/source/overview/reinforcement-learning/rl_existing_scripts.html#rlinf>`__).

`IsaacLab <https://isaac-sim.github.io/IsaacLab/develop/source/overview/reinforcement-learning/rl_existing_scripts.html#rlinf>`__ is NVIDIA's GPU-accelerated robot
learning simulator. You'll use RLinf to PPO-fine-tune GR00T N1.5 or OpenPI π₀.₅ on a
custom Franka cube-stacking task.

Overview
--------

SFT then PPO-fine-tune a VLA on the IsaacLab Franka stack-cube task.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      GR00T N1.5 · π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      Franka stack-cube

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install → download Isaac Sim + an SFT model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Isaac Sim · an SFT checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0``
     - Stack the red block on the blue block, then stack the green block on the red block.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB from a third-person camera and a wrist camera (256×256 by default) plus robot proprioception.
   * - Action
     - 7-dim continuous action: 3D position (x, y, z) + 3D rotation (roll, pitch, yaw) + gripper.
   * - Reward
     - Sparse 0/1 success reward.
   * - Prompt
     - ``Stack the red block on the blue block, then stack the green block on the red block.``

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: GR00T N1.5
      :sync: gr00t

      **1. Install**

      Install RLinf with GR00T N1.5 and IsaacLab:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model gr00t --env isaaclab && source .venv/bin/activate

      Download Isaac Sim 5.1.0 and initialize its shell environment:

      .. code-block:: bash

         mkdir -p isaac_sim && cd isaac_sim
         wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
         unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
         rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
         source ./setup_conda_env.sh

      .. warning::

         Run ``source ./setup_conda_env.sh`` in every new terminal before launching IsaacLab.

      **2. Download**

      Download the GR00T N1.5 SFT checkpoint (trained on human demonstrations from |huggingface| `IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__):

      .. code-block:: bash

         hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir ./RLinf-Gr00t-SFT-Stack-cube

      .. include:: _model_path.rst

      **3. Launch**

      Update ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml`` and launch PPO training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

      The default config separates env, rollout, and actor placement; tune ``cluster.component_placement``, ``rollout.pipeline_stage_num``, and ``actor.enable_offload`` for your GPU memory budget. For standalone evaluation, pass ``isaaclab_franka_stack_cube_ppo_gr00t`` to the unified :doc:`Evaluation CLI <../../evaluations/reference/cli>`.

   .. tab-item:: OpenPI π₀.₅
      :sync: openpi

      **1. Install**

      Install RLinf with OpenPI and IsaacLab:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env isaaclab && source .venv/bin/activate

      Download Isaac Sim 5.1.0 and initialize its shell environment:

      .. code-block:: bash

         mkdir -p isaac_sim && cd isaac_sim
         wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
         unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
         rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
         source ./setup_conda_env.sh

      .. warning::

         Run ``source ./setup_conda_env.sh`` in every new terminal before launching IsaacLab.

      **2. Download**

      Download the OpenPI π₀.₅ SFT checkpoint (trained on human demonstrations from |huggingface| `IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__):

      .. code-block:: bash

         hf download YifWRobotics/RLinf-pi05-SFT-Stack-cube --local-dir ./RLinf-pi05-SFT-Stack-cube

      .. include:: _model_path.rst

      **3. Launch**

      Update ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml`` and launch PPO training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_openpi_pi05

      The default config collocates ``actor,env,rollout: all``; tune ``cluster.component_placement``, ``rollout.pipeline_stage_num``, and ``actor.enable_offload`` for your GPU memory budget. For standalone evaluation, pass ``isaaclab_franka_stack_cube_ppo_openpi_pi05`` to the unified :doc:`Evaluation CLI <../../evaluations/reference/cli>`.

.. note::

   To add a custom IsaacLab task, register the task following :doc:`../../extending/new_env`, then point ``init_params.id`` in an env config such as ``examples/embodiment/config/env/isaaclab_stack_cube.yaml`` at the new task id.

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Model Stage
     - Success Rate
   * - GR00T N1.5 base model (no SFT)
     - 0.000
   * - GR00T N1.5 SFT model
     - 0.654
   * - GR00T N1.5 RL-tuned model (SFT + RL)
     - 0.897
   * - OpenPI π₀.₅ SFT model
     - 0.859
   * - OpenPI π₀.₅ RL-tuned model (SFT + RL)
     - 0.953

.. note::

   **Acknowledgements:** Credit to `Minghui Xu <https://github.com/smallcracker>`__ and `Nan Yang <https://github.com/AquaSage18>`__ for the GR00T N1.5 example, and `Yifan Wu <https://github.com/YifWRobotics>`__ for the OpenPI π₀.₅ example.
