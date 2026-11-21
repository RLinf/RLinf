RL with IsaacLab
================

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

Installation
------------

.. include:: _setup_common.rst

Select ``--model`` for the model you want to run:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Model
     - ``--model``
   * - GR00T N1.5
     - ``gr00t``
   * - OpenPI π₀.₅
     - ``openpi``

.. code:: bash

   bash requirements/install.sh embodied --model <model> --env isaaclab && source .venv/bin/activate

Download Isaac Sim
~~~~~~~~~~~~~~~~~~

Download Isaac Sim 5.1.0 and initialize its shell environment:

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   Run ``source ./setup_conda_env.sh`` in every new terminal before launching IsaacLab.

Download the Model
------------------

Download the SFT checkpoint for the model you plan to fine-tune:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Hugging Face Repo
   * - GR00T N1.5
     - ``RLinf/RLinf-Gr00t-SFT-Stack-cube``
   * - OpenPI π₀.₅
     - ``YifWRobotics/RLinf-pi05-SFT-Stack-cube``

.. code-block:: bash

   hf download <repo> --local-dir <dir>

.. include:: _model_path.rst

The SFT checkpoints come from human demonstrations collected on the IsaacLab stack-cube
task. The dataset is available on |huggingface|
`IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__.

Run It
------

Available training configs:

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - Model
     - Config
     - Command suffix
   * - GR00T N1.5
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml``
     - ``isaaclab_franka_stack_cube_ppo_gr00t``
   * - OpenPI π₀.₅
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml``
     - ``isaaclab_franka_stack_cube_ppo_openpi_pi05``

Launch training with ``run_embodiment.sh``:

.. code:: bash

   # GR00T N1.5
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

   # OpenPI π₀.₅
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_openpi_pi05

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with config fallback and the same suffixes:
``isaaclab_franka_stack_cube_ppo_gr00t`` and
``isaaclab_franka_stack_cube_ppo_openpi_pi05``.

.. note::

   For GR00T, the default config separates env, rollout, and actor placement. For OpenPI,
   the default config collocates ``actor,env,rollout: all``. Tune
   ``cluster.component_placement``, ``rollout.pipeline_stage_num``, and
   ``actor.enable_offload`` for your GPU memory budget.

.. note::

   To add a custom IsaacLab task, implement it under
   ``rlinf/envs/sim/isaaclab/tasks/``, register it in ``rlinf/envs/sim/isaaclab/__init__.py``,
   then point ``init_params.id`` in an env config such as
   ``examples/embodiment/config/env/isaaclab_stack_cube.yaml`` at the new task id.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when you want rollout videos:

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

Enable W&B or SwanLab by adding logger backends:

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]  # or swanlab

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

Acknowledgements
----------------

Credit to `Minghui Xu <https://github.com/smallcracker>`__ and
`Nan Yang <https://github.com/AquaSage18>`__ for the GR00T N1.5 example, and
`Yifan Wu <https://github.com/YifWRobotics>`__ for the OpenPI π₀.₅ example.
