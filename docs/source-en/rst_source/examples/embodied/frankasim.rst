Franka-Sim RL
=============

.. figure:: https://raw.githubusercontent.com/RLinf/serl/refs/heads/RLinf/franka-sim/franka_sim/franka_sim/envs/xmls/robotiq_2f85/2f85.png
   :align: center
   :width: 70%

   Franka-Sim assets from the RLinf SERL fork.

Franka-Sim is a lightweight Franka Panda simulation environment built from the
`SERL <https://rail-berkeley.github.io/serl/docs/sim_quick_start.html>`__ stack.
You'll use RLinf to train either an MLP policy with PPO on state observations or a
CNN policy with asynchronous SAC on RGB observations.

Overview
--------

Train a Franka pick-cube policy with either state-only or vision observations.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP · CNN

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · SAC

   .. grid-item-card:: Tasks
      :text-align: center

      PickCube state · vision

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1 GPU

| **You'll do:** install → optionally download ResNet → launch training → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Franka-Sim assets from the install step.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``PandaPickCube-v0``
     - State-observation pick-cube task for the MLP + PPO recipe.
   * - ``PandaPickCubeVision-v0``
     - RGB-observation pick-cube task for the CNN + asynchronous SAC recipe.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - ``PandaPickCube-v0`` uses proprioceptive state and target position; ``PandaPickCubeVision-v0`` uses RGB images plus state.
   * - Action
     - 4-dim continuous action: 3D end-effector position delta plus gripper control.
   * - Reward
     - Dense task-progress reward.
   * - Prompt
     - Not used by the state MLP recipe; vision policies consume task-conditioned observations from the env wrapper.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: MLP / CNN

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env frankasim && source .venv/bin/activate

      **2. Download**

      Skip this step for the MLP + PPO recipe. For the CNN + SAC recipe, download the pretrained ResNet-10 checkpoint:

      .. code-block:: bash

         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./RLinf-ResNet10-pretrained

      **3. Launch**

      Available recipes:

      .. list-table::
         :header-rows: 1
         :widths: 24 38 20 18

         * - Recipe
           - Config
           - Entrypoint
           - Command suffix
         * - MLP + PPO
           - ``examples/embodiment/config/frankasim_ppo_mlp.yaml``
           - ``run_embodiment.sh``
           - ``frankasim_ppo_mlp``
         * - CNN + SAC
           - ``examples/embodiment/config/frankasim_sac_cnn_async.yaml``
           - ``run_async.sh``
           - ``frankasim_sac_cnn_async``

      .. include:: _model_path.rst

      Launch training with the matching script:

      .. code-block:: bash

         # State-observation MLP + PPO
         bash examples/embodiment/run_embodiment.sh frankasim_ppo_mlp

         # Vision CNN + asynchronous SAC
         bash examples/embodiment/run_async.sh frankasim_sac_cnn_async actor.model.model_path=$PWD/RLinf-ResNet10-pretrained

      .. dropdown:: Advanced configuration (optional)

         Both reference configs run on GPU ``0`` by default. Tune ``cluster.component_placement``, ``env.train.total_num_envs``, and batch sizes when scaling to a larger machine.

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Recipe
     - Reported Behavior
   * - CNN + asynchronous SAC
     - Learns a stable grasping strategy within about one hour in the simulation setup used for the original run.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/frankasim_curve.png
   :align: center
   :width: 90%

   Franka-Sim asynchronous SAC + CNN success-rate curve.
