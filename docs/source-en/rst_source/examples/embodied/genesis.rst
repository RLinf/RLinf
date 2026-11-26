Genesis RL
==========

.. figure:: https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/HeroShot_Final.png
   :align: center
   :width: 90%

   Genesis (image: `Genesis <https://genesis-world.readthedocs.io/>`__).

`Genesis <https://genesis-world.readthedocs.io/>`__ is a GPU-accelerated
multi-physics simulator for robotics. You'll use RLinf to train MLP or CNN
policies with PPO on a Franka cube-pick task.

Overview
--------

Train a Franka Panda policy to pick up a cube in Genesis.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP · CNN

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      CubePick

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1 GPU

| **You'll do:** install → optionally download ResNet → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Genesis dependencies from the install step.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``cube_pick``
     - Control a Franka Panda arm to grasp and lift a cube.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - 16-dim state for MLP; 256×256 RGB plus 16-dim state for CNN.
   * - Action
     - 9-dim continuous action: 7 Franka arm joint positions plus 2 gripper positions.
   * - Reward
     - Dense approach reward plus grasp-success bonus.
   * - Prompt
     - Not used; this is a low-dimensional/CNN policy-control recipe.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: MLP / CNN

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env genesis && source .venv/bin/activate

      **2. Download**

      Skip this step for the MLP + PPO recipe. For the CNN + PPO recipe, download the pretrained ResNet-10 checkpoint:

      .. code-block:: bash

         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./RLinf-ResNet10-pretrained

      **3. Launch**

      Available recipes:

      .. list-table::
         :header-rows: 1
         :widths: 26 48 26

         * - Recipe
           - Config
           - Command suffix
         * - MLP + PPO
           - ``examples/embodiment/config/genesis_cubepick_ppo_mlp.yaml``
           - ``genesis_cubepick_ppo_mlp``
         * - CNN + PPO
           - ``examples/embodiment/config/genesis_cubepick_ppo_cnn.yaml``
           - ``genesis_cubepick_ppo_cnn``

      .. include:: _model_path.rst

      Launch training with ``run_embodiment.sh``:

      .. code-block:: bash

         # State-observation MLP + PPO
         bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_mlp

         # Vision CNN + PPO
         bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_cnn actor.model.model_path=$PWD/RLinf-ResNet10-pretrained

      .. dropdown:: Advanced configuration (optional)

         Both configs run on GPU ``0`` by default. Tune ``cluster.component_placement``, ``env.train.total_num_envs``, and batch sizes for your hardware.

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Recipe
     - Reported Behavior
   * - MLP + PPO
     - With the default ``genesis_cubepick_ppo_mlp`` parameters, ``env/success_once`` reaches about 80%.
