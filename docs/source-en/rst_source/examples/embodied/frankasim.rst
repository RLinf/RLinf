RL with Franka-Sim Benchmark
============================

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

Installation
------------

.. include:: _setup_common.rst

**Docker image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-frankasim

   # For mainland China users:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-frankasim

Switch to the virtual environment inside the image:

.. code:: bash

   source switch_env openvla

**Custom environment**

Install Franka-Sim dependencies:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --model openvla --env frankasim
   source .venv/bin/activate

Download the Model
------------------

Skip this section for the MLP + PPO recipe. For the CNN + SAC recipe, download the
ResNet checkpoint:

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

Then set the same checkpoint path for rollout and actor in
``examples/embodiment/config/frankasim_sac_cnn_async.yaml``:

.. code-block:: yaml

   rollout:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained
   actor:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained

Run It
------

Pick one recipe and launch training:

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

.. code:: bash

   # State-observation PPO recipe
   bash examples/embodiment/run_embodiment.sh frankasim_ppo_mlp

   # Vision SAC recipe
   bash examples/embodiment/run_async.sh frankasim_sac_cnn_async

What this does:

1. Starts the selected embodied training entrypoint.
2. Creates Ray workers for the actor, rollout, and Franka-Sim env components.
3. Runs rollouts, computes task rewards, and updates the selected policy.

.. note::

   Both reference configs run on GPU ``0``. Tune ``cluster.component_placement``,
   ``env.train.total_num_envs``, and batch sizes if you move to a larger machine.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video when you want rollout videos:

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

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
