RL with Genesis Benchmark
=========================

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
      rlinf/rlinf:agentic-rlinf0.3-genesis

   # For mainland China users:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-genesis

**Custom environment**

Install Genesis dependencies:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --env genesis
   source .venv/bin/activate

Download the Model
------------------

Skip this section for the MLP + PPO recipe. For the CNN + PPO recipe, download the
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
``examples/embodiment/config/genesis_cubepick_ppo_cnn.yaml``:

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

.. code:: bash

   bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_mlp
   bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_cnn

What this does:

1. Starts the embodied training entrypoint with the selected Hydra config.
2. Creates Ray workers for the actor, rollout, and Genesis env components.
3. Runs PPO rollouts, computes cube-pick rewards, and updates the selected policy.

.. note::

   Both configs run on GPU ``0`` by default. Tune ``cluster.component_placement``,
   ``env.train.total_num_envs``, and batch sizes for your hardware.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when needed:

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
   * - MLP + PPO
     - With the default ``genesis_cubepick_ppo_mlp`` parameters, ``env/success_once`` reaches about 80%.
