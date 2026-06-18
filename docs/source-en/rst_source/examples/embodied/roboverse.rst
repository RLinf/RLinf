RL with RoboVerse Benchmark
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://roboverseorg.github.io/static/images/teaser.jpg
   :align: center
   :width: 90%

   RoboVerse (image: `RoboVerse <https://roboverseorg.github.io/>`__).

`RoboVerse <https://roboverseorg.github.io/>`__ is a simulator suite for
robot manipulation tasks across multiple backends. You'll use RLinf to
PPO-fine-tune an OpenPI π₀.₅ policy on a RoboVerse kitchen manipulation task.

Overview
--------

Fine-tune OpenPI π₀.₅ on a RoboVerse task with two RGB views and sparse rewards.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      Bowl on cabinet

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4 GPUs

| **You'll do:** install → download resources + model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · RoboVerse resources · an SFT checkpoint.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Task
     - Description
   * - ``libero_90.kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet``
     - Put the black bowl on top of the cabinet in a kitchen scene.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Main camera RGB and wrist-camera RGB at 224×224 plus an 8-dim proprioceptive state.
   * - Action
     - 7-dim continuous action: 3D end-effector position, 3D rotation vector, and gripper.
   * - Reward
     - Sparse task-completion reward.
   * - Prompt
     - Natural-language instruction for the RoboVerse task.

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
      rlinf/rlinf:agentic-rlinf0.2-roboverse

   # For mainland China users:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-roboverse

Switch to the OpenPI virtual environment inside the image:

.. code:: bash

   source switch_env openpi

**Custom environment**

Install RoboVerse with the OpenPI dependencies:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --model openpi --env roboverse
   source .venv/bin/activate

Download the default RoboVerse resources:

.. code:: bash

   cd /path/to/RLinf
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset manity/roboverse_data --local-dir .

Download the Model
------------------

Download the OpenPI π₀.₅ checkpoint used by the reference config:

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT

.. include:: _model_path.rst

Run It
------

Launch the RoboVerse recipe:

.. list-table::
   :header-rows: 1
   :widths: 28 46 26

   * - Recipe
     - Config
     - Command suffix
   * - OpenPI π₀.₅ + PPO
     - ``examples/embodiment/config/roboverse_ppo_openpi_pi05.yaml``
     - ``roboverse_ppo_openpi_pi05``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh roboverse_ppo_openpi_pi05

What this does:

1. Starts the embodied training entrypoint with the RoboVerse Hydra config.
2. Creates Ray workers for the actor, rollout, and RoboVerse env components.
3. Runs PPO rollouts, computes sparse task rewards, and updates the OpenPI policy.

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with config fallback and the same suffix,
``roboverse_ppo_openpi_pi05``.

.. note::

   The default config places actor and rollout on GPUs ``0-1`` and env workers on
   GPUs ``2-3``. Tune ``cluster.component_placement``, ``env.train.total_num_envs``,
   and ``actor.global_batch_size`` for your hardware.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when you want rollout videos:

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

Enable W&B or SwanLab by adding logger backends:

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]  # or swanlab

.. note::

   This page does not publish a fixed RoboVerse success-rate table yet. Use
   ``env/success_once`` and evaluation videos to compare your runs.
