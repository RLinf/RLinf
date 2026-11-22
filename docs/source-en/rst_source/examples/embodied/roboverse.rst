RoboVerse RL
============

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

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀.₅
      :sync: openpi-roboverse

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env roboverse && source .venv/bin/activate

      **2. Download**

      Download the default RoboVerse resources into the RLinf repo root and the OpenPI π₀.₅ SFT checkpoint:

      .. code-block:: bash

         # Download RoboVerse resources
         hf download manity/roboverse_data --repo-type dataset --local-dir ./roboverse_data

         # Download the OpenPI π₀.₅ checkpoint
         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 28 46 26

         * - Recipe
           - Config
           - Command suffix
         * - OpenPI π₀.₅ + PPO
           - ``examples/embodiment/config/roboverse_ppo_openpi_pi05.yaml``
           - ``roboverse_ppo_openpi_pi05``

      .. include:: _model_path.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh roboverse_ppo_openpi_pi05

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with config fallback and the same suffix,
``roboverse_ppo_openpi_pi05``.

.. note::

   The default config places actor and rollout on GPUs ``0-1`` and env workers on
   GPUs ``2-3``. Tune ``cluster.component_placement``, ``env.train.total_num_envs``,
   and ``actor.global_batch_size`` for your hardware.

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

.. note::

   This page does not publish a fixed RoboVerse success-rate table yet. Use
   ``env/success_once`` and evaluation videos to compare your runs.
