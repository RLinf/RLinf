EmbodiChain RL
==============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
   :align: center
   :width: 90%

   EmbodiChain (image: `EmbodiChain <https://github.com/DexForce/EmbodiChain>`__).

`EmbodiChain <https://github.com/DexForce/EmbodiChain>`__ is an embodied
intelligence lab stack that exposes Gym-style RL tasks. You'll use RLinf to train
an MLP actor-critic with PPO on the EmbodiChain CartPole task.

Overview
--------

Train a state-based MLP policy on EmbodiChain CartPole.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      CartPole

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4 GPUs

| **You'll do:** install → launch ``run_embodiment.sh`` → watch rollout rewards.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · EmbodiChain package and task resources.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - CartPole
     - Balance the pole with state observations from ``embodichain_tasks/configs/tasks/classic_control/cart_pole/env.json``.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - A single ``states`` tensor built from ``state_keys: ["qpos", "qvel", "qf"]``.
   * - Action
     - 2-dim continuous action for ``policy_setup: cartpole-delta-qpos``.
   * - Reward
     - Task reward from the EmbodiChain Gym config.
   * - Prompt
     - Not used; this is a low-dimensional state-control recipe.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: MLP

      **1. Install**

      Install EmbodiChain 0.3.0 and its dependencies. The installer gets ``dexsim-engine==0.5.0`` from ``http://pyp.open3dv.site:2345/simple/dexsim-engine/`` and resolves the remaining packages through your configured Python package index (your machine must be able to reach port 2345).

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env embodichain && source .venv/bin/activate

      Use the installed package configs by default. To point at a local EmbodiChain checkout, set ``export EMBODICHAIN_PATH=/path/to/EmbodiChain``.

      **2. Download (Optional)**

      No checkpoint is required; the MLP policy starts from scratch. If a run fails because task resources are missing, download them in the same Python environment:

      .. code-block:: bash

         export EMBODICHAIN_DATA_ROOT="$PWD/data"
         python -m embodichain.data download --name CartPole
         python -m embodichain.data download --name SimResources

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 28 46 26

         * - Recipe
           - Config
           - Command suffix
         * - MLP + PPO
           - ``examples/embodiment/config/embodichain_ppo_cart_pole.yaml``
           - ``embodichain_ppo_cart_pole``

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh embodichain_ppo_cart_pole

.. warning::

   EmbodiChain's ``dexsim`` dependency needs ``libpython3.xx.so``. If you hit
   ``libpython3.11.so`` runtime errors with UV's Python layout, use a Conda
   environment and rerun ``bash requirements/install.sh embodied --env embodichain --no-root``.

.. note::

   Keep ``actor.model.obs_dim``, ``actor.model.action_dim``, and
   ``actor.model.policy_setup`` aligned with the EmbodiChain task config when you
   adapt this recipe to another task.

Visualization and Results
-------------------------

Monitor rollout rewards with ``tensorboard --logdir ./logs --port 6006`` (the default config logs to W&B; pass ``runner.logger.logger_backends='["tensorboard"]'`` to use TensorBoard), enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

.. dropdown:: Evaluation and CI

   EmbodiChain CartPole is also covered by embodied e2e configs under
   ``tests/e2e_tests/embodied/``. Set ``EMBODICHAIN_PATH`` only when you need a
   non-default checkout.
