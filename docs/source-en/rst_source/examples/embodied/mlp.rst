MLP Policy RL
=============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/3_layer_mlp.jpg
   :align: center
   :width: 55%

   A multi-layer perceptron policy.

An **MLP policy** is a lightweight network for robotics control from **low-dimensional
state inputs** (joint angles, end-effector pose, object states). RLinf trains MLP policies
with PPO, SAC, and GRPO across several simulators — useful for quickly validating
environments, training pipelines, and network architectures.

Overview
--------

Train an MLP policy with PPO/SAC/GRPO on low-dimensional state across ManiSkill, LIBERO-Spatial, and FrankaSim.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      ManiSkill · LIBERO · FrankaSim

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · SAC · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      PickCube · LIBERO-Spatial

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · GPUs

| **You'll do:** install the target simulator → pick a config → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the target simulator's dependencies.

Tasks
~~~~~

Select an environment via the ``defaults`` list (``env/<env_name>@env.train`` / ``@env.eval``); override parallel-env count, episode length, and recording under ``env.train`` / ``env.eval``.

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - ManiSkill3
     - PickCube
     - ``maniskill_ppo_mlp`` / ``maniskill_sac_mlp``
     - Low-dimensional state policy training.
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_0_grpo_mlp``
     - GRPO with an MLP policy on a LIBERO spatial task.
   * - MuJoCo / FrankaSim
     - PickCube
     - ``frankasim_ppo_mlp``
     - FrankaSim state-based PPO training.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - Low-dimensional state vectors such as robot joints, end-effector pose, and object states.
   * - Action
     - Continuous robot control commands configured by ``policy_setup``, ``obs_dim``, and ``action_dim``.
   * - Reward
     - Simulator task reward or success signal.
   * - Prompt
     - Not used by the MLP policy; tasks are selected through Hydra configs.

Setup and Run
-------------

.. include:: _setup_common.rst

No pretrained checkpoint is needed; the MLP policy trains from scratch.

.. tab-set::

   .. tab-item:: ManiSkill / LIBERO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate

      **2. Launch**

      Default configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 30 34 36

         * - Recipe
           - Config
           - Launch command suffix
         * - ManiSkill + PPO + MLP
           - ``maniskill_ppo_mlp.yaml``
           - ``maniskill_ppo_mlp``
         * - ManiSkill + SAC + MLP
           - ``maniskill_sac_mlp.yaml``
           - ``maniskill_sac_mlp``
         * - LIBERO-Spatial + GRPO + MLP
           - ``libero_spatial_0_grpo_mlp.yaml``
           - ``libero_spatial_0_grpo_mlp``

      .. code-block:: bash

         # ManiSkill (PPO-MLP)
         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

         # ManiSkill (SAC-MLP)
         bash examples/embodiment/run_embodiment.sh maniskill_sac_mlp

         # LIBERO-Spatial (GRPO-MLP)
         bash examples/embodiment/run_embodiment.sh libero_spatial_0_grpo_mlp

   .. tab-item:: Franka-Sim

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env frankasim && source .venv/bin/activate

      **2. Launch**

      Default config under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 30 34 36

         * - Recipe
           - Config
           - Launch command suffix
         * - Franka-Sim + PPO + MLP
           - ``frankasim_ppo_mlp.yaml``
           - ``frankasim_ppo_mlp``

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh frankasim_ppo_mlp

.. note::

   **SAC specifics.** SAC learns Q-values via Bellman backups with entropy
   regularization (off-policy), so enable the Q-related heads in the config
   (``add_q_head: True``). It also supports **automatic entropy tuning** via
   ``entropy_tuning`` (e.g. ``alpha_type: softplus``) to balance exploration and
   exploitation.

.. dropdown:: Advanced configuration (optional)

   The MLP model is introduced via ``model/mlp_policy@actor.model`` and can be overridden per recipe. Key default fields:

   .. code-block:: yaml

      model_type: "mlp_policy"
      model_path: ""
      policy_setup: "panda-qpos"
      obs_dim: 42
      action_dim: 8
      num_action_chunks: 1
      hidden_dim: 256
      precision: "32"
      add_value_head: True
      is_lora: False
      lora_rank: 32

   For real-robot multi-node deployment (Actor/Policy on GPU servers, Env/Robot on a control PC), see :doc:`franka`.

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.
