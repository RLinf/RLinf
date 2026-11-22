Dexbotic RL
===========

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png
   :align: center
   :width: 90%

   Dexbotic model overview (image: `Dexbotic <https://github.com/dexmal/dexbotic>`__).

`Dexbotic <https://github.com/dexmal/dexbotic>`__ is an open-source VLA toolbox from Dexmal.
RLinf uses the Dexbotic π₀ and DM0 policies as LIBERO action-generation models, then fine-tunes them online with PPO.

Overview
--------

Fine-tune Dexbotic π₀ or DM0 on LIBERO with PPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      LIBERO Spatial · Object · Goal · 10

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install deps → download a Dexbotic checkpoint → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a LIBERO-compatible Dexbotic checkpoint (steps below).

Tasks
~~~~~

Select the model page by matching the environment, task family, and config or checkpoint artifact.

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_ppo_dexbotic_*``
     - Dexbotic pi0/dm0 policies on spatial manipulation tasks.
   * - LIBERO
     - LIBERO-Object
     - ``libero_object_ppo_dexbotic_pi0``
     - Dexbotic pi0 on object manipulation tasks.
   * - LIBERO
     - LIBERO-Goal / LIBERO-10
     - ``libero_goal_ppo_dexbotic_pi0`` / ``libero_10_ppo_dexbotic_pi0``
     - Goal-conditioned and long-horizon LIBERO suites.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - LIBERO camera streams and proprioception packaged for Dexbotic policies.
   * - Action
     - Chunked continuous actions produced by the selected Dexbotic policy backend, including flow-matching / flow-SDE settings.
   * - Reward
     - LIBERO success signal or simulator reward used for PPO updates.
   * - Prompt
     - Natural-language LIBERO instruction consumed by the policy processor.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: π₀

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model dexbotic --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download Dexmal/libero-db-pi0 --local-dir ./libero-db-pi0

      **3. Launch**

      Shipped π₀ configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 36 64

         * - Task suite
           - Config
         * - LIBERO Spatial
           - ``libero_spatial_ppo_dexbotic_pi0.yaml``
         * - LIBERO Object
           - ``libero_object_ppo_dexbotic_pi0.yaml``
         * - LIBERO Goal
           - ``libero_goal_ppo_dexbotic_pi0.yaml``
         * - LIBERO 10
           - ``libero_10_ppo_dexbotic_pi0.yaml``

      .. include:: _model_path.rst

      Launch PPO training with ``run_embodiment.sh``:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_dexbotic_pi0 actor.model.model_path=$PWD/libero-db-pi0

      .. dropdown:: Advanced configuration (optional)

         - **Precision**: ``actor.model.precision`` controls the loaded parameter dtype. Set it to ``bf16`` for BF16 execution and use the same precision in ``rollout.model``; leaving it ``null`` preserves the checkpoint's default dtype.
         - **Action chunks**: π₀ defaults to ``num_action_chunks: 5``.
         - **Standalone evaluation**: run Dexbotic's LIBERO evaluator on a trained checkpoint (or use RLinf's unified :doc:`evaluation <../../evaluations/index>` flow):

           .. code-block:: bash

              python toolkits/standalone_eval_scripts/dexbotic/libero_eval.py \
                --config_name db_pi0_libero \
                --pretrained_path ./libero-db-pi0 \
                --task_suite_name libero_spatial \
                --num_trials_per_task 50 \
                --action_chunk 5 \
                --num_steps 10

         - **Placement and metrics**: see :doc:`Placement <../../concepts/placement>`, :doc:`Execution modes <../../concepts/execution_modes>`, and :doc:`Training metrics <../../reference/metrics>`.

   .. tab-item:: DM0

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model dexbotic --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download Dexmal/DM0-libero --local-dir ./DM0-libero

      **3. Launch**

      The shipped DM0 config is ``examples/embodiment/config/libero_spatial_ppo_dexbotic_dm0.yaml``.

      .. include:: _model_path.rst

      Launch PPO training with ``run_embodiment.sh``:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_dexbotic_dm0 actor.model.model_path=$PWD/DM0-libero

      .. dropdown:: Advanced configuration (optional)

         - **Action chunks**: DM0 defaults to ``num_action_chunks: 10``.
         - **Standalone evaluation**: run Dexbotic's LIBERO evaluator on a trained checkpoint (or use RLinf's unified :doc:`evaluation <../../evaluations/index>` flow):

           .. code-block:: bash

              python toolkits/standalone_eval_scripts/dexbotic/libero_eval.py \
                --config_name dm0_libero \
                --pretrained_path ./DM0-libero \
                --task_suite_name libero_spatial \
                --num_trials_per_task 50 \
                --action_chunk 10 \
                --num_steps 10

         - **Placement and metrics**: see :doc:`Placement <../../concepts/placement>`, :doc:`Execution modes <../../concepts/execution_modes>`, and :doc:`Training metrics <../../reference/metrics>`.

Visualization and Results
-------------------------

Watch ``env/success_once`` in TensorBoard (``tensorboard --logdir ./logs``); set ``env.eval.video_cfg.save_video=True`` to record videos, or see :doc:`Training metrics <../../reference/metrics>` for all logged metrics.
