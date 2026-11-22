MolmoAct2 Evaluation
====================

`MolmoAct2 <https://github.com/allenai/molmoact2>`__ is AllenAI's open
vision-language-action model, served through a LeRobot policy that predicts
continuous actions with a flow-matching action expert. RLinf runs the official
MolmoAct2-LIBERO checkpoint through its LIBERO runner. This integration is
**evaluation only** — the upstream policy is inference-only, so there is no
training path.

Overview
--------

Evaluate the official MolmoAct2-LIBERO checkpoint across the four LIBERO suites.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO

   .. grid-item-card:: Algorithms
      :text-align: center

      Evaluation only

   .. grid-item-card:: Tasks
      :text-align: center

      Spatial · Object · Goal · Long

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1–8 GPUs

| **You'll do:** install → download the checkpoint → launch → watch ``eval/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the MolmoAct2-LIBERO checkpoint (steps below).

Tasks
~~~~~

One config ships per suite. Each runs 20 parallel environments for 25 episodes
each — the full 500-trajectory suite — with a step budget of
``max_episode_steps × 25``.

.. list-table::
   :header-rows: 1
   :widths: 18 40 20 22

   * - Suite
     - Config
     - ``max_episode_steps``
     - Trajectories
   * - Spatial
     - ``libero_spatial_molmoact2_eval``
     - 240
     - 500
   * - Object
     - ``libero_object_molmoact2_eval``
     - 240
     - 500
   * - Goal
     - ``libero_goal_molmoact2_eval``
     - 320
     - 500
   * - Long
     - ``libero_10_molmoact2_eval``
     - 520
     - 500

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - Two camera views — ``main_images`` maps to the agent view and
       ``wrist_images`` to the wrist view — plus the 8-D robot state in ``states``.
   * - Action
     - Continuous 7-DoF actions (6-DoF delta EE + gripper) from the action expert.
       One action is executed per rollout step.
   * - Reward
     - LIBERO task success.
   * - Prompt
     - Natural-language task instruction from ``task_descriptions``.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model molmoact2 --env libero && source .venv/bin/activate

      Set ``MOLMOACT2_LEROBOT_PATH`` before installing to reuse an existing checkout of
      `RLinf/lerobot <https://github.com/RLinf/lerobot/tree/RLinf/molmoact2-hf-inference>`__.

      **2. Download**

      Download the official `allenai/MolmoAct2-LIBERO <https://huggingface.co/allenai/MolmoAct2-LIBERO>`__ checkpoint:

      .. code-block:: bash

         hf download allenai/MolmoAct2-LIBERO --local-dir ./MolmoAct2-LIBERO

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 22 46 32

         * - Suite
           - Config (``evaluations/libero/``)
           - Command suffix
         * - Spatial
           - ``libero_spatial_molmoact2_eval.yaml``
           - ``libero_spatial_molmoact2_eval``
         * - Object
           - ``libero_object_molmoact2_eval.yaml``
           - ``libero_object_molmoact2_eval``
         * - Goal
           - ``libero_goal_molmoact2_eval.yaml``
           - ``libero_goal_molmoact2_eval``
         * - Long
           - ``libero_10_molmoact2_eval.yaml``
           - ``libero_10_molmoact2_eval``

      Set ``rollout.model.model_path`` in the eval config or pass it on the command line (the config has no ``actor`` section, so there is no second path to keep in sync). Evaluate LIBERO-Long:

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_molmoact2_eval \
           rollout.model.model_path=./MolmoAct2-LIBERO

      Swap in any config suffix from the table above to evaluate another suite.

.. warning::

   A full suite covers 500 trajectories and can take several hours. Lower
   ``env.eval.max_steps_per_rollout_epoch`` for a quick check — one episode per
   environment is ``max_episode_steps``.

.. dropdown:: Advanced configuration (optional)

   - Inference settings (``num_steps``, ``norm_tag``, action mode) → the ``molmoact2`` block in ``examples/embodiment/config/model/molmoact2.yaml``.
   - ``rollout.model.precision`` has no effect: MolmoAct2 loads its weights in fp32 upstream.
   - Keep ``rollout.pipeline_stage_num: 1``; the policy keys its per-environment action queues by batch index.
   - Keep ``env.eval.max_episode_steps`` a multiple of the policy's 10-step action queue (240 / 320 / 520 all are), or an episode starts on the previous one's leftover actions.
   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`.

Visualization and Results
-------------------------

The terminal reports ``eval/success_once``. Logs are written to:

.. code-block:: text

   logs/YYYYMMDD-HHMMSS-libero_10_molmoact2_eval/eval_embodiment.log

See :doc:`LIBERO Evaluation <../../evaluations/guides/libero>` for the benchmark
protocol and :doc:`Evaluation Results <../../evaluations/reference/results>` for
metric interpretation.
