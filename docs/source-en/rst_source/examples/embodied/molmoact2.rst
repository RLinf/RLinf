MolmoAct2 Evaluation
====================

Evaluate the official MolmoAct2-LIBERO checkpoint through RLinf's unified LIBERO runner. This integration supports evaluation only.

Installation
------------

Install MolmoAct2 and LIBERO from the repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --model molmoact2 --env libero
   source .venv/bin/activate

What this does:

1. Installs the LIBERO environment and RLinf embodied dependencies.
2. Installs `RLinf/lerobot <https://github.com/RLinf/lerobot/tree/RLinf/molmoact2-hf-inference>`__, RLinf's LeRobot fork whose ``RLinf/molmoact2-hf-inference`` branch provides the MolmoAct2 policy and pins the dependency versions the LIBERO stack needs.

Download the Model
------------------

Download the official `allenai/MolmoAct2-LIBERO <https://huggingface.co/allenai/MolmoAct2-LIBERO>`__ checkpoint:

.. code-block:: bash

   hf download allenai/MolmoAct2-LIBERO \
     --local-dir /path/to/model/MolmoAct2-LIBERO

Run It
------

Launch the ``libero_10_molmoact2_eval`` config and override its placeholder model path:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_10_molmoact2_eval \
     rollout.model.model_path=/path/to/model/MolmoAct2-LIBERO

What this does:

1. Loads the official checkpoint with the MolmoAct2 model adapter.
2. Runs the LIBERO-Long suite with the evaluation settings in ``evaluations/libero/libero_10_molmoact2_eval.yaml``.
3. Writes terminal output and ``eval/success_once`` to the timestamped evaluation log.

.. warning::

   The default config covers the full LIBERO-Long suite and can take several hours. Use ``env.eval`` Hydra overrides when you only need a smoke test.

Other Task Suites
-----------------

One config ships per LIBERO task suite. Each runs 20 parallel environments for 25 episodes each, i.e. the full 500-trajectory suite; the step budget is ``max_steps_per_rollout_epoch = max_episode_steps x 25``.

.. list-table::
   :header-rows: 1
   :widths: 22 38 20 20

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

For example, LIBERO-Spatial:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_molmoact2_eval \
     rollout.model.model_path=/path/to/model/MolmoAct2-LIBERO

Input and Inference Settings
----------------------------

RLinf maps ``main_images`` to the agent view and ``wrist_images`` to the wrist view expected by MolmoAct2-LIBERO, and passes ``states`` and ``task_descriptions`` through unchanged; all four keys are required. The model preset already sets continuous action inference, ``norm_tag: libero``, and ``num_steps: 10`` under its ``molmoact2`` block; you do not need to repeat these values on the command line.

MolmoAct2 loads its weights in fp32 upstream, so ``rollout.model.precision`` has no effect. It also keeps one action queue per batch index, so keep ``rollout.pipeline_stage_num: 1``.

Check Results
-------------

The terminal reports ``eval/success_once``. Logs are written to:

.. code-block:: text

   logs/<timestamp>-libero_10_molmoact2_eval/eval_embodiment.log

See :doc:`LIBERO Evaluation <../../evaluations/guides/libero>` for the benchmark protocol and :doc:`Evaluation Results <../../evaluations/reference/results>` for metric interpretation.
