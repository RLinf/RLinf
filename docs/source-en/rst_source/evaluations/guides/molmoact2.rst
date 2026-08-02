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

1. Configures the MolmoAct2 environment with Python 3.12.
2. Installs the LIBERO environment and RLinf embodied dependencies.
3. Checks out the pinned LeRobot revision used by the MolmoAct2 adapter.

Download the Model
------------------

Download the official `allenai/MolmoAct2-LIBERO <https://huggingface.co/allenai/MolmoAct2-LIBERO>`__ checkpoint:

.. code-block:: bash

   hf download allenai/MolmoAct2-LIBERO \
     --local-dir /path/to/models/MolmoAct2-LIBERO

Run It
------

Launch the ``libero_10_molmoact2_eval`` config and override its placeholder model path:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_10_molmoact2_eval \
     rollout.model.model_path=/path/to/models/MolmoAct2-LIBERO

What this does:

1. Loads the official checkpoint with the MolmoAct2 model adapter.
2. Runs the LIBERO-Long suite with the evaluation settings in ``evaluations/libero/libero_10_molmoact2_eval.yaml``.
3. Writes terminal output and ``eval/success_once`` to the timestamped evaluation log.

.. warning::

   The default config covers the full LIBERO-Long suite and can take several hours. Use ``env.eval`` Hydra overrides when you only need a smoke test.

Input and Inference Settings
----------------------------

RLinf maps ``main_images`` to the agent view and ``wrist_images`` to the wrist view expected by MolmoAct2-LIBERO. The model preset already sets continuous action inference, ``norm_tag: libero``, and ``num_steps: 10``; you do not need to repeat these values on the command line.

Check Results
-------------

The terminal reports ``eval/success_once``. Logs are written to:

.. code-block:: text

   logs/<timestamp>-libero_10_molmoact2_eval/eval_embodiment.log

See :doc:`LIBERO Evaluation <libero>` for the benchmark protocol and :doc:`Evaluation Results <../reference/results>` for metric interpretation.
