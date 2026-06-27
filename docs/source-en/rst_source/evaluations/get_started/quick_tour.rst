Quick Tour
==========

This tutorial walks through your first embodied evaluation in about 5 minutes using **LIBERO Spatial + OpenPI π₀.₅**. The example config is ``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``.

Step 1: Install the Environment
-------------------------------

From the repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero
   source .venv/bin/activate

With ``--env libero``, the installer clones LIBERO into ``.venv/libero`` (or reuses an existing checkout when ``LIBERO_PATH`` is set) and appends it to ``PYTHONPATH`` in ``.venv/bin/activate``. No separate ``LIBERO_PATH`` setup is required.

Step 2: Prepare the Model
-------------------------

Download the SFT checkpoint from Hugging Face (``RLinf/RLinf-Pi05-LIBERO-SFT``) to a local directory:

.. code-block:: bash

   huggingface-cli download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

Model hub: `RLinf/RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__. You can also override ``rollout.model.model_path`` on the command line at launch time.

Step 3: Launch Evaluation
-------------------------

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT

When the config name starts with ``libero_``, you can omit the benchmark argument:

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT

Step 4: Check Results
---------------------

- The terminal prints metrics such as ``eval/success_once`` and ``eval/return``
- Log directory: ``logs/<timestamp>-libero_spatial_openpi_pi05_eval/eval_embodiment.log``
- When ``env.eval.video_cfg.save_video: True``, videos are saved under ``<log_path>/video/eval/``

See :doc:`../reference/results` for more details.

Next Steps
----------

- YAML configuration: :doc:`../reference/configuration`
- Other benchmarks: :doc:`../guides/libero`
- More CLI options: :doc:`../reference/cli`
