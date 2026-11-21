Quick Tour
==========

This tutorial walks through your first embodied evaluation in about 5 minutes using **LIBERO Spatial + OpenPI π₀.₅**. The example config is ``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``.

Step 1: Install the Environment
-------------------------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: OpenPI + LIBERO

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

Step 2: Prepare the Model
-------------------------

Download the SFT checkpoint from Hugging Face (`RLinf/RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__):

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ (LIBERO)

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

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
