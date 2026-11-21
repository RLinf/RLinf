Quick Tour
==========

Run your first embodied evaluation in about 5 minutes using **OpenPI π₀.₅ on LIBERO Spatial** (``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``).

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download`` (or pull a prebuilt image from `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__).

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ on LIBERO Spatial

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

      **2. Download Checkpoint** (`RLinf/RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__)

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      **3. Launch Evaluation**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
           rollout.model.model_path=$PWD/RLinf-Pi05-LIBERO-SFT

Results and Next Steps
----------------------

- The terminal prints metrics such as ``eval/success_once`` and ``eval/return``.
- Logs are saved to ``logs/<timestamp>-libero_spatial_openpi_pi05_eval/eval_embodiment.log``.
- Set ``env.eval.video_cfg.save_video=True`` to record rollout videos under ``<log_path>/video/eval/`` (see :doc:`../reference/results`).

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: YAML Configuration
      :link: ../reference/configuration
      :link-type: doc

      Review Hydra evaluation config fields and overrides.

   .. grid-item-card:: Benchmark Guides
      :link: ../guides/libero
      :link-type: doc

      Run LIBERO, ManiSkill OOD, RoboTwin, BEHAVIOR-1K, and more.

   .. grid-item-card:: CLI Options
      :link: ../reference/cli
      :link-type: doc

      Explore ``run_eval.sh`` flags and override syntax.
