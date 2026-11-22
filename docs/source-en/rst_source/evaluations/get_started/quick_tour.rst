Evaluation Quick Tour
=====================

Run your first embodied evaluation in about 5 minutes using **OpenPI π₀.₅ on LIBERO Spatial** (``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``).

Setup and Run
-------------

If ``.venv`` already holds another model/environment stack, pass ``--venv .venv-eval`` to ``install.sh`` and activate ``.venv-eval`` instead. For mainland China mirrors or Docker Hub images, see :doc:`../../start/installation`.

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ on LIBERO Spatial

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

      **2. Launch Evaluation** (auto-downloads `RLinf/RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__ into the Hugging Face cache on first run)

      For a quick single-GPU check (~5 minutes on a 24 GB GPU), override ``env.eval.total_num_envs=10``:

      .. code-block:: bash

         bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
           rollout.model.model_path=RLinf/RLinf-Pi05-LIBERO-SFT \
           env.eval.total_num_envs=10

      Drop ``env.eval.total_num_envs=10`` to run the full 500-environment suite defined in the YAML. For offline runs, pre-download ``RLinf/RLinf-Pi05-LIBERO-SFT`` to a local directory and pass ``rollout.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT``.

Results and Next Steps
----------------------

- The terminal prints metrics such as ``eval/success_once`` and ``eval/return``.
- Logs are saved to ``logs/<timestamp>-libero_spatial_openpi_pi05_eval/eval_embodiment.log``.
- Rollout videos are saved by default under ``<log_path>/video/eval/`` (``save_video: True`` in the config); pass ``env.eval.video_cfg.save_video=False`` to disable video recording (see :doc:`../reference/results`).

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: YAML Configuration
      :link: ../reference/configuration
      :link-type: doc

      Review Hydra evaluation config fields and overrides.

   .. grid-item-card:: Benchmark Guides
      :link: ../guides/index
      :link-type: doc

      Run LIBERO, ManiSkill OOD, RoboTwin, BEHAVIOR-1K, and more.

   .. grid-item-card:: CLI Options
      :link: ../reference/cli
      :link-type: doc

      Explore ``run_eval.sh`` flags and override syntax.
