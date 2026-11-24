Quick Tour
==========

Run an embodied evaluation with **OpenPI π₀.₅ on LIBERO Spatial** (``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``).

Setup and Run
-------------

If ``.venv`` already holds another model/environment stack, pass ``--venv .venv-eval`` to ``install.sh`` and activate ``.venv-eval`` instead. For mainland China mirrors or Docker Hub images, see :doc:`../../start/installation`.

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ on LIBERO Spatial

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

      **2. Launch Evaluation** (auto-downloads `RLinf/RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__ on first run; runtime ~5 minutes on a 24 GB+ GPU after install and download)

      For a quick check across 1, 2, 4, or 8 GPUs, set ``env.eval.total_num_envs=8``:

      .. code-block:: bash

         bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval env.eval.total_num_envs=8

      To evaluate the full 500-episode suite without exhausting host RAM (each LIBERO env uses ~2 GiB host RAM, so the default 500 parallel envs needs ~1 TiB RAM; see :doc:`../guides/libero`), run 128 parallel envs with auto-reset:

      .. code-block:: bash

         bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
           env.eval.total_num_envs=128 \
           env.eval.max_steps_per_rollout_epoch=960

      For offline runs, download ``RLinf/RLinf-Pi05-LIBERO-SFT`` to a local directory and append ``rollout.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT``.

Results and Next Steps
----------------------

Evaluation prints a summary table to the terminal and appends the same table to ``metrics.log``. Expect ``eval/success_once`` around **0.95+** for ``RLinf/RLinf-Pi05-LIBERO-SFT`` on LIBERO Spatial:

.. code-block:: text

   ╭────────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step:    1/1 │ Progress: ████████████████████ │ 100.0%            │
   │ Elapsed: 03:45      │ ETA: 00:00                     │ Step Time: 225.0s │
   ├────────────────────────────── Evaluation ────────────────────────────────┤
   │ episode_len=142.3   │ num_trajectories=8             │ success_once=1.00 │
   ╰──────────────────────────────────────────────────────────────────────────╯

- Logs and metrics are saved to ``logs/<timestamp>-libero_spatial_openpi_pi05_eval/eval_embodiment.log`` and ``metrics.log``.
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
