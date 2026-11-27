RMBench Evaluation
==================

Overview
--------

Evaluate a fine-tuned π₀.₅ Short Memory checkpoint on bimanual Aloha manipulation tasks in `RMBench <https://rmbench.github.io/>`__, reporting task-level metrics such as ``eval/success_once``.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      RMBench (bimanual Aloha)

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀.₅ (Short Memory)

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/rmbench/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`RMBench Training <../../examples/embodied/rmbench>`

**Prerequisites:** :doc:`Installation </rst_source/start/installation>` · 1 NVIDIA GPU (≥ 24 GB VRAM) with CUDA toolkit installed to build CuRobo, plus the ``.venv`` environment, RMBench checkout, SFT checkpoint, and ``norm_stats.json`` from :doc:`RMBench Training <../../examples/embodied/rmbench>`.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before downloading assets.

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      RLinf uses two virtual environments here: ``.venv`` runs OpenPI SFT, data preparation, and inference, while ``.rmbench-venv`` isolates the SAPIEN/CuRobo simulator worker invoked via ``sim_python`` and downloads the simulator assets.

      **1. Install the Simulator**

      From the RLinf root, install the RMBench simulator environment into ``.rmbench-venv``:

      .. code-block:: bash

         bash requirements/install.sh embodied --env rmbench --venv .rmbench-venv
         source .rmbench-venv/bin/activate

      **2. Download the Assets**

      Download the RMBench assets inside the checkout, then switch back to ``.venv`` for policy inference:

      .. code-block:: bash

         (cd RMBench && bash scripts/_download_assets.sh)
         source .venv/bin/activate

      **3. Run Evaluation**

      Launch ``rmbench_openpi_pi05_eval`` with your local paths via Hydra CLI overrides (defaults to ``observe_and_pickup`` on the ``seen`` split across 100 expert-screened episodes; append ``env.eval.rollout_epoch=1`` to run one episode first):

      .. code-block:: bash

         bash evaluations/run_eval.sh rmbench rmbench_openpi_pi05_eval \
           env.eval.init_params.rmbench_path="$PWD/RMBench" \
           env.eval.init_params.sim_python="$PWD/.rmbench-venv/bin/python" \
           rollout.model.model_path="$PWD/logs/<run>/rmbench_pi05_short_memory_sft/checkpoints/global_step_<N>" \
           rollout.model.openpi_data.norm_stats_path="$PWD/rmbench_lerobot/norm_stats.json"

      The OpenPI policy fixes action sampling with ``rollout.model.openpi.eval_seed``. With Short Memory enabled, the simulator records every control step (including steps inside an action chunk) and feeds eight frames spaced 53 steps apart at 15 Hz.

.. dropdown:: Optional Task, Short Memory, and Video Overrides

   - **Switch tasks:** Pass ``env.eval.init_params.task_name=<task>``, set ``env.eval.max_episode_steps`` to the task limit in ``RMBench/env_cfg/task_config/_eval_step_limit.yml``, and round ``env.eval.max_steps_per_rollout_epoch`` up to a multiple of 32 so the final action chunk completes.
   - **Current-frame checkpoint:** Append ``rollout.model.openpi.short_memory=false env.eval.short_memory=null``.
   - **Record videos:** Append ``env.eval.video_cfg.save_video=true`` to save MP4s under ``logs/rmbench/``.

Results and Troubleshooting
---------------------------

The terminal prints ``eval/success_once``; logs and videos go to ``logs/rmbench/`` (see :doc:`../reference/results` or view them with ``tensorboard --logdir ./logs/rmbench``).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - CuRobo CUDA build fails during ``install.sh``
     - Verify an NVIDIA GPU and matching CUDA toolkit (``nvcc``) are available before building ``.rmbench-venv``.
   * - Missing assets or wrong Python interpreter
     - Pass ``env.eval.init_params.rmbench_path="$PWD/RMBench"`` and ``env.eval.init_params.sim_python="$PWD/.rmbench-venv/bin/python"`` after running ``_download_assets.sh``.
   * - Episode truncated before final chunk
     - Match ``env.eval.max_episode_steps`` to ``_eval_step_limit.yml`` and round ``env.eval.max_steps_per_rollout_epoch`` up to a multiple of 32.
   * - Validation error on parallel envs or reset flags
     - Keep ``env.eval.total_num_envs=1``, ``env.eval.auto_reset=false``, and ``env.eval.ignore_terminations=false`` so seed screening stays deterministic.
