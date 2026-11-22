RoboLab Evaluation
==================

Overview
--------

Evaluate π₀.₅ on RoboLab's 120 DROID joint-position manipulation tasks with RLinf. Install RoboLab and OpenPI in one environment, download the converted PyTorch checkpoint, and run a single task or the full 120-task sweep. See :doc:`the RoboLab overview <../../examples/embodied/robolab>` for observation and action formats.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      RoboLab (120 DROID joint-position tasks)

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀.₅ (``pi05_droid_jointpos``)

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/robolab/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`RoboLab Overview <../../examples/embodied/robolab>`

Setup and Run
-------------

Requires Linux, Python 3.11, ``uv``, and an NVIDIA GPU with a working Vulkan driver (verified on NVIDIA A800 80 GB with driver 570.172.08; see :doc:`the installation guide </rst_source/start/installation>`). In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: Single Task (Default)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robolab && source .venv/bin/activate

      **2. Download Model**

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos --local-dir ./RLinf-Pi05-RoboLab-droid-jointpos

      **3. Evaluate Default Task (BowlStackingLeftOnRightTask, 10 Episodes)**

      .. code-block:: bash

         bash evaluations/run_eval.sh robolab robolab_pi05_droid_jointpos \
           rollout.model.model_path=./RLinf-Pi05-RoboLab-droid-jointpos

   .. tab-item:: Full 120-Task Sweep

      Run all 120 benchmark tasks sequentially on one GPU with ``total_num_envs=10`` and ``rollout_epoch=1`` (1,200 episodes total):

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robolab && source .venv/bin/activate

      **2. Download Model**

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos --local-dir ./RLinf-Pi05-RoboLab-droid-jointpos

      **3. Export Task Table and Evaluate All 120 Tasks**

      .. code-block:: bash

         python -m toolkits.standalone_eval_scripts.robolab.export_tasks
         while IFS=$'\t' read -r task max_steps rest; do
             bash evaluations/run_eval.sh robolab robolab_pi05_droid_jointpos \
                 rollout.model.model_path=./RLinf-Pi05-RoboLab-droid-jointpos \
                 "env.eval.init_params.task=${task}" \
                 "env.eval.max_episode_steps=${max_steps}" \
                 "env.eval.max_steps_per_rollout_epoch=${max_steps}" \
                 "env.eval.init_params.output_dir=./logs/robolab/${task}" || break
         done < <(tail -n +2 robolab_tasks.tsv)

``RLinf/RLinf-Pi05-RoboLab-droid-jointpos`` is the PyTorch conversion of RoboLab's ``pi05_droid_jointpos`` checkpoint (containing ``model.safetensors``, ``config.json``, and ``assets/droid/norm_stats.json``). The 120-task loop stops on failure without automatic resume; copy ``robolab_tasks.tsv``, drop completed rows, and use a fresh output directory per experiment.

.. dropdown:: Export Benchmark Tasks

   Export all 120 tasks from your installed RoboLab version to ``robolab_tasks.tsv`` (use ``--output`` to choose another path):

   .. code-block:: bash

      python -m toolkits.standalone_eval_scripts.robolab.export_tasks

   The TSV lists each task name, native step limit (``max_steps``), difficulty, capability attributes, instruction, and filename:

   .. list-table::
      :header-rows: 1

      * - ``task``
        - ``max_steps``
        - ``difficulty``
        - ``attributes``
        - ``instruction``
        - ``filename``
      * - ``BowlStackingLeftOnRightTask``
        - ``300``
        - ``simple``
        - ``spatial``
        - ``Stack the left bowl on the right bowl``
        - ``benchmark/bowl_stacking_left_on_right.py``
      * - ``BananaInBowlTask``
        - ``750``
        - ``simple``
        - ``semantics``
        - ``Pick up the banana and place it in the bowl``
        - ``benchmark/banana_in_bowl_task.py``

   When switching tasks, copy ``task`` to ``env.eval.init_params.task`` and the same row's ``max_steps`` to both ``env.eval.max_episode_steps`` and ``env.eval.max_steps_per_rollout_epoch``.

.. dropdown:: Configure the Evaluation

   ``evaluations/robolab/robolab_pi05_droid_jointpos.yaml`` defaults to 10 episodes of ``BowlStackingLeftOnRightTask`` (300 steps). Key fields for switching tasks or tuning scale:

   .. list-table::
      :header-rows: 1
      :widths: 38 62

      * - Field
        - What to set
      * - ``rollout.model.model_path``
        - Local checkpoint directory containing ``model.safetensors``, ``config.json``, and ``assets/droid/norm_stats.json``.
      * - ``env.eval.init_params.task``
        - Task name from the TSV ``task`` column (default: ``BowlStackingLeftOnRightTask``).
      * - ``env.eval.max_episode_steps``
        - Native step limit from the same row's ``max_steps`` column (default: ``300``).
      * - ``env.eval.max_steps_per_rollout_epoch``
        - Match ``max_steps`` for the 120 benchmark tasks (default: ``300``).
      * - ``env.eval.total_num_envs``
        - Parallel env count (default: ``10``; lower if GPU memory is tight).
      * - ``env.eval.rollout_epoch``
        - Evaluation rounds (default: ``1``; total episodes = ``rollout_epoch × total_num_envs``).
      * - ``env.eval.init_params.output_dir``
        - Output directory for RoboLab task config and ``run_<index>.hdf5`` trajectories (default: ``./logs/robolab/BowlStackingLeftOnRightTask``).

   All 120 native limits are multiples of the 15-action chunk length. For custom limits, round ``max_steps_per_rollout_epoch`` up to a multiple of 15 (for example, episode limit 17 needs rollout budget 30).

Results and Troubleshooting
---------------------------

Console logs and TensorBoard events go to ``logs/<timestamp>-robolab_pi05_droid_jointpos/`` (see :doc:`../reference/results`), while ``env.eval.init_params.output_dir`` stores RoboLab task configs and ``run_<index>.hdf5`` trajectories. Reported metrics: ``eval/success_once`` (success fraction), ``eval/score`` (mean normalized subtask score in ``[0, 1]`` when ``env.eval.init_params.enable_subtask: true``), ``eval/num_trajectories``, and ``eval/episode_len``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Wrong episode horizon after switching tasks
     - Changing ``env.eval.init_params.task`` alone does not update step limits; copy ``max_steps`` from ``robolab_tasks.tsv`` to both ``max_episode_steps`` and ``max_steps_per_rollout_epoch``.
   * - Custom step-limit validation error
     - Set ``env.eval.max_steps_per_rollout_epoch`` to the smallest multiple of 15 that covers ``max_episode_steps``.
   * - GPU out of memory
     - Lower ``env.eval.total_num_envs`` and raise ``env.eval.rollout_epoch`` (for example, ``1`` env × ``10`` rounds for 10 episodes).
   * - Interrupted 120-task sweep
     - Copy ``robolab_tasks.tsv``, delete completed rows, and rerun the loop with a fresh output directory.
