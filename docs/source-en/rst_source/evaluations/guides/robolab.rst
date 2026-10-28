Evaluate π₀.₅ on RoboLab
========================

Evaluate π₀.₅ on RoboLab's DROID joint-position manipulation tasks with RLinf. Install RoboLab and OpenPI in one environment, download the converted checkpoint, inspect the task metadata, and run a task or the 120-task benchmark. See :doc:`the RoboLab overview <../../examples/embodied/robolab>` for observation and action formats.

Installation
------------

Use Linux, Python 3.11, ``uv``, and an NVIDIA GPU with a working Vulkan driver. The evaluated setup used NVIDIA A800 80 GB GPUs and driver 570.172.08. Follow :doc:`the installation guide </rst_source/start/installation>` for system prerequisites, then run from the RLinf repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robolab
   source .venv/bin/activate

Add ``--use-mirror`` to the install command when needed.

Download the Model
------------------

Download ``RLinf/RLinf-Pi05-RoboLab-droid-jointpos``, the PyTorch conversion of RoboLab's ``pi05_droid_jointpos`` checkpoint. This policy predicts joint positions, so choose the joint-position checkpoint for this configuration.

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
       --local-dir /path/to/model/RLinf-Pi05-RoboLab-droid-jointpos

The checkpoint directory contains ``model.safetensors``, ``config.json``, and ``assets/droid/norm_stats.json``. Set ``rollout.model.model_path`` in ``evaluations/robolab/pi05_droid_jointpos.yaml`` to that directory.

Find the Benchmark Tasks
------------------------

Use the task export script to list the benchmark tasks in your installed RoboLab version. Run it from the RLinf repository root in the activated environment:

.. code-block:: bash

   python -m toolkits.standalone_eval_scripts.robolab.export_tasks

The script creates ``robolab_tasks.tsv`` in the repository root with task names, native step limits (``max_steps``), difficulty labels, capability attributes, instructions, and source filenames. Use ``--output /path/to/tasks.tsv`` to choose another destination. Task implementations are under ``robolab/tasks/<filename>`` in the installed RoboLab checkout.

Configure the Evaluation
------------------------

Edit ``evaluations/robolab/pi05_droid_jointpos.yaml`` before launching an experiment. It defaults to ten episodes of ``BowlStackingLeftOnRightTask`` with a native limit of 300 control steps. For this default task, only the checkpoint path needs to be changed. Configure other tasks using the following fields:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Field
     - What to set
   * - ``rollout.model.model_path``
     - Local checkpoint directory containing the three files listed above.
   * - ``env.eval.init_params.task``
     - Task name from the TSV's ``task`` column. Default: ``BowlStackingLeftOnRightTask``.
   * - ``env.eval.max_episode_steps``
     - Native limit from the same row's ``max_steps`` column. Default: 300.
   * - ``env.eval.max_steps_per_rollout_epoch``
     - Set to the same ``max_steps`` value for the 120 benchmark tasks. Default: 300.
   * - ``env.eval.total_num_envs``
     - Number of parallel environments. Default: 10. Reduce this if GPU memory is limited.
   * - ``env.eval.rollout_epoch``
     - Number of evaluation rounds. Default: 1. Episode count equals this value multiplied by ``total_num_envs``; use 1 environment and 10 rounds to run ten episodes sequentially.
   * - ``env.eval.init_params.output_dir``
     - Directory for task configuration and trajectories. Default: ``./logs/robolab/BowlStackingLeftOnRightTask``. Use a separate directory for each task or experiment.

The export script already converts each task's duration to control steps; no manual time conversion is needed. When changing the task for the single-task command, copy that row's ``max_steps`` into both step fields above. Changing only the task name does not update these fields automatically. For example, switching to ``BananaInBowlTask`` requires setting both fields to 750.

All 120 benchmark limits are multiples of the configured 15-action chunk length. If you choose a custom episode limit, set ``max_steps_per_rollout_epoch`` to the smallest multiple of 15 that covers it: an episode limit of 17 needs a rollout budget of 30. Native task termination can still end an episode before the configured limit.

Run a Task
----------

After saving the YAML configuration, launch the evaluation from the repository root:

.. code-block:: bash

   bash evaluations/run_eval.sh robolab pi05_droid_jointpos

With the default task and episode settings, this runs ten episodes of ``BowlStackingLeftOnRightTask``. The remaining policy and environment defaults are already set in the YAML.

Run All 120 Tasks
-----------------

After exporting the TSV and setting the checkpoint path in the YAML, keep ``total_num_envs: 10`` and ``rollout_epoch: 1`` for 1,200 episodes in total. The loop below runs all tasks sequentially on one GPU. It reads each task name and native step limit from the TSV and overrides the task, both step fields, and output directory automatically:

.. code-block:: bash

   while IFS=$'\t' read -r task max_steps rest; do
       bash evaluations/run_eval.sh robolab pi05_droid_jointpos \
           "env.eval.init_params.task=${task}" \
           "env.eval.max_episode_steps=${max_steps}" \
           "env.eval.max_steps_per_rollout_epoch=${max_steps}" \
           "env.eval.init_params.output_dir=./logs/robolab/${task}" || break
   done < <(tail -n +2 robolab_tasks.tsv)

The loop stops on a failed command. It does not implement automatic resume; remove completed rows from a copy of the TSV to select remaining tasks, and use a fresh output directory for a new experiment. Use the task-level logs when combining results across separate runs.

Read the Results
----------------

The terminal and TensorBoard report the completed episodes. ``evaluations/run_eval.sh`` saves console logs and, by default, TensorBoard events under ``logs/<timestamp>-pi05_droid_jointpos/``. ``env.eval.init_params.output_dir`` controls RoboLab's task configuration and ``run_<index>.hdf5`` trajectory files separately.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Metric
     - Meaning
   * - ``eval/success_once``
     - Fraction of completed episodes that succeeded; multiply by 100 for success percentage.
   * - ``eval/score``
     - Mean normalized subtask score in [0, 1]; successful episodes contribute 1. Enabled by ``env.eval.init_params.enable_subtask: true``.
   * - ``eval/num_trajectories``
     - Number of completed episodes included in the result.
   * - ``eval/episode_len``
     - Mean number of executed control steps per episode.

Aggregate success counts and episode-weighted scores across all tasks. The paper's SPARC and speed metrics are computed from recorded end-effector trajectories rather than emitted by the RLinf runner; speed measures robot motion, not inference throughput. See :doc:`../reference/results` for the shared logging conventions.
