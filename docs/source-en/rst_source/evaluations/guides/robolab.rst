Evaluate π₀.₅ on RoboLab
========================

Evaluate DROID joint-position manipulation tasks through RLinf's environment workers and embodied evaluation runner. Install RoboLab and OpenPI in one environment, download the converted checkpoint, inspect the task metadata, and run a task or the 120-task benchmark. See :doc:`the RoboLab overview <../../examples/embodied/robolab>` for observation and action formats.

Installation
------------

Use Linux, Python 3.11, ``uv``, and an NVIDIA GPU with a working Vulkan driver. The evaluated setup used NVIDIA A800 80 GB GPUs and driver 570.172.08. Follow :doc:`the installation guide </rst_source/start/installation>` for system prerequisites, then run from the RLinf repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robolab
   source .venv/bin/activate

Add ``--use-mirror`` to the install command when needed. The installer creates ``.venv``, installs RoboLab's Isaac Sim 5.0 / Isaac Lab 2.2 dependencies, and installs RLinf OpenPI. RoboLab selects PyTorch 2.7.1 through the installer's shared Torch override mechanism. System graphics libraries and Git LFS are handled by ``requirements/sys_deps.sh``.

The default source checkout is ``.venv/robolab`` and is pinned to ``b6cfcf6ccd2113c4cd1216d35b33024a7d5691da``. Its task metadata and assets must remain beside the installed package. To reuse an existing checkout, set ``ROBOLAB_PATH=/path/to/RoboLab`` before installation; that checkout is used as provided. The adapter discovers the installed source path from package metadata at runtime.

.. note::

   The recorded evaluations validate the existing runtime. A fresh installation and the Docker build have not been validated for this contribution. ``uv pip check --python .venv/bin/python`` reports dependency metadata conflicts in the evaluated environment, including Isaac Sim's Torch pins and the OpenPI Transformers fork. Review the PR installation audit before deploying this setup in a fresh environment.

Download the Model
------------------

Download ``RLinf/RLinf-Pi05-RoboLab-droid-jointpos``, the PyTorch conversion of RoboLab's ``pi05_droid_jointpos`` checkpoint. This policy predicts joint positions, so choose the joint-position checkpoint for this configuration.

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
       --local-dir /path/to/model/RLinf-Pi05-RoboLab-droid-jointpos

The checkpoint directory contains ``model.safetensors``, ``config.json``, and ``assets/droid/norm_stats.json``. Set ``rollout.model.model_path`` in ``evaluations/robolab/pi05_droid_jointpos.yaml`` to that directory, or pass it on the command line as below.

Find the Benchmark Tasks
------------------------

Use the metadata shipped with the installed RoboLab version to choose tasks and their native episode lengths. Export the 120 benchmark tasks, instructions, difficulty labels, capability attributes, and source filenames:

.. code-block:: bash

   mkdir -p data
   python - <<'PYTHON'
   import csv
   import json
   from decimal import Decimal
   from importlib.util import find_spec
   from pathlib import Path

   package = Path(find_spec("robolab").origin).parent
   metadata = package / "tasks/_metadata/task_metadata.json"
   tasks = [t for t in json.loads(metadata.read_text())
            if t["filename"].startswith("benchmark/")]
   with Path("data/robolab_tasks.tsv").open("w", newline="") as stream:
       writer = csv.writer(stream, delimiter="	")
       writer.writerow(["task", "max_steps", "difficulty", "attributes", "instruction", "filename"])
       for task in sorted(tasks, key=lambda t: t["task_name"]):
           writer.writerow([
               task["task_name"], int(Decimal(task["episode_s"]) * 15),
               task["difficulty_label"], task["attributes"],
               task["instruction"], task["filename"],
           ])
   print(f"{len(tasks)} tasks; metadata: {metadata}")
   print("Saved data/robolab_tasks.tsv")
   PYTHON

The DROID configuration runs at 15 control steps per second (simulation timestep 1/120 s, decimation 8), so ``max_steps = episode_s × 15``. ``data/robolab_tasks.tsv`` is a local task index; the original JSON also contains instruction variants and subtask definitions. Find each task implementation under ``robolab/tasks/<filename>`` in the installed source checkout.

Run a Task
----------

Run ten episodes of ``BowlStackingLeftOnRightTask`` with ten parallel environments. This task has a native limit of 300 control steps:

.. code-block:: bash

   bash evaluations/run_eval.sh robolab pi05_droid_jointpos \
       rollout.model.model_path=/path/to/model/RLinf-Pi05-RoboLab-droid-jointpos \
       env.eval.init_params.task=BowlStackingLeftOnRightTask \
       env.eval.max_episode_steps=300 \
       env.eval.max_steps_per_rollout_epoch=300 \
       env.eval.total_num_envs=10 \
       env.eval.rollout_epoch=1 \
       env.eval.init_params.output_dir=./logs/robolab/BowlStackingLeftOnRightTask

RLinf launches ``RoboLabEnv`` through ``EnvWorker`` and runs the policy in the rollout worker. The simulator subprocess belongs to the environment adapter. The Hydra configuration places both workers on GPU 0; use ``cluster.component_placement`` to change placement.

The configuration uses ``env_type: robolab``, ``auto_reset: false``, and ``ignore_terminations: false``. Each rollout epoch resets the whole environment batch. For ten episodes, use ``total_num_envs=10, rollout_epoch=1`` or ``total_num_envs=1, rollout_epoch=10``. ``max_episode_steps`` is an RLinf upper limit; native termination or timeout can end an episode earlier. Set ``max_steps_per_rollout_epoch`` high enough to cover that limit and to a multiple of the 15-action chunk length.

For the reported comparison, use the task's native limit from the metadata. Keep the evaluation configuration's numerical alignment options:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model/RLinf-Pi05-RoboLab-droid-jointpos
       openpi:
         pytorch_numerics: true
         eval_batch_size: 1
         eval_seed: 0
         sample_compile: false
         torch_compile: false

Ten simulation environments and ``eval_batch_size: 1`` have different roles: environments progress in parallel, while the policy processes each active observation separately. Finished environments are skipped. Each task starts a policy random generator with seed 0; this does not reproduce the continuous random stream of a native multi-task server.

Run All 120 Tasks
-----------------

After exporting the TSV, run the tasks sequentially on one GPU. Each task gets its own native limit and ten episodes, for 1,200 episodes in total. Run this Bash loop from the repository root:

.. code-block:: bash

   while IFS=$'	' read -r task max_steps rest; do
       bash evaluations/run_eval.sh robolab pi05_droid_jointpos \
           rollout.model.model_path=/path/to/model/RLinf-Pi05-RoboLab-droid-jointpos \
           "env.eval.init_params.task=${task}" \
           "env.eval.max_episode_steps=${max_steps}" \
           "env.eval.max_steps_per_rollout_epoch=${max_steps}" \
           env.eval.total_num_envs=10 \
           env.eval.rollout_epoch=1 \
           "env.eval.init_params.output_dir=./logs/robolab/${task}" || break
   done < <(tail -n +2 data/robolab_tasks.tsv)

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
     - Mean normalized subtask score in [0, 1]; successful episodes contribute 1. Enabled by ``init_params.enable_subtask: true``.
   * - ``eval/num_trajectories``
     - Number of completed episodes included in the result.
   * - ``eval/episode_len``
     - Mean number of executed control steps per episode.

Aggregate success counts and episode-weighted scores across all tasks. The paper's SPARC and speed metrics are computed from recorded end-effector trajectories rather than emitted by the RLinf runner; speed measures robot motion, not inference throughput. See :doc:`../reference/results` for the shared logging conventions.
