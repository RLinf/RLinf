Offline QAM on OGBench
======================

Train a flow policy from an OGBench offline dataset with native PyTorch QAM, then evaluate its success in the simulator. This recipe uses the AntMaze navigation task and one GPU per training seed.

Overview
--------

The training run learns from a fixed dataset and evaluates saved policies on complete episodes.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      QAM

   .. grid-item-card:: Models
      :text-align: center

      Flow MLP policy

   .. grid-item-card:: Environments / Data
      :text-align: center

      OGBench AntMaze

   .. grid-item-card:: Training
      :text-align: center

      Offline

| **You'll do:** install → train with one random seed → resume or evaluate a checkpoint → watch ``eval/success``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · one NVIDIA GPU · OGBench data downloaded on first run.

Tasks
~~~~~

The example config selects this task and its state and action dimensions:

.. list-table::
   :header-rows: 1

   * - Task
     - State / Action Dimensions
     - Config
   * - ``antmaze-large-navigate-singletask-task1-v0``
     - 29 / 8
     - ``ogbench_qam_offline.yaml``

Evaluation defaults to one environment per worker. It also supports parallel environments when ``env.eval.auto_reset=false`` and ``env.eval.early_stop_on_done=false``. Each environment keeps its own episode seed, return, length, success, and final state. The example uses action chunks of length one. Training supports longer action chunks with padding at episode boundaries; other tasks and chunk lengths require separate benchmarks.

How QAM Works
-------------

An ensemble of critics scores candidate actions. A slow flow policy learns the dataset's actions, and a fast flow policy learns to improve those actions using value gradients propagated through the flow. Evaluation uses the fast policy. See the `official QAM implementation <https://github.com/ColinQiyangLi/qam/tree/2726d767c9a0a7a46d49693f0391f73dc2cf58ac>`_ for the reference algorithm. This recipe covers offline training; online fine-tuning is outside its scope.

Installation
------------

From the RLinf repository root, install the embodied dependencies with OGBench:

.. code-block:: bash

   bash requirements/install.sh embodied --env ogbench
   source .venv/bin/activate

The installer pins OGBench 1.2.1 and MuJoCo 3.3.4. The launch script configures EGL for headless evaluation. Training was validated on NVIDIA A800 GPUs; other accelerator backends have not been benchmarked for this recipe.

The first training run downloads ``antmaze-large-navigate-v0.npz`` and its validation split into ``~/.ogbench/data``. Together they occupy about 255 MiB (267 MB). To reuse a cache on another disk, add ``data.dataset_dir=/path/to/ogbench-data`` to each launch command. Evaluation uses the task environment; its experience is not added to the training dataset.

With the default model, a checkpoint is about 146 MiB. Four checkpoints per seed occupy about 583 MiB. Reserve at least 1 GiB per seed for data, checkpoints, and logs, in addition to the Python environment and package cache. The runs reported below took about 8–11 hours per seed on shared A800 GPUs, including intermediate evaluations but excluding the final evaluation. Runtime depends on hardware and GPU load.

Run It
------

Launch one seed with the standard offline launcher:

.. code-block:: bash

   bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
     actor.seed=10001

The default schedule is 1,000,000 updates at batch size 256, with 50 evaluation episodes every 50,000 updates and a checkpoint every 250,000 updates. The launcher writes the console log to ``logs/<timestamp>-ogbench_qam_offline/run_offline_rl.log`` and prints training and evaluation metrics there.

For three independent seeds, run these commands sequentially:

.. code-block:: bash

   for seed in 10001 20002 30003; do
     bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
       actor.seed="$seed" \
       runner.logger.experiment_name="qam-antmaze-$seed"
   done

Each launch gets a separate log directory. To place a run on GPU 1, add ``cluster.component_placement.actor=1-1``, ``cluster.component_placement.env=1-1``, and ``cluster.component_placement.rollout=1-1``. Use a different available GPU for each concurrent run.

Resume and Evaluation
---------------------

For the single-seed command above, checkpoints are under ``logs/<timestamp>-ogbench_qam_offline/ogbench_qam_offline/checkpoints/``. Replace the path below with that run's step-250000 directory, then resume:

.. code-block:: bash

   QAM_CHECKPOINT=/absolute/path/to/checkpoints/global_step_250000
   bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
     actor.seed=10001 \
     runner.resume_dir="$QAM_CHECKPOINT"

Keep the original seed, model dimensions and architecture, optimizer settings, dataset, and batch size. Repeat any overrides from the original command, especially for a small-model test. The checkpoint restores model parameters, optimizer state, the training noise generator, and the dataset sampler. Choose a final step greater than the saved step to continue training. Evaluation randomness starts a fresh stream on resume; exact training continuation does not imply identical evaluation episodes.

Evaluate that checkpoint without training by adding ``runner.only_eval=true``:

.. code-block:: bash

   bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
     actor.seed=10001 \
     runner.resume_dir="$QAM_CHECKPOINT" \
     runner.only_eval=true

This completes 50 evaluation episodes and writes evaluation metrics at the checkpoint's step in a new log directory. A completed run records ``eval/num_trajectories=50``.

To evaluate two environments in parallel through the standard EnvWorker and RolloutWorker, use:

.. code-block:: bash

   bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
     actor.seed=10001 \
     runner.resume_dir="$QAM_CHECKPOINT" \
     runner.only_eval=true \
     env.eval.total_num_envs=2 \
     env.eval.early_stop_on_done=false

With one evaluation worker, each evaluation epoch runs one episode in each environment. The default ``runner.eval_episodes=50`` therefore completes 100 episodes and records ``eval/num_trajectories=100``. The worker continues until its configured episode limit so that an early success in one environment does not cut off the other.

Visualization and Results
-------------------------

Open TensorBoard to compare the independent runs:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

Each run stores events and its resolved config in ``<log_directory>/tensorboard/``. The following tags show training progress and evaluation results; see :doc:`Training metrics <../../reference/metrics>` for the shared logging conventions.

.. list-table::
   :header-rows: 1

   * - Metric
     - What to Watch
   * - ``train/critic_loss``, ``train/flow_loss``, ``train/adj_loss``
     - Finite losses from the critic, slow policy, and fast policy updates
   * - ``eval/success``
     - Fraction of successful episodes
   * - ``eval/return``, ``eval/episode_len``
     - Mean raw episode return and length
   * - ``eval/num_trajectories``
     - Completed evaluation episodes

The author reported the following offline runs in `PR #1811 <https://github.com/RLinf/RLinf/pull/1811>`_, using the offline integration at commit ``569ee9c`` on A800 GPUs. Each seed reached 1,000,000 updates with the default AntMaze model and was evaluated over 50 episodes. These results have not been revalidated for later revisions.

.. list-table::
   :header-rows: 1

   * - Seed
     - Success
     - Mean Episode Length
   * - 10001
     - 0.96
     - 531.38
   * - 20002
     - 0.74
     - 730.88
   * - 30003
     - 0.92
     - 545.38
   * - Mean ± Sample Standard Deviation
     - 0.8733 ± 0.1172
     - —
