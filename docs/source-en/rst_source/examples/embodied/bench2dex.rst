Bench2Dex
=========

Train and evaluate π₀.₅ policies with RLinf in parallel `Bench2Dex <https://github.com/Bench2Dex/Bench2Dex>`__ Isaac Sim dexterous manipulation environments. Bench2Dex evaluates policies across four generalization channels (``none``, ``cov``, ``inv``, ``inv_cov``); this page walks through Task22 and Task64 as reference setups while supporting other Bench2Dex scenes and robots through configuration.

Overview
--------

The policy consumes stereo and wrist RGB images together with proprioceptive joint positions to predict absolute joint targets.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · Async PPO · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Task22 · Task64

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPUs

| **You'll do:** install deps → download assets, anchors & convert π₀.₅ weights → run critic warmup & PPO → evaluate on four channels → monitor ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · NVIDIA GPUs (8 × 80 GB GPUs for Task22 training and 8-GPU evaluation; 4 × 80 GB GPUs for Task64 training and evaluation).

Tasks, Observation, and Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference recipes cover two bimanual dexterous tasks:

.. list-table::
   :header-rows: 1
   :widths: 34 38 28

   * - Scene
     - Robot
     - Action Dimension
   * - ``22_tool_box_loading``
     - ``multi_panda_with_orca``
     - 40
   * - ``64_sports_ball_cup_sort``
     - ``multi_panda_with_allegro``
     - 32

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Stereo and wrist RGB images plus active joint positions.
   * - Action
     - Absolute joint position targets matching the robot's active joints.
   * - Reward
     - Sparse 0/1 success reward.
   * - Prompt
     - Task instruction from the Bench2Dex scene definition.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Task22 (Critic Warmup + PPO)
      :sync: bench2dex-task22

      Runs 10 epochs of critic warmup (1,400 optimizer steps) from the converted π₀.₅ weights, then launches synchronous PPO for 190 epochs on 8 × 80 GB GPUs in the fixed white-background scene (evaluating and saving a checkpoint every 10 epochs).

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env bench2dex \
           --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
         source .venv-bench2dex-openpi/bin/activate

      **2. Download Assets, Anchors & Convert Weights**

      Download `Bench2Dex/Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`__, Task22 anchors from `Bench2Dex/teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`__, and the official π₀.₅ checkpoint from `Bench2Dex/policy_ckpt <https://huggingface.co/Bench2Dex/policy_ckpt>`__, then convert the checkpoint to PyTorch format. All Task22 training and evaluation configs read ``BENCH2DEX_ASSETS_PATH``, ``BENCH2DEX_ANCHOR_DIR``, and ``BENCH2DEX_PI05_PATH`` directly:

      .. code-block:: bash

         hf download Bench2Dex/Assets --repo-type dataset \
           --local-dir ./data/bench2dex/assets
         hf download Bench2Dex/teleopdata --repo-type dataset \
           --include "dataset/22_tool_box_loading/origin-generalization/**" \
           --local-dir ./data/bench2dex/teleopdata
         hf download Bench2Dex/policy_ckpt \
           --include "22/multi_panda_with_orca/pi05/pi05_base_dex2bench_full/22_pi05_256/2000/**" \
           --local-dir ./data/bench2dex/jax

      Export the local paths and convert the downloaded checkpoint to PyTorch format:

      .. code-block:: bash

         export BENCH2DEX_ASSETS_PATH="$PWD/data/bench2dex/assets"
         export BENCH2DEX_ANCHOR_DIR="$PWD/data/bench2dex/teleopdata/dataset/22_tool_box_loading/origin-generalization"
         export BENCH2DEX_PI05_PATH="$PWD/data/bench2dex/pi05_task22"
         CKPT_SRC=./data/bench2dex/jax/22/multi_panda_with_orca/pi05/pi05_base_dex2bench_full/22_pi05_256/2000
         JAX_PLATFORMS=cpu python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model "$CKPT_SRC" --input-norm-stats "$CKPT_SRC/assets/dex2bench/norm_stats.json" \
           --output-model "$BENCH2DEX_PI05_PATH" \
           --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
           --action-dim 40 --action-horizon 20 --max-token-len 280

      **3. Launch Critic Warmup & PPO**

      Warm up the Task22 critic for 10 epochs (1,400 optimizer updates) using the converted π₀.₅ weights:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh bench2dex_task22_critic_warmup_openpi_pi05 \
           runner.logger.log_path="$PWD/results/task22-warmup"

      After warmup completes, start PPO from the saved local ``full_weights.pt`` with a fresh optimizer:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh bench2dex_task22_ppo_openpi_pi05 \
           runner.ckpt_path="$PWD/results/task22-warmup/bench2dex_task22_critic_warmup_openpi_pi05/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"

      **4. Evaluate on Four Channels**

      Evaluate Task22 across all four generalization channels (8 GPUs, 32 episodes per channel). Pass either ``$BENCH2DEX_PI05_PATH`` or a trained checkpoint's ``full_weights.pt`` to ``rollout.model.model_path``:

      .. code-block:: bash

         python -m evaluations.bench2dex.run_all_channels \
           bench2dex_task22_openpi_pi05_eval --output-root ./results/task22-eval \
           --gpu-id 0,1,2,3,4,5,6,7 --num-envs 32 --num-episodes 32 \
           "rollout.model.model_path=$BENCH2DEX_PI05_PATH" \
           "+rollout.model.openpi_data.norm_stats_path=$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

   .. tab-item:: Task64 (PPO)
      :sync: bench2dex-task64

      Trains or evaluates π₀.₅ on ``64_sports_ball_cup_sort`` with dual Panda arms and Allegro hands (32-dim action space, 4 × 80 GB GPUs, 16 parallel environments).

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env bench2dex \
           --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
         source .venv-bench2dex-openpi/bin/activate

      **2. Download Assets, Anchors & Convert Weights**

      Download `Bench2Dex/Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`__, Task64 anchors from `Bench2Dex/teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`__, and the Task64 π₀.₅ checkpoint from `Bench2Dex/policy_ckpt <https://huggingface.co/Bench2Dex/policy_ckpt>`__, then convert to PyTorch format. All Task64 training and evaluation configs read ``BENCH2DEX_ASSETS_PATH``, ``BENCH2DEX_ANCHOR_DIR``, and ``BENCH2DEX_PI05_PATH`` directly:

      .. code-block:: bash

         hf download Bench2Dex/Assets --repo-type dataset \
           --local-dir ./data/bench2dex/assets
         hf download Bench2Dex/teleopdata --repo-type dataset \
           --include "dataset/64_sports_ball_cup_sort/origin-generalization/**" \
           --local-dir ./data/bench2dex/teleopdata
         hf download Bench2Dex/policy_ckpt \
           --include "64/multi_panda_with_allegro/pi05/pi05_base_dex2bench_full/64_pi05_256/2000/**" \
           --local-dir ./data/bench2dex/jax

      Export the local paths and convert the downloaded checkpoint to PyTorch format:

      .. code-block:: bash

         export BENCH2DEX_ASSETS_PATH="$PWD/data/bench2dex/assets"
         export BENCH2DEX_ANCHOR_DIR="$PWD/data/bench2dex/teleopdata/dataset/64_sports_ball_cup_sort/origin-generalization"
         export BENCH2DEX_PI05_PATH="$PWD/data/bench2dex/pi05_task64"
         CKPT_SRC=./data/bench2dex/jax/64/multi_panda_with_allegro/pi05/pi05_base_dex2bench_full/64_pi05_256/2000
         JAX_PLATFORMS=cpu python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model "$CKPT_SRC" --input-norm-stats "$CKPT_SRC/assets/dex2bench/norm_stats.json" \
           --output-model "$BENCH2DEX_PI05_PATH" \
           --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
           --action-dim 32 --action-horizon 20 --max-token-len 280

      **3. Launch PPO**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh bench2dex_task64_ppo_openpi_pi05

      **4. Evaluate on Four Channels**

      Evaluate Task64 across all four channels on 4 GPUs with 16 parallel environments and 50 episodes per channel:

      .. code-block:: bash

         python -m evaluations.bench2dex.run_all_channels \
           bench2dex_task64_openpi_pi05_eval --output-root ./results/task64-eval \
           --gpu-id 0,1,2,3 --num-envs 16 --num-episodes 50 \
           "rollout.model.model_path=$BENCH2DEX_PI05_PATH" \
           "+rollout.model.openpi_data.norm_stats_path=$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

.. list-table:: Evaluation Channels
   :header-rows: 1
   :widths: 20 80

   * - Channel
     - Scene Variation
   * - ``none``
     - Original anchor scene.
   * - ``cov``
     - Object poses and table height.
   * - ``inv``
     - Appearance, clutter, and camera settings.
   * - ``inv_cov``
     - Combined variations with unseen assets.

.. dropdown:: Other training configs (Async PPO & GRPO)

   Replace the config name in the launch command to run asynchronous PPO (via ``examples/embodiment/run_async.sh``) or GRPO (via ``examples/embodiment/run_embodiment.sh``). Each config reads ``BENCH2DEX_PI05_PATH`` directly:

   .. list-table::
      :header-rows: 1

      * - Task
        - Algorithm
        - Config Name
      * - Task22
        - Asynchronous PPO
        - ``bench2dex_task22_async_ppo_openpi_pi05``
      * - Task64
        - Asynchronous PPO
        - ``bench2dex_task64_async_ppo_openpi_pi05``
      * - Task64
        - GRPO
        - ``bench2dex_task64_grpo_openpi_pi05``

.. dropdown:: Adapting to other Bench2Dex tasks & recording video (optional)

   To run any other Bench2Dex task and robot embodiment, override the task-specific Hydra keys and supply matching anchors and converted weights:

   .. list-table::
      :header-rows: 1
      :widths: 44 56

      * - Hydra Key
        - Purpose
      * - ``env.train.init_params.task`` / ``env.train.init_params.embodiment``
        - Select the Bench2Dex scene name and robot embodiment (mirror on ``env.eval`` when validating).
      * - ``env.train.init_params.anchor_dir``
        - Path to the task's HDF5 scene anchors.
      * - ``actor.model.action_dim`` / ``actor.model.openpi.model_action_dim``
        - Number of active robot joints controlled by the policy.
      * - ``env.train.max_episode_steps`` / ``env.train.max_steps_per_rollout_epoch``
        - Task control-step limit and rollout budget rounded up to a multiple of ``num_action_chunks``.
      * - ``env.train.skip_intermediate_renders`` / ``env.eval.skip_intermediate_renders``
        - Set to ``false`` whenever ``video_cfg.save_video=true`` is enabled.

   See :doc:`Placement </rst_source/concepts/placement>` to adjust GPU placement and :doc:`Checkpoint Resume </rst_source/guides/resume>` to resume from a saved run.

Visualization and Results
-------------------------

Track ``env/success_once`` and ``eval/success_once`` with ``tensorboard --logdir ./results --port 6006``; see :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions. Multi-channel evaluation writes summary success rates to ``success_rates.tsv`` and per-episode records to each channel's ``per_episode.jsonl`` (see :doc:`Evaluation </rst_source/evaluations/index>`).

Task22 Fixed White-Background PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Synchronous PPO on Task22 (22 epochs, 128 training episodes per epoch, 32-episode independent evaluations after epochs 10 and 20, starting from warmup weights with a fresh optimizer):

.. list-table::
   :header-rows: 1
   :widths: 38 20 42

   * - Split
     - Epoch
     - Success Rate
   * - Training rollout
     - 1
     - 86.72% (111/128)
   * - Training rollout
     - 22
     - 96.88% (124/128)
   * - Independent evaluation
     - 10
     - 96.88% (31/32)
   * - Independent evaluation
     - 20
     - 100.00% (32/32)

Task64 Evaluation and Throughput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluated on NVIDIA A800 80 GB GPUs using ``64_sports_ball_cup_sort`` with dual Panda/Allegro (50 episodes per channel for success rate across 3 runs on 4 GPUs / 16 environments; end-to-end physics throughput vs. single-GPU official implementation):

.. list-table::
   :header-rows: 1
   :widths: 16 20 24 22 18

   * - Channel
     - Official π₀.₅
     - RLinf π₀.₅ (3-run mean)
     - RLinf Throughput (4 GPUs / 16 envs)
     - Speedup vs. 1-GPU Official
   * - ``none``
     - 42.00%
     - 34.67%
     - 65.30 steps/s
     - 5.47×
   * - ``cov``
     - 18.00%
     - 16.67%
     - 65.89 steps/s
     - 5.78×
   * - ``inv``
     - 4.00%
     - 16.00%
     - 54.52 steps/s
     - 4.59×
   * - ``inv_cov``
     - 22.00%
     - 18.67%
     - 65.80 steps/s
     - 5.56×
   * - **Overall**
     - **21.50%**
     - **21.33%**
     - —
     - —
