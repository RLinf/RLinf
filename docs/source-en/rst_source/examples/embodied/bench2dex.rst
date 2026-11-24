Bench2Dex
=========

Train and evaluate π₀.₅ policies in parallel Bench2Dex environments. The
examples use Task22 and Task64 and cover all four evaluation channels.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · asynchronous PPO · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Task22 · Task64

   .. grid-item-card:: Hardware
      :text-align: center

      8 × NVIDIA A800 for Task22

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Scene
     - Robot
     - Action Dimension
   * - ``22_tool_box_loading``
     - ``multi_panda_with_orca``
     - 40
   * - ``64_sports_ball_cup_sort``
     - ``multi_panda_with_allegro``
     - 32

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - Observation
     - Stereo and wrist RGB images, with joint positions.
   * - Action
     - Absolute joint position targets.
   * - Reward
     - 1 for success and 0 otherwise.
   * - Prompt
     - Task description from the scene.

Installation
------------

Install the Bench2Dex and OpenPI dependencies, then activate the environment:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env bench2dex \
     --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
   source .venv-bench2dex-openpi/bin/activate

Download the Assets
-------------------

Download `Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`_ and the
task's scene anchors from
`teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`_. Extract
them and set their paths:

.. code-block:: bash

   export BENCH2DEX_ASSETS_PATH="<asset_parent>"
   export BENCH2DEX_ANCHOR_DIR="<task_anchor_directory>"

``<asset_parent>`` contains ``dex2bench_dataset/`` and the extracted asset
folders. ``<task_anchor_directory>`` contains the selected task's HDF5 anchors.

Download the Model
------------------

Download the task's π₀.₅ checkpoint from
`Bench2Dex/policy_ckpt <https://huggingface.co/Bench2Dex/policy_ckpt>`_, then
convert it to PyTorch format. For Task22, run:

.. code-block:: bash

   export BENCH2DEX_PI05_PATH="<converted_checkpoint_directory>"
   JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model "<official_step_directory>" \
     --input-norm-stats "<official_step_directory>/assets/dex2bench/norm_stats.json" \
     --output-model "$BENCH2DEX_PI05_PATH" \
     --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
     --action-dim 40 --action-horizon 20 --max-token-len 280

Replace ``<official_step_directory>`` with the downloaded step directory and
``<converted_checkpoint_directory>`` with the output directory. For Task64,
use its checkpoint and set ``--action-dim 32``.

Run It
------

Train a Policy
~~~~~~~~~~~~~~

The Task22 PPO recipe requires a ``full_weights.pt`` checkpoint prepared by
ten epochs of critic warmup. Set its path and launch training.

.. code-block:: bash

   export BENCH2DEX_WARMUP10_PATH="<warmup10_full_weights.pt>"
   bash examples/embodiment/run_embodiment.sh bench2dex_task22_ppo_openpi_pi05

The recipe trains for 190 epochs in a fixed white-background scene. It
evaluates every ten epochs and saves a checkpoint every two.

During rollout, the policy applies Flow-SDE at one randomly selected step
among the first five denoising steps.

For another training configuration, replace the final argument with a name
from this table. Asynchronous PPO uses ``examples/embodiment/run_async.sh``.

.. list-table::
   :header-rows: 1

   * - Task
     - Algorithm
     - Configuration
   * - Task22
     - Asynchronous PPO
     - ``bench2dex_task22_async_ppo_openpi_pi05``
   * - Task64
     - PPO
     - ``bench2dex_task64_ppo_openpi_pi05``
   * - Task64
     - Asynchronous PPO
     - ``bench2dex_task64_async_ppo_openpi_pi05``
   * - Task64
     - GRPO
     - ``bench2dex_task64_grpo_openpi_pi05``

To run another Bench2Dex task, choose its scene and robot in the environment
config. Use the matching checkpoint, normalization statistics, and scene
anchors. Set the action dimension to the number of joints the policy controls
and ``max_episode_steps`` to the task's control-step budget.

Edit training settings in the selected YAML under
``examples/embodiment/config/``. See
:doc:`Placement <../../concepts/placement>` to change GPU allocation and
:doc:`Checkpoint resume <../../guides/resume>` to continue a saved run.

.. note::

   To record video, set ``skip_intermediate_renders=false`` in the environment
   configuration where recording is enabled.

Evaluate a Policy
~~~~~~~~~~~~~~~~~

Evaluate Task22 on all four channels, with 32 episodes per channel on eight GPUs:

.. code-block:: bash

   python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task22_openpi_pi05_eval --output-root "<evaluation_output>" \
     --gpu-id 0,1,2,3,4,5,6,7 --num-envs 32 --num-episodes 32 \
     "rollout.model.model_path=<model_path>" \
     "+rollout.model.openpi_data.norm_stats_path=$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

Set ``<model_path>`` to the converted model directory or a trained checkpoint's
``global_step_N/actor/model_state_dict/full_weights.pt`` file. For Task64, use
``bench2dex_task64_openpi_pi05_eval`` with its model and anchors.

.. list-table:: Evaluation Channels
   :header-rows: 1

   * - Channel
     - Scene Variation
   * - ``none``
     - Original anchor scene.
   * - ``cov``
     - Object poses and table height.
   * - ``inv``
     - Appearance, clutter and camera settings.
   * - ``inv_cov``
     - Combined variations with unseen assets.

Visualization and Results
-------------------------

Training logs and checkpoints are written under ``logs/``. View training
progress with TensorBoard:

.. code-block:: bash

   tensorboard --logdir logs

``env/success_once`` reports the training success rate. Evaluation results are
summarized in ``<evaluation_output>/success_rates.tsv``, with per-episode
records in each channel's ``per_episode.jsonl``. See
:doc:`Training metrics <../../reference/metrics>` for metric definitions and
:doc:`Evaluation <../../evaluations/index>` for evaluation workflows.
