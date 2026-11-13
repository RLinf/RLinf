Bench2Dex
=========

Train and evaluate π₀.₅ on Bench2Dex Task22 and Task64. Run synchronous or
asynchronous PPO, use grouped GRPO for Task64, and evaluate saved policies
on the four official generalization channels.

Overview
--------

Run independent scenes in parallel and batch their policy observations.
Choose the scene count and GPU placement in the selected recipe. Training
uses complete episodes, including zero-reward padding after a scene ends.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      Synchronous PPO · asynchronous PPO · GRPO · evaluation

   .. grid-item-card:: Tasks
      :text-align: center

      Task22 · Task64

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPUs with RGB rendering

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Scene
     - Robot
     - Active Joints
     - Control Budget
   * - ``22_tool_box_loading``
     - ``multi_panda_with_orca``
     - 40
     - 681
   * - ``64_sports_ball_cup_sort``
     - ``multi_panda_with_allegro``
     - 32
     - 665

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Contract
   * - Observation
     - Stereo and wrist RGB images, joint positions and a task description.
   * - Action
     - Absolute joint targets in the selected robot's active-joint order.
   * - Reward
     - 1 at the first stable success; 0 otherwise. Stage bonuses default to 0.
   * - Prompt
     - Task description from the selected scene.

Installation
------------

Install OpenPI and the scene runtime in a dedicated virtual environment.
Use Bench2Dex commit ``fd90dcc625b66c2e535a8b25bcf75bc81c7b080d``,
Isaac Sim 5.1.0, Isaac Lab v2.3.2, Python 3.11.14 and torch 2.7.0 with CUDA 12.8.

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env bench2dex \
     --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
   source .venv-bench2dex-openpi/bin/activate

Activation sets ``BENCH2DEX_PATH``. Pip-installed Isaac Sim resolves its own
runtime paths. For a standalone installation, set ``ISAAC_PATH``; the
launchers derive ``EXP_PATH`` and ``CARB_APP_PATH`` while preserving explicit
values for either variable.

Download the Assets
-------------------

Download robot and scene assets from
`Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`_ and the selected
task's replay-generalization anchors from
`teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`_. Follow
the `upstream extraction guide <https://bench2dex.github.io/doc/>`_, then set
the extracted asset parent and anchor directory:

.. code-block:: bash

   export BENCH2DEX_ASSETS_PATH="<asset_parent>"
   export BENCH2DEX_ANCHOR_DIR="<task_anchor_directory>"

Replace the placeholders with your directories. The asset parent contains
``dex2bench_dataset/`` and any referenced ``Dex2Assets/`` folder. Use anchors
for the same task as the checkpoint.

Download the Model
------------------

Download the official checkpoint for the selected task using the upstream
guide. Convert its JAX step directory and matching normalization file. This
command converts Task22's 40-dimensional policy:

.. code-block:: bash

   export BENCH2DEX_PI05_PATH="<converted_checkpoint_directory>"
   JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model "<official_step_directory>" \
     --input-norm-stats "<official_step_directory>/assets/dex2bench/norm_stats.json" \
     --output-model "$BENCH2DEX_PI05_PATH" \
     --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
     --action-dim 40 --action-horizon 20 --max-token-len 280

For Task64, use its checkpoint and anchors and set ``--action-dim 32``.
Both tasks use 20-action chunks, four policy views and 280 prompt tokens.
The simulator also initializes the overhead camera used by the official rig.

Run It
------

Train Task22 with PPO
~~~~~~~~~~~~~~~~~~~~~

Start synchronous PPO with the Task22 checkpoint and anchor directory:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task22_ppo_openpi_pi05

The recipe collects 128 trajectories per round with 16 environments and
8 rollout batches. Each trajectory has 700 scheduled action steps.
Physics runs for at most 681 steps. Each round makes three passes over
the rollout data. Actor and critic learning rates are ``2e-6`` and ``2e-5``.

Task22 saves every 2 completed rounds and evaluates 32 episodes every
10 completed rounds. The schedule uses the cumulative round count after
resume. Training and periodic evaluation use the ``none`` scene channel.
Watch TensorBoard's ``env/success_once`` and ``eval/success_once``; see
:doc:`Training metrics <../../reference/metrics>` for their definitions.

To overlap collection and updates, run the asynchronous recipe:

.. code-block:: bash

   bash examples/embodiment/run_async.sh bench2dex_task22_async_ppo_openpi_pi05

Training recipes enable actor and rollout offload. Synchronous training also
releases simulator processes after complete rollouts and rebuilds them for
the next round. The asynchronous recipe keeps training scenes alive.

.. warning::

   Environment offload requires complete episodes and ``auto_reset=false``.
   Keep training environment offload disabled for asynchronous, pipelined or
   decoupled execution.

Training captures one policy observation after each executed action chunk.
For video, tactile data, per-action observation collection or external reward models, set
``skip_intermediate_renders=false`` on the corresponding environment split.

Select a Fixed Scene and Resume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reuse one scene, set both ``env.train.init_params.anchor_hdf5`` and
``env.eval.init_params.anchor_hdf5`` to the same HDF5 anchor and keep the
``none`` channel. Geometry and appearance come from that anchor; select an
anchor that contains the scene you want.

Resume a saved Task22 training checkpoint with the Python entry point:

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
   python "$EMBODIED_PATH/train_embodied_agent.py" \
     --config-path "$EMBODIED_PATH/config" \
     --config-name bench2dex_task22_ppo_openpi_pi05 \
     runner.resume_dir="<checkpoint_directory>"

Use the complete ``global_step_N`` directory to restore model, optimizer and
scheduler state. For asynchronous resume, select ``train_async.py`` and
``bench2dex_task22_async_ppo_openpi_pi05``.

Train Task64 with PPO or GRPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Switch to the Task64 checkpoint and anchors, then select its PPO recipe:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task64_ppo_openpi_pi05

The asynchronous alternative is
``bench2dex_task64_async_ppo_openpi_pi05``. Task64 training samples all scene
variation axes from the ``seen`` asset split. Its episodes have a 665-step
budget, scheduled in 680 action slots. Configure periodic evaluation with
``runner.val_check_interval`` and saving with ``runner.save_interval``.

Run grouped GRPO to compare four independent attempts from each initial scene:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task64_grpo_openpi_pi05

The 16 environments form four groups. Members of a group share the anchor
and scene seed and use independent action noise. Set ``algorithm.group_size``
to change the group size; every environment and actor shard must receive
complete groups. GRPO uses group-relative returns without a value head.

Evaluate a Checkpoint
~~~~~~~~~~~~~~~~~~~~~

Run the official four channels for Task22, 50 episodes per channel:

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task22_openpi_pi05_eval --episodes-per-process 10

Select ``bench2dex_task64_openpi_pi05_eval`` with Task64 assets and weights
for the other task. The launcher runs ``none``, ``cov``, ``inv`` and
``inv_cov`` in sequence. Use ``--gpu-id`` for placement, ``--num-envs`` for
parallel scenes and ``--output-root`` for the results directory.

To evaluate trained weights, add
``rollout.model.model_path=<checkpoint_directory>/actor/model_state_dict/full_weights.pt``
and ``+rollout.model.openpi_data.norm_stats_path=<sft_normalization_file>``
to the command. Keep the normalization file matched to the original task.

.. list-table:: Official Channel Protocol
   :header-rows: 1

   * - Channel
     - Scene Preparation
     - Seed Namespace
   * - ``none``
     - Restore the anchor scene.
     - ``100000000``
   * - ``cov``
     - Restore the anchor; vary object pose and table height.
     - ``100000000``
   * - ``inv``
     - Restore the anchor; vary appearance, clutter and cameras.
     - ``100000000``
   * - ``inv_cov``
     - Sample all seven variation axes from the unseen split.
     - ``200000000``

Parallel evaluation preserves the official episode order and random seeds.
Official evaluation uses CPU JAX noise indexed by
episode and policy query, PyTorch inference on the GPU, and a camera capture
after each executed action.

Visualization and Results
-------------------------

Read ``success_rates.tsv`` in the four-channel output root. Each channel
writes native ``per_episode.jsonl``, ``per_task.json`` and ``summary.json``
alongside RLinf logs and TensorBoard events. Results from parallel scene
shards are merged by episode index, with duplicate and missing episodes
reported as errors. See :doc:`Evaluation <../../evaluations/index>` for
the shared output layout.
