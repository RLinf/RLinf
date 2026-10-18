Bench2Dex
=========

Train and evaluate π₀.₅ on Bench2Dex Task64 with two Panda arms and Allegro
hands. Prepare the scene assets and converted checkpoint, choose synchronous
PPO, asynchronous PPO or GRPO, then evaluate checkpoints on the four official
generalization channels.

Overview
--------

Use the Task64 recipe with one or more independent scene processes per
environment worker. ``Bench2DexVectorEnv`` extends RLinf's ``BaseVectorEnv``
to schedule complete action chunks across those processes, and
the rollout worker receives their observations as a batch. The simulator is
registered as ``env_type: bench2dex``.

Each ``Bench2DexSubprocEnvWorker`` starts one scene with a local ``spawn``
context. Its ``Bench2DexRuntime`` owns and closes that scene's
``SimulationApp``. The vector environment sends a complete chunk to every
active scene before collecting results. Each scene executes its actions in
order and captures an observation after each executed action. The outer
environment batches the observations and pads scenes that finish early.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      Synchronous PPO · asynchronous PPO · GRPO · policy evaluation

   .. grid-item-card:: Tasks
      :text-align: center

      Task64 · dual Panda + Allegro

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPU

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Scene
     - Robot
     - Goal
   * - ``64_sports_ball_cup_sort``
     - ``multi_panda_with_allegro``
     - Sort four balls into two cups.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Contract
   * - Observation
     - Stereo pair, two wrist RGB images, joint positions and task description.
   * - Action
     - Absolute active-joint targets in the configured robot's joint order.
   * - Reward
     - 1 at the first stable success; 0 otherwise.
   * - Prompt
     - Task description from the selected scene YAML.

Installation
------------

Install OpenPI and the scene runtime in a dedicated virtual environment.
The recipe pins Bench2Dex ``fd90dcc625b66c2e535a8b25bcf75bc81c7b080d``,
Isaac Sim 5.1.0, Isaac Lab v2.3.2, Python 3.11.14 and torch 2.7.0 with CUDA 12.8.

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env bench2dex \
     --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
   source .venv-bench2dex-openpi/bin/activate

Activation exports ``BENCH2DEX_PATH``. The pip-installed Isaac Sim package
resolves its runtime paths. A standalone installation can instead supply
``ISAAC_PATH``; the launchers derive its ``apps/`` and ``kit/`` paths.

Download the Assets
-------------------

Prepare the robot and scene assets from
`Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`_ and the
Task64 replay-generalization anchors from
`teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`_.
Follow the `upstream extraction guide <https://bench2dex.github.io/doc/>`_,
then point the recipe at the extracted directories:

.. code-block:: bash

   export BENCH2DEX_ASSETS_PATH=/path/to/bench2dex-data
   export BENCH2DEX_ANCHOR_DIR=/path/to/task64/replay-generalization

The asset parent contains ``dex2bench_dataset/`` and any referenced
``Dex2Assets/`` folder. Anchored channels restore the HDF5 scene and read
live sensor observations.

Download the Model
------------------

Download the official Task64 π₀.₅ checkpoint using the upstream guide.
Convert its JAX step directory, including ``params/`` and the matching
``assets/dex2bench/norm_stats.json``, to RLinf's PyTorch layout:

.. code-block:: bash

   export BENCH2DEX_PI05_PATH=/path/to/task64-pi05-rlinf
   JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/official-pi05-step \
     --input-norm-stats /path/to/official-pi05-step/assets/dex2bench/norm_stats.json \
     --output-model "$BENCH2DEX_PI05_PATH" \
     --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
     --action-dim 32 --action-horizon 20 --max-token-len 280

The recipe loads the converted ``model.safetensors`` and normalization
file. Task64 uses four policy views, 32 active-joint coordinates, a 20-action
chunk and 280 prompt tokens. The aligned evaluator creates the official
five-camera rig, including the overhead view, and captures a preflight frame
before the first policy observation.

Run It
------

Train with PPO
~~~~~~~~~~~~~~

Use the same Task64 checkpoint and normalization file for both training
recipes. Training resamples all scene axes from the ``seen`` asset split,
using its own seed sequence. The official evaluation anchors and ``unseen``
split remain configured separately.

Start synchronous PPO:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task64_ppo_openpi_pi05

This runs ``EmbodiedRunner``: collect a rollout, update the actor and value
head, synchronize weights, then collect the next rollout. The model uses
Flow-SDE exploration with GPU-generated PyTorch noise and trains the action
expert and value head.

To overlap rollout collection with actor updates, use the asynchronous recipe:

.. code-block:: bash

   bash examples/embodiment/run_async.sh bench2dex_task64_async_ppo_openpi_pi05

This uses ``AsyncPPOEmbodiedRunner`` and the existing
``decoupled_actor_critic`` loss. The asynchronous YAML inherits the synchronous
recipe, keeping the model, scene distribution, rollout budget and optimizer
settings identical. It allows one version of policy staleness through
``algorithm.staleness_threshold: 1`` and uses
``algorithm.behave_weight_threshold: 2.0``.

Both recipes start with 16 environments and one rollout per training epoch.
Each episode has a 665-control-step budget, scheduled as 34 chunks of 20
actions. Set ``env.train.total_num_envs`` and
``cluster.component_placement`` in the YAML to choose the scene and worker
counts. The initial actor microbatch is 1 and the global batch is 136.
Each rollout schedules 544 chunk samples.

The training environment uses ``auto_reset=false``. An episode that ends
inside a chunk contributes its executed rewards; subsequent chunks hold the
terminal observation and zero rewards. RLinf's existing done-based loss mask
excludes those subsequent chunks. Both PPO variants use chunk-level rewards
and log-probabilities.

The runner saves checkpoints according to ``runner.save_interval``. To resume
synchronous training from a saved step, pass ``runner.resume_dir`` through the
Python entry point:

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
   python "$EMBODIED_PATH/train_embodied_agent.py" \
     --config-path "$EMBODIED_PATH/config" \
     --config-name bench2dex_task64_ppo_openpi_pi05 \
     runner.resume_dir=/path/to/checkpoints/global_step_40

For asynchronous resume, use ``train_async.py`` and
``bench2dex_task64_async_ppo_openpi_pi05`` with the same ``runner.resume_dir``
argument. Both recipes use independent checkpoint evaluation below. Watch
``env/success_once`` during training; see
:doc:`Training metrics <../../reference/metrics>` for the metric definitions.

Train with GRPO
~~~~~~~~~~~~~~~

Use GRPO to compare independent attempts from the same initial scene. The
recipe keeps PPO's checkpoint, scene distribution and action budget, trains
the action expert with Flow-SDE, and computes advantages from group returns
instead of a value head:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task64_grpo_openpi_pi05

The recipe uses ``EmbodiedRunner`` with ``algorithm.adv_type=grpo`` and
``algorithm.loss_type=actor``. The default placement uses four GPUs (``0-3``); its
16 environments form four groups of four attempts. Set ``algorithm.group_size`` to change the group size;
``env.train.group_size`` follows it. Each environment and actor worker must
receive complete groups.

With ``env.train.use_fixed_reset_state_ids=true``,
``Bench2DexEnv.update_reset_state_ids()`` selects one initial-state index per
group for the next rollout. ``reset()`` sends the repeated indices through
``Bench2DexVectorEnv`` to ``Bench2DexRuntime.reset(reset_state_id=...)``.
Group members use the same anchor and scene seed while the policy samples
independent PyTorch action noise. Scene process IDs remain independent of
initial-state IDs. PPO and evaluation retain their default episode sequence.

Rewards remain 1 at first stable success and 0 otherwise. A group with both
successes and failures supplies relative advantages; uniform-return groups
have zero group-relative advantages. The recipe keeps
``algorithm.filter_rewards=false`` and enables
``algorithm.log_group_success_metrics``. Use the checkpoint evaluation below
for saved policies.

Evaluate a Checkpoint
~~~~~~~~~~~~~~~~~~~~~

Run 50 episodes per channel, with a process restart every 10 episodes:

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task64_openpi_pi05_eval --episodes-per-process 10

To evaluate a training checkpoint, select its weights and keep the SFT
normalization file:

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task64_openpi_pi05_eval --episodes-per-process 10 \
     rollout.model.model_path=/path/to/checkpoints/global_step_40/actor/model_state_dict/full_weights.pt \
     +rollout.model.openpi_data.norm_stats_path="$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

The OpenPI model loader restores the action network for inference and uses the
specified SFT normalization file. Keep ``BENCH2DEX_PI05_PATH`` pointing at the
converted SFT directory.

The launcher runs ``none``, ``cov``, ``inv`` and ``inv_cov`` in
sequence, sharing GPU 0 between simulation and inference. Select another card
with ``--gpu-id``, an output directory with ``--output-root``, or a subset
with ``--channels``. ``--num-episodes`` controls the total episode count per
channel. ``--episodes-per-process`` caps that count per launcher run; zero
keeps the channel in a single run when the environment count divides it.

To run two scenes concurrently in the same environment worker, use:

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task64_openpi_pi05_eval \
     --num-envs 2 --num-episodes 50 --episodes-per-process 20

This command runs 20, 20 and 10 episodes in three launcher runs. Each scene
executes 10, 10 and 5 episodes respectively. The launcher maps ``--num-envs``
to ``env.eval.total_num_envs`` and computes ``env.eval.rollout_epoch`` from
the episode budget. It reduces the last batch's environment count as needed
to execute exactly the requested episodes. Scene 0 takes episodes 1, 3, 5,
and scene 1 takes 2, 4, 6; the official episode seeds and anchor order are
preserved.

The environment returns observations with a leading environment dimension
and chunk outcomes shaped ``[N,C]``. With ``auto_reset=true``, only completed
scenes reset after the chunk; ``final_observation`` and ``final_info`` retain
their terminal data. The evaluation recipe uses ``auto_reset=false``.

The recipe selects ``rollout.model.openpi.noise_rng: jax`` to generate
CPU noise from each episode seed and policy query index. Model inference
continues to use PyTorch on the GPU. Simulator initialization applies the task
seed before creating the simulation context and uses the native physics time
step ``0.0166666``.

.. list-table:: Official Channel Protocol
   :header-rows: 1

   * - Channel
     - Scene preparation
     - Seed namespace
   * - ``none``
     - Restore the anchor scene.
     - ``100000000``
   * - ``cov``
     - Restore the anchor; resample object pose and table height.
     - ``100000000``
   * - ``inv``
     - Restore the anchor; resample background, table surface, lighting, clutter and camera.
     - ``100000000``
   * - ``inv_cov``
     - Sample all seven axes from the unseen split.
     - ``200000000``

Episode seeds follow
``namespace + task_id * 100000 + episode_index - 1``. Task64 starts at
``106400000`` or ``206400000``. Anchors cycle by episode index using the
official ordering. Task64 has a 665-control-step budget; the recipe schedules
680 action slots and stops at the task budget.

To select another task or robot, set the task, embodiment, checkpoint, anchor
directory and step budgets together. The environment derives joint dimensions
from the robot specification.

Visualization and Results
-------------------------

Read ``success_rates.tsv`` in the output root for the four channel rates.
Each channel writes native ``per_episode.jsonl``, ``per_task.json`` and
``summary.json``, plus RLinf logs and TensorBoard events. Successful
reach-and-stop episodes use the upstream stable-success criterion.
Each environment worker writes its scene results through one writer. Runs
with multiple environment workers keep separate ``shards/<id>/`` directories;
the four-channel launcher merges them by episode index and checks for missing
or duplicate episodes.
For the shared evaluation commands and output layout, see
:doc:`Evaluation <../../evaluations/index>`.
