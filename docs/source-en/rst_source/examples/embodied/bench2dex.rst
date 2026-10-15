Bench2Dex
=========

Evaluate π₀.₅ on Bench2Dex Task64 with two Panda arms and Allegro hands.
Prepare the scene assets, replay anchors and converted checkpoint, then run
the four official generalization channels through RLinf.

Overview
--------

Use the provided Task64 recipe with one scene per environment worker.
The simulator is registered as ``env_type: bench2dex``.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      Policy evaluation

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

Run 50 episodes per channel, with a process restart every 10 episodes:

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task64_openpi_pi05_eval --episodes-per-process 10

The launcher runs ``none``, ``cov``, ``inv`` and ``inv_cov`` in
sequence, sharing GPU 0 between simulation and inference. Select another card
with ``--gpu-id``, an output directory with ``--output-root``, or a subset
with ``--channels``. Setting ``--episodes-per-process 0`` runs a channel
in one process; ``--num-episodes`` controls its total episode count.

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
For the shared evaluation commands and output layout, see
:doc:`Evaluation <../../evaluations/index>`.
