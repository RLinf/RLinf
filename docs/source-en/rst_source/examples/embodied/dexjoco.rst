RL with DexJoCo Benchmark
=========================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   Watering a plant with an Allegro hand (image: `DexJoCo <https://github.com/brave-eai/dexjoco>`__).

`DexJoCo <https://github.com/brave-eai/dexjoco>`__ is a MuJoCo benchmark of
single-arm and dual-arm dexterous manipulation with Franka arms and Allegro
hands. You'll use RLinf to train CNN policies with PPO, or fine-tune π₀.₅ on
the official demonstrations and continue with PPO.

Overview
--------

Train Franka and Allegro policies on DexJoCo tasks from camera images and proprioception.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      CNN · π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · PPO

   .. grid-item-card:: Tasks
      :text-align: center

      6 single-arm · 5 dual-arm

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1–8 GPUs

| **You'll do:** install → download datasets + models → launch training → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the DexJoCo package from the install step · DexJoCo LeRobot datasets · a base checkpoint (steps below).

Tasks
~~~~~

Training and evaluation each run one task. Set ``task_name`` in ``env.train``
and ``env.eval``; the description is the task's official prompt:

.. list-table::
   :header-rows: 1
   :widths: 14 24 12 50

   * - Category
     - Task
     - Time limit (steps)
     - Description
   * - Single-arm
     - ``water_plant``
     - 1000
     - Grasp the watering can and apply water to the plant.
   * - Single-arm
     - ``hammer_nail``
     - 1000
     - Use the hammer to drive the nail into the wooden board.
   * - Single-arm
     - ``fold_glasses``
     - 1200
     - Fold the glasses and place them into the case.
   * - Single-arm
     - ``pinch_tongs``
     - 1000
     - Grasp the tongs and perform three consecutive open-close motions.
   * - Single-arm
     - ``pick_bucket``
     - 1200
     - Place the boxed food into the bucket and then lift the bucket.
   * - Single-arm
     - ``click_mouse``
     - 1200
     - Move the mouse to the purple mouse pad and click the left mouse button.
   * - Dual-arm
     - ``bimanual_assembly``
     - 1500
     - Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole.
   * - Dual-arm
     - ``bimanual_hanoi``
     - 1500
     - Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand.
   * - Dual-arm
     - ``bimanual_microwave_cook``
     - 1100
     - Open the microwave door, place the food inside the microwave, close the door, and press the start button.
   * - Dual-arm
     - ``bimanual_photograph``
     - 1000
     - Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand.
   * - Dual-arm
     - ``bimanual_unlock_ipad``
     - 1200
     - Grasp the iPad and enter the password 123 to unlock the device.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB images from ``camera_names`` at 640×640 (downscaled to 128×128 in the CNN recipe), plus 23-dim proprioception ``[pose7, hand16]`` for single-arm tasks or 46-dim ``[right_pose7, left_pose7, right_hand16, left_hand16]`` for dual-arm tasks; each pose is a position and a scalar-first quaternion. Single-arm tasks use ``[front, wrist]`` (``[ego_right, wrist]`` for ``click_mouse``); dual-arm tasks use ``[ego, wrist_left, wrist_right]``.
   * - Action
     - 22-dim continuous action per arm: absolute 3D end-effector position, 3D rotation vector, and 16 Allegro joint targets; the 44-dim dual-arm action lists the right arm first.
   * - Reward
     - Sparse 0/1 success reward.
   * - Prompt
     - Natural-language instruction for the DexJoCo task, used by π₀.₅.

Installation
------------

Clone RLinf and install the environment for the model you want to run:

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # Mainland China users can add --use-mirror.

   # CNN
   bash requirements/install.sh embodied --env dexjoco
   source .venv/bin/activate

   # OpenPI π₀.₅
   # bash requirements/install.sh embodied --model openpi --env dexjoco
   # source .venv/bin/activate

DexJoCo has no PyPI release, so ``--env dexjoco`` clones the
`DexJoCo repository <https://github.com/brave-eai/dexjoco>`__ into
``.venv/dexjoco`` at commit ``8d23b0f`` and installs its ``dexjoco`` package
with ``uv pip install -e``. Set ``DEXJOCO_PATH`` before installing to clone into,
or reuse, a checkout at that path instead. The package holds the task
environments together with their scene XMLs, meshes, textures, and camera
presets, and ``env_type: dexjoco`` builds every environment from it. To add
DexJoCo to an environment that already has RLinf's dependencies:

.. code:: bash

   git clone https://github.com/brave-eai/dexjoco.git
   git -C dexjoco checkout 8d23b0fab23b17a58c4b55f3942e17013aaf8267
   uv pip install -e dexjoco/dexjoco

Download the Datasets
---------------------

The π₀.₅ recipes fine-tune on the
`official LeRobot demonstrations <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__.
Download one task and compute its normalization statistics:

.. code:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
     --include 'dexjoco_lerobot_datasets/water_plant/**' \
     --local-dir /path/to/dexjoco
   PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_dexjoco \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant

What this does:

1. Downloads the ``water_plant`` episodes with front and wrist videos, robot state, actions, and task text.
2. Writes ``norm_stats.json`` into the dataset directory. Use the same file for SFT, PPO, and evaluation.

For dual-arm tasks, pass ``--config-name pi05_dexjoco_dual_arm``; for
``click_mouse``, whose base camera is ``ego_right``, pass
``--config-name pi05_dexjoco_click_mouse``.

Download the Model
------------------

Download the checkpoint for the model you plan to train.

**CNN**

For the CNN + PPO recipe, download the ResNet checkpoint:

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

Then set the same checkpoint path for rollout and actor in
``examples/embodiment/config/dexjoco_ppo_cnn.yaml``:

.. code-block:: yaml

   rollout:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained
   actor:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained

**OpenPI π₀.₅**

DexJoCo publishes the π₀.₅ base checkpoint together with a variant whose
action projections cover the 44-dim dual-arm action. Download the base and
convert it to the RLinf OpenPI layout, as described in :doc:`sft_openpi`:

.. code-block:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download DexJoCo/DexJoCo-Pi05 --include 'pi05_base/params/**' \
     --local-dir /path/to/jax
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/jax/pi05_base \
     --input-norm-stats /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
     --output-model /path/to/pi05_base_openpi \
     --output-norm-stats /path/to/pi05_base_openpi/dexjoco/norm_stats.json \
     --action-dim 32 --action-horizon 30 --max-token-len 250

For dual-arm tasks, download ``pi05_base_action_dim_44/params/**`` instead
and convert it with ``--action-dim 44`` into
``/path/to/pi05_base_action_dim_44_openpi``.

Run It
------

Pick one recipe and launch training:

.. list-table::
   :header-rows: 1
   :widths: 24 38 20 18

   * - Recipe
     - Config
     - Entrypoint
     - Command suffix
   * - CNN + PPO
     - ``examples/embodiment/config/dexjoco_ppo_cnn.yaml``
     - ``run_embodiment.sh``
     - ``dexjoco_ppo_cnn``
   * - π₀.₅ + SFT
     - ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``
     - ``run_vla_sft.sh``
     - ``dexjoco_sft_openpi_pi05``
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``
     - ``run_embodiment.sh``
     - ``dexjoco_ppo_openpi_pi05``

The recipes train ``water_plant``. For another task, override ``task_name``
and ``camera_names`` in both ``env.train`` and ``env.eval``, and set
``max_episode_steps`` to the task's time limit. For the CNN recipe, also set
``env.eval.max_steps_per_rollout_epoch`` to that limit; for the π₀.₅ recipes,
set ``max_steps_per_rollout_epoch`` in both ``env.train`` and ``env.eval`` to
that limit rounded up to a multiple of 30.
Dual-arm tasks use 46-dim state and 44-dim actions: for the CNN, set
``actor.model.state_dim: 46`` and ``actor.model.action_dim: 44``; for π₀.₅,
apply the dual-arm settings in the SFT section.

These ``init_params`` fields control randomization and task options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``init_params.randomize``
     - ``True`` randomizes lighting, table texture, and the ``front`` camera pose (``ego`` on dual-arm tasks).
   * - ``init_params.randomize_dynamics``
     - ``True`` randomizes object dynamics.
   * - ``init_params.password``
     - Digits ``bimanual_unlock_ipad`` expects, default ``[1, 2, 3]``; the π₀.₅ prompt always names 123.

Launch a config with ``run_embodiment.sh``:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

What this does:

1. Loads ``examples/embodiment/config/dexjoco_ppo_cnn.yaml`` and places the actor, rollout, and env workers on GPU ``0``.
2. Starts 8 training environments that return 128×128 images.
3. Runs PPO on the base camera image and proprioception.

.. admonition:: Configure further
   :class: note

   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - All config keys → :doc:`Configuration <../../guides/index>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`
   - Resuming from a checkpoint → :doc:`Resume <../../guides/resume>`

π₀.₅ + SFT
~~~~~~~~~~

Set these fields in ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Field
     - Value
   * - ``data.train_data_paths``
     - The task dataset directory.
   * - ``actor.model.model_path``
     - The converted ``pi05_base_openpi`` directory.
   * - ``actor.model.openpi_data.norm_stats_path``
     - The dataset's ``norm_stats.json``.

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05

The recipe uses FP32 parameters, BF16 computation, and ``no_shard``; see
:doc:`sft_openpi`. Each actor GPU holds the complete model, gradients, and
optimizer state, so plan for 80 GB-class GPUs. For a single 48 GB GPU, set
``actor.micro_batch_size: 1``, ``actor.global_batch_size: 1``, and
``actor.model.precision: bf16``; this changes parameter and optimizer
precision from the FP32 default. The global batch must be divisible by the
micro batch times the number of actor workers. Training saves
``checkpoints/global_step_<N>`` directories, which the PPO and evaluation
configs load directly.

For a dual-arm task, set ``actor.model.openpi.config_name`` to
``pi05_dexjoco_dual_arm``, set ``actor.model.action_dim`` and
``actor.model.openpi.model_action_dim`` to ``44``, and start from the
converted 44-dim base. For ``click_mouse``, set
``actor.model.openpi.config_name`` to ``pi05_dexjoco_click_mouse``.

π₀.₅ + PPO
~~~~~~~~~~

In ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``, set
``actor.model.model_path`` and ``rollout.model.model_path`` to the SFT
``global_step_<N>`` directory, and set
``actor.model.openpi_data.norm_stats_path`` to the statistics used for SFT.

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

What this does:

1. Loads the SFT weights and adds a value head for PPO.
2. Collects rollouts from 16 parallel environments with 30-action chunks.
3. Updates the policy with PPO.

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with ``dexjoco_openpi_pi05_eval``, which
evaluates 50 episodes as in the official DexJoCo protocol and divides them
across 1, 2, or 5 GPUs. Set
``rollout.model.model_path`` to the SFT or PPO checkpoint and
``rollout.model.openpi_data.norm_stats_path`` to the statistics used for
training; for a dual-arm checkpoint, apply the dual-arm settings under
``rollout.model``.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir logs --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when needed:

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Recipe
     - Reported Behavior
   * - π₀.₅ + SFT
     - On ``water_plant``, the step-5000 checkpoint succeeds in 28 of 50 evaluation episodes (56%) in the setup used for the original run (16 H100 GPUs).
   * - π₀.₅ + PPO
     - From that checkpoint, with 128 environments on one 8-GPU H100 node and evaluation after every iteration (``runner.val_check_interval: 1``), success over 64 evaluation episodes is 64.1% after 3 PPO iterations and 57.8% after 5.
