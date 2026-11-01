RL with DexJoCo Benchmark
=========================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   Watering a plant with an Allegro hand (image: `DexJoCo <https://github.com/brave-eai/dexjoco>`__).

Fine-tune CNN and π₀.₅ policies on the official
`DexJoCo <https://github.com/brave-eai/dexjoco>`__ demonstrations, then continue
both with PPO. The MuJoCo benchmark covers single-arm and dual-arm dexterous
manipulation with Franka arms and Allegro hands. This page walks through data
preparation, training, and evaluation.

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
     - RGB from ``camera_names`` at 640×640 (128×128 in the CNN recipe) plus 23-dim proprioception per arm (7-dim end-effector pose and 16 Allegro joints).
   * - Action
     - 22-dim continuous action per arm: absolute 3D end-effector position, 3D rotation vector, and 16 Allegro joint targets; the 44-dim dual-arm action lists the right arm first.
   * - Reward
     - Sparse 0/1 success reward.
   * - Prompt
     - Natural-language instruction for the DexJoCo task, used by π₀.₅.

Installation
------------

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install the environment for the model you want to run:

.. code:: bash

   # Mainland China users can add --use-mirror.

   # CNN
   bash requirements/install.sh embodied --env dexjoco
   source .venv/bin/activate

   # OpenPI π₀.₅
   # bash requirements/install.sh embodied --model openpi --env dexjoco
   # source .venv/bin/activate

Download the Datasets
---------------------

Both recipes start from the
`official LeRobot demonstrations <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__.
Download one task and compute the π₀.₅ normalization statistics:

.. code:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
     --include "dexjoco_lerobot_datasets/water_plant/**" \
     --local-dir /path/to/dexjoco
   PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_dexjoco \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant

What this does:

1. Downloads the ``water_plant`` episodes with front and wrist videos, robot state, actions, and task text.
2. Writes ``norm_stats.json`` into the dataset directory for SFT, PPO, and evaluation.

For dual-arm tasks, pass ``--config-name pi05_dexjoco_dual_arm``; for
``click_mouse``, whose base camera is ``ego_right``, pass
``--config-name pi05_dexjoco_click_mouse``.

The CNN recipes read the same demonstrations as image, state, and action-chunk
arrays. Export one task at the training resolution:

.. code:: bash

   PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant \
     --output /path/to/dexjoco/dexjoco_cnn_datasets/water_plant \
     --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

What this does:

1. Decodes the front and wrist videos once and resizes every frame to 128×128.
2. Writes ``main_images``, ``extra_view_images``, ``states``, ``actions`` (the next 10 actions of each frame), and ``episode_index`` as memory-mapped ``.npy`` files next to a ``meta.json`` with the per-dimension action statistics.

For dual-arm tasks, pass ``--cameras ego wrist_left wrist_right --state-dim 46``.

Download the Model
------------------

Download the checkpoint for the model you plan to train.

**CNN**

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
   hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/params/**" \
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
   * - CNN + SFT
     - ``examples/sft/config/dexjoco_sft_cnn.yaml``
     - ``run_vla_sft.sh``
     - ``dexjoco_sft_cnn``
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

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

What this does:

1. Starts the embodied training entrypoint with the selected Hydra config.
2. Creates Ray workers for the actor, rollout, and DexJoCo env components.
3. Runs PPO rollouts, computes sparse success rewards, and updates the CNN policy.

.. note::

   The recipes train ``water_plant``. For another task, override ``task_name``
   and ``camera_names`` in both ``env.train`` and ``env.eval`` (``[front, wrist]``
   for single-arm tasks, ``[ego_right, wrist]`` for ``click_mouse``, and
   ``[ego, wrist_left, wrist_right]`` for dual-arm tasks), and set
   ``max_episode_steps`` and ``max_steps_per_rollout_epoch`` in both ``env.train``
   and ``env.eval`` to the task's time limit (every limit is a multiple of the
   10-step chunk). Dual-arm tasks use 46-dim state
   and 44-dim actions: for the CNN, set ``actor.model.state_dim: 46`` and
   ``actor.model.action_dim: 44``; for π₀.₅, apply the dual-arm settings in the
   SFT section.

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

The simulator renders the cameras only on the last step of each executed
action chunk, which is the observation the policy acts on; with
``video_cfg.save_video: True`` every step is rendered.

CNN + SFT
~~~~~~~~~

Set these fields in ``examples/sft/config/dexjoco_sft_cnn.yaml``:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Field
     - Value
   * - ``data.train_data_paths``
     - The exported chunk directory.
   * - ``actor.model.model_path``
     - The ``RLinf-ResNet10-pretrained`` directory.
   * - ``actor.model.action_norm_stats``
     - The exported ``meta.json``, whose per-dimension action mean and std normalize the policy's actions.

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn

The policy encodes the 128×128 front and wrist images with the frozen
ResNet-10 backbone, concatenates the 23-dim proprioception, and predicts the
next 10 actions. The loss is the Gaussian negative log-likelihood of the
demonstrated chunk, so the per-dimension action noise that PPO later samples
from is fitted to the demonstrations. The 20,000-step recipe takes about ten
minutes on one 48 GB GPU and saves
``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``.

For a dual-arm task, set ``actor.model.image_num: 3``,
``actor.model.state_dim: 46``, and ``actor.model.action_dim: 44``.

CNN + PPO
~~~~~~~~~

``dexjoco_ppo_cnn`` runs PPO with the same observation and action layout over
128 environments and full 1000-step episodes, and evaluates 64 episodes every
five iterations. Each iteration runs one PPO update epoch with action-level
log-probabilities; the first 50 optimizer steps train the value head alone.
Set ``runner.ckpt_path`` to an SFT ``full_weights.pt`` to
start from the behavior-cloned policy; with ``null`` the policy starts from the
ResNet-10 encoder alone. ``actor.model.action_norm_stats`` must name the same
``meta.json`` as the SFT run.

The reported run starts from SFT step 8000 on 8 H100 GPUs. To reproduce its
placement and evaluation schedule, set these fields in the same config:

.. code:: yaml

   cluster:
     component_placement:
       actor,env,rollout: all
   runner:
     ckpt_path: /path/to/dexjoco_sft_cnn/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
     val_check_interval: 1
     save_interval: 1
   env:
     train:
       seed: 42
     eval:
       seed: 10000
       video_cfg:
         save_video: False
   rollout:
     enable_torch_compile: False
     enable_cuda_graph: False
   actor:
     seed: 42

Launch training after setting the paths and run configuration:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with ``dexjoco_cnn_eval`` and set
``runner.ckpt_path`` to the SFT or PPO ``full_weights.pt``.

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

The reported run starts from SFT step 5000 on 8 H100 GPUs and targets 150
iterations. Set the fields below, keeping the learning-rate schedule at
1000 steps when setting
``runner.max_epochs`` to 150; this preserves the schedule used by the run:

.. code:: yaml

   runner:
     max_epochs: 150
     val_check_interval: 1
     save_interval: 1
   env:
     train:
       seed: 42
     eval:
       seed: 10000
       video_cfg:
         save_video: False
   actor:
     model:
       openpi:
         torch_compile: False
     optim:
       total_training_steps: 1000

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

What this does:

1. Starts the embodied training entrypoint with the selected Hydra config.
2. Creates Ray workers for the actor, rollout, and DexJoCo env components.
3. Runs PPO rollouts, computes sparse success rewards, and updates the OpenPI policy.

Each decision executes the first 10 of the 30 predicted actions and samples
them with 5 denoising steps, which also raises the SFT policy's own success
rate (below). One iteration collects 128 environments × 2 rollout epochs and
runs one PPO epoch over them; the first 50 optimizer steps train the value
head alone (``actor.optim.critic_warmup_steps``).

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with ``dexjoco_openpi_pi05_eval``. Set
``rollout.model.model_path`` to the SFT or PPO checkpoint and
``rollout.model.openpi_data.norm_stats_path`` to the statistics used for
training; for a dual-arm checkpoint, apply the dual-arm settings under
``rollout.model``.

.. note::

   ``dexjoco_ppo_cnn`` runs on GPU ``0`` by default, and the π₀.₅ configs
   colocate ``actor,env,rollout: all``. Tune ``cluster.component_placement``,
   ``env.train.total_num_envs``, and batch sizes for your hardware.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ./logs --port 6006

Watch ``env/success_once`` for training rollouts and ``eval/success_once`` for
evaluation episodes. For every logged metric, see :doc:`Training metrics
<../../reference/metrics>`.

To save the same metrics to Weights & Biases, sign in with ``wandb login`` and
select both logging backends:

.. code:: yaml

   runner:
     logger:
       project_name: rlinf
       experiment_name: dexjoco_ppo_cnn
       logger_backends: ["tensorboard", "wandb"]

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
   * - CNN + SFT
     - On ``water_plant``, the 20,000-step recipe on one 48 GB GPU peaks at step 8000 with 50 of 64 evaluation episodes (78%); steps 10,000–20,000 stay between 66% and 75%.
   * - CNN + PPO
     - Continuing SFT step 8000 with action-level PPO on 8 H100 GPUs gives 42/40/49/48/55 successes out of 64 at iterations 0/10/20/40/80 (65.6%/62.5%/76.6%/75.0%/85.9%). The run uses policy LR 1e-5 and global batch 1600. All checkpoints use the same 64-episode evaluation on 8 workers, with ``env.eval.seed: 10000`` and 10-action chunks. Training is continuing toward the planned 200 iterations.
   * - π₀.₅ + SFT
     - On ``water_plant``, the step-5000 checkpoint (16 H100 GPUs, global batch 32) succeeds in 50 of 64 evaluation episodes (78%) when the policy executes 10 actions per chunk with 5 denoising steps; executing all 30 predicted actions with 10 denoising steps gives 30 of 64 (47%).
   * - π₀.₅ + PPO
     - Continuing SFT step 5000 with the chunk-level recipe on 8 H100 GPUs raises success from 50/64 (78.1%) to 59/64 (92.2%) at PPO iteration 120. Both checkpoints use the same 64-episode evaluation on 8 workers, with ``env.eval.seed: 10000``, 10 executed actions per chunk, and 5 denoising steps.
