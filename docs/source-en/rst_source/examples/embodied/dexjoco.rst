RL with DexJoCo Benchmark
=========================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   Watering a plant with an Allegro hand (image: `DexJoCo <https://github.com/brave-eai/dexjoco>`__).

Fine-tune CNN and π₀.₅ policies on the official DexJoCo ``water_plant``
demonstrations, then continue with PPO. This page covers data preparation,
training, and checkpoint evaluation for the Franka arm and Allegro hand.

Overview
--------

The policy uses front and wrist images to grasp a watering can and water a plant.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      CNN · π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · PPO

   .. grid-item-card:: Task
      :text-align: center

      WaterPlant

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPUs

Tasks
~~~~~

The training and evaluation configs select ``dexjoco_water_plant``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Task
     - Episode limit
     - Description
   * - ``water_plant``
     - 1000 steps
     - Grasp the watering can and apply water to the plant.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Both policies use the following observations, actions, and reward:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Front and wrist RGB images plus 23-dim proprioception: a 7-dim end-effector pose and 16 Allegro joint positions. CNN images are resized to 128×128.
   * - Action
     - 22-dim continuous action: absolute 3D end-effector position, a 3D rotation vector, and 16 Allegro joint targets.
   * - Reward
     - Sparse 0/1 success reward.
   * - Prompt
     - “Grasp the watering can and apply water to the plant.”

Installation
------------

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install the dependencies for your chosen policy and run the commands below from
the RLinf repository root:

.. code:: bash

   # CNN
   bash requirements/install.sh embodied --env dexjoco

   # For π₀.₅, use:
   # bash requirements/install.sh embodied --model openpi --env dexjoco
   source .venv/bin/activate

Download the Datasets
---------------------

Both recipes use the same
`official LeRobot demonstrations <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__.
Download the ``water_plant`` episodes:

.. code:: bash

   hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
     --include "dexjoco_lerobot_datasets/water_plant/**" \
     --local-dir /path/to/dexjoco

The dataset contains front and wrist videos, robot states, actions, and task text.
Each recipe below prepares its own training inputs from this directory.

Run It
------

Choose a policy and follow its SFT, PPO, and evaluation steps with these training
configurations:

.. list-table::
   :header-rows: 1
   :widths: 17 49 18 16

   * - Recipe
     - Config
     - GPUs
     - Training steps
   * - CNN + SFT
     - ``examples/sft/config/dexjoco_sft_cnn.yaml``
     - 1 × 48 GB
     - 20,000 updates
   * - CNN + PPO
     - ``examples/embodiment/config/dexjoco_ppo_cnn.yaml``
     - 8 × H100
     - 200 iterations
   * - π₀.₅ + SFT
     - ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``
     - 16 × H100
     - 5000 updates
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``
     - 8 × H100
     - 150 iterations

CNN: SFT and PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the ResNet-10 backbone and export the demonstrations as images, states,
and 10-action chunks:

.. code:: bash

   hf download RLinf/RLinf-ResNet10-pretrained \
     --local-dir /path/to/RLinf-ResNet10-pretrained
   PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant \
     --output /path/to/dexjoco/dexjoco_cnn_datasets/water_plant \
     --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

The exporter writes the training arrays and ``meta.json``, which contains the
action normalization statistics. Set these paths in
``examples/sft/config/dexjoco_sft_cnn.yaml``:

.. code:: yaml

   data:
     train_data_paths: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant
   actor:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

For SFT, set ``actor.model.sft_loss_type`` to ``mse`` (mean squared error,
default) or ``nll`` (Gaussian negative log-likelihood) in the config, then run:

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn

SFT saves ``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``.
Continue from step 8000 by setting the following fields in
``examples/embodiment/config/dexjoco_ppo_cnn.yaml``:

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
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
     enable_torch_compile: False
     enable_cuda_graph: False
   actor:
     seed: 42
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

PPO uses 128 environments, action-level log-probabilities, learning rate 1e-5,
and global batch 1600. Launch it with:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

Evaluate the saved checkpoint with the same action statistics. The following
command runs 64 episodes across the eight GPUs:

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval \
     runner.ckpt_path=/path/to/dexjoco_ppo_cnn/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt \
     rollout.model.model_path=/path/to/RLinf-ResNet10-pretrained \
     rollout.model.action_norm_stats=/path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json \
     env.eval.seed=10000 env.eval.video_cfg.save_video=False

π₀.₅: SFT and PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the OpenPI normalization statistics and download the π₀.₅ base model:

.. code:: bash

   PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_dexjoco \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant
   hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/params/**" \
     --local-dir /path/to/jax
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/jax/pi05_base \
     --input-norm-stats /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
     --output-model /path/to/pi05_base_openpi \
     --output-norm-stats /path/to/pi05_base_openpi/dexjoco/norm_stats.json \
     --action-dim 32 --action-horizon 30 --max-token-len 250

This produces ``norm_stats.json`` and a model in the RLinf OpenPI layout.
Prepare the two-node cluster using :doc:`/rst_source/guides/multi_node`.
In ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``, replace the ``cluster``
section and set the training fields below:

.. code:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor: 0-15
   runner:
     max_steps: 5000
   data:
     train_data_paths: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant
   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi:
         max_token_len: 250
         torch_compile: False
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

SFT uses global batch 32, micro batch 2, and a 30,000-step learning-rate schedule.
Launch it on the head node:

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05

The resulting ``checkpoints/global_step_5000`` directory supplies the PPO model.
Set these paths in ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``:

.. code:: yaml

   rollout:
     model:
       model_path: /path/to/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
   actor:
     model:
       model_path: /path/to/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

Launch PPO on one node. The default recipe trains for 150 iterations with
token-level log-probabilities, policy learning rate 5e-6, and value learning rate
2e-4. It evaluates and saves a checkpoint every iteration:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

Evaluate a saved checkpoint with the training statistics:

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
     rollout.model.model_path=/path/to/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150 \
     rollout.model.openpi_data.norm_stats_path=/path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

The evaluation config runs 64 episodes on eight workers with seed 10000,
10-action chunks, 5 denoising steps, and a 1000-step episode limit.
See :doc:`/rst_source/evaluations/reference/cli` for evaluation options.

Visualization and Results
-------------------------

Track ``env/success_once`` and ``eval/success_once`` in TensorBoard:

.. code:: bash

   tensorboard --logdir ./logs --port 6006

For Weights & Biases, run ``wandb login`` and select both logging backends:

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3f186472f997ecd29977191f8e4da004cff4a302/pic/dexjoco_cnn_nll_vs_mse.png
   :align: center
   :width: 100%
   :alt: CNN PPO success rates after MSE or NLL SFT on water_plant.

   CNN PPO after MSE or NLL SFT on ``water_plant`` (64 evaluation episodes).

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3f186472f997ecd29977191f8e4da004cff4a302/pic/dexjoco_pi05_sft_ppo.png
   :align: center
   :width: 100%
   :alt: Success rate through 5000 SFT steps and 150 PPO iterations on water_plant.

   π₀.₅ SFT + PPO on ``water_plant`` (64 evaluation episodes).

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Recipe
     - Success rate
   * - CNN + SFT
     - Step 8000: MSE 46/64 (71.9%); NLL 42/64 (65.6%).
   * - CNN + PPO (MSE SFT)
     - Iteration 0: 46/64 (71.9%); iteration 200: 44/64 (68.8%); peak (iteration 123): 60/64 (93.8%).
   * - CNN + PPO (NLL SFT)
     - Iteration 0: 42/64 (65.6%); iteration 200: 59/64 (92.2%); peak (iteration 163): 64/64 (100%).
   * - π₀.₅ + SFT
     - Step 5000: 44/64 (68.75%).
   * - π₀.₅ + PPO
     - Iteration 150: 63/64 (98.44%); peak (iteration 120): 64/64 (100%).
