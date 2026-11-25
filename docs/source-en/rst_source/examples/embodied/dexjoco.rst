DexJoCo RL
==========

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   Watering a plant with an Allegro hand (image: `DexJoCo <https://github.com/brave-eai/dexjoco>`__).

Fine-tune CNN and π₀.₅ policies on the official DexJoCo ``water_plant`` demonstrations, then continue with PPO. DexJoCo provides 11 single-arm and dual-arm dexterous manipulation tasks; this page covers data preparation, SFT, PPO, and evaluation on WaterPlant with a Franka arm and Allegro hand.

Overview
--------

The policy uses front and wrist RGB images to grasp a watering can and water a plant.

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

      WaterPlant

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPUs

| **You'll do:** install deps → download demos & base weights → run SFT → launch PPO → monitor ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · NVIDIA GPUs (1 GPU for CNN SFT, 8 × 80 GB GPUs for PPO/eval, 8–16 × 80 GB GPUs for π₀.₅ SFT).

Tasks, Observation, and Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All recipes run ``dexjoco_water_plant`` (1000-step episode limit):

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Front and wrist RGB images plus 23-dim proprioception (7-dim end-effector pose + 16 Allegro joint positions). CNN images are resized to 128×128.
   * - Action
     - 22-dim continuous action: absolute 3D end-effector position, 3D rotation vector, and 16 Allegro joint targets.
   * - Reward
     - Sparse 0/1 success reward.
   * - Prompt (π₀.₅ only)
     - “Grasp the watering can and apply water to the plant.”

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: CNN (SFT + PPO)
      :sync: dexjoco-cnn

      Uses ``dexjoco_sft_cnn`` (1 × 48 GB GPU, 20,000 steps, Gaussian NLL loss) followed by ``dexjoco_ppo_cnn`` (8 × 80 GB GPUs, 200 iterations, 128 envs).

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env dexjoco && source .venv/bin/activate

      **2. Download & Export Data**

      Download the `official LeRobot demonstrations <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ and ResNet-10 backbone, then export 10-action chunks (generates ``meta.json`` reused across SFT, PPO, and evaluation):

      .. code-block:: bash

         hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
           --include "dexjoco_lerobot_datasets/water_plant/**" \
           --local-dir ./data/dexjoco
         hf download RLinf/RLinf-ResNet10-pretrained \
           --local-dir ./data/RLinf-ResNet10-pretrained
         PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
           --repo-id ./data/dexjoco/dexjoco_lerobot_datasets/water_plant \
           --output ./data/dexjoco/dexjoco_cnn_datasets/water_plant \
           --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

      **3. Launch SFT**

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn \
           data.train_data_paths="$(pwd)/data/dexjoco/dexjoco_cnn_datasets/water_plant" \
           actor.model.model_path="$(pwd)/data/RLinf-ResNet10-pretrained" \
           actor.model.action_norm_stats="$(pwd)/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json"

      **4. Launch PPO**

      Start PPO from the step-8000 SFT checkpoint:

      .. code-block:: bash

         SFT_CKPT=$(ls -d "$(pwd)"/logs/*-dexjoco_sft_cnn/dexjoco_sft_cnn_nll/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt | tail -n 1)
         bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn \
           runner.ckpt_path="$SFT_CKPT" \
           actor.model.model_path="$(pwd)/data/RLinf-ResNet10-pretrained" \
           rollout.model.model_path="$(pwd)/data/RLinf-ResNet10-pretrained" \
           actor.model.action_norm_stats="$(pwd)/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json"

      .. dropdown:: Standalone evaluation & MSE baseline (optional)

         Evaluate any SFT or PPO ``full_weights.pt`` checkpoint (64 episodes on 8 GPUs; see :doc:`/rst_source/evaluations/reference/cli` for CLI flags):

         .. code-block:: bash

            CKPT=$(ls -d "$(pwd)"/logs/*-dexjoco_ppo_cnn/dexjoco_ppo_cnn_nll/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt | tail -n 1)
            bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval \
              runner.ckpt_path="$CKPT" \
              rollout.model.model_path="$(pwd)/data/RLinf-ResNet10-pretrained" \
              rollout.model.action_norm_stats="$(pwd)/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json" \
              runner.logger.experiment_name=dexjoco_cnn_nll_ppo200_eval

         To reproduce the MSE SFT baseline, add ``actor.model.sft_loss_type=mse runner.logger.experiment_name=dexjoco_sft_cnn_mse`` to Step 3 and pass its step-8000 checkpoint to Step 4.

   .. tab-item:: π₀.₅ (SFT + PPO)
      :sync: dexjoco-pi05

      Uses ``dexjoco_sft_openpi_pi05`` (default 2 nodes × 8 H100 GPUs, 5000 steps; see :doc:`/rst_source/guides/multi_node` or pass ``cluster.num_nodes=1`` for 8 GPUs) followed by ``dexjoco_ppo_openpi_pi05`` (8 × 80 GB GPUs, 150 iterations).

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env dexjoco && source .venv/bin/activate

      **2. Download & Convert Base Weights**

      Download the `official LeRobot demonstrations <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ and π₀.₅ base checkpoint, compute normalization statistics, and convert to the RLinf OpenPI layout:

      .. code-block:: bash

         hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
           --include "dexjoco_lerobot_datasets/water_plant/**" \
           --local-dir ./data/dexjoco
         hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/params/**" \
           --local-dir ./data/jax
         PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
           --config-name pi05_dexjoco \
           --repo-id ./data/dexjoco/dexjoco_lerobot_datasets/water_plant
         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model ./data/jax/pi05_base \
           --input-norm-stats ./data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
           --output-model ./data/pi05_base_openpi \
           --output-norm-stats ./data/pi05_base_openpi/dexjoco/norm_stats.json \
           --action-dim 32 --action-horizon 30 --max-token-len 250

      **3. Launch SFT**

      .. code-block:: bash

         # 2-node reproduction (16 GPUs); append cluster.num_nodes=1 for single-node (8 GPUs)
         bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05 \
           data.train_data_paths="$(pwd)/data/dexjoco/dexjoco_lerobot_datasets/water_plant" \
           actor.model.model_path="$(pwd)/data/pi05_base_openpi" \
           actor.model.openpi_data.norm_stats_path="$(pwd)/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json"

      **4. Launch PPO**

      Initialize PPO from the step-5000 SFT checkpoint:

      .. code-block:: bash

         SFT_CKPT=$(ls -d "$(pwd)"/logs/*-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000 | tail -n 1)
         bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05 \
           actor.model.model_path="$SFT_CKPT" \
           rollout.model.model_path="$SFT_CKPT" \
           actor.model.openpi_data.norm_stats_path="$(pwd)/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json"

      .. dropdown:: Standalone evaluation (optional)

         Evaluate any SFT or PPO checkpoint directory (64 episodes on 8 GPUs; see :doc:`/rst_source/evaluations/reference/cli` for CLI flags):

         .. code-block:: bash

            CKPT=$(ls -d "$(pwd)"/logs/*-dexjoco_ppo_openpi_pi05/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150 | tail -n 1)
            bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
              rollout.model.model_path="$CKPT" \
              rollout.model.openpi_data.norm_stats_path="$(pwd)/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json" \
              runner.logger.experiment_name=dexjoco_pi05_ppo150_eval

Visualization and Results
-------------------------

Track ``env/success_once`` and ``eval/success_once`` with ``tensorboard --logdir ./logs --port 6006``; see :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions and Weights & Biases / SwanLab setup.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3f186472f997ecd29977191f8e4da004cff4a302/pic/dexjoco_cnn_nll_vs_mse.png
   :align: center
   :width: 100%
   :alt: CNN PPO success rates after MSE or NLL SFT on water_plant.

   CNN PPO after MSE or NLL SFT on ``water_plant`` (64 evaluation episodes; iteration 0 is the standalone SFT step-8000 evaluation).

.. figure:: https://raw.githubusercontent.com/RLinf/misc/dc2607a2769dfd49cca117cdf42e5ef3f70b8dcf/pic/dexjoco_pi05_sft_ppo.png
   :align: center
   :width: 100%
   :alt: Success rate through 5000 SFT steps and 150 PPO iterations on water_plant.

   π₀.₅ SFT + PPO on ``water_plant`` (64 evaluation episodes).

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Recipe
     - Success rate
   * - CNN + SFT baseline
     - Step 8000 (standalone): MSE 46/64 (71.9%); NLL 42/64 (65.6%).
   * - CNN + PPO (MSE SFT)
     - Iteration 200 (standalone): 44/64 (68.8%); in-training peak (iter 123): 60/64 (93.8%).
   * - CNN + PPO (NLL SFT)
     - Iteration 200 (standalone): 59/64 (92.2%); in-training peak (iter 163): 64/64 (100%).
   * - π₀.₅ + SFT
     - Step 5000 (standalone): 50/64 (78.1%).
   * - π₀.₅ + PPO
     - Iteration 150 (standalone): 63/64 (98.44%); standalone peak (iter 120): 64/64 (100%).
