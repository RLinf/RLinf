DexJoCo RL
==========

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   Watering a plant with an Allegro hand (image: `DexJoCo <https://github.com/brave-eai/dexjoco>`__).

Fine-tune CNN and π₀.₅ policies on the official DexJoCo ``water_plant`` demonstrations, then continue with PPO. This page covers data preparation, SFT, PPO, and checkpoint evaluation for the Franka arm and Allegro hand. DexJoCo provides 11 single-arm and dual-arm tasks; the recipes and results below cover WaterPlant.

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

   .. grid-item-card:: Tasks
      :text-align: center

      WaterPlant

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPUs

| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · NVIDIA GPUs (1–16 GPUs depending on recipe; see table below).

Tasks, Observation, and Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All configs select ``dexjoco_water_plant`` (1000-step episode limit):

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
   :end-before: Then set up the dependencies

Each stage uses one of the checked-in configs below (one SFT step is one optimizer update; one PPO iteration collects rollouts and runs multiple optimizer steps; evaluation runs 64 episodes on 8 workers with seed 10000):

.. list-table::
   :header-rows: 1
   :widths: 17 49 18 16

   * - Recipe
     - Config
     - GPUs (memory per GPU)
     - Run length
   * - CNN + SFT
     - ``examples/sft/config/dexjoco_sft_cnn.yaml``
     - 1 × RTX PRO 5000 Blackwell (48 GB)
     - 20,000 SFT steps
   * - CNN + PPO
     - ``examples/embodiment/config/dexjoco_ppo_cnn.yaml``
     - 8 × H100 (80 GB)
     - 200 PPO iterations
   * - CNN evaluation
     - ``evaluations/dexjoco/dexjoco_cnn_eval.yaml``
     - 8 × H100 (80 GB)
     - 64 episodes
   * - π₀.₅ + SFT
     - ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``
     - 16 × H100 (80 GB, 2 nodes)
     - 5000 SFT steps
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``
     - 8 × H100 (80 GB)
     - 150 PPO iterations
   * - π₀.₅ evaluation
     - ``evaluations/dexjoco/dexjoco_openpi_pi05_eval.yaml``
     - 8 × H100 (80 GB)
     - 64 episodes

Checkpoints are written under ``logs/<timestamp>-<config>/<experiment_name>/checkpoints/global_step_<N>``. Check the launch command logged at the top of ``run_embodiment.log`` for each run's timestamp.

.. tab-set::

   .. tab-item:: CNN (SFT + PPO)
      :sync: dexjoco-cnn

      **1. Install**

      .. code-block:: bash

         bash requirements/install.sh embodied --env dexjoco
         source .venv/bin/activate

      **2. Download & Export Data**

      Download the `official LeRobot demonstrations <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ and the ResNet-10 backbone, then export 10-action chunks:

      .. code-block:: bash

         hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
           --include "dexjoco_lerobot_datasets/water_plant/**" \
           --local-dir ./data/dexjoco
         hf download RLinf/RLinf-ResNet10-pretrained \
           --local-dir ./data/RLinf-ResNet10-pretrained

      .. code-block:: bash

         PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
           --repo-id ./data/dexjoco/dexjoco_lerobot_datasets/water_plant \
           --output ./data/dexjoco/dexjoco_cnn_datasets/water_plant \
           --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

      The exporter saves training arrays and ``meta.json`` (action mean and standard deviation); reuse the same ``meta.json`` across SFT, PPO, and evaluation.

      **3. SFT & Baseline Evaluation**

      Set the paths in ``examples/sft/config/dexjoco_sft_cnn.yaml`` (Gaussian NLL loss, 20,000 steps, saving every 2000 steps) and launch SFT:

      .. code-block:: yaml

         data:
           train_data_paths: /abs/path/to/data/dexjoco/dexjoco_cnn_datasets/water_plant
         actor:
           model:
             model_path: /abs/path/to/data/RLinf-ResNet10-pretrained
             action_norm_stats: /abs/path/to/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn

      Before PPO, evaluate the SFT step-8000 checkpoint by setting ``runner.ckpt_path``, ``rollout.model.model_path``, and ``rollout.model.action_norm_stats`` in ``evaluations/dexjoco/dexjoco_cnn_eval.yaml``:

      .. code-block:: bash

         bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval \
           runner.logger.experiment_name=dexjoco_cnn_nll_sft8000_eval

      **4. PPO & Checkpoint Evaluation**

      Set ``runner.ckpt_path`` (step-8000 SFT ``full_weights.pt``) and actor paths in ``examples/embodiment/config/dexjoco_ppo_cnn.yaml`` (rollout inherits ``actor.model.model_path`` automatically; 128 envs, action-level log-probs, lr ``1e-5``, global batch 1600):

      .. code-block:: yaml

         runner:
           ckpt_path: /abs/path/to/logs/<sft-timestamp>-dexjoco_sft_cnn/dexjoco_sft_cnn_nll/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
         actor:
           model:
             model_path: /abs/path/to/data/RLinf-ResNet10-pretrained
             action_norm_stats: /abs/path/to/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

      To evaluate PPO iteration 200, point ``runner.ckpt_path`` in ``evaluations/dexjoco/dexjoco_cnn_eval.yaml`` at ``global_step_200/.../full_weights.pt`` and run:

      .. code-block:: bash

         bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval \
           runner.logger.experiment_name=dexjoco_cnn_nll_ppo200_eval

      .. dropdown:: CNN MSE loss comparison (optional)

         To reproduce the MSE baseline, reuse the exported dataset and switch the SFT loss to MSE (``actor.model.sft_loss_type=mse``) with a separate experiment name (``dexjoco_sft_cnn_mse``).

         For PPO and evaluation, load the MSE step-8000 checkpoint with experiment names ``dexjoco_ppo_cnn_mse``, ``dexjoco_cnn_mse_sft8000_eval``, and ``dexjoco_cnn_mse_ppo200_eval``.

   .. tab-item:: π₀.₅ (SFT + PPO)
      :sync: dexjoco-pi05

      **1. Install**

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env dexjoco
         source .venv/bin/activate

      **2. Download & Convert Base Weights**

      Download the `official LeRobot demonstrations <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ and π₀.₅ base checkpoint, compute normalization statistics, and convert to the RLinf OpenPI layout:

      .. code-block:: bash

         hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
           --include "dexjoco_lerobot_datasets/water_plant/**" \
           --local-dir ./data/dexjoco
         hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/params/**" \
           --local-dir ./data/jax

      .. code-block:: bash

         PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
           --config-name pi05_dexjoco \
           --repo-id ./data/dexjoco/dexjoco_lerobot_datasets/water_plant
         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model ./data/jax/pi05_base \
           --input-norm-stats ./data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
           --output-model ./data/pi05_base_openpi \
           --output-norm-stats ./data/pi05_base_openpi/dexjoco/norm_stats.json \
           --action-dim 32 --action-horizon 30 --max-token-len 250

      **3. SFT & Baseline Evaluation**

      The reference SFT config ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml`` defaults to **2 nodes × 8 H100 GPUs** (``cluster.num_nodes: 2``; set up the cluster via :doc:`/rst_source/guides/multi_node`). For a single-node quick start, pass ``cluster.num_nodes=1`` on the command line. Set the local paths in ``dexjoco_sft_openpi_pi05.yaml``:

      .. code-block:: yaml

         data:
           train_data_paths: /abs/path/to/data/dexjoco/dexjoco_lerobot_datasets/water_plant
         actor:
           model:
             model_path: /abs/path/to/data/pi05_base_openpi
             openpi_data:
               norm_stats_path: /abs/path/to/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

      .. code-block:: bash

         # 2-node reproduction (16 GPUs, 5000 steps)
         bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05

         # Single-node override (8 GPUs)
         bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05 cluster.num_nodes=1

      Evaluate the step-5000 SFT checkpoint on an 8-GPU node by pointing ``rollout.model.model_path`` and ``norm_stats_path`` in ``evaluations/dexjoco/dexjoco_openpi_pi05_eval.yaml`` to ``global_step_5000``:

      .. code-block:: bash

         bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
           runner.logger.experiment_name=dexjoco_pi05_sft5000_eval

      **4. PPO & Checkpoint Evaluation**

      Initialize PPO from the SFT step-5000 checkpoint in ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml`` (rollout inherits ``actor.model.model_path`` automatically; 150 iterations, token-level log-probs, policy lr 5e-6, value lr 2e-4):

      .. code-block:: yaml

         actor:
           model:
             model_path: /abs/path/to/logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
             openpi_data:
               norm_stats_path: /abs/path/to/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

      To evaluate PPO iteration 150, pass the checkpoint via Hydra override (see :doc:`/rst_source/evaluations/reference/cli` for CLI flags):

      .. code-block:: bash

         bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
           rollout.model.model_path=/abs/path/to/logs/<ppo-timestamp>-dexjoco_ppo_openpi_pi05/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150 \
           runner.logger.experiment_name=dexjoco_pi05_ppo150_eval

Visualization and Results
-------------------------

Track ``env/success_once`` and ``eval/success_once`` with ``tensorboard --logdir ./logs --port 6006``; see :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions and Weights & Biases / SwanLab setup.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3f186472f997ecd29977191f8e4da004cff4a302/pic/dexjoco_cnn_nll_vs_mse.png
   :align: center
   :width: 100%
   :alt: CNN PPO success rates after MSE or NLL SFT on water_plant.

   CNN PPO after MSE or NLL SFT on ``water_plant`` (64 evaluation episodes).

In the CNN figure, iteration 0 is the standalone evaluation of the SFT step-8000 checkpoint; iterations 1–200 are in-training evaluations (TensorBoard step ``N - 1`` corresponds to iteration ``N``). Among ten NLL SFT checkpoints swept on one worker (64 episodes), step 8000 scored highest at 50/64 (78.1%) and 42/64 (65.6%) on the 8-worker evaluation config used for all comparisons below.

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
     - Iteration 200 (standalone): 44/64 (68.8%); in-training peak (iteration 123): 60/64 (93.8%).
   * - CNN + PPO (NLL SFT)
     - Iteration 200 (standalone): 59/64 (92.2%); in-training peak (first reached at iteration 163): 64/64 (100%).
   * - π₀.₅ + SFT
     - Step 5000 (standalone): 50/64 (78.1%).
   * - π₀.₅ + PPO
     - Iteration 150 (standalone): 63/64 (98.44%); standalone peak (iteration 120): 64/64 (100%).
