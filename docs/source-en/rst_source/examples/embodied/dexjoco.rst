RL with the DexJoCo Benchmark
========================================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   Watering a plant with an Allegro hand (image: `DexJoCo <https://github.com/brave-eai/dexjoco>`__).

Fine-tune CNN and π₀.₅ policies on the official DexJoCo ``water_plant``
demonstrations, then continue with PPO. This page covers data preparation,
training, and checkpoint evaluation for the Franka arm and Allegro hand.
DexJoCo provides 11 single-arm and dual-arm tasks; the configurations and results
on this page cover WaterPlant.

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
   * - Prompt (π₀.₅ only)
     - “Grasp the watering can and apply water to the plant.”

Installation
------------

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install the dependencies for your chosen policy and run the commands below from
the RLinf repository root. The installer creates the virtual environment in
``.venv`` under the repository root:

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

Choose a policy and follow the training and checkpoint evaluation steps below.
The checked-in configurations contain the reproduction settings; fill in the
local paths shown below. Each stage uses one of these configurations:

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
     - 16 × H100 (80 GB)
     - 5000 SFT steps
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``
     - 8 × H100 (80 GB)
     - 150 PPO iterations
   * - π₀.₅ evaluation
     - ``evaluations/dexjoco/dexjoco_openpi_pi05_eval.yaml``
     - 8 × H100 (80 GB)
     - 64 episodes

One SFT step is one optimizer update. One PPO iteration collects rollouts and
trains on them for multiple optimizer steps. The run lengths and checkpoints
below use these two counters. Evaluation length is measured in episodes.

Each training launch creates ``logs/<timestamp>-<config>`` under the repository
root. Checkpoints are saved beneath that directory at
``<experiment_name>/checkpoints/global_step_<N>``.
In the paths below, replace ``/path/to/RLinf`` with your repository root and
``<sft-timestamp>`` or ``<ppo-timestamp>`` with the timestamp of the corresponding
run. The launch command in that run's ``run_embodiment.log`` records its log path.

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

The exporter resizes the images and pairs each observation with ten consecutive
actions from its demonstration. It saves these training arrays and ``meta.json``
in the output directory. The metadata contains the action mean and standard
deviation; reuse it for SFT, PPO, and evaluation so all stages use the same
action scale. Set these paths in
``examples/sft/config/dexjoco_sft_cnn.yaml``:

.. code:: yaml

   data:
     train_data_paths: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant
   actor:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

The default SFT configuration uses Gaussian negative log-likelihood (NLL).
Train for 20,000 SFT steps, saving a checkpoint every 2000 steps:

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn

Before PPO, evaluate the SFT step-8000 checkpoint. Set the checkpoint, experiment
name, ResNet path, and action statistics in
``evaluations/dexjoco/dexjoco_cnn_eval.yaml``:

.. code:: yaml

   runner:
     logger:
       experiment_name: dexjoco_cnn_nll_sft8000_eval
     ckpt_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_cnn/dexjoco_sft_cnn_nll/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
   rollout:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

Run the evaluation on eight GPUs. The config uses eight workers and the same
64 initial states, with seed 10000, 10-action chunks, and a 1000-step episode limit:

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval

Set the model, normalization, and selected checkpoint paths in
``examples/embodiment/config/dexjoco_ppo_cnn.yaml``:

.. code:: yaml

   runner:
     ckpt_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_cnn/dexjoco_sft_cnn_nll/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
   rollout:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
   actor:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

PPO uses 128 environments, action-level log-probabilities, policy learning rate
1e-5, and global batch size 1600. Launch it with:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

PPO saves its checkpoints under
``logs/<ppo-timestamp>-dexjoco_ppo_cnn/dexjoco_ppo_cnn_nll/checkpoints``.
To evaluate iteration 200, change only the checkpoint and experiment name
in the same eval YAML:

.. code:: yaml

   runner:
     logger:
       experiment_name: dexjoco_cnn_nll_ppo200_eval
     ckpt_path: /path/to/RLinf/logs/<ppo-timestamp>-dexjoco_ppo_cnn/dexjoco_ppo_cnn_nll/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt

Run the PPO checkpoint evaluation:

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval

π₀.₅: SFT and PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare the π₀.₅ inputs from the original LeRobot dataset. Compute its state
and action normalization statistics, then download and convert the base model:

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

The first command writes ``norm_stats.json`` in the dataset directory. Reuse
this file for SFT, PPO, and evaluation. The conversion writes PyTorch weights
in the RLinf OpenPI layout under ``/path/to/pi05_base_openpi``. SFT reads the
original LeRobot dataset and initializes from these converted base weights.

Prepare two nodes with eight H100 GPUs each using
:doc:`/rst_source/guides/multi_node`. The SFT configuration selects all 16 GPUs
and 5000 SFT steps. Set these paths in
``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``:

.. code:: yaml

   data:
     train_data_paths: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant
   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

SFT uses global batch size 32, micro batch size 2 per GPU, and a 30,000-step
learning-rate schedule. Launch it on the head node:

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05

The reported PPO run starts from the final SFT checkpoint at step 5000.
SFT saves it under
``logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000``.
Run evaluation and PPO on the eight-GPU head node. Ensure that this node can
read the checkpoint directory and the dataset's ``norm_stats.json``.
Set the following paths in
``evaluations/dexjoco/dexjoco_openpi_pi05_eval.yaml``:

.. code:: yaml

   rollout:
     model:
       model_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

Run the SFT baseline evaluation:

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
     runner.logger.experiment_name=dexjoco_pi05_sft5000_eval

The evaluation config uses the same 64 initial states on eight workers with
seed 10000, 10-action chunks, 5 denoising steps, and a 1000-step episode limit.
Use the same config for the PPO checkpoint below.

Initialize PPO from that same SFT checkpoint. Set these paths in
``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``:

.. code:: yaml

   rollout:
     model:
       model_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
   actor:
     model:
       model_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

Launch PPO on the same head node. The default recipe trains for 150 PPO
iterations with token-level log-probabilities, policy learning rate 5e-6, and
value learning rate 2e-4. It evaluates and saves a checkpoint every iteration:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

PPO saves iteration 150 under
``logs/<ppo-timestamp>-dexjoco_ppo_openpi_pi05/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150``.
In the evaluation config used above, change only ``rollout.model.model_path``
to this checkpoint:

.. code:: yaml

   rollout:
     model:
       model_path: /path/to/RLinf/logs/<ppo-timestamp>-dexjoco_ppo_openpi_pi05/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150

Run the PPO evaluation under a separate experiment name:

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
     runner.logger.experiment_name=dexjoco_pi05_ppo150_eval

See :doc:`/rst_source/evaluations/reference/cli` for evaluation options.

CNN: MSE Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the MSE comparison, reuse the exported data and rerun SFT and PPO. In the SFT
config, set ``actor.model.sft_loss_type: mse`` and
``runner.logger.experiment_name: dexjoco_sft_cnn_mse``.
Use ``dexjoco_ppo_cnn_mse`` as the PPO experiment name and load step 8000 from
the MSE run:

.. code:: yaml

   runner:
     logger:
       experiment_name: dexjoco_ppo_cnn_mse
     ckpt_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_cnn/dexjoco_sft_cnn_mse/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt

For the MSE SFT baseline, set the eval YAML to the
``dexjoco_sft_cnn_mse`` step-8000 checkpoint and experiment name
``dexjoco_cnn_mse_sft8000_eval``, then run the same evaluation command.
For the PPO evaluation, set the experiment name
to ``dexjoco_cnn_mse_ppo200_eval`` and
``runner.ckpt_path`` to
``/path/to/RLinf/logs/<ppo-timestamp>-dexjoco_ppo_cnn/dexjoco_ppo_cnn_mse/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt``.
Use the timestamps of the MSE runs and keep the remaining settings unchanged.

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

In the CNN figure, iteration 0 is the standalone evaluation of the SFT step-8000
checkpoint. Iterations 1–200 are in-training evaluations; TensorBoard step
``N - 1`` records the evaluation after PPO iteration ``N``.

The CNN SFT sweep evaluates all ten NLL checkpoints on 64 episodes with one
worker. Step 8000 has the highest success rate, 50/64 (78.1%), and is selected
as the PPO starting point. Its baseline on the eight-worker evaluation config
is 42/64 (65.6%). The SFT and PPO checkpoint comparisons use that same config.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/dc2607a2769dfd49cca117cdf42e5ef3f70b8dcf/pic/dexjoco_pi05_sft_ppo.png
   :align: center
   :width: 100%
   :alt: Success rate through 5000 SFT steps and 150 PPO iterations on water_plant.

   π₀.₅ SFT + PPO on ``water_plant`` (64 evaluation episodes).

The π₀.₅ rows below report standalone evaluations of the SFT and PPO checkpoints
on the same 64 initial states.

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
