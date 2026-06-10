RL with PolaRiS Simulation Platform
=======================================================

This document provides a complete guide for using the **π05 (OpenPI)** model to perform PPO reinforcement learning on the `PolaRiS <https://github.com/arhanjain/polaris>`_ simulation platform within the RLinf framework.

Environment
-----------

**PolaRiS (Policy Learning and Benchmarking in Realistic Simulated Environments)**

PolaRiS is a high-fidelity robotics simulation platform based on Isaac Sim and Gaussian Splatting rendering. It supports various desktop manipulation tasks and provides realistic visual rendering effects.

- **Simulation Platform**: Based on NVIDIA Isaac Sim
- **Rendering**: Real-time Gaussian Splatting, supporting switching between high-quality (expensive) and fast rendering modes.
- **Observation Space**:
  - External camera (desktop view) RGB image (224×224)
  - Wrist camera RGB image (224×224)
  - Robot proprioceptive state: 7-dim joint positions + 1-dim gripper position (8 dims total)
- **Action Space**: 8-dim continuous action
  - 7-dim joint velocity control
  - 1-dim gripper position control
- **Tasks**: Supports various desktop manipulation tasks, such as:
  - TapeIntoContainer
  - PanClean
  - BlockStackKitchen
  - FoodBussing
  - MoveLatteCup
  - OrganizeTools
- **Episode Length**: Default 30 seconds (15Hz sampling rate = 450 steps)

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**
   - Uses GAE (Generalized Advantage Estimation) for advantage estimation
   - Ratio-based policy clipping
   - Value function clipping
   - Entropy regularization

2. **π05 Flow Matching Policy**
   - Based on the OpenPI π05 architecture
   - Flow Matching for action generation (SDE sampling mode)
   - Supports a Value Head for Critic estimation
   - Action Chunking: Generates multiple action steps at once (default 15 steps) and executes them in an open loop.

3. **DROID Data Format**
   - Uses observation key mapping from the DROID dataset format.
   - State encoding: joint positions (7-dim) + gripper position (1-dim)
   - Image encoding: external camera left image + wrist camera left image

Dependency Installation
-----------------------

1. Clone the RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-polaris
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-polaris

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

3. Isaac Sim Download
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before using PolaRiS, you need to download and set up Isaac Sim. Please follow the instructions below:

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip

After downloading, set environment variables via:

.. code-block:: bash

   source ./setup_conda_env.sh

.. warning::

   This step must be done every time you open a new terminal to use Isaac Sim.

Dataset Download
----------------

PolaRiS has two datasets: one for evaluation and one for co-training.

**1. Evaluation Dataset — PolaRiS-Hub**

`PolaRiS-Hub <https://huggingface.co/datasets/owhan/PolaRiS-Hub>`_ contains scene USD files and initial condition configurations used for evaluation.

.. code:: bash

   hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub

After downloading, set the ``POLARIS_DATA_PATH`` environment variable to the dataset path in ``examples/embodiment/run_embodiment.sh`` and ``examples/embodiment/eval_embodiment.sh``:

.. code:: bash

   export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

Alternatively, you can modify ``init_params.dataset_path`` and ``init_params.usd_file`` in the configuration YAML files under ``examples/embodiment/config/env/polaris_droid_*.yaml``.

**2. Co-training Dataset — PolaRiS-datasets**

`PolaRiS-datasets <https://huggingface.co/datasets/owhan/PolaRiS-datasets>`_ contains demonstration data used for co-training fine-tuning of the model.

.. code:: bash

   hf download owhan/PolaRiS-datasets --repo-type=dataset --local-dir ./PolaRiS-datasets

Model Download
--------------

Before starting training, you need to download the corresponding pretrained model:

**Method 1: Download Pre-converted PyTorch Model (Recommended)**

Pre-trained PyTorch models are available on HuggingFace, converted from the original JAX checkpoints.

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-Polaris-droid_jointpos
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Polaris-droid_jointpos

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./checkpoints/RLinf-Pi05-Polaris-droid_jointpos
   hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./checkpoints/RLinf-Pi0-Polaris-droid_jointpos

**Method 2: Download JAX Checkpoint and Convert**

Alternatively, you can download the original JAX checkpoints and convert them to PyTorch format.

.. code:: bash

   # Download JAX checkpoints
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris /path/to/checkpoints/
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris /path/to/checkpoints/

   # Convert π0.5 Polaris to PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi05_droid_jointpos_polaris \
       --config_name pi05_droid_jointpos_polaris \
       --output_path /path/to/checkpoints/pi05_droid_jointpos_polaris_new
   cp -r /path/to/checkpoints/pi05_droid_jointpos_polaris/assets /path/to/checkpoints/pi05_droid_jointpos_polaris_new/

   # Convert π0 Polaris to PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi0_droid_jointpos_polaris \
       --config_name pi0_droid_jointpos_polaris \
       --output_path /path/to/checkpoints/pi0_droid_jointpos_polaris_new
   cp -r /path/to/checkpoints/pi0_droid_jointpos_polaris/assets /path/to/checkpoints/pi0_droid_jointpos_polaris_new/

After downloading, make sure to correctly specify the model path in the configuration yaml file.
The example below uses the π0.5 checkpoint. For π0 configs, use ``RLinf-Pi0-Polaris-droid_jointpos`` instead.

.. code-block:: yaml

   rollout:
     model:
       model_path: "./checkpoints/RLinf-Pi05-Polaris-droid_jointpos"
   actor:
     model:
       model_path: "./checkpoints/RLinf-Pi05-Polaris-droid_jointpos"

Running the Script
------------------

**1. Configuration Files**

PolaRiS currently supports the following training configurations:

- **PPO Training**

  - ``examples/embodiment/config/polaris_ppo_openpi_pi05.yaml``
  - ``examples/embodiment/config/polaris_ppo_openpi.yaml``

- **Evaluation**

  - ``examples/embodiment/config/polaris_openpi_pi05_eval.yaml``
  - ``examples/embodiment/config/polaris_openpi_eval.yaml``

Each task has an independent environment configuration file located under ``examples/embodiment/config/env/``:

- ``polaris_droid_tapeintocontainer.yaml``
- ``polaris_droid_panclean.yaml``
- ``polaris_droid_blockstackkitchen.yaml``
- ``polaris_droid_foodbussing.yaml``
- ``polaris_droid_movelattecup.yaml``
- ``polaris_droid_organizetools.yaml``

**2. Key Parameter Configuration**

Parameters below are located in the training configuration file ``examples/embodiment/config/polaris_ppo_openpi_pi05.yaml``.

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor,rollout,env: 0

You can flexibly configure the GPU allocation for the Actor, Rollout, and Env components.

- **Actor (Training)**: Occupies the most VRAM (weights + gradients + optimizer). It is recommended to place it on the card with the most available memory.
- **Rollout (Inference)**: Only requires model weights and KV Cache.
- **Env (Environment)**: Can share a card with Rollout. The PolaRiS environment requires a GPU for Gaussian Splatting rendering.

.. code-block:: yaml

   actor:
     model:
       num_action_chunks: 15
       action_dim: 8
       openpi:
         config_name: "pi05_droid_polaris"
         num_images_in_input: 2

- ``num_action_chunks: 15``: The model generates 15 action steps at a time.
- ``action_dim: 8``: 7-dim joint velocity + 1-dim gripper position.
- ``config_name: "pi05_droid_polaris"``: Use the PolaRiS configuration with DROID data format.
- ``num_images_in_input: 2``: External camera + wrist camera, total 2 images.

**3. Environment Parameters**

The ``init_params`` are located in the environment configuration files ``examples/embodiment/config/env/polaris_droid_*.yaml``.
The training configuration file references them via Hydra defaults (e.g. ``defaults: - env/polaris_droid_tapeintocontainer@env.train``).

.. code-block:: yaml

   init_params:
     open_loop_horizon: ${actor.model.num_action_chunks}

``open_loop_horizon`` controls the frequency of high-quality Gaussian Splatting rendering. During the execution of an action chunk, high-quality rendering is performed every ``open_loop_horizon`` steps, while intermediate steps use low-quality rendering to speed up the simulation.

**4. Start Training**

.. code-block:: bash

   source /path/to/isaac_sim/setup_conda_env.sh

   # pi05
   bash examples/embodiment/run_embodiment.sh polaris_ppo_openpi_pi05
   # pi0
   bash examples/embodiment/run_embodiment.sh polaris_ppo_openpi

.. note::

   If you have hardcoded ``POLARIS_DATA_PATH`` in your configuration file, please ensure the path is correct.
   You can also set the environment variable before running:

   .. code-block:: bash

      export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

**5. Start Evaluation**

.. code-block:: bash

   source /path/to/isaac_sim/setup_conda_env.sh

   # pi05
   bash examples/embodiment/eval_embodiment.sh polaris_openpi_pi05_eval
   # pi0
   bash examples/embodiment/eval_embodiment.sh polaris_openpi_eval

Visualization and Results
-------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. Key Monitoring Metrics**

- **Environment Metrics**:

  - ``env/success_once``: Task success rate. It is recommended to use this metric to monitor training effectiveness.
  - ``env/return``: Total return per episode.
  - ``env/episode_len``: Actual number of steps per episode.

- **Training Metrics**:

  - ``train/actor/policy_loss``: PPO policy loss.
  - ``train/critic/value_loss``: Value function loss.
  - ``train/actor/approx_kl``: Approximate KL divergence, for monitoring the magnitude of policy updates.

- **Rollout Metrics**:

  - ``rollout/rewards``: Step-wise rewards.
  - ``rollout/advantages_mean``: Mean of the advantage function.

