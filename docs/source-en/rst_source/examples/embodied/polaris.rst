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

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to clone the PolaRiS repository locally first, then edit ``polaris/pyproject.toml`` to match your machine's CUDA version, and then run the install script.

**Step 1:** Clone the PolaRiS repository:

.. code:: bash

   git clone --recursive git@github.com:arhanjain/polaris.git
   export POLARIS_PATH=/path/to/polaris

**Step 2:** Edit ``$POLARIS_PATH/pyproject.toml`` and apply the following changes (shown for CUDA 12.4; adapt version numbers for other CUDA versions):

.. code:: diff

   # 1. Lock Python version to 3.11
   - requires-python = ">=3.11"
   + requires-python = "==3.11.*"

   # 2. Match torch/torchvision version to your CUDA toolkit
   # Pin sympy to avoid version conflict with isaaclab
   #    (CUDA 12.4 example; see https://pytorch.org/get-started/locally/ for other versions)
   -     override-dependencies = [
   -         "pywin32==306; sys_platform == 'win32'",
   -         "torch>=2.9.0", # Change here for different CUDA version
   -         "torchvision>=0.24.0", # Change here for different CUDA version
   -     ]
   +    override-dependencies = [
   +        "pywin32==306; sys_platform == 'win32'",
   +        "torch>=2.6.0", # Change here for different CUDA version
   +        "torchvision>=0.21.0", # Change here for different CUDA version
   +        "sympy==1.13.3"
   +    ]

   # 3. Add flatdict build dependency (missing in upstream, needed for setuptools>=82 compatibility)
   + [tool.uv.extra-build-dependencies]
   + flatdict = ["setuptools<82"]

   # 4. Change torch wheel index to match your CUDA version
   -    url = "https://download.pytorch.org/whl/cu130" # Change here for different CUDA version
   +    url = "https://download.pytorch.org/whl/cu124" # Change here for different CUDA version

**Step 3:** Run the install script:

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

Alternatively, you can let the install script clone and install automatically (but you cannot customize ``pyproject.toml``):

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

.. note::

   After installation, the first run may hang because ``isaacsim`` requires you to accept its EULA.
   Manually import it once to complete the agreement:

   .. code:: bash

      python
      >>> import isaacsim

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

**Method 1: Download Pre-converted PyTorch Model (Recommended)**

Pre-trained PyTorch models are available on HuggingFace, converted from the original JAX checkpoints.

.. code:: bash

   # π0.5 Polaris (Recommended)
   hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./checkpoints/RLinf-Pi05-Polaris-droid_jointpos

   # π0 Polaris
   hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./checkpoints/RLinf-Pi0-Polaris-droid_jointpos

**Method 2: Download JAX Checkpoint and Convert**

PolaRiS provides model variants trained on the DROID dataset, stored in Google Cloud Storage (GCS). You need to download the JAX checkpoint and convert it to PyTorch format.

**2.1 Download JAX Checkpoint**

.. code:: bash

   # π0.5 Polaris (Recommended)
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris /path/to/checkpoints/

   # π0 Polaris
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris /path/to/checkpoints/

**2.2 Convert to PyTorch Format**

The downloaded JAX checkpoint needs to be converted to PyTorch format to be used in RLinf:

.. code:: bash

   cd path/to/polaris/third_party/openpi
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   source .venv/bin/activate

   # π0.5 Polaris → PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi05_droid_jointpos_polaris \
       --config_name pi05_droid_jointpos_polaris \
       --output_path /path/to/checkpoints/pi05_droid_jointpos_polaris_new
   # Copy assets
   cp -r /path/to/checkpoints/pi05_droid_jointpos_polaris/assets /path/to/checkpoints/pi05_droid_jointpos_polaris_new/

   # π0 Polaris → PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi0_droid_jointpos_polaris \
       --config_name pi0_droid_jointpos_polaris \
       --output_path /path/to/checkpoints/pi0_droid_jointpos_polaris_new
   # Copy assets
   cp -r /path/to/checkpoints/pi0_droid_jointpos_polaris/assets /path/to/checkpoints/pi0_droid_jointpos_polaris_new/

The mapping between models and the ``config_name`` in YAML is as follows:

.. list-table:: **Model Checkpoint to config_name Mapping**
   :header-rows: 1
   :widths: 30 30

   * - Model
     - RLinf YAML ``config_name``
   * - π0.5 Polaris
     - ``pi05_droid_polaris``
   * - π0 Polaris
     - ``pi0_droid_polaris``

**3. Configure Model Path**

After downloading or converting, set the model path in the YAML configuration file:

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

  - ``examples/embodiment/config/polaris_train_ppo_openpi.yaml``
  - ``examples/embodiment/config/polaris_train_ppo_openpi_pi0.yaml``

- **Evaluation**

  - ``examples/embodiment/config/polaris_eval_openpi.yaml``
  - ``examples/embodiment/config/polaris_eval_openpi_pi0.yaml``

Each task has an independent environment configuration file located under ``examples/embodiment/config/env/``:

- ``polaris_droid_tapeintocontainer.yaml``
- ``polaris_droid_panclean.yaml``
- ``polaris_droid_blockstackkitchen.yaml``
- ``polaris_droid_foodbussing.yaml``
- ``polaris_droid_movelattecup.yaml``
- ``polaris_droid_organizetools.yaml``

**2. Key Parameter Configuration**

Parameters below are located in the training configuration file ``examples/embodiment/config/polaris_train_ppo_openpi.yaml``.

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

   # pi05
   bash examples/embodiment/run_embodiment.sh polaris_train_ppo_openpi
   # pi0
   bash examples/embodiment/run_embodiment.sh polaris_train_ppo_openpi_pi0

.. note::

   If you have hardcoded ``POLARIS_DATA_PATH`` in your configuration file, please ensure the path is correct.
   You can also set the environment variable before running:

   .. code-block:: bash

      export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

**5. Start Evaluation**

.. code-block:: bash

   # pi05
   bash examples/embodiment/eval_embodiment.sh polaris_eval_openpi
   # pi0
   bash examples/embodiment/eval_embodiment.sh polaris_eval_openpi_pi0

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

