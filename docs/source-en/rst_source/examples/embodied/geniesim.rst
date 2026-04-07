RL with GenieSim Place Workpiece
================================

This document provides a comprehensive guide to running the **Place Workpiece** task
on the **GenieSim** simulation platform within the **RLinf** framework. It covers the
full pipeline from Docker image building, SpaceMouse demonstration collection, to SAC
human-in-the-loop training.

Key features:

1. **Isaac Sim + MuJoCo dual-simulator**: Isaac Sim for high-fidelity rendering, MuJoCo for high-frequency physics;
2. **SpaceMouse Human-in-the-Loop**: The human operator can intervene during training in real time;
3. **RLPD + BC regularization**: Demonstrations feed both the demo buffer (symmetric sampling) and behavioral cloning loss;
4. **SAC + CNN Encoder**: ResNet-10 encoder for camera images, SAC for continuous action optimization.

Environment
-----------

**GenieSim Place Workpiece Environment**

- **Simulation**: GenieSim (Isaac Sim rendering + MuJoCo physics)
- **Task**: Control the G2 robot's right arm to place a workpiece into a target slot
- **Observation Space**:

  - Right wrist camera RGB image (128×128, resized from 480×480)
  - Right-arm EE state (12-dim: EE position, orientation, linear velocity, angular velocity)

- **Action Space**: 7-dimensional continuous delta actions

  - 3D position delta (dx, dy, dz)
  - 3D rotation delta (droll, dpitch, dyaw)
  - 1D gripper command

- **Reward**:

  - ``r_alive``: Exponentially decaying reward based on 3D distance and orientation error to target
  - ``r_below``: Penalty when workpiece drops below target height, preventing excessive downward pressure
  - ``r_success``: Sparse reward when the workpiece is placed at the target and held still

Algorithm
---------

Core algorithm components:

1. **SAC (Soft Actor-Critic)**

   - Dual critic networks to reduce Q-value overestimation;
   - Automatic entropy temperature tuning;
   - RLPD symmetric sampling: 50% from replay buffer, 50% from demo buffer.

2. **Human-in-the-Loop**

   - env_0 accepts real-time SpaceMouse intervention;
   - Intervened actions are stored in the replay buffer and extracted to the demo buffer;
   - BC regularization loss guides the policy toward expert behavior.

3. **CNN Encoder**

   - ResNet-10 encodes the right wrist camera image (128×128);
   - Image features are concatenated with right-arm EE state (12-dim) and fed into Actor/Critic.

Prerequisites
-------------

Hardware
~~~~~~~~

- NVIDIA GPU (RTX 3090+ recommended, VRAM ≥ 24GB)
- 3Dconnexion SpaceMouse (required for data collection)

Software
~~~~~~~~

- Ubuntu 22.04 / 24.04
- Docker with NVIDIA Container Toolkit
- NVIDIA driver ≥ 535

Repository Setup
~~~~~~~~~~~~~~~~

.. code:: bash

   mkdir workspace && cd workspace
   git clone https://github.com/AgibotTech/genie_sim.git
   git clone -b dev/geniesim https://github.com/RLinf/RLinf.git

.. code:: text

   workspace/
   ├── genie_sim/     # GenieSim repository
   └── RLinf/         # RLinf repository

Both repositories should reside under the same parent directory. For GenieSim
installation and asset downloads, refer to the
`GenieSim documentation <https://agibot-world.com/sim-evaluation/docs/#/v3>`_ .

Dependency Installation
-----------------------

1. Build Base Image
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd workspace/
   bash genie_sim/scripts/build_geniesim_rlinf_image.sh

Output: ``geniesim-rlinf:latest``. The build script uses ``genie_sim/scripts/`` as the build context, avoiding large asset directories.

2. Build Training Image
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd workspace/RLinf
   docker build \
     --build-arg BUILD_TARGET=embodied-geniesim \
     -t geniesim-rlinf-train:latest \
     -f docker/Dockerfile \
     .

3. Verify Image
~~~~~~~~~~~~~~~~

.. code:: bash

   docker run --rm --gpus all geniesim-rlinf-train:latest nvidia-smi

Assets Download
---------------

Download the CNN encoder pretrained weights:

.. code:: bash

   cd workspace/RLinf/examples/embodiment/config
   # For mainland China users, set:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir .

Running the Script
------------------

**1. Collect Demonstration Data**

Connect the SpaceMouse via USB, then run:

.. code:: bash

   cd workspace/
   bash RLinf/rlinf/envs/geniesim/scripts/run.sh collect --num-demos 50

SpaceMouse controls:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Action
     - Effect
   * - Translate device
     - Move right arm end-effector (x/y/z)
   * - Rotate device
     - Rotate right arm end-effector (roll/pitch/yaw)
   * - Press left button
     - End trajectory → save demo → environment resets
   * - Press right button
     - End trajectory → discard demo → environment resets

Collected demos are saved to ``workspace/genie_sim/sac_demo/``.

**2. Convert Demonstration Data**

.. code:: bash

   bash RLinf/rlinf/envs/geniesim/scripts/run.sh convert

Reads from ``genie_sim/sac_demo/`` by default and outputs to ``genie_sim/sac_demo_buffer/``.

**3. Start Training**

.. code:: bash

   bash RLinf/rlinf/envs/geniesim/scripts/run.sh train

During training, you can use the SpaceMouse to intervene on env_0 in real time. The remaining environments are driven by the policy.

**4. Key Configuration Files**

- **Environment config**: ``examples/embodiment/config/env/geniesim_place_workpiece.yaml``
- **Training config**: ``examples/embodiment/config/geniesim_sac_spacemouse.yaml``
- **Model config**: ``examples/embodiment/config/model/cnn_policy.yaml``

**5. Hydra Overrides**

.. code:: bash

   # Adjust discount factor
   bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.gamma=0.97

   # Adjust BC regularization coefficient
   bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.bc_coef=5.0

   # Change log path
   bash RLinf/rlinf/envs/geniesim/scripts/run.sh train runner.logger.log_path=../my_results

**6. Interactive Shell**

.. code:: bash

   bash RLinf/rlinf/envs/geniesim/scripts/run.sh shell

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code:: bash

   tensorboard --logdir workspace/results/

**2. Key Metrics**

- **Training metrics**:

  - ``critic_loss``: Critic loss, should gradually decrease
  - ``actor_loss``: Actor policy loss
  - ``q_values``: Q-value estimates, should rise steadily without exploding
  - ``entropy``: Policy entropy, indicates exploration level
  - ``bc_loss``: Behavioral cloning loss (only with demo buffer)

- **Environment metrics**:

  - ``eval/success_rate``: Evaluation success rate
  - ``env/episode_len``: Episode length
  - ``env/reward``: Step-level reward
