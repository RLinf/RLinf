RL on Dexbotic Models
======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a guide to fine-tuning the **Dexbotic** VLA models with
reinforcement learning using the RLinf framework. Dexbotic (`<https://github.com/dexmal/dexbotic>`__) is an open-source
Vision-Language-Action toolbox from Dexmal, a unified implementation of various embodied models. This example covers the LIBERO Spatial
benchmark with the Dexbotic π\ :sub:`0`\ model.

The primary objective is to develop a model capable of robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions via flow-based
   diffusion denoising.
4. **Reinforcement Learning**: Optimizing the policy via PPO with environment
   feedback.

Environment
-----------

**LIBERO Environment**

- **Environment**: LIBERO simulation benchmark built on top of *robosuite*
  (MuJoCo).
- **Task**: Command a 7-DoF robotic arm to perform household manipulation skills
  (pick-and-place, stacking, spatial rearrangement).
- **Observation**: RGB images (typical resolutions 128 × 128 or 224 × 224)
  captured by off-screen cameras placed around the workspace.
- **Action Space**: 7-dimensional continuous actions
  - 3D end-effector position control (x, y, z)
  - 3D rotation control (roll, pitch, yaw)
  - Gripper control (open / close)

**Task Description Format**

Dexbotic uses the environment-provided natural-language task description as the
language model input.

**Data Structure**

- **Images**: Main-view and wrist-view RGB tensors, each of shape
  ``[batch_size, 224, 224, 3]``
- **States**: End-effector pose (position + orientation) and gripper state.
- **Task Descriptions**: Natural-language instructions
- **Actions**: Action chunks of length 50 (configurable); actions are replanned
  every N steps.

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)
   - Policy clipping with ratio limits
   - Value function clipping
   - Entropy regularization

2. **Dexbotic (π\ :sub:`0.5`\ -based VLA)**

   - Flow-matching / flow-SDE action generation
   - Diffusion denoising for action chunks
   - Value head for critic function
   - Configurable ``noise_method`` (e.g. ``flow_sde``), ``noise_level``, and
     ``num_steps`` for denoising

Dependency Installation
-----------------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use the Docker image for LIBERO-based embodied training:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env dexbotic

**Option 2: Custom Environment**

Install dependencies directly in your environment:

.. code:: bash

   bash requirements/install.sh embodied --model dexbotic --env maniskill_libero
   source .venv/bin/activate

Model Download
--------------

Before starting training, download the Dexbotic SFT model from HuggingFace:

.. code:: bash

   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/Dexmal/libero-db-pi0

   # Method 2: Using huggingface-hub
   pip install huggingface-hub
   huggingface-cli download Dexmal/libero-db-pi0 --local-dir libero-db-pi0

Then set ``rollout.model.model_path`` and ``actor.model.model_path`` in your
configuration to the local path (e.g. ``/path/to/model/Dexbotic-Pi05-SFT`` or
``./libero-db-pi0``).

Quick Start
-----------

**Configuration File**

- **Dexbotic + PPO + LIBERO Spatial**:
  ``examples/embodiment/config/libero_spatial_ppo_dexbotic_pi0.yaml``

**Key Config Snippets**

.. code:: yaml

   rollout:
     model:
       model_path: "/path/to/model/Dexbotic-Pi05-SFT"
   actor:
     model:
       model_path: "/path/to/model/Dexbotic-Pi05-SFT"
       num_action_chunks: 5
       num_steps: 4
       action_dim: 7
       add_value_head: True
       dexbotic:
         num_images_in_input: 2
         noise_level: 0.5
         noise_method: "flow_sde"
         train_expert_only: True
         detach_critic_input: True

**Launch Command**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_dexbotic_pi0

Evaluation
----------

Dexbotic provides a dedicated evaluation script for LIBERO in
``toolkits/eval_scripts_dexbotic/``:

.. code-block:: bash

   python toolkits/eval_scripts_dexbotic/libero_eval.py \
      --config libero_spatial_dexbotic \
      --ckpt_path /path/to/checkpoint \
      --action_chunk_size 50 \
      --num_diffusion_steps 10

You can also use RLinf's unified VLA evaluation flow; refer to the
:doc:`VLA Evaluation Documentation <../../start/vla-eval>` for details.

Visualization and Results
-------------------------

**TensorBoard Logging**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**Key Metrics**

- **Training**: ``train/actor/policy_loss``, ``train/critic/value_loss``,
  ``train/actor/approx_kl``
- **Environment**: ``env/success_once`` (episodic success rate),
  ``env/episode_len``
