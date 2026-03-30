RL on Lingbot-VLA Models
=========================

This document describes how to integrate Lingbot-VLA as a native plugin into the RLinf framework and perform end-to-end policy evaluation and reinforcement learning fine-tuning in the RoboTwin 2.0 simulation environment. Unlike the traditional WebSocket communication mode, native integration completely embeds Lingbot-VLA into RLinf's Python memory space, enabling the most efficient interaction and training.

The primary objective is to equip the model with the following capabilities:

* **Visual Understanding**: Process multi-view RGB images from robot cameras (e.g., head, wrist).
* **Language Comprehension**: Understand and generalize natural-language task descriptions.
* **Action Generation**: Directly autoregressively generate high-dimensional continuous action chunks via a large model backbone (based on Qwen2.5-VL).
* **Native Interaction**: Perform zero-latency tensor-level interaction with the RoboTwin simulation environment directly within the RLinf framework.

Environment
-----------

**RoboTwin Environment**

* **Environment**: RoboTwin 2.0 physical simulation benchmark built on Sapien.
* **Task**: Command dual-arm/single-arm robots (e.g., ALOHA) to perform complex household and manipulation skills (e.g., ``click_bell``, ``open_microwave``, ``stack_blocks_three``, etc.).
* **Observation**: RGB images captured from multiple camera views.
* **Action Space**: 14-dimensional continuous actions (for dual-arm ALOHA), including absolute poses (x, y, z, roll, pitch, yaw) of both arms and gripper openness.

Task Description Format
-----------------------

Lingbot-VLA directly uses the environment-provided natural-language task description as the text prompt input for the Vision-Language Model (VLM).

Data Structure
--------------

* **Images**: RGB images from the head and left/right wrist views.
* **Task Descriptions**: Natural-language instructions (e.g., "click the bell").
* **Actions**: Action chunks of length 50 (configurable), executed in an open-loop/closed-loop policy based on historical observations.

Algorithm
---------

**Core Algorithm Components**

* **GRPO (Group Relative Policy Optimization)**
    * Advantage estimation using group-based relative rewards.
    * Policy clipping with ratio limits.
    * KL divergence regularization.

* **Lingbot-VLA (Qwen2.5-VL based)**
    * Autoregressive action chunk generation via a Vision-Language backbone.
    * Flow-SDE (Stochastic Differential Equation) based action denoising / generation.
    * Configurable ``noise_method`` (e.g., ``flow_sde``), ``noise_level``, and ``num_steps`` for denoising.

Dependency Installation
-----------------------

To ensure perfect compatibility between the high-version Torch (2.8.0) and RLinf (Python 3.10), we have encapsulated the complex dependency isolation logic into an installation script. Please follow the steps below to build a hybrid environment.

1. Clone the RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, clone the RLinf repository and enter the main directory:

.. code-block:: bash

    git clone https://github.com/RLinf/RLinf.git
    cd RLinf
    export RLINF_PATH=$(pwd)

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Run embodied training based on RoboTwin using the Docker image:

.. code-block:: bash

    docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:embodied-rlinf0.2-robotwin

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code-block:: bash

    source switch_env lingbotvla

**Option 2: Custom Environment**

Install the Lingbot-VLA native environment and RoboTwin base dependencies in one command (the script will automatically pull the lingbot-vla source code to the `.venv/lingbot-vla` directory and handle all high-risk dependency conflicts):

.. code-block:: bash

    bash requirements/install.sh embodied --model lingbotvla --env robotwin --use-mirror
    source .venv/bin/activate

RoboTwin Repository Clone and Assets Download
---------------------------------------------

RoboTwin Assets are asset files required by the RoboTwin environment and need to be downloaded from HuggingFace.

.. code-block:: bash

   # 1. Clone RoboTwin repository
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   
   # 2. Download and extract Assets files
   bash script/_download_assets.sh

Model Download
--------------

Before starting training, download the Lingbot-VLA base weights and the Qwen backbone model from HuggingFace:

.. code-block:: bash

    # Method 1: Using git clone
    git lfs install
    git clone https://huggingface.co/robbyant/lingbot-vla-4b
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

    # Method 2: Using huggingface-hub
    pip install huggingface-hub
    huggingface-cli download robbyant/lingbot-vla-4b --local-dir lingbot-vla-4b
    huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B-Instruct
    

Then set ``rollout.model.model_path`` and ``actor.model.model_path`` in the configuration to your local model path (e.g., ``/path/to/model/lingbot-vla-4b`` or ``./lingbot-vla-4b``), and **be sure to** set the corresponding ``tokenizer_path`` to the downloaded Tokenizer path (e.g., ``/path/to/model/Qwen2.5-VL-3B-Instruct``). Otherwise, the Rollout node will throw an error when parsing text instructions.

Quick Start
-----------

Configuration Files
~~~~~~~~~~~~~~~~~~~

RLinf supports full-parameter Supervised Fine-Tuning (SFT) and reinforcement learning alignment (GRPO) for Lingbot-VLA. Relevant configuration files are as follows:

* **SFT (Behavior Cloning)**:
  ``examples/sft/config/robotwin_sft_lingbotvla.yaml``
* **GRPO (Reinforcement Learning)**:
  ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``

Key Config Snippets (SFT)
^^^^^^^^^^^^^^^^^^^^^^^^^

The core of the SFT phase lies in specifying the offline dataset path (LeRobot Parquet format), the FSDP training backend, and the batch size.

.. code-block:: yaml

    runner:
      task_type: sft
      max_epochs: 30000

    data:
      # Path to the converted LeRobot format offline dataset
      train_data_paths: "/path/to/lerobot_data"

    actor:
      training_backend: "fsdp"
      micro_batch_size: 1
      global_batch_size: 8
      model:
        model_type: "lingbotvla"
        model_path: "path/to/lingbot_model"
        tokenizer_path: "/path/to/model/Qwen2.5-VL-3B-Instruct"
        precision: bf16
        num_action_chunks: 50
        action_dim: 14

Key Config Snippets (GRPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The top-level file dynamically assembles the environment and model via Hydra, and directly overrides the core SDE sampling parameters required for GRPO reinforcement learning under ``actor.model``.

**Note**: Because Lingbot-VLA uses the unified global normalization keys (e.g., ``action.arm.position``) from ``robotwin_50.json``, there is **no need to configure or override** ``unnorm_key`` when switching between different tasks, enabling truly smooth multi-task transfer.

.. code-block:: yaml

    rollout:
      model:
        model_type: "lingbotvla"

    actor:
      model:
        model_path: "/path/to/lingbot_sft_model"
        tokenizer_path: "/path/to/model/Qwen2.5-VL-3B-Instruct"
        model_type: "lingbotvla"
        lingbotvla:
            config_path: "/path/to/lingbot-vla-4b"
        action_dim: 14
        num_action_chunks: 50
        num_steps: 10              
        noise_method: "flow_sde"   
        noise_level: 0.5           
        action_env_dim: 14         

Launch Commands
~~~~~~~~~~~~~~~

To start training with the selected configuration, run the corresponding launch script.

**Note**: Since the default tasks use a dual-arm robot, please ensure you declare the robot platform as ALOHA in your terminal before executing any launch scripts. Otherwise, the environment will fail to load the action space correctly:

.. code-block:: bash

    export ROBOT_PLATFORM="ALOHA"
    # Set ROBOTWIN_PATH environment variable
    export ROBOTWIN_PATH=/path/to/RoboTwin
    # Enter the lingbot-vla directory automatically generated by install.sh
    export LINGBOT_VLA_PATH=$(python -c "import lingbotvla; import os; print(os.path.dirname(lingbotvla.__path__[0]))")

    
**1. Launch SFT Training**

Perform supervised fine-tuning using the converted offline data:

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvla

**2. Launch GRPO Training**

For example, to fine-tune the SFT-trained model with the GRPO algorithm on the RoboTwin Click Bell task:

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla

Evaluation
----------

Lingbot-VLA provides an end-to-end evaluation script for various tasks in the RoboTwin environment (using the bell task as an example):

.. code-block:: bash

    export ROBOT_PLATFORM="ALOHA"
    bash examples/embodiment/eval_embodiment.sh robotwin_click_bell_grpo_lingbotvla_eval

For RLinf's unified VLA evaluation flow, please refer to the :doc:`VLA Evaluation Documentation <../../start/vla-eval>`.

Visualization and Results
-------------------------

**TensorBoard Logging**

.. code-block:: bash

    tensorboard --logdir ../results --port 6006

**Key Metrics**

* **Training**: ``train/actor/policy_loss``, ``train/actor/entropy_loss``, ``train/actor/approx_kl``
* **Environment**: ``env/success_once`` (episodic success rate), ``env/episode_len``, ``env/reward``
