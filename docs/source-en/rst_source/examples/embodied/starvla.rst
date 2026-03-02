RL on StarVLA Models
====================

This page explains how to run reinforcement learning fine-tuning for **StarVLA**
in **RLinf**. StarVLA is an open-source Vision-Language-Action (VLA) toolkit
that composes a VLM backbone and an action head in a modular way. This example
uses **LIBERO Spatial** with the **QwenOFT** setup.

Training Goal Overview
----------------------

This training setup focuses on:

1. Vision understanding from RGB observations.
2. Language understanding from natural-language task instructions.
3. Action generation by reading VLM hidden states and regressing an action
   chunk in parallel with an MLP action head.
4. RL optimization with **GRPO** based on environment feedback.

Environment and Interface Conventions
-------------------------------------

LIBERO Environment
^^^^^^^^^^^^^^^^^^

* **Environment**: LIBERO benchmark based on robosuite / MuJoCo.
* **Task**: control a 7-DoF robot arm for household manipulation skills.
* **Observation**: multi-view RGB images with optional proprio/state.
* **Action space**: continuous actions, commonly 7-D
  (6-D end-effector delta pose + 1-D gripper).
* **Robot platform**: RLinf selects platform-dependent action dimensions and
  (un)normalization behavior via ``ROBOT_PLATFORM``. This page assumes
  ``ROBOT_PLATFORM=libero``.

Task Description Format
^^^^^^^^^^^^^^^^^^^^^^^

StarVLA directly consumes environment-provided natural-language task
descriptions as language-model input.

Environment Observation Structure (``env_obs``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the RLinf StarVLA wrapper, ``env_obs`` is expected to be a dict in
batch-first format (dimension 0 is batch size ``B``).

Required fields:

* ``main_images``: main-view RGB, ``torch.uint8``, shape ``[B, H, W, 3]``.
* ``states``: proprio/state tensor, ``torch.float32``, shape ``[B, D_state]``.
* ``task_descriptions``: natural-language descriptions, ``list[str]`` with
  length ``B``.

Optional fields:

* ``wrist_images``: wrist-view RGB, ``torch.uint8``, shape ``[B, H, W, 3]``.
* ``extra_view_images``: additional RGB views, recommended shape
  ``[B, V, H, W, 3]`` where ``V`` is number of extra views. A single extra
  view may also be provided as ``[B, H, W, 3]`` and is treated as ``V=1``.

In default LIBERO usage, ``states`` is commonly:

* end-effector position ``(x, y, z)`` (3-D)
* end-effector axis-angle ``(rx, ry, rz)`` (3-D)
* gripper state (originally 2-D)

So ``D_state`` is often ``3 + 3 + 2 = 8``. If a checkpoint expects 7-D state,
the wrapper compresses the 2-D gripper state into:

``[x, y, z, rx, ry, rz, g_mean]`` where ``g_mean = 0.5 * (g0 + g1)``.

Action Chunk Interface
^^^^^^^^^^^^^^^^^^^^^^

StarVLA inference outputs chunked actions:

* ``actions``: ``torch.float32``, shape ``[B, T, D_action]``
* ``T = actor.model.num_action_chunks``: chunk length / planning horizon
* ``D_action = actor.model.action_dim``: action dimension (commonly 7 on LIBERO)

Rollout typically follows a receding-horizon strategy: each policy forward pass
predicts ``T`` actions; the environment executes the first ``N`` steps
(``1 <= N <= T``), then replans.

Algorithm Notes
---------------

* **StarVLA (QwenOFT)**: the VLM backbone provides multimodal understanding,
  while the MLP action head regresses continuous chunked actions in parallel
  (non-diffusion decoding).
* **GRPO**: policy optimization with environment feedback, integrated with
  RLinf embodied training and LIBERO.

Installation
------------

1. Clone RLinf
^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env starvla

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model starvla --env maniskill_libero
   source .venv/bin/activate

Model Download
----------------

Before training, download the required StarVLA checkpoint and base VLM:

* ``StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1``
* ``Qwen/Qwen2.5-VL-3B-Instruct``

.. code-block:: bash

   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1
   git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   uv pip install huggingface-hub
   hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
   hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

.. note::

   After download, update ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` so
   ``framework.qwenvl.base_vlm`` points to your local
   ``Qwen2.5-VL-3B-Instruct`` path.

Quickstart
----------

Config file
^^^^^^^^^^^

StarVLA + GRPO + LIBERO (10 tasks) example config:

* ``examples/embodiment/config/libero_10_grpo_starvla.yaml``

Key config snippet
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   rollout:
     model:
       model_path: "/path/to/model/Qwen2.5-VL-OFT-LIBERO-4in1"

   actor:
     model:
       model_path: "/path/to/model/Qwen2.5-VL-OFT-LIBERO-4in1"
       action_dim: 7
       num_action_chunks: 8
       starvla:
         framework_name: "QwenOFT"
         expected_action_dim: ${actor.model.action_dim}
         expected_num_action_chunks: ${actor.model.num_action_chunks}

Run training
^^^^^^^^^^^^

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_grpo_starvla

Evaluation
----------

For evaluation, we recommend RLinf's unified VLA evaluation workflow
(see the Embodied VLA Evaluation tutorial in RLinf docs).
