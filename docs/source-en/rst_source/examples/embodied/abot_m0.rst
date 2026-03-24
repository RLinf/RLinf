RL on ABot-M0 Model
====================

This document describes how to integrate ABot-M0 as a native plugin into RLinf and run end-to-end embodied RL smoke validation on LIBERO tasks.
Unlike external serving mode, native integration keeps ABot-M0 in RLinf's Python process space for direct tensor-level interaction.

The current objective of this page is smoke-level validation:

* **Dependency Wiring**: Verify RLinf + ABot-Manipulation + VGGT are importable in one environment.
* **Native Rollout**: Verify ABot-M0 can generate action chunks inside RLinf rollout workers.
* **Actor-Rollout Sync**: Verify policy weight sync and training loop run without parameter mismatch errors.
* **Smoke Training**: Verify minimal PPO loop on LIBERO config can start and complete.

Environment
-----------

**LIBERO Environment**

* **Environment**: LIBERO benchmark through RLinf embodied pipeline.
* **Task**: Language-conditioned manipulation tasks from LIBERO suites.
* **Observation**: Multi-view RGB images and robot state.
* **Action Space**: Continuous action chunks in ABot-M0 policy format.

Data Structure
--------------

* **Images**: Multi-view RGB inputs mapped to ``main_images``.
* **Task Descriptions**: Natural-language instructions mapped to ``task_descriptions``.
* **States**: Robot state mapped to ``states`` and normalized to ABot expected layout.

Algorithm
---------

**Core Components**

* **PPO (actor_critic)**
   * Advantage estimation using GAE (Generalized Advantage Estimation).
   * Policy clipping with ratio limits.
   * Value function clipping.
   * Entropy regularization.

* **ABot-M0 Native Policy**
   * General-purpose VLA model for robotic manipulation with cross-embodiment training.
   * Action Manifold Learning (AML) for efficient and stable continuous action prediction.
   * Modular perception design that integrates VLM semantics and optional 3D priors (via ABot-Manipulation and VGGT).
   * RLinf-native wrapper for rollout action generation and training-time logprob/value recomputation.

Dependency Installation
-----------------------

1. Dependency Source Options (ABot and VGGT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf supports both dependency-source workflows (implemented in ``requirements/install.sh``):

* **Option A (manual clone + explicit path)**: clone ABot-Manipulation and VGGT in advance, then export ``ABOT_PATH`` and ``VGGT_PATH``.
* **Option B (script auto-clone)**: do not set these env vars; the install script will clone ABot and VGGT under the venv directory automatically.

Option A example:

.. code-block:: bash

   git clone https://github.com/amap-cvlab/ABot-Manipulation.git
   git clone https://github.com/facebookresearch/vggt.git

   cd <path_to_RLinf>
   export ABOT_PATH=<path_to_ABot-Manipulation>
   export VGGT_PATH=<path_to_vggt>

2. Install by Script (Custom Environment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env maniskill_libero --install-rlinf
   source .venv/bin/activate

3. Download ABot-M0 Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download ABot-M0 LIBERO checkpoint from:
``https://huggingface.co/acvlab/ABot-M0-LIBERO/tree/main``

Example with huggingface-cli:

.. code-block:: bash

   pip install -U "huggingface_hub[cli]"
   huggingface-cli download acvlab/ABot-M0-LIBERO \
     --local-dir <path_to_ABot-M0-LIBERO>

4. Configure ``model_path`` in ABot Smoke YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set both fields in:
``examples/embodiment/config/libero_10_ppo_abot_m0_smoke.yaml``

* ``rollout.model.model_path``
* ``actor.model.model_path``

to your local ABot-M0 checkpoint path.

5. Import Smoke Check
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import rlinf; import ABot; import vggt; print('IMPORT_SMOKE_OK')"

If the command prints ``IMPORT_SMOKE_OK``, the package-level dependency wiring is valid.

Docker Installation (Recommended for Reproducibility)
-----------------------------------------------------

You can also validate ABot-M0 with Docker-based environment setup:

.. code-block:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

   docker run -it --gpus all \
     --shm-size 100g \
     --net=host \
     --name rlinf-abot \
     -e NVIDIA_DRIVER_CAPABILITIES=all \
       -v <path_to_RLinf>:/workspace/RLinf \
       -v <path_to_ABot-Manipulation>:/workspace/ABot-Manipulation \
       -v <path_to_vggt>:/workspace/vggt \
     rlinf/rlinf:agentic-rlinf0.2-maniskill_libero /bin/bash

Inside container:

.. code-block:: bash

   cd /workspace/RLinf
   export ABOT_PATH=/workspace/ABot-Manipulation
   export VGGT_PATH=/workspace/vggt
   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env maniskill_libero --install-rlinf
   source .venv/bin/activate
   python -c "import rlinf; import ABot; import vggt; print('IMPORT_SMOKE_OK')"

Quick Start (Smoke Test)
------------------------

Use the ABot smoke config file:

.. code-block:: bash

   # examples/embodiment/config/libero_10_ppo_abot_m0_smoke.yaml

Set runtime environment variables:

.. code-block:: bash

   export REPO_PATH=$(pwd)
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
   export CUDA_VISIBLE_DEVICES=0,1
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export ROBOT_PLATFORM=LIBERO

Start Ray and run smoke training:

.. code-block:: bash

   ray stop || true
   ray start --head --port=6379

   python examples/embodiment/train_embodied_agent.py \
     --config-name libero_10_ppo_abot_m0_smoke \
     runner.max_epochs=3 \
     algorithm.rollout_epoch=2 \
     env.train.total_num_envs=4 \
     env.eval.total_num_envs=2 \
     actor.micro_batch_size=2 \
     actor.global_batch_size=8

   ray stop

Current Validation Status
-------------------------

* **Passed**: The minimal smoke command above has been verified.
* **Pending**: Full Docker-path smoke verification is not completed yet.

Known Dependency Compatibility Notes
------------------------------------

During script-based installation, package version mismatches may appear depending on resolver outcomes and prior environment state.
There may be a ``peft`` version compatibility issue during installation, because RLinf has a fixed top-level ``peft`` override in dependency resolution.

For ABot-M0, we recommend using ``peft==0.18.1`` in the runtime environment.
After the install script finishes, explicitly install the ABot-M0 PEFT version:

.. code-block:: bash

   source .venv/bin/activate
   uv pip install peft==0.18.1

Then verify:

.. code-block:: bash

   python -c "import peft; print(peft.__version__)"

Visualization
-------------

.. code-block:: bash

   tensorboard --logdir logs --port 6006
