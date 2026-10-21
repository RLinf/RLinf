RL with RoboDojo Benchmark
==========================

`RoboDojo <https://robodojo-benchmark.github.io/>`__ is an Isaac Sim-based
benchmark for dual-arm manipulation. It includes tabletop rearrangement,
pick-and-place, and tool-use tasks for robots such as ARX-X5. RLinf exposes
RoboDojo as a Gym environment for VLA rollout, training, and evaluation.

Overview
--------

Run RoboDojo on Isaac Sim and drive a dual-arm robot through RLinf's unified
environment interface.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Simulator
      :text-align: center

      RoboDojo · Isaac Sim 5.1

   .. grid-item-card:: Robot
      :text-align: center

      Dual-arm ARX-X5

   .. grid-item-card:: Example task
      :text-align: center

      Put Bottles Into Dustbin

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPU · Linux x86_64

| **You'll do:** install → prepare scene assets → verify the installation →
  launch a single-environment task.
| **Prerequisites:** :doc:`Base installation </rst_source/start/installation>` ·
  an NVIDIA CUDA GPU · at least 80 GB of free disk space.

Tasks
~~~~~

RLinf currently provides a
``put_bottles_into_dustbin`` environment configuration as the integration and
verification example.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``put_bottles_into_dustbin``
     - Grab bottles from the table with both arms and place them in a dustbin.

The configuration is located at:

.. code-block:: text

   examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml

Other RoboDojo tasks use the same environment interface. Change
``task_config.task_name`` and the corresponding task parameters.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Specification
   * - ``main_images``
     - Head-camera RGB, ``[B, H, W, 3]`` uint8.
   * - ``wrist_images``
     - Optional left/right wrist-camera RGB, ``[B, n, H, W, 3]`` uint8, or ``None``.
   * - ``states``
     - Left/right arm joints and gripper states, ``[B, 14]`` float32 by default.
   * - ``task_descriptions``
     - Natural-language instructions generated from the current scene.
   * - ``actions``
     - 14 continuous values by default: left arm 6, right arm 6, left gripper 1,
       right gripper 1.
   * - Reward
     - Sparse task-completion reward by default.

Installation
------------

RoboDojo currently supports Linux x86_64 with an NVIDIA CUDA GPU. CPU-only,
macOS, and AMD ROCm execution are not supported. Check the driver and disk
space before installing:

.. code-block:: bash

   nvidia-smi
   uname -m
   df -h .

**Custom environment**

From the RLinf repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --env robodojo
   source .venv/bin/activate

Users in mainland China can add ``--use-mirror``. To include OpenPI:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robodojo

The installer installs PyTorch 2.7, Isaac Sim 5.1, pinned IsaacLab and cuRobo,
and applies the validated compatibility pins. Do not install
``rlinf-robodojo-runtime`` separately; a plain dependency resolve can produce
an environment that cannot start.

.. note::

   The installer sets ``OMNI_KIT_ACCEPT_EULA=YES``. Continuing with the
   installation means accepting the applicable NVIDIA Omniverse license terms.

Download Scene Assets
~~~~~~~~~~~~~~~~~~~~~

Dependency installation does not download scene assets. Set a persistent asset
root, then download the approximately 39 GB ``Assets/`` tree:

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   ROBODOJO_DOWNLOAD_ASSETS=1 \
     bash requirements/install.sh embodied --env robodojo
   source .venv/bin/activate

You can also place existing assets in this directory. The required layout is:

.. code-block:: text

   /path/to/robodojo/
   └── Assets/
       ├── Robots/
       ├── Object/
       ├── Material/
       └── Eval_Layout/

.. note::

   ``ROBODOJO_ASSETS_ROOT`` must point to the parent of ``Assets/``, not to
   ``Assets/`` itself. Set ``HF_ENDPOINT`` before installation when using a
   Hugging Face mirror.

Verify the Installation
~~~~~~~~~~~~~~~~~~~~~~~

This check validates the Python package and CUDA without starting the simulator:

.. code-block:: bash

   python - <<'PY'
   import torch
   from robodojo_runtime import assets_root, source_root
   from robodojo_runtime.bridge import VectorEnv

   print("torch:", torch.__version__)
   print("cuda:", torch.cuda.is_available())
   print("source:", source_root())
   print("assets:", assets_root())
   PY

Run It
------

**Single-environment launch**

Start with one environment to verify Isaac Sim, cameras, and the downloaded
assets:

.. code-block:: bash

   python - <<'PY'
   import numpy as np
   from robodojo_runtime.bridge import VectorEnv

   env = VectorEnv(
       {
           "task_name": "put_bottles_into_dustbin",
           "env_cfg_type": "arx_x5",
           "headless": True,
           "cuda_device": 0,
           "max_episode_steps": 700,
       },
       n_envs=1,
       env_seeds=[0],
   )
   try:
       env.reset()
       obs = env.get_obs()[0]
       print(obs["full_image"].shape, obs["state"].shape)
       env.step(np.zeros((1, 1, 14), dtype=np.float32))
   finally:
       env.close()
   PY

Do not start a policy server, video recording, and multiple environments during
the first launch. Once reset, step, and close work for one environment, increase
``env.train.total_num_envs``.

**Use in a training configuration**

Reference the environment preset from your training configuration:

.. code-block:: text

   examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml

The policy's ``action_dim`` must equal the sum of ``action_dims``. The example
uses ``[6, 6, 1, 1]``, which is a 14-dimensional action.

Visualization and Results
-------------------------

RoboDojo is launched through the same embodied training entrypoint as other
RLinf environments. The primary metric is ``env/success_once``; see
:doc:`Training metrics <../../reference/metrics>` for the full metric list.

Enable rollout videos in the environment configuration when needed:

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

Common Problems
---------------

**``robodojo_runtime`` is not found**

Activate the virtual environment created by the installer. If you used
``--venv``, activate that environment.

**Dependency resolution fails**

Do not install Isaac Sim, IsaacLab, cuRobo, or the RoboDojo runtime manually.
Remove the failed environment and rerun the installation command so the pinned
versions and compatibility overrides are applied together.

**``pip check`` reports Isaac Sim/IsaacLab version differences**

These packages use overlapping exact version pins. The installer selects the
runtime-tested combination. Do not manually upgrade ``starlette``,
``uvicorn``, or ``typing-extensions``.

**``torch.cuda.is_available()`` returns ``False``**

Run ``nvidia-smi`` to check the driver and GPU visibility. RoboDojo does not
currently support CPU-only, macOS, or AMD ROCm execution.

**``Assets``, ``cam_head``, or task layouts are missing**

Verify that ``ROBODOJO_ASSETS_ROOT`` points to the parent of ``Assets/`` and
that all four required subdirectories exist. Remove incomplete downloads and
download them again.

**GPU memory is insufficient**

Set the environment count to one and stop other training or inference
processes. RoboDojo starts a separate Isaac Sim process per parallel
environment, so multi-environment settings depend on the measured per-scene
memory use.

**GPU memory remains allocated after exit**

Call ``env.close()`` from a ``try/finally`` block. The RoboDojo adapter supports
a complete close and recreate, not suspend and resume.
