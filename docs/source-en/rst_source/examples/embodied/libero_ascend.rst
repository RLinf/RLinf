RL with LIBERO on Ascend CANN
=============================

This page covers the Ascend CANN-specific setup for running the LIBERO RL
example in RLinf. It focuses on dependency installation and runtime access to
the host Ascend driver. The LIBERO task description, PPO/GRPO algorithm details,
model download, configuration files, metrics, and results are platform
independent; for those sections, refer to :doc:`RL with LIBERO Benchmark <libero>`.

Dependency Installation
-----------------------

The key difference from the NVIDIA workflow is that dependencies must be
installed with the Ascend backend. ``install.sh`` installs the CPU PyTorch wheel
and then adds the matching ``torch-npu`` package for Ascend.

Option 1: Docker Image
~~~~~~~~~~~~~~~~~~~~~~

Use an Ascend LIBERO image, or build one from the RLinf Dockerfile. The
container must be run in privileged mode and the host Ascend driver must be
mounted into the container:

.. code-block:: bash

   docker run -it --rm \
      --privileged \
      --ipc=host \
      --shm-size 20g \
      --network host \
      --name rlinf-ascend-libero \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-libero-cann9.0
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-libero-cann9.0

Inside the container, switch to the OpenVLA-OFT environment:

.. code-block:: bash

   source switch_env openvla-oft

If you build the Docker image yourself, pass the Ascend platform and CANN image
version explicitly. ``CANN_VER`` includes the hardware tag used by the base
image:

.. code-block:: bash

   docker build \
      --build-arg PLATFORM=ascend \
      --build-arg CANN_VER=9.0.0-910b \
      --build-arg UBUNTU_VER=22.04 \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-cann9 .

The Dockerfile uses the CANN base image:

.. code-block:: text

   swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VER}-ubuntu${UBUNTU_VER}-py3.11

Option 2: Native Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install dependencies with ``install.sh`` and pass ``--platform ascend``:

.. code-block:: bash

   bash requirements/install.sh --platform ascend embodied --model openvla-oft --env libero
   source .venv/bin/activate

For faster downloads in mainland China, add ``--use-mirror``:

.. code-block:: bash

   bash requirements/install.sh --use-mirror --platform ascend embodied --model openvla-oft --env libero

LIBERO CPU Rendering
--------------------

Use CPU rendering for LIBERO on Ascend. Set both rendering variables before
launching the training script:

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

The helper script ``examples/embodiment/run_embodiment.sh`` respects these
environment variables. If they are unset, it keeps the default EGL behavior used
by other examples.

Launch LIBERO on Ascend
-----------------------

After the dependencies and model paths are ready, run the same LIBERO
configuration described in :doc:`the main LIBERO guide <libero>`, but keep OSMesa
enabled:

.. code-block:: bash

   MUJOCO_GL=osmesa \
   PYOPENGL_PLATFORM=osmesa \
   ROBOT_PLATFORM=LIBERO \
   bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

For PPO, use the PPO config from the original LIBERO page:

.. code-block:: bash

   MUJOCO_GL=osmesa \
   PYOPENGL_PLATFORM=osmesa \
   ROBOT_PLATFORM=LIBERO \
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openvlaoft

What Stays the Same
-------------------

- Use the same LIBERO configs documented in :doc:`RL with LIBERO Benchmark <libero>`.
- Use the same model download and ``model_path`` configuration flow.
- Use the same PPO/GRPO algorithm settings and placement concepts.
- Monitor the same training, rollout, and environment metrics.
