RL with LIBERO on AMD ROCm
==========================

This page covers the AMD ROCm-specific setup for running the LIBERO RL example
in RLinf. It intentionally focuses on dependency installation and runtime
environment variables. The LIBERO task description, PPO/GRPO algorithm details,
model download, configuration files, metrics, and results are platform
independent; for those sections, refer to :doc:`RL with LIBERO Benchmark <libero>`.

Dependency Installation
-----------------------

The key difference from the NVIDIA workflow is that dependencies must be
installed with the ROCm backend and LIBERO should use CPU rendering through
OSMesa.

Option 1: Docker Image
~~~~~~~~~~~~~~~~~~~~~~

Use the ROCm LIBERO image when possible:

.. code-block:: bash

   docker run -it --rm \
      --device=/dev/kfd \
      --device=/dev/dri \
      --group-add video \
      --ipc=host \
      --shm-size 20g \
      --network host \
      --name rlinf-amd-libero \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4
      # rocm7.2.3: rlinf/rlinf:agentic-rlinf0.2-libero-rocm7.2.3

Inside the container, switch to the OpenVLA-OFT environment:

.. code-block:: bash

   source switch_env openvla-oft

If you build the Docker image yourself, pass the AMD platform and the ROCm
version explicitly:

.. code-block:: bash

   docker build \
      --build-arg PLATFORM=amd \
      --build-arg ROCM_VER=6.4 \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-rocm6.4 .

When building in an environment where the AMD device is not visible, ROCm
extension builds such as ``flash-attn`` cannot auto-detect the GPU architecture.
Pass the target architecture list explicitly:

.. code-block:: bash

   docker build \
      --build-arg PLATFORM=amd \
      --build-arg ROCM_VER=6.4 \
      --build-arg 'ROCM_ARCHS=gfx90a;gfx942' \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-rocm6.4 .

Choose the ``gfx`` values that match the target AMD GPUs. RLinf forwards
``ROCM_ARCHS`` to ROCm build variables such as ``GPU_ARCHS`` so source builds do
not fall back to ``--offload-arch=native`` during Docker builds.

Option 2: Native Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install dependencies with ``install.sh`` and pass ``--platform amd``:

.. code-block:: bash

   bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env libero
   source .venv/bin/activate

If ROCm is installed in the standard location, ``--rocm`` can usually be omitted
and the installer will auto-detect it. For faster downloads in mainland China,
add ``--use-mirror``:

.. code-block:: bash

   bash requirements/install.sh --use-mirror --platform amd --rocm 6.4 embodied --model openvla-oft --env libero

LIBERO CPU Rendering
--------------------

Use CPU rendering for LIBERO on AMD. Set both rendering variables before
launching the training script:

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

The helper script ``examples/embodiment/run_embodiment.sh`` respects these
environment variables. If they are unset, it keeps the default EGL behavior used
by other examples.

Launch LIBERO on AMD
--------------------

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
