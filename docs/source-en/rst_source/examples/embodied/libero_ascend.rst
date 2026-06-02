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
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /var/log/npu:/usr/slog \
      -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
      -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
      -v .:/workspace/RLinf \
      swr.cn-north-9.myhuaweicloud.com/rlinf/rlinf_npu:v1.0.1-910b
      # above is 910b image，for 910c(a3) image please use below image:
      # swr.cn-north-9.myhuaweicloud.com/rlinf/rlinf_npu:v1.0.1-a3

If you don't want to use privileged flag, then you need to add serval devices, and manually add NPU:

.. code-block:: bash

      # adding text below to the above command
      --device=/dev/davinci_manager \
      --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      --device=dev/davinci0 # first npu

If you build the Docker image yourself, pass the Ascend platform and CANN image
version explicitly. ``CANN_VER`` includes the hardware tag used by the base
image:

.. code-block:: bash

   docker build \
      --build-arg PLATFORM=ascend \
      --build-arg CANN_VER=8.5.0-910b \
      --build-arg UBUNTU_VER=22.04 \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-cann9 .

The Dockerfile uses the CANN base image:

.. code-block:: text

   swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VER}-ubuntu${UBUNTU_VER}-py3.11

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
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

For PPO, use the PPO config from the original LIBERO page:

.. code-block:: bash

   MUJOCO_GL=osmesa \
   PYOPENGL_PLATFORM=osmesa \
   ROBOT_PLATFORM=LIBERO \
   bash examples/embodiment/run_embodiment.sh ibero_10_ppo_openpi_pi05

What Stays the Same
-------------------

- Use the same LIBERO configs documented in :doc:`RL with LIBERO Benchmark <libero>`.
- Use the same model download and ``model_path`` configuration flow.
- Use the same PPO/GRPO algorithm settings and placement concepts.
- Monitor the same training, rollout, and environment metrics.
