The ROCm LIBERO container needs access to the AMD kernel and rendering devices.
Start it from the repository root on the host:

.. code-block:: bash

   docker run -it --rm \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --ipc=host --shm-size 20g --network host \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-rocm7.2 bash

This image builds on ROCm 7.2 and carries every ManiSkill and LIBERO model
environment. The earlier LIBERO-only images remain available as
``agentic-rlinf0.3-libero-rocm6.4`` and ``agentic-rlinf0.3-libero-rocm7.2.3``.
For downloads from mainland China, use ``docker.1ms.run/rlinf/rlinf`` with the
same tag.

To build the shared model image from your checkout, run this command on the
host, then substitute ``rlinf:embodied-maniskill_libero-rocm7.2`` in the
container command above:

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=amd \
      --build-arg ROCM_VER=7.2 \
      --build-arg 'ROCM_ARCHS=gfx90a;gfx942' \
      --build-arg BUILD_TARGET=embodied-maniskill_libero \
      -t rlinf:embodied-maniskill_libero-rocm7.2 .

.. warning::

   Match ``ROCM_ARCHS`` to the target GPUs. During Docker builds the devices may
   be invisible, so extensions such as ``flash-attn`` need explicit architecture
   values. The Dockerfile forwards them to the ROCm build tools.
