The Ascend LIBERO image exposes the host NPU drivers to the model environment.
From the repository root on the host, start the container with those drivers
mounted:

.. code-block:: bash

   docker run -it --rm \
      --privileged \
      --ipc=host \
      --shm-size 20g \
      --network host \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /var/log/npu:/usr/slog \
      -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
      -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
      -v "$PWD":/workspace/RLinf \
      -w /workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-libero-cann9.0 bash

For downloads from mainland China, the image is also available under
``docker.1ms.run/rlinf/rlinf`` with the same tag. To expose specific NPUs,
replace ``--privileged`` with the following device arguments, adding a
``/dev/davinciN`` entry for each NPU you will use:

.. code-block:: text

   --device=/dev/davinci_manager
   --device=/dev/devmm_svm
   --device=/dev/hisi_hdc
   --device=/dev/davinci0

To build the image from your checkout, run this on the host and substitute
``rlinf-libero-cann9`` for the image in the command above:

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=ascend \
      --build-arg CANN_VER=9.0.0-910b \
      --build-arg UBUNTU_VER=22.04 \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-cann9 .

``CANN_VER`` includes the hardware suffix in the Ascend base-image tag.
The Dockerfile also accepts ``ASCEND_BASE_IMAGE`` to select a different full
base-image reference.
