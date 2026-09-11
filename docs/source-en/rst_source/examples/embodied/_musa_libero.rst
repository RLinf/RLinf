MUSA relies on the PyTorch and ``torch_musa`` pair in the Moore Threads
training-suite image. Start that image with the ``mthreads`` runtime so the
driver libraries and devices are available inside the container:

.. code-block:: bash

   export MUSA_IMAGE=registry.mthreads.com/mcctest/ai/training-suite:v2.1.5-musa4.3.7
   docker run -it --rm --runtime=mthreads \
      --ipc=host --shm-size=100g \
      -e MTHREADS_VISIBLE_DEVICES=all \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      "$MUSA_IMAGE" bash

Verify device access before installing a model environment:

.. code-block:: bash

   mthreads-gmi
   python -c "import torch, torch_musa; print(torch.musa.device_count())"

.. warning::

   Resolve a zero device count before training. Keep the image's matching
   PyTorch and ``torch_musa`` packages; installing a generic torch wheel can
   replace the MUSA build. The embodied recipes use the Hugging Face rollout
   backend because the MUSA install skips the CUDA-only vLLM and SGLang kernels.

To build an RLinf image from your checkout, use BuildKit on the host:

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=musa \
      --build-arg MUSA_VER=v2.1.5-musa4.3.7 \
      --build-arg BUILD_TARGET=embodied-maniskill_libero \
      -t rlinf:embodied-maniskill_libero-musa .

The runtime injects the MUSA driver libraries, so image-build steps cannot
import torch before a device-aware container starts. BuildKit also avoids
resolving unrelated CUDA and ROCm base-image stages.
