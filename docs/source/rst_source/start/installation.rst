Installation
============

RLinf supports multiple backend engines for both training and inference. As of now, the following configurations are available:

- **Megatron** and **SGLang** for training LLMs on MATH tasks.
- **FSDP** and **Huggingface** for training VLAs on LIBERO and ManiSkill3.

Backend Engines
---------------

1. **Training Engines**

   - **FSDP**: A simple and efficient training engine that is beginner-friendly, widely compatible, easy to use, and supports native PyTorch modules.

   - **Megatron**: Designed for experienced developers seeking maximum performance. It supports a variety of parallel configurations and offers SOTA training speed and scalability.

2. **Inference Engines**

   - **SGLang**: A mature and widely adopted inference engine that offers many advanced features and optimizations.

   - **Huggingface**: Easy to use, with native APIs provided by the Huggingface ecosystem.

Installation Methods
--------------------

RLinf provides two installation options. We **recommend using Docker**, as it provides the fastest and most reproducible environment.

However, if your system is incompatible with the Docker image, you can also install RLinf manually in a Python environment.

Install from Docker Image
-------------------------

We provide two pre-built Docker images optimized for different backend engine combinations:

- The official image for **Megatron** and **SGLang**: ``rlinf/rlinf:math-rlinf0.1-torch2.5.1-sglang0.4.4-vllm0.7.1-megatron0.11.0-te2.1``
- The official image for **FSDP** and **Huggingface**: ``rlinf/rlinf:agentic-openvla-rlinf0.1-torch2.5.1`` and ``rlinf/rlinf:agentic-openvlaoft-rlinf0.1-torch2.5.1``.

Once you've identified the appropriate image for your setup, pull the Docker image:

.. code-block:: bash

   docker pull rlinf/rlinf:CHOSEN_IMAGE

Then, start the container using the pulled image:

.. code-block:: bash

   docker run -it --gpus all \
      --shm-size 100g \
      --net=host \
      --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      --name rlinf \
      rlinf/rlinf:CHOSEN_IMAGE /bin/bash

Inside the container, clone the RLinf repository:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

.. tip::

   For multi-node training, make sure to clone the repository in shared storage so that every node has access to it.



Install from Custom Environment
-------------------------------

TODO:

1. Install CUDA and cuDNN.
2. Create a new conda environment.
3. Install dependencies such as Megatron and SGLang manually.
4. Install Python dependencies:

.. code-block:: bash

   pip install -r requirements.txt
