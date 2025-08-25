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

This installation is divided into two steps depending on the experiments you wish to run.

First, for all experiments, follow the :ref:`Common Dependencies <common-dependencies>` section to install common dependencies.

Second, for experiments depending on Megatron and SGLang/vLLM like Math, follow the :ref:`Megatron and SGLang/vLLM Dependencies <megatron-and-sglang-vllm-dependencies>` section to install Megatron-related dependencies.  
For embodied experiments, follow the :ref:`Embodied Dependencies <embodied-dependencies>` to install OpenVLA or Pi0 dependencies.

.. _common-dependencies:

Common Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using `uv <https://docs.astral.sh/uv/>`_ to install the necessary Python dependencies.  
If you are using `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_, you can also install ``uv`` via ``pip``.

.. code-block:: shell

   conda create -n rlinf python=3.11.10 -y
   conda activate rlinf
   pip install --upgrade uv

After installing ``uv``, create a virtual environment and install PyTorch as well as the common dependencies.

.. code-block:: shell

   uv venv
   source .venv/bin/activate
   UV_TORCH_BACKEND=auto uv sync

.. _megatron-and-sglang-vllm-dependencies:

Megatron and SGLang/vLLM Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following to install Megatron, SGLang/vLLM and their dependencies.

.. code-block:: shell

   uv sync --extra sgl_vllm
   mkdir -p /opt && git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.11.0 /opt/Megatron-LM
   APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install -r requirements/megatron.txt --no-build-isolation

Before using Megatron, make sure its path is added to the ``PYTHONPATH`` environment variable.

.. code-block:: shell

   export PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH

.. _embodied-dependencies:

Embodied Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For embodied experiments, first install the necessary system dependencies (currently only Debian/Ubuntu ``apt`` package management is supported).

.. code-block:: shell

   bash requirements/install_embodied_deps.sh
   uv sync --extra embodied

Next, depending on the experiment types, install the ``openvla`` or ``pi0`` dependencies.

.. code-block:: shell

   # For OpenVLA/OpenVLA-oft experiments
   UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla.txt --no-build-isolation

   # For Pi0 experiment
   UV_TORCH_BACKEND=auto uv pip install -r requirements/pi0.txt --no-build-isolation
