Installation
============

RLinf supports multiple backend engines for both training and inference. As of now, the following configurations are available:

- **Megatron** and **SGLang/vLLM** for training LLMs on MATH tasks.
- **FSDP** and **Huggingface** for training VLAs on LIBERO and ManiSkill3.

Backend Engines
---------------

1. **Training Engines**

   - **FSDP**: A simple and efficient training engine that is beginner-friendly, widely compatible, easy to use, and supports native PyTorch modules.

   - **Megatron**: Designed for experienced developers seeking maximum performance. It supports a variety of parallel configurations and offers SOTA training speed and scalability.

2. **Inference Engines**

   - **SGLang/vLLM**: A mature and widely adopted inference engine that offers many advanced features and optimizations.

   - **Huggingface**: Easy to use, with native APIs provided by the Huggingface ecosystem.

Installation Methods
--------------------

RLinf provides two installation options. We **recommend using Docker**, as it provides the fastest and most reproducible environment.
However, if your system is incompatible with the Docker image, you can also install RLinf manually in a Python environment.

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~~~

The following hardware configuration has been extensively tested:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Configuration
   * - GPU
     - 8xH100 per node
   * - CPU
     - 192 cores per node
   * - Memory
     - 1.8TB per node
   * - Network
     - NVLink + RoCE / IB 3.2 Tbps 
   * - Storage
     - | 1TB local storage for single-node experiments
       | 10TB shared storage (NAS) for distributed experiments


Software Requirements
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Version
   * - Operating System
     - Ubuntu 22.04
   * - NVIDIA Driver
     - 535.183.06
   * - CUDA
     - 12.4 
   * - Docker
     - 26.0.0
   * - NVIDIA Container Toolkit
     - 1.17.8


Install from Docker Image
-------------------------

We provide two official Docker images optimized for different backend configurations:

- **Megatron + SGLang/vLLM**:  

  - ``rlinf/rlinf:math-rlinf0.1-torch2.5.1-sglang0.4.4-vllm0.7.1-megatron0.11.0-te2.1`` (used for enhancing LLM reasoning on MATH tasks)

- **FSDP + Huggingface**:  

  - ``rlinf/rlinf:agentic-openvla-rlinf0.1-torch2.5.1`` (for the OpenVLA model)  
  - ``rlinf/rlinf:agentic-openvlaoft-rlinf0.1-torch2.5.1`` (for the OpenVLA-OFT model)


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

Installation is divided into three parts depending on the type of experiments you plan to run.

First, for all experiments, follow the :ref:`Common Dependencies <common-dependencies>` section to install the shared dependencies.  
This already includes the full backend setup for **FSDP + Huggingface**.

Second, for experiments using **Megatron** and **SGLang/vLLM** backends,  
follow the :ref:`Megatron and SGLang/vLLM Dependencies <megatron-and-sglang-vllm-dependencies>` section to install all required packages.  

Third, for embodied intelligence experiments (e.g., OpenVLA, OpenVLA-OFT and Pi0),  
follow the :ref:`Embodied Dependencies <embodied-dependencies>` section to install their specific dependencies.

.. _common-dependencies:

Common Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using `uv <https://docs.astral.sh/uv/>`_ to install the required Python packages.  
If you are using `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_, you can install ``uv`` via ``pip``.

.. code-block:: shell

   conda create -n rlinf python=3.11.10 -y
   conda activate rlinf
   pip install --upgrade uv

After installing ``uv``, create a virtual environment and install PyTorch along with the common dependencies:

.. code-block:: shell

   uv venv
   source .venv/bin/activate
   UV_TORCH_BACKEND=auto uv sync

.. _megatron-and-sglang-vllm-dependencies:

Megatron and SGLang/vLLM Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following commands to install Megatron, SGLang or vLLM, and their dependencies:

.. code-block:: shell

   uv sync --extra sgl_vllm
   mkdir -p /opt && git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.13.0 /opt/Megatron-LM
   APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install -r requirements/megatron.txt --no-build-isolation

Before using Megatron, ensure its path is added to the ``PYTHONPATH`` environment variable:

.. code-block:: shell

   export PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH

SGLang installation:

.. code-block:: shell

   uv sync --extra sglang

vLLM installation:

.. code-block:: shell

   uv sync --extra vllm

.. _embodied-dependencies:

Additional Embodied Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For embodied experiments, first install the necessary system dependencies (currently only supported on Debian/Ubuntu via ``apt``):

.. code-block:: shell

   bash requirements/install_embodied_deps.sh
   uv sync --extra embodied

Then, depending on the experiment type, install the required packages for ``openvla``, ``openvla-oft`` and ``pi0``:

.. code-block:: shell

   # For OpenVLA/OpenVLA-oft experiments
   UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla.txt --no-build-isolation

   # For Pi0 experiments
   UV_TORCH_BACKEND=auto uv pip install -r requirements/pi0.txt --no-build-isolation

