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

Installation Methods
--------------------

RLinf provides two installation options. We **recommend using Docker**, as it provides the fastest and most reproducible environment.
However, if your system is incompatible with the Docker image, you can also install RLinf manually in a Python environment.


Installation Method 1: Docker Image
-------------------------

We provide two official Docker images optimized for different backend configurations:

- **Math reasoning with Megatron + SGLang/vLLM**:  

  - ``rlinf/rlinf:math-rlinf0.1-torch2.5.1-sglang0.4.4-vllm0.7.1-megatron0.11.0-te2.1`` (used for enhancing LLM reasoning on MATH tasks)

- **Embodied with FSDP + Huggingface**:  

  - ``rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0`` (for the OpenVLA/OpenVLA-OFT/openpi model)

Once you've identified the appropriate image for your setup, pull the Docker image:

.. code-block:: bash

   docker pull rlinf/rlinf:CHOSEN_IMAGE

Then, start the container using the pulled image:

.. code-block:: bash

   docker run -it --gpus all \
      --shm-size 100g \
      --net=host \
      --name rlinf \
      rlinf/rlinf:CHOSEN_IMAGE /bin/bash

Inside the container, clone the RLinf repository:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

The embodied image contains multiple Python virtual environments (venv) located in the `/opt/venv` directory for different models, namely ``openvla``, ``openvla-oft``, and ``openpi``.
The default environment is set to ``openvla``.
To switch to the desired venv, use the built-in script `switch_env`:

.. code-block:: bash

   source switch_env <env_name>
   # source switch_env openvla
   # source switch_env openvla-oft
   # source switch_env openpi

.. tip::

   - For multi-node training, make sure to clone the repository in shared storage so that every node has access to it.
   - To use ManiSkill settings, refer to the README at ``https://huggingface.co/datasets/RLinf/maniskill_assets`` for instructions on downloading the required files.

Installation Method 2: UV Custom Environment
-------------------------------
**If you have already used the Docker image, you can skip the following steps.**

Installation is divided into two parts depending on the type of experiments you plan to run.

First, for all experiments, follow the :ref:`Common Dependencies <common-dependencies>` section to install the shared dependencies.

Next, install the specific dependencies based on your experiment type.

* For reasoning experiments using **Megatron** and **SGLang/vLLM** backends, follow the :ref:`Megatron and SGLang/vLLM Dependencies <megatron-and-sglang-vllm-dependencies>` section to install all required packages.  

* For embodied intelligence experiments (e.g., OpenVLA, OpenVLA-OFT and OpenPI), follow the :ref:`Embodied Dependencies <embodied-dependencies>` section to install their specific dependencies.

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

.. note::
  If you are running embodied experiments, there is no need to install these dependencies.
  Please proceed directly to the :ref:`Embodied Dependencies <embodied-dependencies>` section.

Run the following commands to install Megatron, SGLang or vLLM, and their dependencies:

.. code-block:: shell

   uv sync --extra sglang-vllm
   mkdir -p /opt && git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.13.0 /opt/Megatron-LM
   APEX_CPP_EXT=1 APEX_CUDA_EXT=1 NVCC_APPEND_FLAGS="--threads 24" APEX_PARALLEL_BUILD=24 uv pip install -r requirements/megatron.txt --no-build-isolation

Before using Megatron, ensure its path is added to the ``PYTHONPATH`` environment variable:

.. code-block:: shell

   export PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH

.. _embodied-dependencies:

Embodied Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For embodied experiments, first install the necessary system dependencies (currently only supported on Debian/Ubuntu via ``apt``):

.. code-block:: shell

   uv sync --extra embodied
   bash requirements/install_embodied_deps.sh # Must be run after the above command

Then, depending on the experiment type, install the required packages for ``openvla``, ``openvla-oft`` and ``openpi``:

.. code-block:: shell

   # For OpenVLA experiments
   UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla.txt --no-build-isolation

   # For OpenVLA-oft experiment
   UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla_oft.txt --no-build-isolation

   # For openpi experiment
   UV_TORCH_BACKEND=auto GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements/openpi.txt
   cp -r .venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   TOKENIZER_DIR=/root/.cache/openpi/big_vision/ && mkdir -p $TOKENIZER_DIR && gsutil -m cp -r gs://big_vision/paligemma_tokenizer.model $TOKENIZER_DIR

Finally, Run the following to install the LIBERO dependency.

.. code-block:: shell

  mkdir -p /opt && git clone https://github.com/RLinf/LIBERO.git /opt/libero

Before using LIBERO, make sure its path is added to the `PYTHONPATH` environment variables.

.. code-block:: shell
  
  export PYTHONPATH=/opt/libero:$PYTHONPATH