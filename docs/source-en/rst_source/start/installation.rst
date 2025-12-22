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
--------------------------------------------------

We provide two official Docker images optimized for different backend configurations:

- **Math reasoning with Megatron + SGLang/vLLM**:  

  - ``rlinf/rlinf:math-rlinf0.1-torch2.5.1-sglang0.4.4-vllm0.7.1-megatron0.11.0-te2.1`` (used for enhancing LLM reasoning on MATH tasks)

- **Embodied with FSDP + Huggingface**:  

  - ``rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0`` (for the OpenVLA/OpenVLA-OFT/openpi model)

Once you've identified the appropriate image for your setup, pull the Docker image:

.. code-block:: bash

   docker pull rlinf/rlinf:CHOSEN_IMAGE

Then, start the container using the pulled image:

.. warning::

  1. Ensure the docker is started with `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` to enable GPU support, especially the `graphics` capability for rendering in embodied experiments.

  2. Do not override the `/root` and `/opt` directories in the container (with `-v` or `--volume` of `docker run`), as they contain important asset files and environments. If your platform requires mounting `/root`, run `link_assets` in the container after starting it to restore the asset links in the `/root` directory.

  3. Avoid changing the `$HOME` environment variable (e.g., `docker run -e HOME=/new_home`), which should be `/root` by default. ManiSkill and other tools rely on this path to find the assets. If `$HOME` is changed before running scripts in the docker image, make sure to relink the assets to the new `$HOME` by executing `link_assets`.

.. code-block:: bash

   docker run -it --gpus all \
      --shm-size 100g \
      --net=host \
      --name rlinf \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
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

.. note::

  Both the `link_assets` and `switch_env` scripts are built-in utilities in the Docker image provided by us. You can find them in `/usr/local/bin`.

.. tip::

   - For multi-node training, make sure to clone the repository in shared storage so that every node has access to it.
   - To use ManiSkill settings, refer to the README at ``https://huggingface.co/datasets/RLinf/maniskill_assets`` for instructions on downloading the required files.

Installation Method 2: UV Custom Environment
--------------------------------------------------------------
**If you have already used the Docker image, you can skip the following steps.**

We recommend using `uv <https://docs.astral.sh/uv/>`_ to install the required Python packages.  
You can install ``uv`` via ``pip``.

.. code-block:: shell

   pip install --upgrade uv

After installing `uv`, you can install the dependencies for the target experiments using the `install.sh` script under the `requirements/` folder.
The script accepts one argument which specifies the target experiment, including `openvla`, `openvla-oft`, `openpi`, and `reason`.
For example, to install the dependencies for the openvla experiment, you would run:

.. note:: 

  This script needs to be run from the root directory of the RLinf repository. Please ensure you are not running it from within the `requirements/` directory.

.. code-block:: shell
  
  bash requirements/install.sh openvla

This will create a virtual environment under the current path named `.venv`.
To activate the virtual environment, you can use the following command:

.. code-block:: shell
  
  source .venv/bin/activate

To deactivate the virtual environment, simply run:

.. code-block:: shell

  deactivate