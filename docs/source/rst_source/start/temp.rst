Example For Quickstart
========================

Installation
============

Hardware Requirements
---------------------

The following configuration has been thoroughly tested.

+-----------------------------------------------+---------------------------------------------------------------+
| Resource                                      | Specification                                                 |
+===============================================+===============================================================+
| GPU                                           | XXX × XXX GB HBM per node                                     |
+-----------------------------------------------+---------------------------------------------------------------+
| CPU                                           | XXX cores per node                                            |
+-----------------------------------------------+---------------------------------------------------------------+
| System memory                                 | XXX TB per node                                               |
+-----------------------------------------------+---------------------------------------------------------------+
| High-speed interconnect                       | XXX Gbps (e.g. NVSwitch / InfiniBand / RoCE)                  |
+-----------------------------------------------+---------------------------------------------------------------+
| Local storage (single-node)                   | XXX TB                                                        |
+-----------------------------------------------+---------------------------------------------------------------+
| Shared storage (distributed experiments)      | XXX TB (NAS / SAN / Ceph)                                     |
+-----------------------------------------------+---------------------------------------------------------------+

Software Requirements
---------------------

+-----------------------------------------------+---------------------------------------------------------------+
| Component                                     | Version / Notes                                               |
+===============================================+===============================================================+
| Operating System                              | Linux XXX (tested)                                            |
+-----------------------------------------------+---------------------------------------------------------------+
| NVIDIA Driver                                 | XXX                                                           |
+-----------------------------------------------+---------------------------------------------------------------+
| CUDA Toolkit                                  | XXX                                                           |
+-----------------------------------------------+---------------------------------------------------------------+
| Docker Engine                                 | XXX or later *(Docker workflow)*                              |
+-----------------------------------------------+---------------------------------------------------------------+
| Conda                                         | XXX *(Conda workflow)*                                        |
+-----------------------------------------------+---------------------------------------------------------------+
| Python                                        | ≥ 3.9                                                         |
+-----------------------------------------------+---------------------------------------------------------------+





Create an Environment
---------------------

We recommend **one** of the two methods below.

Method 1 — Docker (reproducible & portable)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. **Pull or build the Docker image**

   .. code-block:: bash

      # pull a pre-built image
      docker pull RLinf:latest

#. **Run the container**

   .. code-block:: bash

      docker run -it --rm --gpus all \
        --shm-size 2g \
        -v $(pwd):/workspace/megatron-RLinf \
        -v XXX_DATASET_DIR:/workspace/dataset \
        -v XXX_MODEL_DIR:/workspace/model \
        --name RLinf \
        RLInf:latest /bin/bash

      docker run -it --rm --gpus all \
        --shm-size 2g \
        -v $(pwd):/workspace/megatron-RLinf \
        --name RLinf \
        rlinf:latest /bin/bash

   a30: 
      dataset: /mnt/public/dataset/boba_106k_0319_prompt_1024.jsonl
      model: /mnt/public/dataset/DeepSeek-R1-Distill-Qwen-1.5B/


   Inside the container you should see the project under `/workspace/megatron-RLinf`.

Method 2 — Conda + Pip (bare-metal or custom VMs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. **Create a new Conda environment**

   .. code-block:: bash

      conda create -y -n RLinf python=3.9
      conda activate RLinf


#. **Install RLinf and its dependencies**

   .. code-block:: bash

      pip install -r requirements.txt  # if you generated one
      export PYTHONPATH=$(pwd)         # set PYTHONPATH so that infini-rl can be correctly imported

#. **Verify the installation**

   .. code-block:: bash

      python -c "import infinigence_rl, torch; print('RLinf', infinigence_rl.__version__); print('CUDA available', torch.cuda.is_available())"

If the version prints successfully and `torch.cuda.is_available()` returns `True`, you are ready to start training.

Start Your First Training Job
=============================

This page explains how to launch **GRPO** training with the provided `bash` script and *which* parameters you must customise.

We recommend trying **single-node × 8 GPUs** first; distributed multi-node tweaks are identical except for setting `NNODES`, `MASTER_ADDR`, and network ports.

Prepare Data & Model
--------------------

1. **Dataset** – download or preprocess:

   .. code-block:: bash

      # TODO:
      python tools/download_dataset.py \
          --name XXX_DATASET_NAME \
          --output /mnt/public/dataset

2. **Base Model** – obtain from Hugging Face / ModelScope / internal storage:

   .. code-block:: bash

      # TODO:
      python -c "import transformers, os; transformers.pipeline('text-generation', model='XXX_BASE_MODEL')"

Update `CHECKPOINT_LOAD_PATH` to point to the downloaded checkpoint directory.

Launch Training
---------------

From inside the repository root:

.. code-block:: bash

   cd /workspace/megatron-RLinf
   bash scripts/run_grpo.sh

Key Flags Explained
~~~~~~~~~~~~~~~~~~~

+ `MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` – micro per GPU vs effective global tokens.  
+ `SEQ_LEN` – prompt + response length (9216 by default).  
+ Offloading (`OFFLOAD_*`) – enable ZeRO-3 style CPU offload when GPU memory is tight.  
+ Sampling params (`TEMPERATURE`, `TOP_P`, etc.) – govern how roll-out responses are generated.  
+ `SAVE_INTERVAL` – checkpoint frequency (in training steps).  

The script auto-derives:

* `WORLD_SIZE = NNODES × GPUS_PER_NODE`
* Port offsets for vLLM (`WEIGHT_PORT_BASE`, `TOKEN_PORT_BASE`)
* Log file names with time-stamps

Monitoring & Output
-------------------

* **Stdout / stderr** are tee-ed to `logs/MEGA-*.log` per rank.  
* **TensorBoard**: if `TENSORBOARD_ENABLE=True`, run  
  ::

     tensorboard --logdir logs/ --port 6006

Next Steps
----------

* Tweak micro-batch size or enable optimizer offload if you hit out-of-memory.  
* Scale-out: set `NNODES > 1`, make sure `MASTER_ADDR` and network ports are reachable.  
* Explore other algorithms (`PPO`, `DPO`) in the `examples/` directory.

You are now running your first post-training job with RLinf — happy fine-tuning!

new
====

mkdir rlinf_test

cd rlinf_test

.. docker run -it --rm --gpus all \
..   --shm-size 2g \
..   -v $(pwd):/workspace/megatron-RLinf \
..   -v /mnt/public/dataset/boba_106k_0319_prompt_1024.jsonl:/workspace/dataset/boba_106k_0319_prompt_1024.jsonl \
..   -v /mnt/public/dataset/DeepSeek-R1-Distill-Qwen-1.5B:/workspace/model \
..   --name RLinf \
..   rlinf:latest /bin/bash
  
.. docker run -it --rm --gpus all \
..   --shm-size 2g \
..   --user $(id -u):$(id -g) \
..   -v "$(pwd)":/workspace/megatron-RLinf \
..   -v /mnt/public/dataset/boba_106k_0319_prompt_1024.jsonl:/workspace/dataset/boba_106k_0319_prompt_1024.jsonl \
..   -v /mnt/public/dataset/DeepSeek-R1-Distill-Qwen-1.5B:/workspace/model \
..   -w /workspace/megatron-RLinf \
..   --name RLinf \
..   rlinf:latest /bin/bash

docker run -it --gpus all \
  --shm-size 80g \
  -v "$(pwd)":/workspace/megatron-RLinf \
  -w /workspace/megatron-RLinf \
  --name RLinf \
  rlinf:latest /bin/bash

git clone https://gitlab.infini-ai.com/SysComm/megatron-infinigence-rl.git

cd megatron-infinigence-rl

export HTTP_PROXY='http://10.10.20.100:1089'
export HTTPS_PROXY='http://10.10.20.100:1089'

hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   --local-dir /workspace/model/DeepSeek-R1-Distill-Qwen-1.5B
hf download inclusionAI/AReaL-boba-Data --repo-type=dataset --local-dir /workspace/data/boba

docker start RLinf
docker exec -it RLinf /bin/bash
