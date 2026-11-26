rStar2 RL
=========

Reproduce the multi-turn code-execution RL experiments from `rStar2-Agent: Agentic Reasoning Technical Report <https://arxiv.org/abs/2508.20722>`__ in RLinf, training Qwen2.5-7B-Instruct to solve math problems by calling a local Python execution server. After 150 steps, RLinf reaches **45.79** average accuracy across AIME 24, AIME 25, and Math 500.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-7B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO (multi-turn tool RL)

   .. grid-item-card:: Tools
      :text-align: center

      Code Judge server · Math-Verify

   .. grid-item-card:: Hardware
      :text-align: center

      8×H100 (Megatron)

| **What you'll do:** install deps and Code Judge → download model and start judge service → launch ``run_rstar2.sh``.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · Redis and ``tmux`` for the local Code Judge service.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Qwen2.5-7B-Instruct (Megatron · 8×H100)

      **1. Install**

      Install the agentic environment, reward packages, Redis, and Code Judge (adapted from `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/blob/c12e3cbce8dceb70e9c9b16252bfd5675ec3129c/recipe/rstar2_agent/README.md>`__):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install math-verify sympy pylatexenc
         sudo apt-get update -y && sudo apt-get install redis -y
         git clone https://github.com/0xWJ/code-judge examples/agent/rstar2/code-judge
         uv pip install -r examples/agent/rstar2/code-judge/requirements.txt
         uv pip install -e examples/agent/rstar2/code-judge

      **2. Download and Start Code Judge**

      Download the base model and generate the processed training dataset (saved to ``./dapo/train.jsonl``):

      .. code-block:: bash

         hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct
         python examples/agent/rstar2/data_process/process_train_dataset.py

      Start Redis, the Code Judge HTTP server (master node only), and the Code Judge workers. On multi-node setups, set ``MASTER_ADDR`` to the master node IP:

      .. code-block:: bash

         export WORKSPACE=$(pwd)
         export MASTER_ADDR=127.0.0.1
         redis-server --daemonize yes --protected-mode no --bind 0.0.0.0
         tmux new-session -d -s server \
           'cd $WORKSPACE/examples/agent/rstar2/code-judge && MAX_EXECUTION_TIME=4 REDIS_URI="redis://$MASTER_ADDR:6379" RUN_WORKERS=0 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 16 2>&1 | tee server.log'
         tmux new-session -d -s worker \
           'cd $WORKSPACE/examples/agent/rstar2/code-judge && MAX_EXECUTION_TIME=4 REDIS_URI="redis://$MASTER_ADDR:6379" MAX_WORKERS=64 python run_workers.py 2>&1 | tee worker.log'

      **3. Launch**

      Launch training with ``rstar2-qwen2.5-7b-megatron.yaml`` by passing the model and dataset paths on the command line:

      .. code-block:: bash

         bash examples/agent/rstar2/run_rstar2.sh rstar2-qwen2.5-7b-megatron \
           rollout.model.model_path=./Qwen2.5-7B-Instruct \
           data.train_data_paths='["./dapo/train.jsonl"]'

      .. dropdown:: Advanced configuration (optional)

         To persist paths or enable validation in ``examples/agent/rstar2/config/rstar2-qwen2.5-7b-megatron.yaml``, update the fields below and keep ``algorithm.recompute_logprobs: False`` and ``algorithm.shuffle_rollout: False``:

         .. code-block:: yaml

            data:
              train_data_paths: ["./dapo/train.jsonl"]
              val_data_paths: ["./dapo/train.jsonl"]

            rollout:
              model:
                model_path: ./Qwen2.5-7B-Instruct

Visualization and Results
-------------------------

Track reward and response-length curves in TensorBoard (``tensorboard --logdir ../results``), or see :doc:`Training metrics </rst_source/reference/metrics>` for W&B/SwanLab and metric definitions.

The figures and table below compare RLinf and Verl after 150 training steps with default settings.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rstar2-RLinf-7b.jpg
   :width: 80%
   :align: center
   :alt: Qwen2.5-7B-Instruct in RLinf

   Qwen2.5-7B-Instruct in RLinf

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rstar2-Verl-7b.jpg
   :width: 80%
   :align: center
   :alt: Qwen2.5-7B-Instruct in Verl

   Qwen2.5-7B-Instruct in Verl

.. list-table:: **7B model results (150 steps)**
   :header-rows: 1
   :widths: 35 15 15 15 20

   * - Engine
     - AIME 24
     - AIME 25
     - Math 500
     - Average
   * - `RLinf <https://github.com/RLinf/RLinf>`_
     - **33.65**
     - 24.11
     - **79.60**
     - **45.79**
   * - `Verl <https://github.com/volcengine/verl>`_
     - 31.77
     - **25.94**
     - 76.20
     - 44.64

**Reference:** `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/pull/3397>`__.
