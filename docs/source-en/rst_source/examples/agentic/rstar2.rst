rStar2 RL
=========

Multi-turn RL combined with tool invocation extends the interaction boundaries of Large Language Models (LLMs) to the real world. Reproduce the experiments from `rStar2-Agent: Agentic Reasoning Technical Report <https://arxiv.org/abs/2508.20722>`__ in RLinf, training LLMs to answer questions by invoking code-execution tools.

Overview
--------

Use this recipe to reproduce rStar2-style agentic reasoning with code execution
tools and Megatron training.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-7B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      Multi-turn RL with tool invocation

   .. grid-item-card:: Tools
      :text-align: center

      Code Judge server and Math-Verify rewards

   .. grid-item-card:: Hardware
      :text-align: center

      Reference run on 8×H100

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__. See :doc:`RLinf Installation </rst_source/start/installation>` for all options.

.. tab-set::

   .. tab-item:: Qwen2.5-7B-Instruct (Megatron · 8×H100)

      **1. Install**

      Install RLinf, Code Judge (see `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/blob/c12e3cbce8dceb70e9c9b16252bfd5675ec3129c/recipe/rstar2_agent/README.md>`__), and reward dependencies (``math-verify``, ``sympy``, ``pylatexenc``):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install math-verify sympy pylatexenc

         # Install Code Judge dependencies
         sudo apt-get update -y && sudo apt-get install redis -y
         git clone https://github.com/0xWJ/code-judge examples/agent/rstar2/code-judge
         uv pip install -r examples/agent/rstar2/code-judge/requirements.txt
         uv pip install -e examples/agent/rstar2/code-judge

      Start Redis and export the workspace address (set ``MASTER_ADDR`` to the master IP on multi-node setups):

      .. code-block:: bash

         export WORKSPACE=$(pwd)
         export MASTER_ADDR=127.0.0.1
         redis-server --daemonize yes --protected-mode no --bind 0.0.0.0

      Start the Code Judge server (master node only):

      .. code-block:: bash

         tmux new-session -d -s server \
             'cd $WORKSPACE/examples/agent/rstar2/code-judge && MAX_EXECUTION_TIME=4 REDIS_URI="redis://$MASTER_ADDR:6379" RUN_WORKERS=0 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 16 2>&1 | tee server.log'

      Start the Code Judge workers:

      .. code-block:: bash

         tmux new-session -d -s worker \
             'cd $WORKSPACE/examples/agent/rstar2/code-judge && MAX_EXECUTION_TIME=4 REDIS_URI="redis://$MASTER_ADDR:6379" MAX_WORKERS=64 python run_workers.py 2>&1 | tee worker.log'

      **2. Download**

      Download the base model and prepare the training dataset:

      .. code-block:: bash

         hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct
         python examples/agent/rstar2/data_process/process_train_dataset.py

      **3. Launch**

      Update ``data.train_data_paths``, ``data.val_data_paths``, and ``rollout.model.model_path`` in ``examples/agent/rstar2/config/rstar2-qwen2.5-7b-megatron.yaml`` (keep ``algorithm.recompute_logprobs: False`` and ``algorithm.shuffle_rollout: False``):

      .. code-block:: yaml

         data:
           train_data_paths: ["/path/to/train.jsonl"]
           val_data_paths: ["/path/to/train.jsonl"]

         rollout:
           model:
             model_path: /path/to/model/Qwen2.5-7B-Instruct

      Start training:

      .. code-block:: bash

         bash examples/agent/rstar2/run_rstar2.sh rstar2-qwen2.5-7b-megatron


Visualization and Results
-------------------------

Below shows the comparison of reward curves and response length curves between RLinf and Verl.

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

\* We retrain the model using the default settings for 150 steps.

.. list-table:: **7 B model results**
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

.. note::

   **References:** `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/pull/3397>`__.
