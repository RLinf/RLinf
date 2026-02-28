rStar2的强化学习训练
=======================

结合工具调用的 Multi-turn RL 被证明能够将大语言模型（LLM）的交互边界扩展到真实世界。本文档介绍了如何在 RLinf 框架下复现论文 `rStar2-Agent: Agentic Reasoning Technical Report <https://arxiv.org/abs/2508.20722>`__ 的实验，使用强化学习（RL）来训练大语言模型（LLM）通过调用代码运行工具回答问题。

环境
----

RLinf环境
~~~~~~~~~

RLinf 环境配置参照 `RLinf Installation <https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html>`__

Code judge运行环境
~~~~~~~~~~~~~~~~~~

我们使用 rStar2 示例中的 code judge 工具，安装过程参考 `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/blob/c12e3cbce8dceb70e9c9b16252bfd5675ec3129c/recipe/rstar2_agent/README.md>`__

.. code-block:: bash

   cd examples/rstar2

   # install code judge
   sudo apt-get update -y && sudo apt-get install redis -y
   git clone https://github.com/0xWJ/code-judge
   pip install -r code-judge/requirements.txt
   pip install -e code-judge

   # install rstar2_agent requirements
   pip install -r requirements.txt

   cd ../..

Code Judge 服务器设置
^^^^^^^^^^^^^^^^^^^^^

rStar2-Agent 使用 Code Judge 作为工具调用服务器来执行模型生成的 Python 代码。

**1. 启动 Redis 服务器**

.. code-block:: bash

   sudo apt-get update -y && sudo apt-get install redis -y
   redis-server --daemonize yes --protected-mode no --bind 0.0.0.0

**2. 启动 Code Judge Server**

.. code-block:: bash

   # Start the main server (master node only)
   # Environment variables can be configured as per: https://github.com/0xWJ/code-judge/blob/main/app/config.py
   # Replace $WORKSPACE and $MASTER_ADDR with your actual paths

   tmux new-session -d -s server \
   'cd $WORKSPACE/examples/rstar2/code-judge && \
      MAX_EXECUTION_TIME=4 \
      REDIS_URI="redis://$MASTER_ADDR:6379" \
      RUN_WORKERS=0 \
      uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 16 \
      2>&1 | tee server.log'

**3. 启动 Code Judge Workers**

.. code-block:: bash

   # Launch workers (can be deployed on multiple nodes for increased parallelism)
   # Adjust MAX_WORKERS based on your CPU count per node

   tmux new-session -d -s worker \
   'cd $WORKSPACE/examples/rstar2/code-judge && \
      MAX_EXECUTION_TIME=4 \
      REDIS_URI="redis://$MASTER_ADDR:6379" \
      MAX_WORKERS=64 \
      python run_workers.py \
      2>&1 | tee worker.log'


Reward计算工具
~~~~~~~~~~~~~~

我们使用 Math-Verify 辅助进行 reward 计算，需通过 pip 安装

.. code-block:: bash

   pip install math-verify

我们同时使用了简单规则进行reward计算，以确保reward计算的正确性， 计算需安装依赖。

.. code-block:: bash

   pip install sympy
   pip install pylatexenc

在8*H100上训练
--------------

通过 ``examples/rstar2/data_process/process_train_dataset.py`` 下载训练集，并将路径写入 ``examples/rstar2/config/rstar2-qwen2.5-7b-megatron.yaml``

.. code-block:: yaml

   data:
     # ……
     train_data_paths: ["/path/to/train.jsonl"]
     val_data_paths: ["/path/to/train.jsonl"]

修改 ``examples/rstar2/config/rstar2-qwen2.5-7b-megatron.yaml`` 中 ``rollout.model.model_path`` 的路径

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.5
     model:
       model_path: /path/to/model/Qwen2.5-7B-Instruct
       model_type: qwen2.5

由于 down sample 逻辑不适配目前 inference 逻辑，``recompute_logprobs`` 应当设置为 ``False``

.. code-block:: yaml

   algorithm:
      # ……
      recompute_logprobs: False
      shuffle_rollout: False

启动训练
~~~~~~~~

运行 ``examples/rstar2/run_rstar2.sh`` 启动训练。


训练曲线
--------

下面展示 RLinf 与 Verl 的 reward 曲线和 response 长度曲线对比。

.. figure:: https://github.com/RLinf/misc/raw/main/pic/rstar2-RLinf-7b.jpg
   :width: 80%
   :align: center
   :alt: Qwen2.5-7B-Instruct in RLinf

   Qwen2.5-7B-Instruct in RLinf

.. figure:: https://github.com/RLinf/misc/raw/main/pic/rstar2-Verl-7b.jpg
   :width: 80%
   :align: center
   :alt: Qwen2.5-7B-Instruct in Verl

   Qwen2.5-7B-Instruct in Verl

\* 我们使用默认配置对模型进行了 150 步重训，以和Verl的效果对齐。

.. list-table:: **7B验证结果对比**
   :header-rows: 1
   :widths: 35 15 15 15 20

   * - 框架
     - AIME 24
     - AIME 25
     - Math 500
     - 平均值
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

References
----------

- rStar2 & veRL-SGLang: https://github.com/volcengine/verl/pull/3397
