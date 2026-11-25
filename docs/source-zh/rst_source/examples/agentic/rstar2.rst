rStar2 强化学习
===============

复现 `rStar2-Agent: Agentic Reasoning Technical Report <https://arxiv.org/abs/2508.20722>`__ 中的多轮代码执行强化学习实验，在 RLinf 中训练 Qwen2.5-7B-Instruct 调用本地 Python 执行服务解答数学题。默认配置训练 150 步后，RLinf 在 AIME 24、AIME 25 与 Math 500 上达到 **45.79** 的平均得分。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-7B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      GRPO（多轮工具调用 RL）

   .. grid-item-card:: 工具
      :text-align: center

      Code Judge 服务 · Math-Verify

   .. grid-item-card:: 硬件
      :text-align: center

      8×H100（Megatron）

| **你将完成：** 安装依赖与 Code Judge → 下载模型并启动代码执行服务 → 运行 ``run_rstar2.sh``。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · 本地 Code Judge 服务需要 Redis 和 ``tmux``。

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: Qwen2.5-7B-Instruct（Megatron · 8×H100）

      **1. 安装环境**

      安装 agentic 环境、Reward 依赖、Redis 和 Code Judge（参考 `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/blob/c12e3cbce8dceb70e9c9b16252bfd5675ec3129c/recipe/rstar2_agent/README.md>`__）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install math-verify sympy pylatexenc
         sudo apt-get update -y && sudo apt-get install redis -y
         git clone https://github.com/0xWJ/code-judge examples/agent/rstar2/code-judge
         uv pip install -r examples/agent/rstar2/code-judge/requirements.txt
         uv pip install -e examples/agent/rstar2/code-judge

      **2. 下载模型并启动 Code Judge**

      下载基础模型并生成处理后的训练数据（默认输出到 ``./dapo/train.jsonl``）：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct
         python examples/agent/rstar2/data_process/process_train_dataset.py

      启动 Redis、Code Judge HTTP 服务（仅主节点运行）及执行 worker；多机部署时将 ``MASTER_ADDR`` 改为主节点 IP：

      .. code-block:: bash

         export WORKSPACE=$(pwd)
         export MASTER_ADDR=127.0.0.1
         redis-server --daemonize yes --protected-mode no --bind 0.0.0.0
         tmux new-session -d -s server \
           'cd $WORKSPACE/examples/agent/rstar2/code-judge && MAX_EXECUTION_TIME=4 REDIS_URI="redis://$MASTER_ADDR:6379" RUN_WORKERS=0 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 16 2>&1 | tee server.log'
         tmux new-session -d -s worker \
           'cd $WORKSPACE/examples/agent/rstar2/code-judge && MAX_EXECUTION_TIME=4 REDIS_URI="redis://$MASTER_ADDR:6379" MAX_WORKERS=64 python run_workers.py 2>&1 | tee worker.log'

      **3. 启动训练**

      通过命令行传入模型和数据集路径，使用 ``rstar2-qwen2.5-7b-megatron.yaml`` 启动训练：

      .. code-block:: bash

         bash examples/agent/rstar2/run_rstar2.sh rstar2-qwen2.5-7b-megatron \
           rollout.model.model_path=./Qwen2.5-7B-Instruct \
           data.train_data_paths='["./dapo/train.jsonl"]'

      .. dropdown:: 进阶配置（可选）

         如需将路径写入配置文件或开启验证，编辑 ``examples/agent/rstar2/config/rstar2-qwen2.5-7b-megatron.yaml``，并保持 ``algorithm.recompute_logprobs: False`` 与 ``algorithm.shuffle_rollout: False``：

         .. code-block:: yaml

            data:
              train_data_paths: ["./dapo/train.jsonl"]
              val_data_paths: ["./dapo/train.jsonl"]

            rollout:
              model:
                model_path: ./Qwen2.5-7B-Instruct

可视化与结果
------------

运行 ``tensorboard --logdir ../results`` 查看 Reward 与响应长度曲线；更多指标与 W&B/SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

下图和表格展示默认配置训练 150 步后 RLinf 与 Verl 的效果对比。

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

.. list-table:: **7B 模型验证结果（150 步）**
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

**参考：** `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/pull/3397>`__。
