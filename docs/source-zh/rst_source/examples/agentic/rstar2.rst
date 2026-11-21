rStar2 强化学习
===============

使用本配方复现 `rStar2-Agent: Agentic Reasoning Technical Report <https://arxiv.org/abs/2508.20722>`__ 的多轮工具调用实验，通过强化学习训练大语言模型调用代码运行工具回答问题。

概览
----------------------------------------

使用本配方复现 rStar2 风格的智能体推理，包含代码执行工具与 Megatron 训练。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-7B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      带工具调用的多轮强化学习

   .. grid-item-card:: 工具
      :text-align: center

      Code Judge server 与 Math-Verify reward

   .. grid-item-card:: 硬件
      :text-align: center

      参考运行使用 8×H100

一键上手
----------------------------------------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__，完整选项见 :doc:`RLinf Installation </rst_source/start/installation>`。

.. tab-set::

   .. tab-item:: Qwen2.5-7B-Instruct（Megatron · 8×H100）

      **1. 安装环境**

      安装 RLinf、Code Judge（参考 `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/blob/c12e3cbce8dceb70e9c9b16252bfd5675ec3129c/recipe/rstar2_agent/README.md>`__）以及 reward 计算依赖（``math-verify``、``sympy``、``pylatexenc``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install math-verify sympy pylatexenc

         # 安装 Code Judge 与 rStar2 agent 依赖
         sudo apt-get update -y && sudo apt-get install redis -y
         git clone https://github.com/0xWJ/code-judge examples/agent/rstar2/code-judge
         uv pip install -r examples/agent/rstar2/code-judge/requirements.txt
         uv pip install -e examples/agent/rstar2/code-judge
         uv pip install -r examples/agent/rstar2/requirements.txt

      启动 Redis、Code Judge Server（仅 master 节点）和 Code Judge Workers（将 ``$WORKSPACE`` 和 ``$MASTER_ADDR`` 替换为实际值）：

      .. code-block:: bash

         redis-server --daemonize yes --protected-mode no --bind 0.0.0.0

         tmux new-session -d -s server \
         'cd $WORKSPACE/examples/agent/rstar2/code-judge && \
            MAX_EXECUTION_TIME=4 \
            REDIS_URI="redis://$MASTER_ADDR:6379" \
            RUN_WORKERS=0 \
            uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 16 \
            2>&1 | tee server.log'

         tmux new-session -d -s worker \
         'cd $WORKSPACE/examples/agent/rstar2/code-judge && \
            MAX_EXECUTION_TIME=4 \
            REDIS_URI="redis://$MASTER_ADDR:6379" \
            MAX_WORKERS=64 \
            python run_workers.py \
            2>&1 | tee worker.log'

      **2. 下载模型与数据**

      下载基础模型并准备训练数据集：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct
         python examples/agent/rstar2/data_process/process_train_dataset.py

      **3. 启动训练**

      在 ``examples/agent/rstar2/config/rstar2-qwen2.5-7b-megatron.yaml`` 中填写 ``data.train_data_paths``、``data.val_data_paths`` 和 ``rollout.model.model_path``\ （保持 ``algorithm.recompute_logprobs: False`` 和 ``algorithm.shuffle_rollout: False``）：

      .. code-block:: yaml

         data:
           train_data_paths: ["/path/to/train.jsonl"]
           val_data_paths: ["/path/to/train.jsonl"]

         rollout:
           model:
             model_path: /path/to/model/Qwen2.5-7B-Instruct

      启动训练：

      .. code-block:: bash

         bash examples/agent/rstar2/run_rstar2.sh


可视化与结果
----------------------------------------

下面展示 RLinf 与 Verl 的 reward 曲线和 response 长度曲线对比。

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

\* 我们使用默认配置将模型重训了 150 步，以对齐 Verl 的效果。

.. list-table:: **7B 验证结果对比**
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

.. note::

   **参考：** `rStar2 & veRL-SGLang <https://github.com/volcengine/verl/pull/3397>`__。
