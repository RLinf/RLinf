AgentLightning 强化学习
=======================

``calc_x`` 是 RLinf 中的 AgentLightning 示例，用于训练一个会做数学题的 agent。agent 会读取题目，生成推理过程与答案，并根据反馈做强化学习更新。

概述
----------------------------------------

使用本配方通过 Agent Lightning 与 RLinf 分布式训练器训练带计算器工具的数学智能体。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-1.5B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      多轮智能体强化学习

   .. grid-item-card:: 工具
      :text-align: center

      MCP calculator 与 AutoGen agent chat

   .. grid-item-card:: 硬件
      :text-align: center

      一个节点，至少一张 40 GB GPU

安装与运行
----------
AgentLightning 0.3.0 需要使用 RLinf 约束的 OpenTelemetry 版本，因为 OpenTelemetry 1.45 修改了将 rollout span 路由到 Lightning Store 的 exporter 实现。中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__，其他安装选项和支持的平台请参考 :doc:`RLinf Installation </rst_source/start/installation>`。

.. tab-set::

   .. tab-item:: Qwen2.5-1.5B-Instruct

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install -r requirements/agentic/agentlightning_constraints.txt
         uv pip install "agentlightning==0.3.0" "litellm<1.95" \
           "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0" \
           "mcp-server-calculator" "openai==2.32.0"

      **2. 下载模型与数据**

      下载基础模型，并从 `Google Drive <https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view>`_ 下载解压 ``calc_x`` 数据集：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct

      **3. 启动训练**

      在 ``examples/agent/agentlightning/calc_x/config/`` 中选择配置文件，并填写 ``rollout.model.model_path``、``data.train_data_paths`` 和 ``data.val_data_paths``：

      .. list-table::
         :header-rows: 1
         :widths: 26 44 30

         * - 模式
           - 配置文件
           - 优势估计
         * - 多轮
           - ``qwen2.5-1.5b-enginehttp-multiturn.yaml``
           - 轮次级
         * - 轨迹级
           - ``qwen2.5-1.5b-enginehttp-trajectory.yaml``
           - 轨迹级

      .. code-block:: yaml

         rollout:
           model:
             model_path: /path/to/model/Qwen2.5-1.5B-Instruct

         data:
           train_data_paths: ["/path/to/train.parquet"]
           val_data_paths: ["/path/to/test.parquet"]

      在仓库根目录启动训练（或通过命令行 Hydra overrides 覆盖路径）：

      .. code-block:: bash

         # 多轮模式
         bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn

         # 轨迹级模式
         bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory

         # 或不修改配置文件，直接命令行覆盖路径
         bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn \
           rollout.model.model_path=/path/to/Qwen2.5-1.5B-Instruct \
           data.train_data_paths='["/path/to/train.parquet"]' \
           data.val_data_paths='["/path/to/test.parquet"]'

可视化与结果
----------------------------------------

以下为一次 ``calc_x`` 训练运行的指标曲线示例（具体曲线会因配置与随机种子而有所不同）：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentlightning_calcx.png
   :width: 90%
   :align: center
   :alt: AgentLightning calc_x 训练曲线

   AgentLightning ``calc_x`` 训练曲线

评测
----------------------------------------

HF 评测时在对应的 ``*_eval.yaml`` 里设置 ``rollout.model.model_path``。例如：

.. code-block:: bash

   bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn_eval
   bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory_eval
