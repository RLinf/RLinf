AgentLightning 的强化学习训练（calc_x）
============================================

``calc_x`` 是 RLinf 中的 AgentLightning 示例，用于训练一个会做数学题的 agent。
agent 会读取题目，生成推理过程与答案，并根据反馈做强化学习更新。

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

安装
----------------------------------------

AgentLightning 0.3.0 需要使用 RLinf 约束的 OpenTelemetry 版本，因为 OpenTelemetry 1.45 修改了将 rollout span 路由到 Lightning Store 的 exporter 实现。创建基础环境后，请先安装兼容的 OpenTelemetry 依赖，再安装本示例的其他依赖：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh agentic
   source .venv/bin/activate
   uv pip install -r requirements/agentic/agentlightning_constraints.txt
   uv pip install "agentlightning==0.3.0" "litellm<1.95" \
     "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0" \
     "mcp-server-calculator" "openai==2.32.0"

其他安装选项和支持的平台请参考 :doc:`RLinf Installation </rst_source/start/installation>`。

数据准备
----------------------------------------

下载并解压 ``calc_x`` 数据集（Google Drive），下载链接见 `这里 <https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view>`_。

运行
----------------------------------------

进入示例目录：

.. code-block:: bash

   cd /path/to/RLinf/examples/agent/agentlightning/calc_x

先选择要运行的配置文件，并在同一个配置文件中设置模型与数据集路径。下面的训练命令使用
``config/qwen2.5-1.5b-enginehttp-multiturn.yaml``：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model/Qwen2.5-1.5B-Instruct

   data:
     train_data_paths: ["/path/to/train.parquet"]
     val_data_paths: ["/path/to/test.parquet"]

启动训练：

.. code-block:: bash

   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn

也可以不修改配置文件，直接通过命令行传入 Hydra overrides：

.. code-block:: bash

   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn \
     rollout.model.model_path=/path/to/Qwen2.5-1.5B-Instruct \
     data.train_data_paths='["/path/to/train.parquet"]' \
     data.val_data_paths='["/path/to/test.parquet"]'

如果需要使用 trajectory-level advantages 训练，则使用对应的 trajectory 配置，并在其中设置
相同路径或通过 overrides 传入：

.. code-block:: bash

   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory

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

   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn_eval
   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory_eval
