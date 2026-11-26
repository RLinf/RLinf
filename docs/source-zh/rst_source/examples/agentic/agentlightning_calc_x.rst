AgentLightning 强化学习
=======================

在 ``calc_x`` 任务上结合 Agent Lightning 工作流编排与 RLinf 分布式训练，训练调用 MCP 计算器的数学 Agent。模型在多轮交互中完成推理并按轮次级或轨迹级 GRPO 更新 Qwen2.5-1.5B-Instruct。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-1.5B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      GRPO（轮次级或轨迹级）

   .. grid-item-card:: 工具
      :text-align: center

      MCP calculator · AutoGen

   .. grid-item-card:: 硬件
      :text-align: center

      单节点 · 至少 1 张 40 GB GPU

| **你将完成：** 安装 Agent Lightning 依赖 → 下载模型与 ``calc_x`` 数据集 → 运行 ``run_calc_x.sh``。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · 从 Google Drive 获取的 ``calc_x`` parquet 数据集。

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: Qwen2.5-1.5B-Instruct

      **1. 安装环境**

      安装 agentic 环境与 Agent Lightning 依赖。先安装 ``requirements/agentic/agentlightning_constraints.txt`` 锁定 OpenTelemetry 版本，以兼容 AgentLightning 0.3.0 向 Lightning Store 上报 rollout span：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install -r requirements/agentic/agentlightning_constraints.txt
         uv pip install "agentlightning==0.3.0" "litellm<1.95" \
           "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0" \
           "mcp-server-calculator" "openai==2.32.0"

      **2. 下载模型与数据**

      下载基础模型，并从 `Google Drive <https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view>`_ 下载解压 ``calc_x`` 数据集（含 ``train.parquet`` 与 ``test.parquet``）至 ``./calc_x/``：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct

      **3. 启动训练**

      在 ``examples/agent/agentlightning/calc_x/config/`` 中按需选择轮次级或轨迹级 Advantage 配置：

      .. list-table::
         :header-rows: 1
         :widths: 24 48 28

         * - 模式
           - 配置文件
           - Advantage 模式
         * - 多轮
           - ``qwen2.5-1.5b-enginehttp-multiturn.yaml``
           - 轮次级
         * - 轨迹级
           - ``qwen2.5-1.5b-enginehttp-trajectory.yaml``
           - 轨迹级

      在仓库根目录通过命令行传入模型和数据集路径启动训练（切换为轨迹级模式时改为 ``qwen2.5-1.5b-enginehttp-trajectory``）：

      .. code-block:: bash

         bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn \
           rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
           data.train_data_paths='["./calc_x/train.parquet"]' \
           data.val_data_paths='["./calc_x/test.parquet"]'

      .. dropdown:: 独立评测与 YAML 配置（可选）

         如需直接修改配置文件而非命令行传参，在对应 YAML 中填写 ``rollout.model.model_path``、``data.train_data_paths`` 和 ``data.val_data_paths`` 后，仅传配置名运行 ``run_calc_x.sh`` 即可。

         对 Hugging Face checkpoint 做独立评测时，使用对应的 ``*_eval`` 配置并指定 ``rollout.model.model_path``：

         .. code-block:: bash

            # 多轮模式评测
            bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn_eval \
              rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
              data.val_data_paths='["./calc_x/test.parquet"]'

            # 轨迹级模式评测
            bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory_eval \
              rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
              data.val_data_paths='["./calc_x/test.parquet"]'

可视化与结果
------------

运行 ``tensorboard --logdir ../results`` 查看训练曲线；更多指标与 W&B/SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

以下为一次 ``calc_x`` 运行的指标曲线示例（结果会随配置与随机种子变化）：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentlightning_calcx.png
   :width: 90%
   :align: center
   :alt: AgentLightning calc_x 训练曲线

   AgentLightning ``calc_x`` 训练曲线
