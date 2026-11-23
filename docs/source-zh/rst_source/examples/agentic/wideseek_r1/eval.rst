WideSeek-R1 评测
========================================

本页介绍如何在 RLinf 中评测 WideSeek-R1。

提供的脚本支持两种评测设置：

- WideSearch 基准评测。
- 标准 QA 评测。

参考配置使用的是 Qwen3 系列稠密模型。

概述
----------------------------------------

使用本页评测已发布的 WideSeek-R1 checkpoint，或你自己的 Qwen3 稠密模型。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      WideSeek-R1-4B 或 Qwen3 系列稠密模型

   .. grid-item-card:: 基准
      :text-align: center

      WideSearch 与标准 QA

   .. grid-item-card:: 工具
      :text-align: center

      WideSearch 使用在线搜索；QA 使用离线检索

   .. grid-item-card:: 输出
      :text-align: center

      ``metric.json`` 与生成结果

.. contents::
   :depth: 2
   :local:

前置条件
--------

评测前，请先完成 :doc:`环境与工具配置 <tools>`：

- RLinf 环境已安装。
- 评判模型已启动。
- 已配置相应的工具后端。

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。启动评测前，请先按 :doc:`环境与工具配置 <tools>` 启动评判模型并配置对应工具后端（WideSearch 使用在线搜索，标准 QA 使用离线检索）。

.. note::

   已发布的 `WideSeek-R1-4B <https://huggingface.co/RLinf/WideSeek-R1-4b>`__ checkpoint 权重实际以 fp16 存储，但其 ``config.json`` 声明为 ``bfloat16``。默认跟随 config dtype 的推理框架（vLLM/SGLang/HF Transformers 的 ``dtype="auto"``）会把权重上转为 bf16，在 WideSearch 上约损失 3 分 avg@4（55.9 vs 59.0）。在 RLinf 之外部署该 checkpoint 时，请显式传入 ``--dtype float16``。在 RLinf 内部无需额外设置：自带的评测与训练配置已指定 ``rollout.model.precision: fp16`` 并会传给 rollout 引擎（见 `issue #1662 <https://github.com/RLinf/RLinf/issues/1662>`__）。

.. tab-set::

   .. tab-item:: WideSearch 基准

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载已发布的 checkpoint（或使用自己的 Qwen3 稠密模型）和格式化评测集 `WideSeek-R1-test-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-test-data>`__：

      .. code-block:: bash

         hf download RLinf/WideSeek-R1-4b --local-dir ./WideSeek-R1-4b
         hf download --repo-type dataset RLinf/WideSeek-R1-test-data --local-dir ./WideSeek-R1-test-data

      **3. 启动评测**

      更新 ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch.yaml``\ （保持 ``data.is_markdown: True``；``data_size: -1`` 评测全部 200 条样本，使用 8 卡生成与 8 卡评判约需 **7 小时**；设置 ``agentloop.llm_ip``）：

      .. code-block:: yaml

         rollout:
           model:
             model_type: qwen3
             model_path: /PATH/TO/MODEL

         data:
           is_markdown: True
           val_data_paths: /PATH/TO/EVAL/WIDESEARCH/DATASET
           data_size: -1

      启动评测（多节点启动见 :ref:`wideseek-r1-multinode`）：

      .. code-block:: bash

         bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_widesearch

   .. tab-item:: 标准 QA

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载已发布的 checkpoint 和 `ASearcher-test-data <https://huggingface.co/datasets/inclusionAI/ASearcher-test-data>`__\ （含单跳与多跳 QA 任务）：

      .. code-block:: bash

         hf download RLinf/WideSeek-R1-4b --local-dir ./WideSeek-R1-4b
         hf download --repo-type dataset inclusionAI/ASearcher-test-data --local-dir ./ASearcher-test-data

      **3. 启动评测**

      更新 ``examples/agent/wideseek_r1/config/eval_qwen3_qa.yaml``\ （保持 ``data.is_markdown: False``；建议先设置较小的 ``data_size`` 快速验证流程；设置 ``agentloop.llm_ip``）：

      .. code-block:: yaml

         rollout:
           model:
             model_type: qwen3
             model_path: /PATH/TO/MODEL

         data:
           is_markdown: False
           val_data_paths: /PATH/TO/EVAL/QA/DATASET
           data_size: -1

      启动评测（多节点启动见 :ref:`wideseek-r1-multinode`）：

      .. code-block:: bash

         bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_qa

输出文件
----------------------------------------

评测输出会写入：

.. code-block:: text

   ${runner.output_dir}/${runner.experiment_name}

重要文件包括：

- ``metric.json``：聚合指标，例如输出长度和工具使用情况。
- ``allresult.json``：完整的多轮交互日志。
- ``responses/``：每个样本的最终模型回答。

对于标准 QA 评测，``metric.json`` 还包含最终的 LLM 评判结果。

对于 WideSearch 评测，RLinf 会保存生成的回答，以便使用官方 WideSearch 评测流程打分。

额外的 WideSearch 打分
----------------------------------------

如果需要最终的 WideSearch 基准分数，请使用专门的评测仓库：

- `WideSeek-R1-Eval <https://github.com/RLinf/WideSeek-R1-Eval>`__

完整流程请参见该仓库的 README。

双引擎评测
----------------------------------------

WideSeek-R1 还支持在多智能体设定下使用两个独立模型实例评测，从而让 planner 和 worker 角色使用不同模型。

请使用 ``examples/agent/wideseek_r1/config/eval_qwen3_qa_2eng.yaml``。相关字段如下：

.. code-block:: yaml

   agentloop:
     fixed_role: worker  # planner or worker

   rollout:
     use_fixed_worker: True

``use_fixed_worker`` 用于启用第二个模型实例。``fixed_role`` 用于选择哪个角色使用该第二模型。

然后你可以分别在 ``rollout.model.model_path`` 和 ``rollout_fixed_worker.model.model_path`` 下设置不同的模型路径。

说明
----------------------------------------

与训练时相同，``agentloop.workflow`` 用于控制评测采用单智能体还是多智能体执行：

- ``mas``：多智能体评测。
- ``sa``：单智能体评测。

单智能体模式旨在与 `ASearcher <https://github.com/inclusionAI/ASearcher>`__ 保持可比性。
