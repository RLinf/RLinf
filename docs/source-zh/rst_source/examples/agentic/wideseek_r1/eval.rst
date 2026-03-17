评测
====

本页介绍如何在 RLinf 中评测 WideSeek-R1。

提供的脚本支持两种评测设置：

- WideSearch 基准评测。
- 标准 QA 评测。

参考配置使用的是 Qwen3 系列稠密模型。

.. contents::
   :depth: 2
   :local:

前置条件
--------

评测前，请确保以下组件已准备就绪：

- RLinf 环境已安装。参见 :doc:`../../../start/installation`。
- 评判模型服务已启动。参见 :doc:`index`。
- 已配置相应的工具后端。参见 :doc:`tools`。

下载模型
--------

已发布的 checkpoint 可从以下地址获取：

- `WideSeek-R1-4B <https://huggingface.co/RLinf/WideSeek-R1-4b>`__

你也可以评测你自己的 Qwen3 系列稠密模型。

下载模型后，在评测配置中设置本地模型路径：

.. code-block:: yaml

   rollout:
     model:
       model_type: qwen3
       model_path: /PATH/TO/MODEL

评测数据集
----------

WideSeek-R1 当前支持两类评测数据集。

WideSearch 基准
~~~~~~~~~~~~~~~

请使用 Hugging Face 上提供的格式化 WideSearch 评测集：

- `WideSeek-R1-test-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-test-data>`__

与原始未处理的基准相比，这个版本已被转换为 RLinf 所需的格式，并包含若干数据修复。

请按如下方式更新 `examples/agent/wideseek_r1/config/eval_qwen3_widesearch.yaml`：

.. code-block:: yaml

   data:
     is_markdown: True
     val_data_paths: /PATH/TO/EVAL/WIDESEARCH/DATASET
     data_size: -1

关键字段说明：

- ``is_markdown`` 对于 WideSearch 数据集应保持为 ``True``。
- ``val_data_paths`` 指向评测数据集。
- ``data_size: -1`` 表示在完整数据集上进行评测。

如果只是快速检查流程是否正常，建议先使用较小的 ``data_size``。

在参考配置中，使用 8 张 GPU 做生成、8 张 GPU 运行评判模型，对 200 条 WideSearch 样本做完整评测大约需要 **7 小时** 。

标准 QA 评测
~~~~~~~~~~~~

对于标准 QA 评测，请使用 ASearcher 发布的数据集：

- `ASearcher-test-data <https://huggingface.co/datasets/inclusionAI/ASearcher-test-data>`__

该数据集同时包含单跳任务（如 Natural Questions）和多跳任务（如 HotpotQA）。

请按如下方式更新 `examples/agent/wideseek_r1/config/eval_qwen3_qa.yaml`：

.. code-block:: yaml

   data:
     is_markdown: False
     val_data_paths: /PATH/TO/EVAL/QA/DATASET
     data_size: -1

这里 ``is_markdown`` 必须为 ``False``。

与 WideSearch 评测相比，标准 QA 评测速度更快。
建议先在部分标准 QA 数据上运行评测，以进行快速的基本正确性检查。

运行评测
--------

启动评测前，请确认以下各项：

- ``rollout.model.model_path`` 指向你要评测的模型。
- ``data.val_data_paths`` 指向正确的数据集。
- ``agentloop.llm_ip`` 设置正确。
- 所需工具已配置完成。参见 :doc:`tools`。

然后运行以下命令之一：

.. code-block:: bash

   bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_widesearch
   bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_qa

输出文件
--------

评测输出会写入：

.. code-block:: text

   ${runner.output_dir}/${runner.experiment_name}

重要文件包括：

- ``metric.json``：聚合指标，例如输出长度和工具使用情况。
- ``allresult.json``：完整的多轮交互日志。
- ``responses/``：每个样本的最终模型回答。

对于标准 QA 评测，``metric.json`` 还包含最终的 LLM 评判结果。

对于 WideSearch 评测，RLinf 会保存生成的回答，以便使用官方 WideSearch 评测流程进行打分。

额外的 WideSearch 打分
------------------------

如果需要最终的 WideSearch 基准分数，请使用专门的评测仓库：

- `WideSeek-R1-Eval <https://github.com/RLinf/WideSeek-R1-Eval>`__

完整流程请参见该仓库的 README。

双引擎评测
----------

WideSeek-R1 还支持在多智能体设定下使用两个独立模型实例进行评测，从而让 planner 和 worker 角色使用不同模型。

请使用 `examples/agent/wideseek_r1/config/eval_qwen3_qa_2eng.yaml`。相关字段如下：

.. code-block:: yaml

   agentloop:
     fixed_role: worker  # planner or worker

   rollout:
     use_fixed_worker: True

``use_fixed_worker`` 用于启用第二个模型实例。``fixed_role`` 用于选择哪个角色使用该第二模型。

然后你可以分别在 ``rollout.model.model_path`` 和 ``rollout_fixed_worker.model.model_path`` 下设置不同的模型路径。

说明
----

与训练时相同，``agentloop.workflow`` 用于控制评测采用单智能体还是多智能体执行：

- ``mas``：多智能体评测。
- ``sa``：单智能体评测。

单智能体模式旨在与 `ASearcher <https://github.com/inclusionAI/ASearcher>`__ 保持可比性。
