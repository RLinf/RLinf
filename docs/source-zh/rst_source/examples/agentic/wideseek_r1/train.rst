训练
====

本页介绍如何在 RLinf 中复现 WideSeek-R1 的训练流程。

参考配置使用 ``Qwen3-4B``，但当前流程也兼容 Qwen3 系列中的其他稠密模型。

.. contents::
   :depth: 2
   :local:

前置条件
--------

开始训练前，请确保以下组件已准备就绪：

- RLinf 环境已安装。参见 :doc:`../../../start/installation`。
- 评判模型服务已启动。参见 :doc:`index`。
- 离线检索工具已配置完成。参见 :doc:`tools`。

下载基础模型
------------

主实验使用 `Qwen3-4B <https://huggingface.co/Qwen/Qwen3-4B>`__。

下载模型后，更新 `examples/agent/wideseek_r1/config/train_qwen3_hybrid.yaml` 中的本地模型路径：

.. code-block:: yaml

   rollout:
     model:
       model_type: qwen3
       model_path: /PATH/TO/MODEL

下载训练数据
------------

WideSeek-R1 训练使用一个 2 万条样本的混合数据集，该数据集将广域信息检索数据与标准 QA 数据结合在一起。数据集可在 Hugging Face 获取：

- `WideSeek-R1-train-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data>`__

主实验使用该数据集中的 ``hybrid_20k.jsonl``。

下载数据后，更新 `examples/agent/wideseek_r1/config/train_qwen3_hybrid.yaml` 中的数据集路径：

.. code-block:: yaml

   data:
     train_data_paths: /PATH/TO/TRAIN/DATASET/hybrid_20k.jsonl
     is_hybrid: True

关于 ``is_hybrid`` 和 ``is_markdown``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``is_hybrid`` 表示训练集是否混合了 WideSearch 风格数据和标准 QA 数据。

- 对于提供的混合训练集，请设置 ``is_hybrid: True``。
- 如果你使用的是单一来源数据集进行训练，例如 ``width_20k`` 或 ``depth_20k``，请设置 ``is_hybrid: False``。
- 当 ``is_hybrid`` 为 ``False`` 时，请确保 ``data.is_markdown`` 与所使用数据集的格式一致（``width_20k`` 为 True，``depth_20k`` 为 False）。

启动训练
--------

开始训练前，请确认以下各项均正确无误：

- ``rollout.model.model_path`` 指向已下载的基础模型。
- ``data.train_data_paths`` 指向训练数据集。
- ``agentloop.llm_ip`` 设置正确。
- 离线工具已配置且可访问。参见 :doc:`tools`。

然后运行：

.. code-block:: bash

   bash examples/agent/wideseek_r1/run_train.sh train_qwen3_hybrid

输出
----

训练输出会写入：

.. code-block:: text

   ${runner.output_dir}/${runner.experiment_name}

你可以查看该目录中的 TensorBoard 文件来监控训练指标。

说明
----

WideSeek-R1 同时支持单智能体和多智能体执行模式。可通过 YAML 配置中的 ``agentloop.workflow`` 进行切换：

- ``mas``：多智能体训练。
- ``sa``：单智能体训练。

单智能体模式旨在与 `ASearcher <https://github.com/inclusionAI/ASearcher>`__ 保持可比性。
