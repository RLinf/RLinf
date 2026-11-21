WideSeek-R1 训练
========================================

本页介绍如何在 RLinf 中复现 WideSeek-R1 的训练流程。

参考配置使用 ``Qwen3-4B``，但当前流程也兼容 Qwen3 系列中的其他稠密模型。

概述
----------------------------------------

使用本页准备 WideSeek-R1 训练所需的模型、数据与配置。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen3-4B

   .. grid-item-card:: 算法
      :text-align: center

      Hybrid multi-agent RL

   .. grid-item-card:: 数据
      :text-align: center

      WideSeek-R1-train-data 中的 ``hybrid_20k.jsonl``

   .. grid-item-card:: 工具
      :text-align: center

      离线检索与 judge model server

.. contents::
   :depth: 2
   :local:

前置条件
--------

开始训练前，请先完成 :doc:`环境与工具配置 <tools>`：

- RLinf 环境已安装。
- 评判模型已启动。
- 离线检索工具已配置完成。

一键上手
----------------------------------------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。启动训练前，请先按 :doc:`环境与工具配置 <tools>` 启动评判模型和离线检索工具。

.. tab-set::

   .. tab-item:: Qwen3-4B（混合多智能体强化学习）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载 `Qwen3-4B <https://huggingface.co/Qwen/Qwen3-4B>`__ 和 `WideSeek-R1-train-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data>`__\ （主实验使用其中的 ``hybrid_20k.jsonl``；另有公开语料资源 `WideSeek-R1-Corpus <https://huggingface.co/datasets/RLinf/WideSeek-R1-Corpus>`__ 供查阅）：

      .. code-block:: bash

         hf download Qwen/Qwen3-4B --local-dir ./Qwen3-4B
         hf download --repo-type dataset RLinf/WideSeek-R1-train-data --local-dir ./WideSeek-R1-train-data

      **3. 启动训练**

      在 ``examples/agent/wideseek_r1/config/train_qwen3_hybrid.yaml`` 中填写 ``rollout.model.model_path``、``data.train_data_paths`` 和 ``agentloop.llm_ip``：

      .. code-block:: yaml

         rollout:
           model:
             model_type: qwen3
             model_path: /PATH/TO/MODEL

         data:
           train_data_paths: /PATH/TO/TRAIN/DATASET/hybrid_20k.jsonl
           is_hybrid: True

      启动训练（多节点启动见 :ref:`wideseek-r1-multinode`）：

      .. code-block:: bash

         bash examples/agent/wideseek_r1/run_train.sh train_qwen3_hybrid

混合与单一来源数据集
----------------------------------------

``is_hybrid`` 表示训练集是否混合了 WideSearch 风格数据和标准 QA 数据。

- 对于提供的混合训练集，请设置 ``is_hybrid: True``。
- 如果你使用单一来源数据集训练，例如 ``width_20k`` 或 ``depth_20k``，请设置 ``is_hybrid: False``。
- ``is_hybrid`` 设为 ``False`` 时，请确保 ``data.is_markdown`` 与所使用数据集的格式一致（``width_20k`` 为 ``True``，``depth_20k`` 为 ``False``）。

可视化与结果
----------------------------------------

训练输出会写入：

.. code-block:: text

   ${runner.output_dir}/${runner.experiment_name}

你可以查看该目录中的 TensorBoard 文件来监控训练指标。

说明
----------------------------------------

WideSeek-R1 同时支持单智能体和多智能体执行模式。可通过 YAML 配置中的 ``agentloop.workflow`` 切换：

- ``mas``：多智能体训练。
- ``sa``：单智能体训练。

单智能体模式旨在与 `ASearcher <https://github.com/inclusionAI/ASearcher>`__ 保持可比性。
