.. _wideseek-r1-example:

WideSeek-R1
===========

WideSeek-R1 是一个面向广域信息检索任务的主智能体与子智能体框架，通过多智能体强化学习（MARL）进行训练。它通过共享 LLM、隔离的智能体上下文以及专用工具，实现了可扩展的编排与并行执行。

在 WideSearch 基准上，WideSeek-R1-4B 的 item F1 分数达到 ``40.0%``。这一结果可与单智能体 DeepSeek-R1-671B 相当，并且随着并行子智能体数量的增加仍在持续提升。

有关完整方法和实验结果，请参见 :doc:`WideSeek-R1 论文页面 <../../../resources/publications/wideseek_r1>`、`项目主页 <https://wideseek-r1.github.io>`__、`arXiv 论文 <https://arxiv.org/abs/2602.04634>`__，以及 `RLinf 中的示例代码 <https://github.com/RLinf/RLinf/tree/main/examples/agent/wideseek_r1>`__。

本页介绍该示例，并引导你进入环境与工具配置、训练与评测。请先完成 :doc:`环境与工具配置 <tools>`，再打开与当前步骤对应的页面。

概述
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen3-4B 与 Qwen3 系列稠密模型

   .. grid-item-card:: 算法
      :text-align: center

      面向广域信息检索的多智能体强化学习

   .. grid-item-card:: 工具
      :text-align: center

      在线网页搜索或离线 Qdrant 检索

   .. grid-item-card:: 硬件
      :text-align: center

      单节点快速开始或多节点扩展

选择页面
--------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 环境与工具配置
      :link: tools
      :link-type: doc

      安装依赖栈、配置搜索后端、启动评判模型，并做多节点扩展。

   .. grid-item-card:: 训练
      :link: train
      :link-type: doc

      准备模型和数据，然后启动 hybrid 多智能体强化学习。

   .. grid-item-card:: 评测
      :link: eval
      :link-type: doc

      使用匹配的工具后端，在 WideSearch 或标准 QA 上评测。

.. toctree::
   :hidden:
   :maxdepth: 2

   环境与工具配置 <tools>
   训练 <train>
   评测 <eval>
