指南
====

指南面向实际操作流程：配置训练、扩展运行、管理数据与 checkpoint，以及调优性能。

选择工作流
----------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 配置
      :link: configure/index
      :link-type: doc

      设置 Hydra 配置、日志与不同工作负载的专用选项。

   .. grid-item-card:: 启动与扩展
      :link: launch-scale/index
      :link-type: doc

      在多节点、异构硬件、云边协同或真实机器人上运行。

   .. grid-item-card:: 数据与 Checkpoint
      :link: data-checkpoints/index
      :link-type: doc

      采集数据、转换 checkpoint，并恢复训练。

   .. grid-item-card:: 性能
      :link: performance/index
      :link-type: doc

      调整 placement、调度、并行、profiling 与 adapter 训练。

   .. grid-item-card:: 硬件后端
      :link: hardware/index
      :link-type: doc

      在 AMD ROCm 或 Ascend CANN 上运行 RLinf。

   .. grid-item-card:: 智能体工作流
      :link: agent-workflows/index
      :link-type: doc

      运行智能体与推理工作负载。

   .. grid-item-card:: Rollout 引擎
      :link: rollout-engines/index
      :link-type: doc

      启动 sglang server 与 router，向外暴露统一的推理入口。

.. toctree::
   :hidden:

   配置 <configure/index>
   启动与扩展 <launch-scale/index>
   数据与 Checkpoint <data-checkpoints/index>
   性能 <performance/index>
   硬件后端 <hardware/index>
   智能体工作流 <agent-workflows/index>
   Rollout 引擎 <rollout-engines/index>
