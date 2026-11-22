指南
====

指南面向实际操作流程：配置训练、扩展运行、管理数据与 checkpoint，以及调优性能。

选择工作流
----------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 启动与扩展
      :link: launch-scale/index
      :link-type: doc

      在多节点、真机、异构硬件或云边协同环境上运行。

   .. grid-item-card:: 数据与 Checkpoint
      :link: data-checkpoints/index
      :link-type: doc

      恢复训练、转换 checkpoint，以及采集示教数据。

   .. grid-item-card:: 性能与并行
      :link: performance/index
      :link-type: doc

      调整 LoRA、profiling、流水线模式、RTC、5D 并行与调度。

   .. grid-item-card:: 环境与日志
      :link: configure/index
      :link-type: doc

      配置 TensorBoard、W&B 与 SwanLab 实验日志，以及异构训练环境。

   .. grid-item-card:: Rollout 引擎
      :link: rollout-engines/index
      :link-type: doc

      切换 SGLang / vLLM 版本、启动 SGLang 服务，或使用 PhyAI / ApxInf 具身后端。
