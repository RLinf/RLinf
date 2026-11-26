核心概念
========

通过核心概念了解 RLinf 的执行流、调度排布与 Robotics 硬件抽象。直接从最贴近当前问题的主题开始阅读即可。

选择主题
--------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 架构总览
      :link: overview
      :link-type: doc

      用 5 分钟建立 RL 训练循环、部署模式与数据格式的全局视图。

   .. grid-item-card:: 执行模型
      :link: execution-model/index
      :link-type: doc

      了解执行流、Worker、Cluster、Channel 与集合通信。

   .. grid-item-card:: 调度与排布
      :link: scheduling-model/index
      :link-type: doc

      了解 Placement 策略、执行模式与 Replay Buffer。

   .. grid-item-card:: Robotics 架构
      :link: robotics-model/index
      :link-type: doc

      了解真机部件接口、连接生命周期与真机任务环境。

.. toctree::
   :hidden:

   架构总览 <overview>
   执行模型 <execution-model/index>
   调度与排布 <scheduling-model/index>
   Robotics 架构 <robotics-model/index>
