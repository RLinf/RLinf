概念
====

在调优 placement、worker 或通信之前，先阅读概念页了解 RLinf 的执行与调度模型。

选择概念区域
------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 执行模型
      :link: execution-model/index
      :link-type: doc

      理解任务流程、worker、cluster、channel 与 collective。

   .. grid-item-card:: 调度模型
      :link: scheduling-model/index
      :link-type: doc

      理解 placement 策略、执行模式与 replay buffer。

.. toctree::
   :hidden:

   执行模型 <execution-model/index>
   调度模型 <scheduling-model/index>
