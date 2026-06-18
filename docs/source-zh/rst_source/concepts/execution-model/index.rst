执行模型
========

当你需要理解 RLinf 任务如何运行，以及 worker 如何交换工作时，使用这些概念页。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 概念
     - 内容
   * - :doc:`RLinf 执行流程 <../execution_flow>`
     - 跨代码、进程与核心抽象的端到端任务流程。
   * - :doc:`M2Flow 编程流程 <../flow>`
     - 将逻辑与调度分离的宏观到微观模型。
   * - :doc:`Worker 与 WorkerGroup <../worker>`
     - 计算单元，以及驱动 worker group 的句柄。
   * - :doc:`Cluster <../cluster>`
     - 集群抽象与资源模型。
   * - :doc:`Channel <../channel>`
     - 用于 worker 间数据交换的异步通道。
   * - :doc:`集合通信 <../collective>`
     - 集合通信操作与异步工作句柄。

.. toctree::
   :hidden:

   RLinf 执行流程 <../execution_flow>
   M2Flow 编程流程 <../flow>
   Worker 与 WorkerGroup <../worker>
   Cluster <../cluster>
   Channel <../channel>
   集合通信 <../collective>
