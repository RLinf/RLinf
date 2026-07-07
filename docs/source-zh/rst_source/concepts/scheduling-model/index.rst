调度模型
========

当你需要理解任务运行位置，以及 RLinf 如何存储轨迹数据时，使用这些概念页。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 概念
     - 内容
   * - :doc:`Placement <../placement>`
     - worker 如何映射到节点与 GPU。
   * - :doc:`执行模式 <../execution_modes>`
     - 共享式、分离式与混合式 placement 的权衡。
   * - :doc:`Replay Buffer <../replay_buffer>`
     - 轨迹回放缓冲区的设计与采样。

.. toctree::
   :hidden:

   Placement <../placement>
   执行模式 <../execution_modes>
   Replay Buffer <../replay_buffer>
