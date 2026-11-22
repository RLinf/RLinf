启动与扩展
==========

在单机默认运行之外启动任务，或将 RLinf 接入真机硬件时，使用这些指南。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`多节点训练 <../multi_node>`
     - 在多个节点上启动训练。
   * - :doc:`异构集群 <../hetero>`
     - 配置混合硬件与节点分组。
   * - :doc:`云边协同 <../cloud_edge>`
     - 在云端与边缘之间拆分推理与训练。
   * - :doc:`真机多机启动 <../realworld_robot>`
     - 跨控制节点与 GPU 节点启动真机强化学习。
   * - :doc:`遥操作 <../teleoperation>`
     - 让操作者在 rollout 中途从 policy 手里接管。

.. toctree::
   :hidden:

   多节点训练 <../multi_node>
   异构集群 <../hetero>
   云边协同 <../cloud_edge>
   真机多机启动 <../realworld_robot>
   遥操作 <../teleoperation>
