高级集成
========

当扩展内容涉及后端集成或权重传输，而不是添加主要模型或环境时，
使用这些指南。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`Megatron-Bridge <../mbridge>`
     - 使用 Megatron-Bridge actor 后端。
   * - :doc:`权重同步 <../weight_syncer>`
     - 优化具身训练中 actor 到 rollout 的权重同步。
   * - :doc:`将 session server 接入 runner <../session_server_runner>`
     - 在异步 runner 以外的 runner 中使用 session server。

.. toctree::
   :hidden:

   Megatron-Bridge <../mbridge>
   权重同步 <../weight_syncer>
   将 session server 接入 runner <../session_server_runner>
