高级集成
========

当扩展内容涉及后端集成、权重传输或奖励模型工作流，而不是添加主要模型或环境时，
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
   * - :doc:`奖励模型流程 <../reward_model>`
     - 使用图像分类与 VLM 奖励模型。

.. toctree::
   :hidden:

   Megatron-Bridge <../mbridge>
   权重同步 <../weight_syncer>
   奖励模型流程 <../reward_model>
