高级集成
========

需要底层后端定制时使用以下指南：包括手动编写 Megatron 模型适配（Megatron-Bridge 暂未覆盖你的模型结构时参考 :doc:`Megatron-Bridge <../mbridge>`\ 之外的底层方案）以及优化 actor 到 rollout 的权重同步传输。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`手动接入 Megatron 模型 <../new_model_megatron>`
     - 手动编写 Megatron actor 适配与 SGLang 权重重切分逻辑。
   * - :doc:`权重同步 <../weight_syncer>`
     - 优化具身训练中 actor 到 rollout 的权重同步。

.. toctree::
   :hidden:

   手动接入 Megatron 模型 <../new_model_megatron>
   权重同步 <../weight_syncer>
