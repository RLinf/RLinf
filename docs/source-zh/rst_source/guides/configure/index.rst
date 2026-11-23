配置与运行
==========

跑通首个任务后从这里开始：接入日志监控、配置断点续训、导出 Checkpoint、用 LoRA 降低显存占用，或在一次训练中混合多个环境。

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: 日志与可视化
      :link: ../logger
      :link-type: doc

      将训练曲线与评测视频接入 TensorBoard、Weights & Biases 或 SwanLab。

   .. grid-item-card:: 断点续训
      :link: ../resume
      :link-type: doc

      定期保存训练状态，并在任务中断后从检查点恢复。

   .. grid-item-card:: Checkpoint 转换
      :link: ../convertor
      :link-type: doc

      将 FSDP 或 Megatron 训练产物转换为 Hugging Face 或评测格式。

   .. grid-item-card:: LoRA 微调
      :link: ../lora
      :link-type: doc

      通过 LoRA 适配器显著降低大模型与 VLA 训练显存。

   .. grid-item-card:: 多 Env 混合训练
      :link: ../hetero_envs
      :link-type: doc

      在单次运行中同时使用多个不同的仿真或真机 Env 配置。

.. toctree::
   :hidden:

   日志与可视化 <../logger>
   断点续训 <../resume>
   Checkpoint 转换 <../convertor>
   LoRA 微调 <../lora>
   多 Env 混合训练 <../hetero_envs>
