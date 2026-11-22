环境与日志
==========

启动训练前需要配置异构环境或实验日志时，使用这些指南。

.. note::

   完整的 YAML 字段说明与指标命名空间见 :doc:`配置与指标 <../../reference/training/index>`；VLA 与扩散模型的 adapter 配置见 :doc:`LoRA 微调 <../lora>`。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`异构环境 <../hetero_envs>`
     - 在一次运行中让同一个 policy 使用不同的环境配置，例如 ManiSkill 和 LIBERO。
   * - :doc:`日志与可视化 <../logger>`
     - TensorBoard、W&B 与 SwanLab 配置，包括离线日志与多节点 API 密钥传递。

.. toctree::
   :hidden:

   异构环境 <../hetero_envs>
   日志与可视化 <../logger>
