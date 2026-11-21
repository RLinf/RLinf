配置
====

当你需要在启动前修改 YAML、日志或工作负载专用设置时，使用这些指南。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`基础配置 <../basic_config>`
     - RLinf 配置文件的结构与覆盖方式。
   * - :doc:`具身智能配置 <../embodiment_config>`
     - 具身训练专用配置项。
   * - :doc:`智能体配置 <../agentic_config>`
     - 智能体和推理训练专用配置项。
   * - :doc:`日志 <../logger>`
     - TensorBoard、wandb 和 swanlab 日志设置。
   * - :doc:`理解 success_once 指标的波动 <../eval_success_once_noise>`
     - 区分具身评估指标中的统计噪声与真实训练不稳定。

.. toctree::
   :hidden:

   基础配置 <../basic_config>
   具身智能配置 <../embodiment_config>
   智能体配置 <../agentic_config>
   日志 <../logger>
   理解 success_once 指标的波动 <../eval_success_once_noise>
