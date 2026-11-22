配置
====

启动前需要修改 YAML、日志或工作负载专用设置时，使用这些指南。可以先看一页式的 :doc:`训练配置 <../../reference/configuration>` 速查最小 YAML 预设与最常调的 5 个配置项，再按下方指南查阅完整字段。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`基础配置 <../basic_config>`
     - RLinf 配置文件的结构与覆盖方式。
   * - :doc:`具身智能配置 <../embodiment_config>`
     - 具身训练专用配置项。
   * - :doc:`异构环境 <../hetero_envs>`
     - 在一次运行中让同一个 policy 使用不同的环境配置，例如 ManiSkill 和 LIBERO。
   * - :doc:`智能体配置 <../agentic_config>`
     - 智能体和推理训练专用配置项。
   * - :doc:`日志与可视化 <../logger>`
     - TensorBoard、wandb 和 swanlab 日志设置。

.. toctree::
   :hidden:

   基础配置 <../basic_config>
   具身智能配置 <../embodiment_config>
   异构环境 <../hetero_envs>
   智能体配置 <../agentic_config>
   日志与可视化 <../logger>
