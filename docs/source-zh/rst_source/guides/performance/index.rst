性能
====

当延迟、吞吐、显存、placement 或大模型训练效率成为瓶颈时，使用这些指南。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`RTC <../rtc>`
     - 将策略推理与动作块执行重叠，隐藏推理延迟，支持仿真与真机。
   * - :doc:`Env Decoupled Mode <../env_decoupled_mode>`
     - 解耦 Env Worker 与 Rollout Worker，用于具身任务中的动态 rollout 调度。
   * - :doc:`观测压缩 <../obs_compression>`
     - 在 Env 到 Rollout 通道上无损压缩图像观测，节省带宽。
   * - :doc:`LoRA <../lora>`
     - 使用 LoRA adapter 训练。
   * - :doc:`自动 Placement <../auto_placement>`
     - 为训练负载自动选择最优 placement。
   * - :doc:`动态调度 <../dynamic_scheduling>`
     - 训练过程中动态调度资源。
   * - :doc:`Profiling <../profile>`
     - 对 Ray worker 进程进行系统级 profiling。
   * - :doc:`5D 并行 <../5D>`
     - 为大模型配置 5D 并行。

.. toctree::
   :hidden:

   RTC <../rtc>
   Env Decoupled Mode <../env_decoupled_mode>
   观测压缩 <../obs_compression>
   LoRA <../lora>
   自动 Placement <../auto_placement>
   动态调度 <../dynamic_scheduling>
   Profiling <../profile>
   5D 并行 <../5D>
