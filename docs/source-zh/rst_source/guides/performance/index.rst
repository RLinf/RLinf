性能
====

当吞吐、显存、placement 或大模型训练效率成为瓶颈时，使用这些指南。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`自动 Placement <../auto_placement>`
     - 为训练负载自动选择最优 placement。
   * - :doc:`动态调度 <../dynamic_scheduling>`
     - 训练过程中动态调度资源。
   * - :doc:`Profiling <../profile>`
     - 对 Ray worker 进程进行系统级 profiling。
   * - :doc:`5D 并行 <../5D>`
     - 为大模型配置 5D 并行。
   * - :doc:`LoRA <../lora>`
     - 使用 LoRA adapter 训练。
   * - :doc:`Env Decoupled Mode <../env_decoupled_mode>`
     - 解耦 Env Worker 与 Rollout Worker，用于具身任务中的动态 rollout 调度。
   * - :doc:`真机仿真器 <real_simulator>`
     - 在仿真中模拟观测延迟和网络条件。

.. toctree::
   :hidden:

   自动 Placement <../auto_placement>
   动态调度 <../dynamic_scheduling>
   Profiling <../profile>
   5D 并行 <../5D>
   LoRA <../lora>
   Env Decoupled Mode <../env_decoupled_mode>
   真机仿真器 <real_simulator>
