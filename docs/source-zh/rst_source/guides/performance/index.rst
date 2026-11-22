性能与并行
==========

延迟、吞吐、显存、placement 或大模型训练效率成为瓶颈时，按下方场景选择对应指南。

通用调优
--------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 指南
     - 适用现象（何时选用）
     - 说明
   * - :doc:`LoRA <../lora>`
     - 全参数微调显存不足。
     - 为具身 FSDP VLA（``is_lora`` / ``merged_loras``）或扩散模型（``use_lora``）开启低秩适配训练。
   * - :doc:`Profiling <../profile>`
     - 不确定瓶颈在环境、rollout、通信还是 actor 训练。
     - 对 Ray worker 进程做系统级 profiling 与时间线采集。

具身吞吐优化
------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 指南
     - 适用现象（何时选用）
     - 说明
   * - :doc:`训练流水线模式 <../training_pipeline>`
     - actor GPU 空等整轮 rollout 结束，或大尺寸轨迹一次性传输形成阻塞。
     - 按阶段切分 rollout，并将 micro-batch 流式发往 FSDP actor 实现训推重叠。
   * - :doc:`环境解耦模式 <../env_decoupled_mode>`
     - 各环境步时差异大，慢环境拖累 rollout GPU 等待。
     - 解耦 env worker 与 rollout worker，实现动态 rollout 调度。
   * - :doc:`观测压缩 <../obs_compression>`
     - 高分辨率图像观测占满 env 到 rollout 通道的网络或进程间带宽。
     - 使用 ``lz4`` 或 ``zstd`` 在 env 到 rollout 通道上无损压缩图像观测。
   * - :doc:`RTC <../rtc>`
     - 真机或仿真执行动作块之间因推理延迟出现停顿。
     - 将 policy 推理与动作块执行重叠，隐藏推理延迟。

大模型与智能体
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 指南
     - 适用现象（何时选用）
     - 说明
   * - :doc:`5D 并行 <../5D>`
     - 大尺寸 LLM 或 MoE 模型单卡装不下，或长序列显存与速度受限。
     - 配置 Megatron TP、DP、PP、SP、CP 与 MoE 专家并行（EP），附 GPU 规模速查表。
   * - :doc:`异步智能体 RL <../async_agentic>`
     - 多轮 agent rollout 耗时长，同步步进下训练 GPU 大量闲置。
     - 让多轮 rollout 与 Megatron 训练重叠，并用两个准入上限控制数据陈旧度。
   * - :doc:`自动 Placement <../auto_placement>`
     - 需要为 Megatron 数学推理任务选择更快的共置或分离 GPU 划分方案。
     - 根据实测单步耗时自动搜索最优 ``cluster.component_placement``。
   * - :doc:`FUSCO <../../examples/system/fusco>`
     - Megatron MoE 训练中 All-to-All 通信耗时占比过高。
     - 融合 token 重排与 All-to-All 通信，加速 Megatron MoE 训练。
   * - :doc:`动态调度 <../dynamic_scheduling>`
     - 分离式 Megatron-LM 0.11 流水线中 rollout 远早于 actor 完成。
     - 训练运行期在 rollout、inference 与 actor 间动态迁移 GPU（旧版 Megatron-LM 0.11 环境）。

.. toctree::
   :hidden:

   LoRA <../lora>
   Profiling <../profile>
   训练流水线模式 <../training_pipeline>
   环境解耦模式 <../env_decoupled_mode>
   观测压缩 <../obs_compression>
   RTC <../rtc>
   5D 并行 <../5D>
   异步智能体 RL <../async_agentic>
   自动 Placement <../auto_placement>
   FUSCO <../../examples/system/fusco>
   动态调度 <../dynamic_scheduling>
