博客
====

阅读 RLinf 的工程实践、系统优化与模型集成文章。

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: MUSA 复现 WoVR
      :link: musa_wovr
      :link-type: doc

      摩尔线程 S5000 上的 MUSA 适配、精度对齐，以及 8 卡与 16 卡性能实践。

   .. grid-item-card:: WorldLoop 奖励模型
      :link: worldloop_topreward
      :link-type: doc

      为 WoVR 世界模型 RL 接入无需训练的 VLM 奖励。

   .. grid-item-card:: SGLang 集成 Cosmos3
      :link: cosmos3_sft_and_sglang_evaluate
      :link-type: doc

      微调 Cosmos3，并通过 SGLang 并行推理完成评测。

   .. grid-item-card:: BEHAVIOR 25× 优化
      :link: behavior_system_optimization
      :link-type: doc

      加速家庭操作任务中的环境交互。

   .. grid-item-card:: Ascend 具身优化
      :link: ascend_embodied_optimization
      :link-type: doc

      通过复用注意力控制信息、重叠 FSDP 参数通信并减少主机-设备同步，提升昇腾 NPU 上的具身 RL 吞吐。

   .. grid-item-card:: Cursor 代码 RL
      :link: build_a_coding_online_rl_case
      :link-type: doc

      将代码补全反馈接入在线 policy 训练。

   .. grid-item-card:: 与 veRL 对比
      :link: compare_with_verl
      :link-type: doc

      对照测试环境、配置与吞吐性能结果。

.. toctree::
   :hidden:
   :maxdepth: 1

   MUSA 复现 WoVR <musa_wovr>
   WorldLoop 奖励模型 <worldloop_topreward>
   SGLang 集成 Cosmos3 <cosmos3_sft_and_sglang_evaluate>
   BEHAVIOR 25× 优化 <behavior_system_optimization>
   Ascend 具身优化 <ascend_embodied_optimization>
   Cursor 代码 RL <build_a_coding_online_rl_case>
   与 veRL 对比 <compare_with_verl>
