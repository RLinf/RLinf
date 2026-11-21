博客
====

阅读 RLinf 的工程实践、系统优化与模型集成文章。

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 在摩尔线程 S5000 上复现 WoVR
      :link: musa_wovr
      :link-type: doc

      MUSA 适配、精度对齐，以及 8 卡与 16 卡性能实践。

   .. grid-item-card:: WorldLoop 与 WoVR
      :link: worldloop_topreward
      :link-type: doc

      为世界模型 RL 接入无需训练的 VLM 奖励。

   .. grid-item-card:: Cosmos3 与 SGLang
      :link: cosmos3_sft_and_sglang_evaluate
      :link-type: doc

      微调 Cosmos3，并通过并行推理完成评测。

   .. grid-item-card:: BEHAVIOR 系统优化
      :link: behavior_system_optimization
      :link-type: doc

      加速家庭操作任务中的环境交互。

   .. grid-item-card:: 具身模型的 Ascend 优化
      :link: ascend_embodied_optimization
      :link-type: doc

      从完整调用的数据依赖，解释控制信息复用、通信重叠和执行图的设计与收益。

   .. grid-item-card:: 代码补全在线 RL
      :link: build_a_coding_online_rl_case
      :link-type: doc

      将代码补全反馈接入在线 policy 训练。

   .. grid-item-card:: 与 veRL 对比
      :link: compare_with_verl
      :link-type: doc

      对照测试环境、配置与性能结果。

.. toctree::
   :hidden:
   :maxdepth: 1

   RLinf × MUSA：复现 WoVR <musa_wovr>
   RLinf × WorldLoop：VLM 奖励 <worldloop_topreward>
   RLinf × SGLang：Cosmos3 实践 <cosmos3_sft_and_sglang_evaluate>
   BEHAVIOR 25 倍系统优化 <behavior_system_optimization>
   昇腾 Ascend 具身优化 <ascend_embodied_optimization>
   Cursor 在线代码强化学习 <build_a_coding_online_rl_case>
   与 veRL 对比 <compare_with_verl>
