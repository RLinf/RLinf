示例库
========================================

本节展示了 **RLinf 目前支持的示例集合**，
展示该框架如何应用于不同场景，并演示其在实际中的高效性。示例库会随着时间不断扩展，涵盖新的场景和任务，以展示 RLinf 的多样性和可扩展性。

具身智能是 RLinf 的核心方向。具身示例被拆分为以下五个入口，便于按你的实际起点快速定位：

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 模拟器
      :link: simulators_index
      :link-type: doc

      以模拟器 / 基准为出发点 —— LIBERO、ManiSkill、RoboTwin、IsaacLab 等。

   .. grid-item-card:: 真机
      :link: real_world_index
      :link-type: doc

      在真实机器人硬件上运行 —— Franka 系列，以及 GimArm、XSquare Turtle2、DOS-W1。

   .. grid-item-card:: 模型
      :link: vla_wam_index
      :link-type: doc

      对某个模型家族做 RL 微调 —— π₀、GR00T、Lingbot-VLA、OpenSora、Wan 等。

   .. grid-item-card:: SFT
      :link: sft_index
      :link-type: doc

      用于产出 RL 冷启动检查点的监督微调（SFT）配方。

   .. grid-item-card:: 算法
      :link: methods_index
      :link-type: doc

      以训练算法为主线 —— DAgger、RECAP、DSRL、IQL 离线 RL、仿真-真机协同训练、MLP / SAC-Flow。

具身之外：

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 智能体
      :link: agentic/index
      :link-type: doc

      数学推理与智能体 AI 工作流，涵盖单智能体与多智能体设置。

   .. grid-item-card:: 系统
      :link: system/index
      :link-type: doc

      计算资源的灵活与动态调度，并将任务分配到最合适的硬件设备。

.. toctree::
   :hidden:
   :maxdepth: 2

   模拟器 <simulators_index>
   真机 <real_world_index>
   模型 <vla_wam_index>
   SFT <sft_index>
   算法 <methods_index>
   智能体 <agentic/index>
   系统 <system/index>
