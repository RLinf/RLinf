示例库
========

本节展示了 **RLinf 目前支持的示例集合**，
展示该框架如何应用于不同场景，并演示其在实际中的高效性。示例库会随着时间不断扩展，涵盖新的场景和任务，以展示 RLinf 的多样性和可扩展性。

具身智能是 RLinf 的核心方向，具身示例被拆分为以下五个入口，便于按你的实际起点快速定位：

- :doc:`simulators_index` —— 当你以模拟器 / 基准（LIBERO、ManiSkill、RoboTwin、IsaacLab 等）为出发点时选择本节。

- :doc:`real_world_index` —— 当你拥有真实硬件（Franka、灵巧手、移动双臂平台等）时选择本节。

- :doc:`vla_wam_index` —— 当你想对某个模型家族（π₀、GR00T、Lingbot-VLA、OpenSora、Wan 等）做 RL 微调时选择本节。

- :doc:`sft_index` —— 用于产出 RL 冷启动检查点的 SFT 配方。

- :doc:`methods_index` —— 以训练算法为主线的示例（DAgger、RECAP、DSRL、IQL 离线 RL、仿真-真机协同训练、MLP / SAC-Flow 策略等）。

具身之外：

- :doc:`agentic/index` —— 覆盖数学推理与智能体 AI 工作流的训练示例，包含单智能体与多智能体设置。

- :doc:`system/index` —— 展示计算资源的灵活与动态调度，以及任务分配到最合适硬件设备的示例。

.. toctree::
   :hidden:
   :maxdepth: 2

   simulators_index
   real_world_index
   vla_wam_index
   sft_index
   methods_index
   agentic/index
   system/index

