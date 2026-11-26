示例库
========================================

本节汇总 RLinf 支持的各类训练与部署示例。按你的任务起点选择对应的入口：

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

   .. grid-item-card:: 具身模型
      :link: vla_wam_index
      :link-type: doc

      对具身模型做 RL 微调 —— π₀、GR00T、StarVLA、Lingbot-VLA 等。

   .. grid-item-card:: 世界模型
      :link: world_model_index
      :link-type: doc

      训练和微调世界模型 —— OpenSora、Wan 等。

   .. grid-item-card:: 具身智能体
      :link: embodied_agent/index
      :link-type: doc

      以多轮工具调用范式训练推理 VLM 具身智能体 —— 包含 VLM-as-Policy 与后续高层规划大脑。

   .. grid-item-card:: LLM 智能体
      :link: agentic/index
      :link-type: doc

      数学推理与 LLM 智能体 AI 工作流，涵盖单智能体与多智能体设置。

   .. grid-item-card:: SFT
      :link: sft_index
      :link-type: doc

      用于产出 RL 冷启动检查点的监督微调（SFT）配方。

   .. grid-item-card:: RL
      :link: methods_index
      :link-type: doc

      以训练算法为主线 —— DAgger、RECAP、DSRL、IQL 离线 RL、仿真-真机协同训练、MLP / SAC-Flow。

   .. grid-item-card:: 奖励模型
      :link: embodied/reward_model_index
      :link-type: doc

      构建用于仿真与真实机器人强化学习流程的奖励模型。

   .. grid-item-card:: 图像与视频生成
      :link: video_models
      :link-type: doc

      准备 SD3 与 Wan2.2 模型，再选择 Diffusion-NFT 或 DiffusionART 训练。

.. toctree::
   :hidden:
   :maxdepth: 2

   模拟器 <simulators_index>
   真机 <real_world_index>
   具身模型 <vla_wam_index>
   世界模型 <world_model_index>
   具身智能体 <embodied_agent/index>
   LLM 智能体 <agentic/index>
   SFT <sft_index>
   RL <methods_index>
   奖励模型 <embodied/reward_model_index>
   图像与视频生成 <video_models>
