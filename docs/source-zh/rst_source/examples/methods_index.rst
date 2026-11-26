RL 算法示例
===========

按训练方法而不是具体基准或模型选型时，从本节进入。涵盖离策略与离线 RL、在线蒸馏、仿真-真机协同训练以及噪声空间 policy 调控。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 仿真-真机协同训练
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-co/overview.png
      :img-alt: 仿真-真机协同训练概览
      :link: embodied/co_training
      :link-type: doc

      结合仿真 PPO 与真机 SFT，提升 Sim-to-Real 迁移表现。

   .. grid-item-card:: DSRL
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dsrl.png
      :img-alt: Pi0 噪声空间强化学习
      :link: embodied/dsrl
      :link-type: doc

      用轻量级 SAC 在噪声空间调控冻结的 π₀ 扩散 policy。

   .. grid-item-card:: FlexRLT
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_zh.png
      :img-alt: FlexRLT 冻结 VLA 与可训练 actor-critic
      :link: embodied/flexrlt
      :link-type: doc

      基于冻结 VLA prefix 特征在线训练轻量 actor-critic，并用 STEAM 做阶段路由。

   .. grid-item-card:: IQL（D4RL）
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/d4rl.png
      :img-alt: D4RL 上的 IQL
      :link: embodied/iql_d4rl
      :link-type: doc

      在 D4RL 基准上训练离线 Implicit Q-Learning policy。

   .. grid-item-card:: OPD
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/opd.png
      :img-alt: 在线策略蒸馏
      :link: embodied/opd
      :link-type: doc

      在 LIBERO 上利用 teacher logprob 对 student on-policy 动作 token 做蒸馏。

   .. grid-item-card:: RECAP
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/recap.png
      :img-alt: RECAP 概览
      :link: embodied/recap
      :link-type: doc

      用优势条件引导做离线 policy 优化。

   .. grid-item-card:: RL Token
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT.png
      :img-alt: RL Token
      :link: embodied/rlt
      :link-type: doc

      借助冻结的 VLA prefix 特征启动在线 RL，并训练轻量级 actor-critic。

   .. grid-item-card:: SAC-Flow
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-overview.png
      :img-alt: SAC-Flow 概览
      :link: embodied/sac_flow
      :link-type: doc

      在 ManiSkill 仿真或真实 Franka 机械臂上用 SAC 训练 Flow Matching policy。

   .. grid-item-card:: STEAM
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/steam.png
      :img-alt: STEAM 集成优势建模
      :link: embodied/steam
      :link-type: doc

      用集成优势预测器对离线样本做保守评估。

   .. grid-item-card:: SuperMimic
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/supermimic.jpg
      :img-alt: SuperMimic 概览
      :link: embodied/supermimic
      :link-type: doc

      在 MuJoCo Warp 中通过可微世界模型训练 G1 全身动作跟踪 policy。

.. toctree::
   :hidden:
   :maxdepth: 2

   Co-Training <embodied/co_training>
   DSRL <embodied/dsrl>
   FlexRLT <embodied/flexrlt>
   IQL (D4RL) <embodied/iql_d4rl>
   OPD <embodied/opd>
   RECAP <embodied/recap>
   RL Token <embodied/rlt>
   SAC-Flow <embodied/sac_flow>
   STEAM <embodied/steam>
   SuperMimic <embodied/supermimic>
