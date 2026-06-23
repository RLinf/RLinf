算法
====

当你需要查看支持的 RL 算法目标、适用范围与实现说明时，使用这些参考页。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 算法
     - 简介
   * - :doc:`PPO <ppo>`
     - Proximal Policy Optimization。
   * - :doc:`GRPO <grpo>`
     - Group Relative Policy Optimization。
   * - :doc:`DAPO <dapo>`
     - 解耦裁剪与动态采样的策略优化。
   * - :doc:`Reinforce++ <reinforce>`
     - 增强版 REINFORCE 基线。
   * - :doc:`SAC <sac>`
     - Soft Actor-Critic。
   * - :doc:`CrossQ <crossq>`
     - 无需 target 网络的高样本效率离策略 RL。
   * - :doc:`RLPD <rlpd>`
     - 利用先验数据的强化学习。
   * - :doc:`IQL <iql>`
     - 面向离线 RL 的 Implicit Q-Learning。
   * - :doc:`Async PPO <async_ppo>`
     - 异步流水线化的 PPO。

.. toctree::
   :hidden:

   PPO <ppo>
   GRPO <grpo>
   DAPO <dapo>
   Reinforce++ <reinforce>
   SAC <sac>
   CrossQ <crossq>
   RLPD <rlpd>
   IQL <iql>
   Async PPO <async_ppo>
