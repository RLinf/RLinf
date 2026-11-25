算法参考
========

本页汇总 RLinf 支持的强化学习算法，用于横向对比适用场景、是否需要 critic 与离线数据，以及对应的起步配方与配置。

算法选型矩阵
------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 18 11 11 30

   * - 算法
     - 类别
     - 适用任务
     - 需要 Critic？
     - 离线数据？
     - 起步配方 / 配置
   * - :doc:`Async PPO <async_ppo>`
     - 异步 on-policy
     - 具身（仿真、真机）
     - 是
     - 否
     - ``examples/embodiment/config/maniskill_async_ppo_openvla.yaml``、``examples/embodiment/config/libero_spatial_async_ppo_openpi_pi05.yaml``
   * - :doc:`CrossQ <crossq>`
     - 无 target 网络 off-policy
     - 具身
     - 是（BRN Q 网络）
     - 可选
     - :doc:`MLP Policy <../../examples/embodied/mlp>`\ （``examples/embodiment/config/maniskill_crossq_mlp.yaml``）
   * - :doc:`DAPO <dapo>`
     - 无 critic 非对称裁剪
     - 推理、VQA
     - 否
     - 否
     - :doc:`Moonlight-16B <../../examples/agentic/math_reasoning/moonlight>`\ （``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``）、``examples/reasoning/config/vqa/qwen3-vl-2b-grpo-fsdp-geo3k.yaml``
   * - :doc:`GRPO <grpo>`
     - 无 critic 组内相对优化
     - 具身、推理、智能体、VQA
     - 否
     - 否
     - :doc:`LIBERO <../../examples/embodied/libero>`\ （``examples/embodiment/config/libero_10_grpo_openpi.yaml``）、:doc:`数学推理 <../../examples/agentic/math_reasoning/reasoning>`\ （``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml``）
   * - :doc:`IQL <iql>`
     - 离线 RL
     - 具身（D4RL）
     - 是（Q + V）
     - 是（纯离线）
     - :doc:`D4RL IQL <../../examples/embodied/iql_d4rl>`\ （``examples/embodiment/config/d4rl_iql_mujoco.yaml``）
   * - :doc:`PPO <ppo>`
     - On-policy actor-critic
     - 具身、推理、智能体
     - 是
     - 否
     - :doc:`ManiSkill <../../examples/embodied/maniskill>`\ （``examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml``）、:doc:`数学推理 PPO <../../examples/agentic/math_reasoning/reasoning_ppo>`\ （``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml``）
   * - :doc:`REINFORCE++ <reinforce>`
     - 无 critic 策略梯度
     - 推理
     - 否
     - 否
     - :doc:`数学推理 <../../examples/agentic/math_reasoning/reasoning>`\ （``examples/reasoning/config/math/qwen2.5-1.5b-reinpp-megatron.yaml``）
   * - :doc:`RLPD <rlpd>`
     - Off-policy + 先验演示
     - 具身（真机、仿真）
     - 是（Q 集成）
     - 是（50/50 混合）
     - :doc:`Franka 真机 RL <../../examples/embodied/franka>`\ （``examples/embodiment/config/realworld_pnp_rlpd_cnn_async.yaml``）
   * - :doc:`SAC <sac>`
     - Off-policy 最大熵
     - 具身（仿真、真机）
     - 是（Q 网络）
     - 可选
     - :doc:`MLP Policy <../../examples/embodied/mlp>`\ （``examples/embodiment/config/maniskill_sac_mlp.yaml``）

.. toctree::
   :hidden:

   Async PPO <async_ppo>
   CrossQ <crossq>
   DAPO <dapo>
   GRPO <grpo>
   IQL <iql>
   PPO <ppo>
   REINFORCE++ <reinforce>
   RLPD <rlpd>
   SAC <sac>
