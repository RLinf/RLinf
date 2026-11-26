Algorithms
==========

Use this reference to compare supported RL algorithms and look up each method's objective, configuration keys, and starting recipes.

Algorithm Decision Matrix
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 18 11 11 30

   * - Algorithm
     - Family
     - Workloads
     - Needs Critic?
     - Offline Data?
     - Starting Recipe / Config
   * - :doc:`Async PPO <async_ppo>`
     - Asynchronous on-policy
     - Embodied (Sim, Real-World)
     - Yes
     - No
     - ``examples/embodiment/config/maniskill_async_ppo_openvla.yaml``, ``examples/embodiment/config/libero_spatial_async_ppo_openpi_pi05.yaml``
   * - :doc:`CrossQ <crossq>`
     - Off-policy (no target net)
     - Embodied
     - Yes (BRN Q-networks)
     - Optional
     - :doc:`MLP Policy <../../examples/embodied/mlp>` (``examples/embodiment/config/maniskill_crossq_mlp.yaml``)
   * - :doc:`DAPO <dapo>`
     - Critic-free asymmetric-clip
     - Reasoning, VQA
     - No
     - No
     - :doc:`Moonlight-16B <../../examples/agentic/math_reasoning/moonlight>` (``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``), ``examples/reasoning/config/vqa/qwen3-vl-2b-grpo-fsdp-geo3k.yaml``
   * - :doc:`GRPO <grpo>`
     - Critic-free group-relative
     - Embodied, Reasoning, Agentic, VQA
     - No
     - No
     - :doc:`LIBERO <../../examples/embodied/libero>` (``examples/embodiment/config/libero_10_grpo_openpi.yaml``), :doc:`Math Reasoning <../../examples/agentic/math_reasoning/reasoning>` (``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml``)
   * - :doc:`IQL <iql>`
     - Offline RL
     - Embodied (D4RL)
     - Yes (Q + V)
     - Yes (Offline only)
     - :doc:`D4RL IQL <../../examples/embodied/iql_d4rl>` (``examples/embodiment/config/d4rl_iql_mujoco.yaml``)
   * - :doc:`PPO <ppo>`
     - On-policy actor-critic
     - Embodied, Reasoning, Agentic
     - Yes
     - No
     - :doc:`ManiSkill <../../examples/embodied/maniskill>` (``examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml``), :doc:`Math PPO <../../examples/agentic/math_reasoning/reasoning_ppo>` (``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml``)
   * - :doc:`REINFORCE++ <reinforce>`
     - Critic-free policy gradient
     - Reasoning
     - No
     - No
     - :doc:`Math Reasoning <../../examples/agentic/math_reasoning/reasoning>` (``examples/reasoning/config/math/qwen2.5-1.5b-reinpp-megatron.yaml``)
   * - :doc:`RLPD <rlpd>`
     - Off-policy + prior demos
     - Embodied (Real-World, Sim)
     - Yes (Q-ensemble)
     - Yes (50/50 mix)
     - :doc:`Franka Real-World RL <../../examples/embodied/franka>` (``examples/embodiment/config/realworld_pnp_rlpd_cnn_async.yaml``)
   * - :doc:`SAC <sac>`
     - Off-policy max-entropy
     - Embodied (Sim, Real-World)
     - Yes (Q-networks)
     - Optional
     - :doc:`MLP Policy <../../examples/embodied/mlp>` (``examples/embodiment/config/maniskill_sac_mlp.yaml``)
   * - :doc:`SuperMimic <supermimic>`
     - Differentiable world model
     - Embodied (G1 tracking)
     - Yes
     - No
     - :doc:`SuperMimic <../../examples/embodied/supermimic>` (``examples/embodiment/config/supermimic_lafan1_gencoord.yaml``)

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
   SuperMimic <supermimic>
