自定义环境奖励
==============

本页介绍如何自定义具身环境每步送入 RL 训练（GRPO、PPO 等）的奖励：
稠密奖励、分层（阶段化）奖励结构，以及完全自定义的奖励函数——
全部通过 YAML 选择，无需修改环境代码。

步进奖励如何进入训练
--------------------

每个具身环境在 ``_calc_step_reward`` 中（例如
``rlinf/envs/maniskill/maniskill_env.py``）把仿真器的原始奖励和每步
``info`` 字典转换为训练奖励。该奖励构成轨迹回报，随后由
``algorithm.adv_type`` 选择的优势函数（如 ``grpo``、``gae``）消费。

奖励函数由 ``env.<split>.reward_mode`` 选择。对 ManiSkill 环境：

- ``default`` —— 内置的硬编码分层奖励
  （抓取 ``0.1`` + 持续抓取 ``0.1`` + 门控成功 ``1.0``）；
- ``raw`` —— 透传仿真器的原始（通常为稠密）奖励；
- ``only_success`` —— 稀疏奖励，任务成功时为 ``1.0``；
- 其他任意名称 —— 在步进奖励注册表（``rlinf/envs/rewards``）中查找，
  见下文。

用 YAML 配置加权分层稠密奖励
----------------------------

内置的 ``weighted_components`` 奖励完全通过配置组合出稠密、分层的
奖励。每个分量引用环境步进 ``info`` 字典中的一个张量（特殊名称
``raw`` 引用仿真器的原始奖励）：

.. code-block:: yaml

   env:
     train:
       reward_mode: weighted_components
       reward_components:
         # 第一层：接触 / 抓取检测
         is_src_obj_grasped: 0.3
         # 第二层：基于距离信号的稠密位姿对齐塑形
         gripper_carrot_dist:
           weight: 0.5
           transform: one_minus_tanh
           scale: 5.0
         # 第三层：任务完成，以抓取阶段为门控
         success:
           weight: 1.0
           requires: [is_src_obj_grasped]

每个条目要么是 ``name: weight``，要么是包含以下键的字典：

- ``weight`` （必填）—— 分量系数；
- ``transform`` —— 加权前对信号做变换：
  ``none`` （默认）、``neg`` （``-scale * v``，用于惩罚项）、
  ``neg_exp`` （``exp(-scale * v)``）和 ``one_minus_tanh``
  （``1 - tanh(scale * v)``）——后两者把距离转换为在零处取峰值的
  稠密奖励；
- ``scale`` —— 变换陡峭程度（默认 ``1.0``）；
- ``requires`` —— 一组必须全部为真的布尔 ``info`` 键，分量才计入
  奖励；分层 / 阶段化结构由此表达。

可用的分量名称取决于任务 ``evaluate()`` 写入 ``info`` 的内容：
ManiSkill 的 put-on 任务包括 ``is_src_obj_grasped``、
``consecutive_grasp``、``success``、``gripper_carrot_dist``、
``carrot_plate_dist`` 等。若分量名称不在 ``info`` 中，会报错并列出
可用键。

注册完全自定义的奖励函数
------------------------

当加权分量表达能力不足时，可注册自己的函数。函数接收原始奖励、
步进 ``info`` 字典和环境配置节点，返回形状为 ``[num_envs]`` 的
浮点张量：

.. code-block:: python

   # my_pkg/my_rewards.py
   import torch
   from rlinf.envs.rewards import register_env_reward

   @register_env_reward("staged_pick_place")
   def staged_pick_place(*, raw_reward, info, cfg=None):
       reward = 0.3 * info["is_src_obj_grasped"].float()
       reward += 0.5 * (1.0 - torch.tanh(5.0 * info["carrot_plate_dist"]))
       reward += 1.0 * (info["success"] & info["is_src_obj_grasped"]).float()
       return reward

然后在 YAML 中选择它。若模块位于 RLinf 源码树之外，用
``reward_fn_module`` 指向该模块，使装饰器在查找前先执行：

.. code-block:: yaml

   env:
     train:
       reward_mode: staged_pick_place
       reward_fn_module: my_pkg.my_rewards

注意事项
--------

- ``env.<split>.use_rel_reward: True`` 对所有奖励模式生效：环境输出
  的是奖励的每步 *差分* ，因此单调的阶段化奖励会变成每达成一个阶段
  发放一次的奖励。
- 基于组的优势（``algorithm.adv_type: grpo``）在组内对轨迹回报做
  归一化，因此只有轨迹间的相对奖励尺度有意义；使用 ``gae`` 时，
  每步稠密塑形会直接影响价值目标。
- 对于 ManiSkill 之外的环境，可用相同方式让其
  ``_calc_step_reward`` 经过 ``rlinf.envs.rewards.get_env_reward_fn``
  以接入注册表。
