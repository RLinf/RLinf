近端策略优化 (PPO)
==================================

1. 引言
---------------

近端策略优化 (PPO) 是目前应用最广泛的强化学习 (RL) 算法之一。  
它包含两个核心部分：  

- Actor（策略模型）：根据当前状态生成动作。  
- Critic（价值模型）：评估所选动作的价值。  

PPO 是一种稳定的策略梯度方法，它在原始策略梯度 (Policy Gradient) 的基础上改进。  
它通过限制策略更新的步长，提高了训练的稳定性和效率。  
此外，PPO 使用广义优势估计 (GAE) 来降低价值估计的方差。  

在 RLHF (Reinforcement Learning from Human Feedback) 的早期阶段，PPO 得到了广泛应用。  
然而，由于需要一个大型 Critic 模型（通常是另一种 LLM），它会带来高昂的计算成本和训练开销。  

更多细节请参考原始论文  
`PPO <https://arxiv.org/abs/1707.06347>`_ 以及它在 RLHF 中的应用  
`InstructGPT <https://arxiv.org/abs/2203.02155>`_。


2. 目标函数
----------------------

设策略为 :math:`\pi_\theta`。  
对于包含问答对 :math:`(q,a)` 的数据集 :math:`\mathcal{D}`，  
PPO 的目标函数定义如下：  

.. math::

   J_{\mathrm{PPO}}(\theta)
   = \mathbb{E}_{(q,a)\sim\mathcal{D},\, o_{\le t}\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
   \Big[
     \min\!\Big(
       r_t(\theta)\,\hat{A}_t,\;
       \mathrm{clip}\,\big(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\,\hat{A}_t
     \Big)
   \Big],

其中：  

- :math:`r_t(\theta) = \dfrac{\pi_\theta(o_t \mid q, o_{<t})}
  {\pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})}`  
  表示重要性采样比率，用来比较新旧策略。  

- :math:`\varepsilon` 是裁剪范围，一个超参数，用于防止更新过大。  

- :math:`\hat{A}_t` 是时间步 :math:`t` 的优势估计。  

使用广义优势估计 (GAE) 时，优势的计算公式为：  

.. math::

   \hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
   = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l},
   \qquad
   \delta_l = R_l + \gamma V(s_{l+1}) - V(s_l),
   \quad 0 \le \gamma, \lambda \le 1.

其中：  

- :math:`\gamma` （折扣因子）和 :math:`\lambda` （GAE 参数）是超参数。  
- :math:`V(s)` 是 Critic 模型给出的价值估计。  


3. 配置
-----------------

我们的框架支持在 LLM 推理任务和具身任务中使用 PPO。

3.1. 具身任务
~~~~~~~~~~~~~~~~~

下面首先给出一个具身任务的示例配置：

.. code-block:: yaml

   algorithm:

      # 核心 PPO 设置（建议不要修改）
      normalize_advantages: True
      group_size: 1
      adv_type: gae
      loss_type: actor_critic
      loss_agg_func: "token-mean"

      # 算法参数（通常需要调优）

      rollout_micro_batch_size: 256
      logprob_forward_micro_batch_size: 16  # 较大的 batch_size 可以提高稳定性。
                                            # 请根据算力和模型大小调整。

      entropy_bonus: 0          # 可选：鼓励探索
      clip_ratio_high: 0.2      # PPO 裁剪参数 (epsilon)
      clip_ratio_low: 0.2       # 应与 clip_ratio_high 保持一致
      value_clip: 0.2           # 稳定价值函数更新

      gamma: 0.99               # GAE 的折扣因子
      gae_lambda: 0.95          # GAE 的 Lambda 参数

      huber_delta: 10.0         # 价值训练中 Huber 损失的 Delta 参数

Rollout 与 Actor 的重要性采样修正
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当 rollout 和 actor 使用不同的推理实现时，即使同步相同权重，二者的动作对数概率也可能不同。同步具身 PPO 可以用 actor 重放已采样的 transition，再应用与 reasoning 训练相同的重要性采样修正：

.. code-block:: yaml

   algorithm:
      importance_sampling_fix: True
      importance_sampling_clip: 1.0
      recompute_logprobs: True

启用 ``algorithm.recompute_logprobs`` 后，actor 会在训练前重算旧 policy 的 log-probability。设 :math:`q` 为 rollout 的行为 policy，:math:`p` 为 actor 重放的 policy；修正时将 advantage 乘以 :math:`\min(p/q,\;c)`，并用 :math:`p` 作为 PPO 的旧 policy 锚点。概率比按 ``logprob_type`` 配置的粒度计算；value 预测和 GAE 仍使用 rollout 返回的 value。

该模式要求使用同步、非流水线的具身训练，``loss_type`` 为 ``actor`` 或 ``actor_critic``。通过 ``actor/importance_sampling_weight``、``actor/importance_sampling_clip_fraction``、``actor/behav_approx_kl`` 和 ``actor/recomputed_logprob_abs_diff`` 监控 rollout 与 actor 的差异。

3.2. LLM 推理任务
~~~~~~~~~~~~~~~~~
LLM 推理任务的配置与具身任务的配置有相似之处。

.. code-block:: yaml

    algorithm:
       # 组数应为1
       group_size: 1

       # 优势函数
       adv_type: gae
       gamma: 1
       gae_lambda: 1
       normalize_advantages: True

       # actor loss 的类型。此处与具身任务不同，
       # 因为 actor 与 critic 不在同一个模型中，不能进行联合 backward
       loss_type: actor
       loss_agg_func: "token-mean"

       # 用于actor loss
       clip_ratio_c: 3.0
       clip_ratio_low: 0.2
       clip_ratio_high: 0.2

       # 用于critic loss
       value_clip: 0.2           # 稳定价值函数更新

另外，在 LLM 推理任务中，我们使用独立的 critic 模型，而不是让 actor 模型和 critic 模型共享 backbone；这意味着配置中除了像 GRPO 一样有 actor 组件，还需要添加一个 critic 组件，并为其设置相应的 placement:

.. code-block:: yaml

    cluster:
      num_nodes: 1
      component_placement:
        # 注意相比于 GRPO 新增了一个 critic 组件
        actor,critic,rollout,reward: all
                
    actor:
      group_name: "ActorGroup"
      training_backend: megatron
      ...
    
    critic:
      use_critic_model: true # 该参数用于指出 critic 是一个完整的模型，而非只有一个 value head
      group_name: "CriticGroup"
      training_backend: megatron
      # critic 的其余部分配置和 GRPO 配置中的 actor 配置非常类似
      ...
       

3.3. 多轮 Agent 任务
~~~~~~~~~~~~~~~~~~~~

Agent 在生成文本和调用工具之间交替执行时，需要沿完整 episode 的 action token 计算 GAE。工具返回的文本保留在 observation 上下文中，actor 和 critic 的 loss mask 排除这些位置。:doc:`Search-R1 PPO 配置 </rst_source/examples/agentic/searchr1/ppo>` 展示了这条训练路径。

训练流程
^^^^^^^^

每个迭代使用固定的 rollout policy，两个模型完成更新后才开始下一个 rollout batch：

1. 使用当前 actor 权重生成整个 batch 的 episode。
2. Critic 预测 old values，计算固定 advantages 和 returns。
3. Critic 使用这些 returns 更新，随后 actor 使用固定 advantages 和 old-policy logprobs 更新。
4. 安装更新后的 actor 权重，再生成下一个 batch。

实现说明
^^^^^^^^

多轮实现注册为 ``algorithm.adv_type: gae_dynamic``。它按照 prompt 长度排列同一 episode 的 turn，将各轮生成的 action 位置连接起来，仅在最后一个 action 应用一次最终答案 reward，再调用已有 GAE 实现，将结果还原到原来的 turn 布局。折扣按生成的 action token 推进。完整 episode 以及因轮数或 token 上限而结束的 episode，未来 bootstrap value 均为零。

Critic worker 在更新前计算目标，并跨数据并行 rank 汇总有效 action token 的统计量来归一化 advantages。Returns 保持原来的尺度。Actor 只接收固定 advantages；old values 和 returns 留在 critic 侧用于 value 训练。

当前 Search-R1 接入使用 Megatron actor/critic TP=PP=CP=1 和 SGLang，actor、critic、rollout 共用相同 GPU。训练前 offload rollout 引擎可以腾出显存，容纳两个训练模型。这是共享 placement 的显存管理选择；同步 PPO 要求模型更新和权重安装在下一个 batch 开始前完成。各角色的 ``offload_weight``、``offload_grad`` 和 ``offload_optimizer`` 控制训练状态的驻留。

4. 注意事项
-----------

- 使用奖励归一化来稳定训练。  
- 监控 KL 散度以检测策略是否更新过度。  
- 对于大型 LLM，增加 batch size 可以减少方差。  
