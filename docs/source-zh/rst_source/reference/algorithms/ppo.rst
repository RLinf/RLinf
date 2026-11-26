近端策略优化（PPO）
===================

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
~~~~~~~~~~~~~

具身 PPO 默认预设位于 ``examples/embodiment/config/algorithm/ppo.yaml``。完整配方参见 :doc:`ManiSkill <../../examples/embodied/maniskill>` 与 ``examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml``：

.. code-block:: yaml

   algorithm:
      # 核心 PPO 设置
      normalize_advantages: True
      group_size: 1
      adv_type: gae
      loss_type: actor_critic
      loss_agg_func: "token-mean"

      # 算法参数
      kl_beta: 0.0
      entropy_bonus: 0          # 可选：鼓励探索
      clip_ratio_high: 0.2      # PPO 裁剪参数 (epsilon)
      clip_ratio_low: 0.2       # 应与 clip_ratio_high 保持一致
      clip_ratio_c: 3.0         # 双重裁剪上界
      value_clip: 0.2           # 稳定价值函数更新

      gamma: 0.99               # GAE 折扣因子
      gae_lambda: 0.95          # GAE Lambda 参数
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

按 KL 收紧概率比的裁剪范围
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Actor 在每个动作块选中的去噪转移上，计算从 rollout policy 到当前 policy 的条件 KL。

KL 超过目标的动作块才收紧 ratio clip 的范围。上下界各自与 1 的距离都乘以“目标 KL 除以实测 KL”的平方根，最少保留原距离的四分之一。例如，目标 KL 为 0.05 时，KL 为 0.20 的动作块会把两个距离都缩短一半。

这个选项适用于同步 FSDP OpenPI PPO 或 GRPO，rollout 使用 Hugging Face。奖励和 log-prob 都按动作块计算。使用噪声强度大于 0 的 Flow-SDE，关闭 joint log-prob，并让 rollout 和 actor 的去噪步数一致。``algorithm.recompute_logprobs``\ 保持关闭，ratio 以 rollout 返回的 log-prob 为基准。

在训练 YAML 中设置有限的正数作为 KL 目标。省略该项或设为 ``null``\ 即可关闭。

.. code:: yaml

  algorithm:
    kl_clip_target: 0.05

``train/actor/conditional_kl_mean``\ 记录有效动作块的平均 KL，``train/actor/kl_clip/tight_frac``\ 记录有效动作块中裁剪范围收紧的比例。``train/actor/kl_clip/extra_frac``\ 记录有效动作块中因范围收紧才进入 PPO 裁剪分支的比例。两个比例都按整轮更新的有效动作块统计。实际更新预算仍由训练配置决定。

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

模型和数据准备参见 :doc:`Search-R1 训练 </rst_source/examples/agentic/searchr1/index>`。使用 ``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml``，关键配置如下：

.. code-block:: yaml

   runner:
     synchronous: true

   algorithm:
     group_size: 1
     adv_type: gae_dynamic
     gamma: 1.0
     gae_lambda: 1.0
     normalize_advantages: true
     use_valid_token_scale: true
     kl_beta: 0.0
     loss_scales: []
     recompute_logprobs: false

   critic:
     use_critic_model: true

   rollout:
     return_logprobs: true

通过 Search-R1 训练脚本启动 PPO：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo

4. 注意事项
-----------

- 使用奖励归一化来稳定训练。  
- 监控 KL 散度以检测策略是否更新过度。  
- 对于大型 LLM，增加 batch size 可以减少方差。  
