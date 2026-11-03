DRPO
====

使用Decoupled Reward Policy Optimization（DRPO），分别利用答案正确性和回答长度训练推理模型。本页说明目标函数、支持的 FSDP 配置，以及如何检查准确率与回答长度之间的权衡。

目标函数
--------

为每道题采样一组回答，用二元验证器判断正确性。较短的正确回答获得更大的正向权重；错误回答根据当前生成概率获得负向权重。长度只影响正确回答之间的相对权重。

设 :math:`s_i` 为回答有效 token 的当前平均对数概率，:math:`L_i` 为回答长度，:math:`C` 为配置的回答容量（``runner.seq_length - data.max_prompt_length``）。对每道题，用 :math:`P` 和 :math:`N` 表示正确与错误回答集合：

.. math::

   w_i = \frac{\exp(-L_i/(C\lambda))}{\sum_{j\in P}\exp(-L_j/(C\lambda))},
   \qquad
   \ell_q = -\sum_{i\in P} w_i s_i
      + \tau\log\left(\frac{1}{|N|}\sum_{i\in N}\exp(s_i/\tau)\right).

按照 `作者实现 <https://github.com/Optimization-AI/DRPO>`_，没有同时包含两类回答的组贡献为零，但平均分母仍包含全部题组。空回答不计入任一类别；padding 不参与回答分数或 KL 的计算。

旧 policy KL 估计 :math:`K` 在同步微批的全部 actor rank 间按有效 token 加权。损失加入 :math:`\frac{\beta}{2}[K-\delta]_+^2`，其梯度与官方代码停止梯度的 :math:`\beta[K-\delta]_+` 系数一致；记录的损失是真实罚函数目标，而非官方代码用于产生梯度的替代表达式。梯度累积平均各微批目标，包括各自的 KL 罚项。因此，即使全局 batch size 相同，改变微批大小仍可能改变约束估计。

配置与运行
----------

先按 :doc:`../../examples/agentic/math_reasoning/reasoning` 准备推理环境、模型和数据，再从 ``examples/reasoning/config/math/qwen2.5-1.5b-drpo-fsdp.yaml`` 开始。设置示例中的模型、tokenizer 及训练/验证数据路径后运行：

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-drpo-fsdp

共用推理入口完成回答采样、答案验证，再通过 ``algorithm.loss_type: drpo`` 更新 actor。示例使用 SGLang、共置 FSDP actor，以及输出 0/1 的数学验证器。``raw`` advantage 设置用于满足共用 runner 接口；DRPO 直接根据奖励与当前对数概率计算目标。

保持每道题的回答组完整：``training_batch_size_per_gpu`` 必须为 ``group_size`` 的整数倍，各 rank 的 rollout batch 必须能均匀划分为配置的 mini-batch 和 micro-batch。将 ``shuffle_rollout``、``normalize_advantages``、``runner.enable_dynamic_batch_size``、``actor.enable_dp_load_balance`` 和 ``actor.model.variable_seq_lengths`` 设为 false。数据加载器仍可在训练 batch 之间打乱题目。本实现支持共置推理 FSDP 训练；这些检查防止回答级打乱或负载均衡悄悄改变分组。

.. list-table:: DRPO 参数
   :header-rows: 1
   :widths: 35 15 50

   * - ``algorithm`` 下的字段
     - 默认值
     - 含义
   * - ``drpo_lambda``
     - 0.1
     - 正的长度权重温度；越小越偏向较短的正确回答。较大的有限值接近均匀正向权重（DisCO）。
   * - ``drpo_tau``
     - 10.0
     - 正的负样本分数温度；越小越强调当前生成概率较高的错误回答。
   * - ``drpo_beta``
     - 1000.0
     - 非负的旧 policy KL 罚项系数；零表示禁用此罚项。
   * - ``drpo_delta``
     - 0.0001
     - 非负的 KL 阈值，超过后罚项生效。
   * - ``drpo_kl_type``
     - ``kl``
     - ``kl`` 使用旧概率减当前概率的对数均值，与官方训练脚本一致；``low_var_kl`` 使用 ``exp(current-old)-1-(current-old)``。采样 ``kl`` 估计可能为负。

检查训练
--------

同时观察奖励、回答长度、``actor/ppo_kl``、``actor/drpo_constraint`` 和 ``actor/drpo_mixed_group_fraction``。如果没有题组同时包含正确和错误回答，判别目标项为零。先确认模型、题目和采样设置能产生混合题组，再判断零梯度是否来自实现错误。

使用相同的模型初始化、题目、解码配置和 seed，对比长度感知训练与较大 ``drpo_lambda`` 的对照组，同时评估留出集准确率和平均生成 token 数，并报告耗时及生成 token 总量。回答缩短本身不能证明质量与成本的权衡更好。示例是可运行的集成配置，不代表复现论文的 8k 上下文训练或其报告的收益。

参考：`DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization <https://arxiv.org/abs/2510.04474>`_，公式 (7)。
