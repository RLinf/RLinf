Group Sequence Policy Optimization（GSPO）
==========================================

使用 GSPO，以完整回答为单位计算重要性比率和裁剪，训练推理模型。本页说明优化目标、现有推理 runner 的配置方式，以及保持各回答等权所需的批次设置。

优化目标
--------

GSPO 比较当前 policy 与采样 policy 对同一完整回答的概率。先计算有效 token 的平均 log 概率差，再取指数，从而按回答长度归一化比率：

.. math::

   s_i = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
       (\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
       -\log\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t}))\right).

复用 GRPO 优势估计器，在同一 prompt 的回答组内归一化奖励。记回答优势为 :math:`\hat A_i`，最小化的损失为：

.. math::

   L = -\frac{1}{B}\sum_{i=1}^{B}\min\left(
       s_i\hat A_i,
       \operatorname{clip}(s_i,1-\epsilon_{\mathrm{low}},
       1+\epsilon_{\mathrm{high}})\hat A_i\right).

计算比率与归约优势时均排除 padding，各回答权重相同，与长度无关。空回答的 policy loss 和梯度为零，但仍计入批次平均的分母。比率、裁剪比例和近似 KL 指标在非空回答上取平均。

公式对应 `GSPO 论文 <https://arxiv.org/abs/2507.18071>`_ 的式 5–7。本实现采用每条回答共享一个优势值的原始序列目标，与论文中允许逐 token 优势的 GSPO-token 变体不同。

配置
----

从 ``examples/reasoning/config/math/qwen2.5-1.5b-gspo-fsdp.yaml`` 开始配置。该文件继承 GRPO FSDP 示例的数据和组件放置设置，将初始预算缩小为 16 个 prompt、每个 prompt 四条回答、总长度 2,048 token（prompt 最多 512 token）。需要更长推理的任务应增加上下文预算，截断会影响奖励与训练质量。将 ``actor.model.model_path`` 改为本地模型 checkpoint 路径，rollout 和 tokenizer 路径会自动跟随；另设置 ``data.train_data_paths`` 和 ``data.val_data_paths``。推理环境与数据准备参见 :doc:`../../examples/agentic/math_reasoning/reasoning_ppo`。

以下配置选择 GSPO loss，并继续使用组内相对优势：

.. code-block:: yaml

   runner:
     task_type: reasoning
     enable_dynamic_batch_size: False
   algorithm:
     adv_type: grpo
     loss_type: gspo
     loss_agg_func: seq-mean-token-mean
     group_size: 4
     normalize_advantages: False
     use_valid_token_scale: False
     importance_sampling_fix: False
     clip_ratio_low: 0.0003
     clip_ratio_high: 0.0004
     clip_ratio_c: null

上下裁剪值来自论文实验，实际使用时应针对任务调整。``group_size`` 必须大于一。GRPO 优势估计器已完成组内归一化，额外的批次级优势归一化会改变目标。序列平均设置也决定 actor 如何归约可选的 entropy 和参考 KL 项。

修改路径和组件放置后，启动现有推理 runner：

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-gspo-fsdp

此命令依次运行 rollout、组内优势计算与 GSPO actor 更新。示例保留在 actor 上重新计算旧 log 概率的设置。结合任务奖励观察 ``actor/ratio`` 与 ``actor/clip_fraction``，这两个指标在 GSPO 中描述完整回答。

参数调整
--------

* ``algorithm.group_size`` 控制每个 prompt 的回答数，必须大于一；``data.rollout_batch_size`` 统计 prompt 数，而非回答数。
* ``algorithm.n_minibatches`` 控制每次 rollout 的优化器 minibatch 数。runner 由 ``rollout_batch_size * group_size / n_minibatches`` 推导 actor 全局批量，请保证整除。
* ``algorithm.training_batch_size_per_gpu`` 控制实际固定微批大小，runner 会将它复制到 ``actor.micro_batch_size``，因此仅修改后者不会改变推理微批。推导出的全局批量必须能被微批大小与 actor world size 的乘积整除。
* ``algorithm.clip_ratio_low`` 与 ``clip_ratio_high`` 可以不同；某一侧设为 ``null`` 时使用 ``ratio_clip_eps``。值必须有限且非负，下裁剪宽度必须小于一。
* ``algorithm.recompute_logprobs: True`` 使用 actor 重算的旧 log 概率；``False`` 使用 rollout 提供的 log 概率，也会暴露两个引擎之间的数值差异。
* ``algorithm.kl_beta`` 与 ``entropy_bonus`` 保留现有 actor 正则化逻辑，是序列 policy 目标之外的附加项。示例保留 fp32 actor 参数与优化器状态，使用 bf16 前向计算和 rollout，以及 fp32 梯度归约。

当前 reasoning 训练循环不会仅因设置 ``runner.val_check_interval`` 就执行 held-out 评估。请单独评估导出的 checkpoint，不能将训练奖励当作验证集准确率。

支持的设置
----------

使用固定大小的微批，使梯度累积保持序列等权平均。配置检查会拒绝动态微批、按 token 数缩放 loss、逐 token 的重要性采样修正、PPO 双重裁剪以及 log 比率截断。GSPO loss 通过推理 actor 接口接收完整回答；具身 action chunk 不在本次支持范围内。

可选的 entropy 与参考 KL 项使用现有 actor 归约方式，启用时要求各回答的 mask 非空。默认示例将两个系数均设为零。
