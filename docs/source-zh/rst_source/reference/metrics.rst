训练指标
========

RLinf 通过 :doc:`MetricLogger <../guides/logger>` 在 ``train/``\ 、``rollout/``\ 、``env/`` 和 ``time/`` 命名空间下记录指标。本页按命名空间列出各项定义。

.. tip::

   具身任务可以对照查看 ``env/success_once`` 和 ``env/return_mean``\ 。每个回合成功得 1、失败得 0 时，两项指标都等于回合成功率。``rollout/returns_mean`` 是价值训练目标，不是回合总回报。

训练指标 —— ``train/``
----------------------

策略与价值优化的统计量，每次 actor 更新时记录。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``train/actor/approx_kl``
     - 新旧策略之间的近似 KL 散度。
   * - ``train/actor/clip_fraction``
     - 概率比被裁剪的更新比例。
   * - ``train/actor/dual_clip_fraction``
     - 有效 policy 损失项中，受 Dual-Clip 限制的比例。关闭 Dual-Clip 时为 0；统计时排除 padding。
   * - ``train/actor/clipped_ratio``
     - 被裁剪后概率比的均值。
   * - ``train/actor/grad_norm``
     - actor 的梯度范数。
   * - ``train/actor/lr``
     - 当前学习率。
   * - ``train/actor/policy_loss``
     - PPO / GRPO 策略损失。
   * - ``train/critic/value_loss``
     - 价值函数损失。
   * - ``train/critic/value_clip_ratio``
     - 价值目标更新被裁剪的比例。
   * - ``train/critic/explained_variance``
     - 价值预测的可解释方差（越接近 1 越好）。
   * - ``train/entropy_loss``
     - 策略熵。
   * - ``train/loss``
     - 训练总损失（actor + critic + 熵正则）。

Rollout 指标 —— ``rollout/``
----------------------------

以下统计量通过 loss mask 排除 padding。均值和总体标准差按各 rank 的有效元素数量加权，每个有效元素权重相同。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``rollout/advantages_max``
     - 该批次中优势的最大值。
   * - ``rollout/advantages_mean``
     - 该批次中优势的均值。
   * - ``rollout/advantages_std``
     - 训练所用优势的总体标准差。启用优势归一化时，统计归一化后的优势。
   * - ``rollout/advantages_min``
     - 该批次中优势的最小值。
   * - ``rollout/rewards``
     - 保留的旧指标：存储奖励元素的均值，在 chunk 内求和之前计算。
   * - ``rollout/rewards_mean``\ 、``rollout/rewards_std``
     - 奖励的均值和总体标准差。采用 chunk 级奖励时，先对每个 chunk 内的动作奖励求和。
   * - ``rollout/returns_mean``\ 、``rollout/returns_std``
     - 价值训练目标的均值和总体标准差。GAE 的目标包含折扣奖励和 bootstrap 价值。
   * - ``rollout/returns_min``\ 、``rollout/returns_max``
     - 价值训练目标的最小值和最大值。
   * - ``rollout/values_mean``\ 、``rollout/values_std``
     - 采样到的状态转移中，GAE 所用价值预测的均值和总体标准差。不包含末尾的 bootstrap 价值。
   * - ``rollout/advantages_raw_mean``\ 、``rollout/advantages_raw_std``
     - GAE 目标与价值预测之差的均值和总体标准差，在优势归一化前计算。仅在关闭 return 归一化时记录。
   * - ``rollout/loss_mask_fraction``
     - loss mask 保留的 rollout 元素比例。

环境指标 —— ``env/``
--------------------

来自模拟器的任务级信号。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``env/success_once``
     - **推荐。** 未归一化的回合成功率——最能反映真实任务表现。
   * - ``env/episode_len``
     - 回合中实际经历的环境步数。
   * - ``env/return``
     - 已结束回合的平均总回报，按实际环境奖励求和，不含价值 bootstrap。
   * - ``env/return_mean``\ 、``env/return_std``
     - 回合总回报的均值和总体标准差。均值与 ``env/return`` 相同；各 rank 的已结束回合权重相同。
   * - ``env/reward``
     - step 级奖励（中间步为 ``0``，成功时为 ``1``）。日志值按回合步数归一化，难以直接反映真实表现。

独立评测在 ``eval/`` 命名空间下记录同样的回合指标。回合总回报只取 0 或 1，且成功率为 :math:`p` 时，均值为 :math:`p`，总体标准差为 :math:`\sqrt{p(1-p)}`。

Dual-Clip 配置
--------------

以下选项用于具身 PPO 的 actor 损失。

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 选项
     - 类型
     - 默认值与含义
   * - ``algorithm.use_dual_clip``
     - 布尔值
     - 未设置时使用 actor 的默认值。设为 ``false`` 可关闭 Dual-Clip，Task22 配方采用此设置。
   * - ``algorithm.clip_ratio_c``
     - 浮点数或 null
     - 启用 Dual-Clip 时默认为 3.0。设为大于 1 的有限值可限制 policy 损失；设为 null 则关闭此限制。

如何选择日志后端（TensorBoard、Weights & Biases、SwanLab）以及配置 ``runner.logger``，参见 :doc:`日志 <../guides/logger>` 教程。
