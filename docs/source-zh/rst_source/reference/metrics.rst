训练指标
========

本页按训练、rollout 和回合统计列出指标。标准差均指总体标准差。日志后端配置见 :doc:`日志 <../guides/logger>`\ 。

训练指标
--------

``train/`` 命名空间记录 policy 和价值函数的训练指标。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``train/actor/approx_kl``
     - 新旧 policy 之间的近似 KL 散度。
   * - ``train/actor/clip_fraction``
     - 有效元素中受 PPO 裁剪影响的比例。
   * - ``train/actor/dual_clip_fraction``
     - 有效元素中受 Dual-Clip 限制的比例；关闭时为 0。
   * - ``train/actor/clipped_ratio``
     - 裁剪后的概率比均值。
   * - ``train/actor/grad_norm``
     - actor 的梯度范数。
   * - ``train/actor/lr``
     - 学习率。
   * - ``train/actor/policy_loss``
     - policy 损失。
   * - ``train/critic/value_loss``
     - 价值函数损失。
   * - ``train/critic/value_clip_ratio``
     - 价值更新中受裁剪影响的比例。
   * - ``train/critic/explained_variance``
     - 价值预测的可解释方差。
   * - ``train/entropy_loss``
     - policy 熵。
   * - ``train/loss``
     - 训练总损失。

Rollout 指标
------------

``rollout/`` 命名空间统计有效训练样本。GAE return 用作价值函数训练目标，回合总回报见下方的环境指标。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``rollout/advantages_max``
     - 训练优势的最大值。
   * - ``rollout/advantages_mean``
     - 训练优势的均值。
   * - ``rollout/advantages_min``
     - 训练优势的最小值。
   * - ``rollout/advantages_std``
     - 训练优势的标准差。
   * - ``rollout/rewards``
     - 按动作统计的奖励均值。
   * - ``rollout/rewards_mean``\ 、``rollout/rewards_std``
     - 按配置的奖励粒度统计的均值和标准差。
   * - ``rollout/returns_mean``\ 、``rollout/returns_std``
     - 价值函数训练目标的均值和标准差。
   * - ``rollout/returns_min``\ 、``rollout/returns_max``
     - 价值函数训练目标的最小值和最大值。
   * - ``rollout/values_mean``\ 、``rollout/values_std``
     - 采样状态的价值预测均值和标准差。
   * - ``rollout/advantages_raw_mean``\ 、``rollout/advantages_raw_std``
     - 未归一化 GAE 优势的均值和标准差，仅在关闭 return 归一化时提供。
   * - ``rollout/loss_mask_fraction``
     - 参与训练的 rollout 元素比例。

回合指标
--------

环境通过 ``env/`` 命名空间记录已结束回合的统计量。独立评测在 ``eval/`` 命名空间下记录同样的指标。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``env/success_once``
     - 至少一次达到成功条件的回合比例。
   * - ``env/episode_len``
     - 平均回合长度，单位为环境步数。
   * - ``env/return``\ 、``env/return_mean``
     - 回合总回报的均值。
   * - ``env/return_std``
     - 回合总回报的标准差。
   * - ``env/reward``
     - 每个环境步的平均奖励。

Dual-Clip 配置
--------------

以下选项控制具身 PPO 的 Dual-Clip。

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 选项
     - 类型
     - 默认值与含义
   * - ``algorithm.use_dual_clip``
     - 布尔值
     - 未设置时使用 actor 的默认值。``false`` 表示关闭 Dual-Clip。
   * - ``algorithm.clip_ratio_c``
     - 浮点数或 null
     - 启用时默认为 3.0，接受大于 1 的有限值；null 表示关闭此限制。
