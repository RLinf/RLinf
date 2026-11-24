训练指标
========

RLinf 通过 :doc:`MetricLogger <../guides/logger>` 在五个核心命名空间下记录标量指标：``train/``、``rollout/``、``env/``、``eval/`` 与 ``time/``。本页统一给出各指标的含义和排障关注点，示例页面直接链接到此页。

.. tip::

   具身任务优先关注 **``env/success_once``**\ （训练）与 **``eval/success_once``**\ （验证）。稀疏奖励下，步级 ``env/reward`` 会被回合长度稀释，不如回合成功率直观。

``train/``
----------

每次 actor 更新时记录的 policy 与价值函数优化统计量（on-policy PPO / GRPO）。

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - 指标
     - 含义
     - 排障关注点
   * - ``train/actor/approx_kl``
     - 新旧 policy 之间的近似 KL 散度。
     - 超过 ``0.05``–``0.1`` 说明单步更新过大，应调小学习率或裁剪阈值。
   * - ``train/actor/clip_fraction``
     - 概率比触发裁剪的样本比例。
     - 持续高于 ``0.3`` 说明每轮 policy 变化过快。
   * - ``train/actor/clipped_ratio``
     - 裁剪后概率比的均值。
     - 正常训练应保持在 ``1.0`` 附近。
   * - ``train/actor/grad_norm``
     - 裁剪前 actor 梯度范数。
     - 频繁尖峰或 ``NaN`` 提示数值不稳定或奖励尺度异常。
   * - ``train/actor/lr``
     - 当前 actor 学习率。
     - 核对 warmup 与衰减曲线是否符合配置。
   * - ``train/actor/policy_loss``
     - PPO / GRPO policy 梯度损失。
     - 数值本身可能震荡，应结合奖励与 ``env/success_once`` 判断收敛。
   * - ``train/critic/value_loss``
     - 价值函数的 Huber 损失。
     - 训练前期应稳步下降；持续居高或发散会拖累优势估计。
   * - ``train/critic/value_clip_ratio``
     - 价值更新触发裁剪的比例。
     - 过高说明 ``value_clip`` 过紧或 critic 学习率偏大。
   * - ``train/critic/explained_variance``
     - 价值预测对回报方差的解释比例（越接近 ``1.0`` 越好）。
     - 接近或小于 ``0`` 说明 critic 尚未拟合回报。
   * - ``train/entropy_loss``
     - policy 熵（开启熵计算时记录）。
     - 迅速跌向 ``0`` 提示熵坍缩与过早收敛。
   * - ``train/loss``
     - 总损失（actor + critic + 熵正则）。
     - 重点监控是否出现 ``NaN`` / ``Inf``，细节看上方分项。

``rollout/``
------------

优化前在 rollout 阶段收集的奖励与优势统计量。

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - 指标
     - 含义
     - 排障关注点
   * - ``rollout/rewards``
     - rollout chunk 或生成回复的平均奖励。
     - 应随训练上升；长期为 0 说明 policy 未探索到有效奖励。
   * - ``rollout/advantages_mean``
     - 批次优势均值。
     - 开启优势归一化时应接近 ``0.0``。
   * - ``rollout/advantages_max``
     - 批次优势最大值。
     - 未归一化时若出现极大离群值会冲击梯度。
   * - ``rollout/advantages_min``
     - 批次优势最小值。
     - 与 ``advantages_max`` 对照检查奖励分布是否偏斜。
   * - ``rollout/loss_mask_fraction``
     - 批次中有效（未 mask）token 或步数的比例。
     - 接近 ``0`` 提示轨迹被提前截断或格式异常。

``env/``
--------

训练环境 worker 上报的任务级指标。

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - 指标
     - 含义
     - 排障关注点
   * - ``env/success_once``
     - **核心具身指标。** 未归一化的回合成功率。
     - 应随训练稳步提升，并与 ``eval/success_once`` 对照观察泛化。
   * - ``env/episode_len``
     - 回合平均环境步数。
     - 随着任务完成变快，通常会逐步下降。
   * - ``env/return``
     - 回合累计回报。
     - 稀疏奖励下在成功前接近 0。
   * - ``env/reward``
     - 按回合长度平均后的步级奖励。
     - 稀疏奖励任务受回合长度稀释，建议以 ``env/success_once`` 为准。

``eval/``
---------

``runner.val_check_interval`` 触发验证（或独立评测）时记录的指标。

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - 指标
     - 含义
     - 排障关注点
   * - ``eval/success_once``
     - 验证环境或任务划分上的回合成功率。
     - 选 checkpoint 的核心依据；与 ``env/success_once`` 差距过大提示过拟合或探索噪声。
   * - ``eval/return``
     - 验证回合平均回报。
     - 稠密奖励任务中反映去除探索噪声后的真实表现。
   * - ``eval/episode_len``
     - 验证回合平均步数。
     - 成功率提高且步数下降说明动作更利落。
   * - ``eval/num_trajectories``
     - 跨评估 worker 汇总的轨迹总数。
     - 核对是否与配置的 ``eval`` worker 与回合数一致。

``time/``
---------

Runner 与 worker 计时器上报的阶段耗时（单位：秒），用于定位系统瓶颈。

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - 指标
     - 含义
     - 排障关注点
   * - ``time/step``
     - 单个全局训练步的总耗时。
     - 端到端吞吐指标；与分阶段耗时之和对比可评估流水线重叠效果。
   * - ``time/sync_weights`` / ``time/update_rollout_weights``
     - actor 到 rollout worker 的权重同步耗时。
     - 占比过高时检查 placement 策略或权重同步频率。
   * - ``time/generate_rollouts``
     - rollout 推理与环境交互总耗时。
     - 占比主导说明仿真步进或推理吞吐成为瓶颈。
   * - ``time/cal_adv_and_returns``
     - 优势与回报计算耗时。
     - 通常很短；异常偏高提示主机与设备间同步开销。
   * - ``time/actor_training``
     - actor 前向、反向与优化器更新耗时。
     - 占比过高可增加 actor 卡数或调整 ``micro_batch_size``。
   * - ``time/eval``
     - 单次验证耗时。
     - 耗时过长时可调大 ``val_check_interval``。
   * - ``time/env/*``、``time/rollout/*``、``time/actor/*``、``time/reward/*``
     - 按组件汇总的 worker 细粒度耗时（开启 ``runner.per_worker_log: True`` 时同时记录各 rank 耗时）。
     - 用于排查慢节点或 GPU 分配不均。

Off-Policy 与 SAC
-----------------

由 ``EmbodiedSACFSDPPolicy`` 与 ``AsyncEmbodiedSACFSDPPolicy`` 记录（涵盖 SAC、CrossQ 与 RLPD）：

.. list-table::
   :header-rows: 1
   :widths: 30 36 34

   * - 指标
     - 含义
     - 排障关注点
   * - ``train/sac/critic_loss``
     - Q 网络 Bellman 回归损失。
     - 发散说明 Q 值过估计，需检查 ``tau``、``gamma`` 或 LayerNorm/BRN 配置。
   * - ``train/sac/actor_loss``
     - 最大熵 policy 损失。
     - 需与 ``train/sac/critic_loss`` 和 ``env/success_once`` 联合观察。
   * - ``train/sac/alpha`` / ``train/sac/alpha_loss``
     - 自适应熵温度系数及其优化损失。
     - ``alpha`` 跌至 ``0`` 或暴涨时调整 ``target_entropy`` 或 ``initial_alpha``。
   * - ``train/actor/entropy``
     - policy 动作熵。
     - 开启自动调温时应逐渐稳定在 ``-target_entropy`` 附近。
   * - ``train/critic/lr`` / ``train/critic/grad_norm``
     - critic 学习率与梯度范数。
     - 高 UTD 更新时关注梯度是否突增。
   * - ``train/replay_channel_qsize``
     - 异步 actor channel 中等待消费的轨迹队列长度。
     - 队列持续堆积说明 actor 训练速度慢于采样速度。

Async PPO 样本陈旧度
--------------------

在 :doc:`Async PPO <algorithms/async_ppo>` 运行时由 ``compute_decoupled_ppo_actor_loss`` 记录：

.. list-table::
   :header-rows: 1
   :widths: 30 36 34

   * - 指标
     - 含义
     - 排障关注点
   * - ``train/actor/proximal_approx_kl``
     - 当前 policy 与近端锚点 policy 之间的近似 KL。
     - 主更新步长监控项（对应同步 PPO 的 ``approx_kl``）。
   * - ``train/actor/behav_approx_kl``
     - 近端锚点与采样行为 policy 之间的近似 KL。
     - 持续偏高说明 rollout 样本过旧，应调小 ``staleness_threshold``。
   * - ``train/actor/behav_clip_fraction``
     - 被 ``behave_weight_threshold`` 过滤掉的样本比例。
     - 过高说明大量陈旧样本被丢弃、有效吞吐下降。
   * - ``train/actor/dual_clip_fraction``
     - 触发双重裁剪上界 ``clip_ratio_c`` 的比例。
     - 持续偏高说明单步策略偏移过大。
   * - ``train/actor/average_version`` / ``train/actor/current_version``
     - 训练批次的平均行为 policy 版本与当前 actor 版本。
     - 差值 ``current_version - average_version`` 直接量化样本滞后步数。

推理与智能体
------------

由推理与智能体 runner（``reasoning_runner.py``、``agent_runner.py``、``async_agent_runner.py``）记录：

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - 指标
     - 含义
     - 排障关注点
   * - ``actor/training/*`` / ``critic/training/*``
     - 每个 minibatch 的 actor 与 critic 训练指标（损失、KL、裁剪比例、梯度范数）。
     - 判读方式与上方 ``train/actor/*``、``train/critic/*`` 一致。
   * - ``rollout/equal_reward_group_ratio`` / ``rollout/zero_reward_group_ratio``
     - 组内奖励完全相同或全为 0 的 prompt 分组比例。
     - 过高说明 GRPO 缺少有效对比信号，可检查采样温度或题目难度。
   * - ``rollout/data_staleness/mean`` / ``rollout/data_staleness/max``
     - 异步智能体 rollout 中已完成分组的版本滞后均值与最大值。
     - 保持在受控范围内，避免异步训练出现严重的 off-policy 漂移。
   * - ``rollout/buffer/size`` / ``rollout/inflight_groups``
     - 异步缓冲池内待消费分组数与正在采样的分组数。
     - 排查 rollout 到 actor 的反压或采样 worker 空转。
   * - ``time/rollout``、``time/reward``、``time/actor/training``、``time/actor/inference``、``time/critic/training``
     - LLM 生成、奖励评分、参考/旧 logprob 推理与训练各阶段耗时。
     - 用于调整 rollout、inference 与 training worker 的 GPU 配比。

日志后端（TensorBoard、Weights & Biases、SwanLab）选择与 ``runner.logger`` 配置见 :doc:`日志 <../guides/logger>` 指南。
