训练指标
========

RLinf 通过 :doc:`MetricLogger <../guides/logger>` 在若干命名空间下记录指标——``train/``、``rollout/``、
``env/`` 与 ``time/``。本页统一给出它们的定义；示例页面直接链接到此处，而不再重复说明。

.. tip::

   对具身任务而言，最有用的单一指标是 **``env/success_once``** —— 未归一化的回合成功率。在稀疏
   奖励下，其他 ``env/*`` 指标往往难以直接解读（见下文）。

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

rollout 阶段收集的优势与奖励统计量。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``rollout/advantages_max``
     - 该批次中优势的最大值。
   * - ``rollout/advantages_mean``
     - 该批次中优势的均值。
   * - ``rollout/advantages_min``
     - 该批次中优势的最小值。
   * - ``rollout/rewards``
     - 一个 rollout chunk 的奖励。

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
     - 回合总回报。稀疏奖励下在成功结束前几乎为 0，训练过程中参考价值有限。
   * - ``env/reward``
     - step 级奖励（中间步为 ``0``，成功时为 ``1``）。日志值按回合步数归一化，难以直接反映真实表现。

如何选择日志后端（TensorBoard、Weights & Biases、SwanLab）以及配置 ``runner.logger``，参见
:doc:`日志 <../guides/logger>` 教程。
