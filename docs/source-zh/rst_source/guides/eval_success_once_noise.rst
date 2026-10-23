理解 ``success_once`` 指标的波动
================================

本指南解释具身评估指标(如 ``eval/env/success_once``)为何会在验证步之间
波动(例如在 RoboCasa 上用 PPO 微调 pi05 时在 0.5 与 0.6 之间振荡),以及
如何区分统计噪声与真实的训练不稳定。

``success_once`` 的计算方式
---------------------------

每个 episode 的 ``success_once`` 由环境封装层维护(参见
``rlinf/envs/sim/robocasa/robocasa_env.py``):它记录该 episode 在终止前
是否**至少成功过一次**。

验证阶段,``rlinf.utils.metric_utils.compute_evaluate_metrics`` 会把所有
评估 rollout epoch 与 rollout stage 的 episode 汇总为对二值结果的**一个
简单均值**,同时通过 ``eval/num_trajectories`` 上报参与统计的 episode 总数。

为什么指标会振荡
----------------

二值结果的均值服从**二项分布**。即使策略与环境种子完全固定,随机策略
(自回归动作采样)的重复评估也会产生不同的 episode,估计值带有采样噪声:

.. math::

   \mathrm{std}(\hat{p}) = \sqrt{\frac{p\,(1-p)}{n}}

其中 :math:`p` 是真实成功率,:math:`n` 是评估 episode 数。当
:math:`p \approx 0.55` 时:

====================================  =========================  ========================
每次评估 episode 数(``n``)          估计值标准差                典型 95% 噪声区间
====================================  =========================  ========================
16(如 4 env × 4 rollout epoch)      ±0.124                     大约 0.30 – 0.80
100                                   ±0.050                     大约 0.45 – 0.65
1000                                  ±0.016                     大约 0.52 – 0.58
====================================  =========================  ========================

因此,在评估批次较小时,``env/success_once`` 在 0.5 与 0.6 之间振荡
**完全落在预期采样噪声范围内**,本身并不代表训练不稳定。约 20 个 PPO 步
内曲线平坦或 noisy,更可能是评估统计功效不足,或策略尚未越过噪声底线的
真实提升。

RoboCasa 的环境种子已是确定性的
(``seed = cfg.seed + global_env_id``,见
``rlinf/envs/sim/robocasa/robocasa_env.py``),场景/复位布局不会引入额外
方差;剩余方差来自 episode 数量少以及策略自身的随机性。

使用内置工具诊断
----------------

RLinf 提供 ``tools/diagnose_success_once_noise.py``:

.. code-block:: bash

   # 单个评估点的噪声区间:16 个 episode 下,相对 0.55 多大波动属于噪声?
   python tools/diagnose_success_once_noise.py single --rate 0.55 --episodes 16

   # 相邻两个评估点(0.50 与 0.60)的差异是否显著?
   python tools/diagnose_success_once_noise.py compare \
       --rate-a 0.50 --episodes-a 16 --rate-b 0.60 --episodes-b 16

   # 要达到 ±0.03 的置信区间需要多少 episode?
   python tools/diagnose_success_once_noise.py plan --target-width 0.06 --rate 0.55

``compare`` 子命令执行两比例 z 检验。只有当它报告差异统计显著时,才应把
``success_once`` 的下降视为训练不稳定的证据。

请在日志中查看 ``eval/num_trajectories``,它给出适用于你本次运行的 ``n``。

推荐的配置调整
--------------

1. **增大评估批次。** 提高 ``env.eval.total_num_envs`` 和/或
   ``env.eval.rollout_epoch``,使其乘积达到 ``plan`` 子命令给出的 episode
   数(p = 0.55 时 ±0.03 区间约需 1000 个 episode;约 100 个 episode 即可
   将噪声区间缩至 ±0.05 左右)。
2. **保持种子固定。** 各验证步之间保持 ``env.eval.seed`` 不变,使步间差异
   归因于策略变化而非场景变化。
3. **谨慎平滑,而非掩盖。** 若对验证步做滑动平均绘图,请注明窗口大小,
   并记住滑动平均不会降低单个点的噪声。
4. **用 z 检验代替肉眼比较。** 对波动做出反应之前,先用 ``compare``
   子命令检验相邻两个评估点。
