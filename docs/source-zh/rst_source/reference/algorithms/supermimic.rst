SuperMimic
==========

SuperMimic 将想象轨迹上的回报沿着可微世界模型反向传播，用于训练人形机器人的全身动作跟踪 policy。RLinf 在 MuJoCo Warp 上为 29 自由度 Unitree G1 实现了论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*\ （Liao 等，`arXiv:2610.28480 <https://arxiv.org/abs/2610.28480>`__，`项目主页 <https://supermimic.github.io>`__），状态布局与优化器设置沿用 `HybridRobotics/supermimic <https://github.com/HybridRobotics/supermimic>`__\ （Apache-2.0）。

安装与训练步骤见 :doc:`../../examples/embodied/supermimic`，仿真环境说明见 :doc:`../../examples/embodied/g1_mimic`，独立评测见 :doc:`../../evaluations/guides/supermimic`。

训练循环
--------

每次全局更新在所有 GPU rank 间同步执行三个阶段：

1. **采集。** Policy 在 MuJoCo Warp 中叠加探索噪声 :math:`\epsilon \sim \mathcal{N}(0, 0.1^2)` 执行动作，并将实际执行的带噪动作存入各自 rank 的 replay buffer；回合重置时由按回报重置采样器抽取起始帧。
2. **拟合世界模型。** 从 replay 状态出发，按记录动作自回归展开 ``k = 8`` 步，每次更新训练 10 个批次（全局每批 2048 个窗口）。
3. **优化 policy 与 critic。** 从 replay 起点在冻结世界模型中想象 ``H = 32`` 步，反向传播想象回报更新 policy；随后在同一批想象轨迹上计算 GAE 目标，训练 critic 5 个 epoch、每轮 4 个小批。

默认设置（``algorithm.replay_ratio = 16``、全局 4096 个环境、每次更新 ``10 × 2048`` 个窗口）下，每个控制步触发 0.4 次全局更新（6000 次更新约采集 15000 个控制步）。一次更新期间 replay 不变，因此所有 10 个世界模型批次和想象起点均在更新开始前从所有合法的 ``k`` 步窗口起点中一次性抽完（窗口在最后一步前不跨过回合边界）。

状态表示与动力学积分
--------------------

``actor.model.representation`` 选择 replay 存储、世界模型和 policy 共用的状态布局，checkpoint 只能加载到训练时对应的布局中。

.. list-table::
   :header-rows: 1
   :widths: 15 24 25 20 16

   * - 状态布局
     - 存储状态
     - 编码状态
     - 预测增量
     - 部位 / 关节恢复
   * - ``gencoord``
     - 71 维仿真状态
     - 68 维 ``[z, gravity_b, v_b, w_b, q, qd]``
     - 35 维 ``[dv_b, dw_b, dqd]``
     - 可微 G1 正运动学
   * - ``maxcoord``
     - 182 维 14 个跟踪部位世界位姿与速度
     - 214 维骨盆系下各部位位置、6 维旋转、速度及骨盆高度与重力
     - 84 维每部位 ``[dv, dw]``
     - 链式逆运动学

在单个控制步 :math:`\Delta t = 0.02\text{ s}`\ （包含 :math:`D = 4` 个物理子步）内，给定预测的速度增量 :math:`\Delta v`，速度累加完整增量，位置和朝向按子步平均半隐式权重积分：

.. math::

   v_{t+1} = v_t + \Delta v, \qquad
   p_{t+1} = p_t + \Delta t \left(v_t + \frac{D + 1}{2D} \Delta v\right).

目标函数
--------

世界模型损失（Bryson 加权）
~~~~~~~~~~~~~~~~~~~~~~~~~~~

在自回归 ``k`` 步窗口上，损失函数将预测状态通道 :math:`\hat{x}_c` 与真实状态 :math:`x_c` 按通道组 :math:`c`\ （维度 :math:`d_c`，名义误差 :math:`\sigma_c`）做加权 MSE：

.. math::

   \mathcal{L}_{\text{wm}} = \frac{\sum_c d_c \, \sigma_c^{-2} \, \text{MSE}(\hat{x}_c, x_c)}{\sum_c d_c}.

.. list-table:: 名义误差
   :header-rows: 1
   :widths: 22 14 22 14 28

   * - 通道组
     - :math:`\sigma_c`
     - 通道组
     - :math:`\sigma_c`
     - 单位
   * - ``root_pos``
     - ``0.01``
     - ``body_pos``
     - ``0.05``
     - m
   * - ``root_ori``
     - ``0.02``
     - ``body_rot``
     - ``0.1``
     - rad
   * - ``joint_pos``
     - ``0.1``
     - ``body_vel``
     - ``1.0``
     - rad / (m/s)
   * - ``joint_vel``
     - ``3.14``
     - ``body_avel``
     - ``3.14``
     - rad/s

跟踪奖励与软终止概率
~~~~~~~~~~~~~~~~~~~~

仿真器与想象 rollout 使用同一套单步跟踪奖励，覆盖骨盆根节点和 14 个跟踪部位（身体部位的位置与朝向参考先重锚到机器人骨盆当前的水平位置与偏航角）：

.. math::

   r_t = \Delta t \left( \sum_{i=1}^{6} w_i \exp\!\left(-\frac{e_i}{\sigma_i^2}\right) - 0.1 \|a_t - a_{t-1}\|_2^2 - 2.0 \sum_{j=1}^{29} \delta_{\text{lim}, j} \right),

其中 :math:`\Delta t = 0.02`，:math:`\delta_{\text{lim}, j}` 为关节角超出软限位的偏移量；由于 :math:`\sum w_i = 5`，单步奖励上限为 :math:`r_{\max} = 0.1`，6 个跟踪核参数如下：

.. list-table:: 跟踪核参数
   :header-rows: 1
   :widths: 44 28 28

   * - 跟踪核
     - 权重 :math:`w_i`
     - 标准差 :math:`\sigma_i`
   * - 根位置 / 根朝向
     - 0.5 / 0.5
     - 0.3 / 0.4
   * - 部位位置 / 部位朝向
     - 1.0 / 1.0
     - 0.3 / 0.4
   * - 部位线速度 / 部位角速度
     - 1.0 / 1.0
     - 1.0 / 3.14

仿真中的硬失败条件为 :math:`|z_{\text{pelvis}} - z_{\text{ref}}| > 0.25\text{ m}` 或 :math:`|g_z - g_{z,\text{ref}}| > 0.8`。想象轨迹中将其替换为平滑可微的软终止概率（平滑系数 :math:`\tau = 0.05`）：

.. math::

   \hat{d}_t = \sigma\!\left(\frac{\max\!\left(\frac{|z_{\text{pelvis}} - z_{\text{ref}}|}{0.25},\; \frac{|g_z - g_{z,\text{ref}}|}{0.8}\right) - 1}{\tau}\right).

想象回报与 Critic 目标
~~~~~~~~~~~~~~~~~~~~~~

冻结世界模型和 critic 后，policy 最大化步长 :math:`H = 32`\ （:math:`\gamma = 0.99`）的软存活折扣想象回报：

.. math::

   c_0 = 1, \quad c_{t+1} = c_t (1 - \hat{d}_t), \qquad
   J(\theta) = \mathbb{E}\!\left[ \sum_{t=0}^{H-1} \gamma^t c_t r_t + \gamma^H c_H V_\psi(s_H) \right].

梯度同时穿过 :math:`r_t`、:math:`\hat{d}_t` 与末端估值 :math:`V_\psi(s_H)`。随后 critic 在同一批想象轨迹上反向计算固定 GAE 目标（:math:`\gamma = 0.99`，:math:`\lambda = 0.95`）并拟合，价值预测裁剪在最大折扣回报 :math:`r_{\max} / (1 - \gamma) = 10.0` 以内：

.. math::

   \delta_t = r_t + \gamma (1 - \hat{d}_t) V(s_{t+1}) - V(s_t), \qquad
   A_t = \delta_t + \gamma \lambda (1 - \hat{d}_t) A_{t+1}, \qquad
   \hat{G}_t = A_t + V(s_t).

网络结构与输入
--------------

三个网络均由 ``Linear -> LayerNorm -> GELU`` 模块堆叠接最后一层 ``Linear`` 构成，并对状态和目标输入维护运行期均值方差归一化。

* **目标输入：** 从当前参考帧起每隔 5 个控制步取 1 帧、共 10 帧（超出片段末尾停在末帧）。每帧含关节角、关节速度及相对当前根节点的 6 维根旋转（actor 目标 640 维）；世界模型与 critic 目标额外拼接相对根位置（670 维）。
* **仿真物理上下文（33 维）：** 归一化足底摩擦（1）、躯干质心偏移（3）与 29 个关节编码器偏置（29）。
* **Dynamics（世界模型）：** 输入 ``encoding_dim`` + 29（PD 关节目标）+ 33（上下文），隐层宽度 ``[2048, 2048, 2048]``。
* **MotionPolicy（actor）：** 目标编码器 ``640 -> [2048, 1024, 512, 256] -> 32``\ （``tanh`` 潜变量）；解码器 ``32 + encoding_dim -> [2048, 1024, 512, 256] -> 29``。
* **Critic：** 输入 ``encoding_dim`` + 670 + 33，隐层宽度 ``[256, 128] -> 1``。

优化器与执行
------------

* **双优化器：** 各网络隐藏层线性矩阵使用 Muon（``adjust_lr_fn: match_rms_adamw``、权重衰减 ``0.1``、动量 ``0.95``、Nesterov ``True``）；其余参数（首尾层、归一化层与偏置）使用 fused Adam，actor 的 :math:`\beta = (0.9, 0.95)`，世界模型与 critic 的 :math:`\beta = (0.9, 0.999)`。学习率恒定 :math:`2\times 10^{-4}`。
* **梯度裁剪与异常检查：** 在 GPU 上按全局范数 ``1.0`` 裁剪（不触发主机同步），actor 额外对每个元素裁剪到 ``0.1``。一旦出现非有限梯度，指标拷回主机时训练直接终止。
* **CUDA 图：** 开启 ``actor.cuda_graph``\ （默认 ``True``）后，前 3 次更新按常规方式执行，第 4 次将三个阶段录成单张 CUDA 图并持续回放。
* **bf16 混合精度：** 默认通过 FSDP2 ``mixed_precision.param_dtype: bf16``\ （梯度归约保持 fp32、不转换前向输入）在每阶段 unshard 时将网络权重转为 bf16，主权重、优化器状态、梯度归约、运动学、奖励与损失仍保持 float32；rollout 推理则由 ``rollout.amp_autocast`` 启用 bf16。

并行训练（FSDP2）
-----------------

每张 GPU 上运行一个 env、一个 rollout 和一个 actor worker，env rank :math:`r` 仅与同号 rollout、actor rank :math:`r` 通信。全局环境数与两种批大小均分到各 rank，每个 actor rank 从本地 replay 抽样各自份额。

所有 actor worker 组成一个 FSDP2 组，动力学网络、policy 与 critic 作为三个独立 FSDP 单元（``module_classes_to_wrap: [Dynamics, MotionPolicy, Critic]``）。由于完整模型在 float32 下约 64 MB，默认配置使用 ``sharding_strategy: no_shard`` 和 ``reshard_after_forward: False``，各卡保留完整权重并用 all-reduce 同步梯度；切换到分片策略（如 ``full_shard``）时，unshard 执行 all-gather，反向则用 reduce-scatter 归约梯度。每个阶段开始前只将本阶段参数设为可训练，并仅 unshard 本阶段读取的网络：

* **世界模型阶段：** unshard ``Dynamics``。
* **Actor 阶段：** unshard 全部三个网络，其中 ``Dynamics`` 与 ``Critic`` 冻结。
* **Critic 阶段：** 仅 unshard ``Critic``。

全局梯度范数在 FSDP 参数上计算，重置分箱统计与输入归一化均值和方差在 env rank 间归约同步。

按回报重置采样器
----------------

动作库时间轴按 0.2 秒（50 Hz 下 10 帧）切分为时间分箱，每个分箱 :math:`b` 维护从对应分箱出发的折扣回报估计 :math:`G_b`\ （上限 :math:`G_{\max} = 0.1 / (1 - \gamma) = 10`）。每 20 个控制步按三项混合刷新重置分布：

.. math::

   p(b) = 0.5 \, \mathcal{U}(b) + 0.4 \, \text{softmax}\!\left(-25 \frac{G_b}{G_{\max}}\right)_b + 0.1 \, p_{\text{keep}}(b),

其中 :math:`p_{\text{keep}}` 随时间累积 softmax 平方分布，使历史高难度区段持续保留采样权重。在刷新步结束的回合仍使用刷新前的分布抽样，与官方实现一致。回报统计覆盖片段的全部帧，起始帧只在各片段的前 ``frames - 2`` 帧内按分箱权重抽取。

配置参考
--------

``examples/embodiment/config/supermimic_lafan1_gencoord.yaml`` 中的核心算法与 actor 配置如下：

.. code-block:: yaml

   algorithm:
     adv_type: null
     loss_type: supermimic
     k: 8                       # 世界模型展开窗口长度
     H: 32                      # policy 想象步长
     gamma: 0.99
     lam: 0.95
     replay_ratio: 16           # 每条转移被世界模型平均抽样次数
     wm_batches: 10
     wm_window_batch: 2048      # 全局每批窗口数
     imagine_batch: 4096        # 全局每次更新的想象 rollout 数
     critic_epochs: 5
     critic_minibatches: 4
     replay_length: 1000

   actor:
     cuda_graph: True
     model:
       representation: gencoord  # gencoord 或 maxcoord
     fsdp_config:
       strategy: fsdp2
       sharding_strategy: no_shard
       reshard_after_forward: False
       wrap_policy:
         module_classes_to_wrap: [Dynamics, MotionPolicy, Critic]
       mixed_precision:
         param_dtype: bf16
         reduce_dtype: fp32
         cast_forward_inputs: False
       amp_autocast:
         enabled: False
         precision: bf16

与论文和官方代码的差异
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - 项目
     - 论文或官方代码
     - RLinf 实现
   * - 状态布局
     - 官方代码提供 GenCoord 与 MaxCoord
     - 一致；``supermimic_lafan1_gencoord`` 与 ``supermimic_lafan1_maxcoord`` 只有状态布局不同
   * - 世界模型损失
     - 论文写为不加权平方误差
     - Bryson 通道加权（见上文名义误差表）
   * - 目标窗口
     - 论文说明 10 个未来帧
     - 10 帧，间隔 5 个控制步
   * - 重置分箱
     - 论文为 1 秒分箱
     - 0.2 秒分箱，50% / 40% / 10% 混合分布
   * - 数值精度
     - 论文图 5 为 TF32
     - 默认用 FSDP2 ``mixed_precision.param_dtype`` 开启训练 bf16 矩阵乘（rollout 推理用 ``rollout.amp_autocast``），主权重、优化器状态、运动学与损失保持 float32；将 ``param_dtype`` 设为 ``null`` 并将 ``rollout.amp_autocast.enabled`` 设为 ``False`` 即回退全 TF32
   * - CUDA 图
     - 官方分别录为 2 张图（世界模型；actor + critic）
     - 默认开启（``actor.cuda_graph: True``），将 3 个阶段录为 1 张图
   * - 仿真评分时机
     - 官方环境用最后一个物理子步前的身体位姿计算奖励与失败
     - 最后一个物理子步后先刷新身体位姿，再计算奖励与失败
   * - Critic 小批
     - 官方先将想象样本截断为 4 的整数倍再打乱
     - 先打乱全部样本再截断
   * - 并行方式
     - 官方入口为单进程单卡
     - 多卡 FSDP2，每卡一组 env、rollout 与 actor worker，全局批量一致
   * - 默认数据
     - 论文主结果为过滤后的 BONES-SEED；官方默认下载单条 W&B 片段
     - 本地 40 条 LAFAN1 片段

已知限制
--------

* 尚未实现速度跟踪。
* 站立推力恢复采用 mjlab 屈膝站立姿态，而非论文第 4.3 节的 T-pose。
* 多节点训练与变更 GPU 数量续训尚未验证。

致谢
----

感谢 `CuboiLeo <https://github.com/CuboiLeo>`__\ （`SuperMimic-Reproduced <https://github.com/CuboiLeo/SuperMimic-Reproduced>`__）和 `zita-ch <https://github.com/zita-ch>`__\ （`supermimic_vibed <https://github.com/zita-ch/supermimic_vibed>`__）公开各自的社区复现，本集成参考了相关实现。
