SuperMimic
==========

SuperMimic 让想象出来的回报沿着一个可微的世界模型反向传播，用来训练人形机器人的全身动作跟踪 policy。RLinf 在 MuJoCo Warp 上为 Unitree G1 实现了论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*\ （Liao 等，arXiv:2610.28480，`项目主页 <https://supermimic.github.io>`_）。状态表示和优化器跟随 `HybridRobotics/supermimic <https://github.com/HybridRobotics/supermimic>`_\ （Apache-2.0）。本页讲算法本身，以及和论文不同的地方。安装、数据和命令见 :doc:`../../examples/embodied/supermimic`。

训练循环
--------

每次更新分三步：

1. **采集。** Policy 在仿真里执行动作，训练噪声是 :math:`\mathcal{N}(0, 0.1^2)`，replay 存实际执行的带噪动作。回合从按回报采样的参考帧开始。
2. **拟合世界模型**，用 replay 里的 ``k = 8`` 步窗口。一次更新有 10 个批次，每批 2048 个窗口（全部 GPU 的总和）。
3. **改进 policy。** 在世界模型里想象 ``H = 32`` 步并反向传播，然后用想象轨迹上的 GAE 目标拟合 critic，5 个 epoch、每轮 4 个小批。

``algorithm.replay_ratio = 16`` 是每条转移平均被世界模型抽到的次数。和上面的批量、4096 个环境一起，每个控制步欠 0.4 次全局更新。所以 ``runner.max_epochs=16000`` 是 40000 个控制步。

世界模型
--------

两种表示由 ``actor.model.representation`` 选择，检查点不能交叉加载。

* ``gencoord`` 存 71 维仿真状态。编码是 68 维 ``[z, gravity_b, v_b, w_b, q, qd]``，网络预测 35 维速度增量 ``[dv_b, dw_b, dqd]``。身体位姿由 G1 运动学树从预测状态算出来。
* ``maxcoord`` 存 14 个被跟踪身体的世界位姿和速度（182 维）。编码是骨盆系下各身体的位置、6 维旋转和速度，再加骨盆高度和重力（214 维）。网络预测每个身体的 ``[dv, dw]``\ （84 维）。关节角由逆运动学解回。

位置积分用 :math:`(D+1)/(2D)` 倍的增量，:math:`D = 4` 是一个控制步里的物理子步；速度用整个增量。窗口从记录的状态出发，按记录的动作自回归展开。状态损失按通道分组，用 Bryson 规则加权：每组的均方误差除以该组的名义误差平方，再按通道数归一。名义误差写在 ``rlinf/algorithms/supermimic/objectives.py`` 的 ``NOMINAL_ERRORS`` 里。这不是论文里不加权的 :math:`\|\hat s - s\|^2`。

奖励由预测状态的可微正运动学算出。锚点是骨盆：根项比较骨盆，身体位姿项把参考重锚到骨盆的水平位置和偏航。跟踪核是根位置、根朝向、身体位置、身体朝向、身体线速度和身体角速度，权重之和为 5，再乘 :math:`\Delta t = 0.02`，所以单步奖励上限是 0.1。另有动作变化惩罚和关节限位惩罚。

终止看骨盆：高度误差绝对值超过 0.25 m，或倾斜误差绝对值超过 0.8，记为失败。倾斜误差是重力方向在机器人与参考的骨盆坐标系下的 z 分量之差。10 秒的训练时长上限和参考片段结束都会结束回合，但不算失败。

Actor 与 critic
---------------

Policy 的输入是当前状态编码和 640 维 actor Goal，不是多帧本体感觉历史。Goal 从当前参考帧起，每隔 5 个控制步取一帧，共 10 帧，超出片段末尾就停在最后一帧。内容是各帧的关节角和关节速度，再加上各帧根朝向相对当前根的 6 维旋转。世界模型的 Goal 再拼上相对根位置，为 670 维。编码和 Goal 用运行中的均值和方差归一化。

想象回报用软存活权重 :math:`c_0 = 1`、:math:`c_{t+1} = c_t (1 - \hat d_{t+1})`。梯度经过奖励、经过终止概率、经过末端价值。这一步里世界模型和 critic 冻结。三个网络都是 ``Linear -> LayerNorm -> GELU`` 堆叠再接一个线性层。Policy 的编码器和解码器隐藏宽度是 2048、1024、512、256。动力学网络是 2048、2048、2048。Critic 是 256、128。潜变量 32 维。

Critic 拟合同一条想象轨迹上从后往前算出的 GAE 回报，:math:`\gamma = 0.99`、:math:`\lambda = 0.95`。拟合时目标保持不变，价值被夹在能达到的最大折扣回报以内。

优化器
------

每个阶段有两个优化器。隐藏层矩阵用 Muon，``adjust_lr_fn`` 为 ``match_rms_adamw``，权重衰减 0.1，动量 0.95。其余参数（首尾层、范数和偏置）用 fused Adam。Actor 的 Adam 用 :math:`\beta = (0.9, 0.95)`，世界模型和 critic 用 :math:`(0.9, 0.999)`。学习率 :math:`2\times 10^{-4}`，常数调度。

梯度先按全局范数裁到 1.0。Actor 再把每个元素夹到 0.1。范数留在 GPU 上，这一步不读回 CPU。非有限梯度不会被跳过；一批更新结束、指标拷回主机时如果出现非有限值，训练停止。官方实现把整次更新放进 CUDA graph。当前这条 FSDP2 训练路径还不能重放那次捕获，所以更新仍按普通步骤执行。差别在速度，不改变上面的更新公式。

按回报重置
----------

动作时间轴切成 0.2 秒一箱（50 Hz 下 10 帧）。每箱维护从该箱出发的折扣回报。重置分布是三项混合：50% 在箱子上均匀抽，40% 按 :math:`\mathrm{softmax}(-25\, G_b / G_{max})` 抽，10% 抽向这个 softmax 的平方所累积的分布。:math:`G_{max} = 0.1 / (1 - \gamma) = 10`。论文写的是 1 秒一箱。

和论文、官方代码的差别
----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 项
     - 论文或官方代码
     - 本实现
   * - 状态
     - 官方代码有 GenCoord 和 MaxCoord
     - 同官方。两份 YAML 只差这一项
   * - 世界模型损失
     - 论文是不加权平方
     - 按名义误差做 Bryson 加权
   * - Goal
     - 论文写 10 个未来帧
     - 10 帧，间隔 5 个控制步
   * - 重置箱
     - 论文是 1 秒
     - 0.2 秒，50% / 40% / 10%
   * - 精度
     - 论文图 5 是 TF32
     - 权重 float32，矩阵乘 TF32。bf16 只在打开 ``amp_autocast`` 时用于网络矩阵乘
   * - 优化器
     - 官方是 Muon 加 fused Adam，整次更新在 CUDA graph 里
     - Muon 加 fused Adam，范数留在 GPU 上。这条 FSDP2 路径没有做整次更新捕获
   * - 并行
     - 官方是单进程
     - 每张 GPU 一组 env、rollout、FSDP2 actor
   * - 已报告的数据
     - 论文主结果是过滤后的 BONES-SEED；官方默认下载一条 wandb 片段
     - 本地的 40 条 LAFAN1。评测数字见使用指南

实现入口
--------

训练从 ``examples/embodiment/train_embodied_agent.py`` 启动，评测从 ``evaluations/eval_embodied_agent.py`` 启动。``algorithm.loss_type`` 为 ``supermimic`` 时选用 SuperMimic 的 runner。

* Runner：``rlinf/runners/supermimic_runner.py`` 和 ``rlinf/runners/supermimic_eval_runner.py``。
* Env worker：``rlinf/workers/env/supermimic_env_worker.py``。
* Rollout worker：``rlinf/workers/rollout/hf/supermimic_rollout_worker.py``。
* Actor worker：``rlinf/workers/actor/fsdp_supermimic_policy_worker.py``。
* 网络：``rlinf/models/embodiment/supermimic/``。
* 损失、replay、重置采样：``rlinf/algorithms/supermimic/``。
* 仿真、参考动作、奖励和评测：``rlinf/envs/sim/g1_mimic/``。

已知限制
--------

速度跟踪没有实现。站立推力恢复用的是 mjlab 的屈膝站立姿态，不是论文 §4.3 的 T-pose。BONES-SEED 转换器还没有拿官方 CSV 核对过。多节点和换 GPU 数恢复没有验证。使用指南里的跟踪表是第 2000 次和第 4000 次更新的检查点，训练目标是 16000 次更新。
