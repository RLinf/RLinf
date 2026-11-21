SuperMimic
==========

SuperMimic 让想象出来的回报沿着一个可微的世界模型反向传播，用来训练人形机器人的全身动作跟踪 policy。RLinf 在 MuJoCo Warp 上为 Unitree G1 实现了论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*\ （Liao 等，arXiv:2610.28480，`项目主页 <https://supermimic.github.io>`_）。状态表示和优化器跟随 `HybridRobotics/supermimic <https://github.com/HybridRobotics/supermimic>`_\ （Apache-2.0）。本页讲算法本身，以及和论文不同的地方。安装、数据和命令见 :doc:`../../examples/embodied/supermimic`。

训练循环
--------

每次更新分三步：

1. **采集。** Policy 在仿真里执行动作，训练噪声是 :math:`\mathcal{N}(0, 0.1^2)`，replay 存实际执行的带噪动作。回合从按回报采样的参考帧开始。
2. **拟合世界模型**，用 replay 里的 ``k = 8`` 步窗口。一次更新有 10 个批次，每批 2048 个窗口（全部 GPU 的总和）。
3. **改进 policy。** 在世界模型里想象 ``H = 32`` 步并反向传播，然后用想象轨迹上的 GAE 目标拟合 critic，5 个 epoch、每轮 4 个小批。

``algorithm.replay_ratio = 16`` 是世界模型平均抽到每条转移的次数。和上面的批量、4096 个环境一起，每个控制步对应 0.4 次全局更新，所以使用指南里 6000 次更新的训练约采集 15000 个控制步。

一次更新期间 replay 不变。因此每次更新在训练前就抽好 10 个世界模型批次和想象的起点，合法的 ``k`` 步窗口起点表只建一次，10 个批次共用。窗口在最后一步之前不会跨过回合结束。抽样时先在有合法窗口的环境里均匀抽环境，再在这个环境里均匀抽起点。

世界模型
--------

两种表示由 ``actor.model.representation`` 选择，checkpoint 不能交叉加载。

* ``gencoord`` 存 71 维仿真状态。编码是 68 维 ``[z, gravity_b, v_b, w_b, q, qd]``，网络预测 35 维速度增量 ``[dv_b, dw_b, dqd]``。身体位姿由 G1 运动学树从预测状态算出来。
* ``maxcoord`` 存 14 个跟踪部位的世界位姿和速度（182 维）。编码是骨盆系下各身体的位置、6 维旋转和速度，再加骨盆高度和重力（214 维）。网络预测每个身体的 ``[dv, dw]``\ （84 维）。关节角由逆运动学解回。

位置积分用 :math:`(D+1)/(2D)` 倍的增量，:math:`D = 4` 是一个控制步里的物理子步；速度用整个增量。窗口从记录的状态出发，按记录的动作自回归展开。状态损失按通道分组，用 Bryson 规则加权：每组的均方误差除以该组的名义误差平方，再按通道数归一。名义误差写在 ``rlinf/algorithms/supermimic/objectives.py`` 的 ``NOMINAL_ERRORS`` 里。这不是论文里不加权的 :math:`\|\hat s - s\|^2`。

奖励由预测状态的可微正运动学算出。锚点是骨盆：根项比较骨盆，身体位姿项把参考重锚到骨盆的水平位置和偏航。跟踪核是根位置、根朝向、身体位置、身体朝向、身体线速度和身体角速度，权重之和为 5，再乘 :math:`\Delta t = 0.02`，所以单步奖励上限是 0.1。另有动作变化惩罚和关节限位惩罚。

失败判定看骨盆：高度误差绝对值超过 0.25 m，或倾斜误差绝对值超过 0.8，记为失败。倾斜误差是重力方向在机器人与参考的骨盆坐标系下的 z 分量之差。10 秒的训练时长上限和参考片段结束都会结束回合，但不算失败。

Actor 与 critic
---------------

Policy 的输入是当前状态编码和 640 维 actor Goal，不是多帧本体感觉历史。Goal 从当前参考帧起，每隔 5 个控制步取一帧，共 10 帧，超出片段末尾就停在最后一帧。内容是各帧的关节角和关节速度，再加上各帧根朝向相对当前根的 6 维旋转。世界模型的 Goal 再拼上相对根位置，为 670 维。编码和 Goal 用运行中的均值和方差归一化。

想象回报用软存活权重 :math:`c_0 = 1`、:math:`c_{t+1} = c_t (1 - \hat d_{t+1})`。梯度经过奖励、经过终止概率、经过末端价值。这一步里世界模型和 critic 冻结。三个网络都是 ``Linear -> LayerNorm -> GELU`` 堆叠再接一个线性层。Policy 的编码器和解码器隐藏宽度是 2048、1024、512、256。动力学网络是 2048、2048、2048。Critic 是 256、128。潜变量 32 维。

Critic 拟合同一条想象轨迹上从后往前算出的 GAE 回报，:math:`\gamma = 0.99`、:math:`\lambda = 0.95`。拟合时目标保持不变，价值裁剪到能达到的最大折扣回报以内。

优化器
------

每个阶段有两个优化器。隐藏层矩阵用 Muon，``adjust_lr_fn`` 为 ``match_rms_adamw``，权重衰减 0.1，动量 0.95。其余参数（首尾层、归一化层和偏置）用 fused Adam。Actor 的 Adam 用 :math:`\beta = (0.9, 0.95)`，世界模型和 critic 用 :math:`(0.9, 0.999)`。学习率 :math:`2\times 10^{-4}`，常数调度。

梯度先按全局范数裁到 1.0。Actor 再把每个元素夹到 0.1。范数留在 GPU 上，这一步不读回主机。出现非有限梯度时不跳过；一批更新结束、指标拷回主机时如果出现非有限值，训练停止。``actor.cuda_graph`` 默认关闭。打开后，前三次更新按普通方式执行，第四次录成一张 CUDA 图，之后每次更新都回放它。官方代码把世界模型更新和 actor 加 critic 的更新分别录成两张图，RLinf 把三个阶段录进一张。回放只影响速度，不改变上面的更新公式。``actor.fsdp_config.amp_autocast.enabled`` 默认也关闭；打开后网络矩阵乘用 bf16，权重、损失和优化器状态仍是 float32。

并行训练
--------

每张 GPU 上各有一个 env、rollout 和 actor worker，env rank :math:`r` 只和 rollout、actor rank :math:`r` 交换数据。环境数和两种批量都是全局值，平均分到各 rank。每个 actor rank 用自己这组环境的 replay，从中抽取每个批次里属于自己的那一份。

Actor worker 组成一个 FSDP2 组，动力学网络、policy 和 critic 是三个 FSDP 单元（``module_classes_to_wrap: [Dynamics, MotionPolicy, Critic]``）。每个阶段开始前，所有单元先 reshard，再把本阶段的参数设为可训练。阶段内每次前向把它要读的网络 all-gather 一次，8 步世界模型窗口或 32 步想象都在收集好的权重上运行，不必每步收集：

* 世界模型阶段收集动力学网络；
* actor 阶段收集三个网络，其中动力学网络和 critic 冻结；
* critic 阶段只收集 critic。

梯度由 FSDP 做 reduce-scatter。裁剪前的范数由 RLinf 的 ``get_grad_norm`` 在本阶段的分片参数上计算。重置箱统计和输入归一化的矩在 env rank 之间合并。

按回报重置
----------

动作时间轴切成 0.2 秒一箱（50 Hz 下 10 帧）。每箱维护从该箱出发的折扣回报。重置分布是三项混合：50% 在箱子上均匀抽，40% 按 :math:`\mathrm{softmax}(-25\, G_b / G_{max})` 抽，10% 抽向这个 softmax 的平方所累积的分布。:math:`G_{max} = 0.1 / (1 - \gamma) = 10`。论文写的是 1 秒一箱。分布每 20 个控制步刷新一次；在刷新那一步结束的回合，用刷新前的分布抽起点，和官方代码一致。回报按时间轴的全部帧分箱，起点却在前 ``frames - 2`` 帧里按所选箱的份额抽取，所以抽到的帧可能落在相邻箱。官方采样器的帧映射相同。

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
     - 官方是 Muon 加 fused Adam，世界模型更新和 actor 加 critic 的更新分别录成两张 CUDA 图
     - Muon 加 fused Adam，范数留在 GPU 上。默认关闭；``actor.cuda_graph=True`` 时每次更新回放一张 CUDA 图
   * - 仿真评分
     - 官方环境用最后一个物理子步之前的身体位姿算奖励和失败
     - 最后一个子步之后先刷新身体位姿，再算奖励和失败
   * - Critic 小批
     - 官方先把想象出的行截成 4 的整数倍再打乱，末尾几行永远拟合不到
     - 先打乱全部行再截断。只有行数不是 ``critic_minibatches`` 的整数倍时才有差别
   * - 并行
     - 官方是单 GPU 上的单进程
     - 每张 GPU 一组 env、rollout、FSDP2 actor，三个网络各是一个 FSDP 单元。批量是全局值，GPU 数不改变更新
   * - 已报告的数据
     - 论文主结果是过滤后的 BONES-SEED；官方默认下载一条 wandb 片段
     - 本地的 40 条 LAFAN1

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

速度跟踪没有实现。站立推力恢复用的是 mjlab 的屈膝站立姿态，不是论文 §4.3 的 T-pose。BONES-SEED 转换器还没有拿官方 CSV 核对过。多节点训练以及换 GPU 数续训都还没有验证。

致谢
----

感谢 `CuboiLeo <https://github.com/CuboiLeo>`__\ （`SuperMimic-Reproduced <https://github.com/CuboiLeo/SuperMimic-Reproduced>`__）和 `zita-ch <https://github.com/zita-ch>`__\ （`supermimic_vibed <https://github.com/zita-ch/supermimic_vibed>`__）公开各自的社区复现，本集成参考过它们。设置以上文所述的官方代码和论文为准。
