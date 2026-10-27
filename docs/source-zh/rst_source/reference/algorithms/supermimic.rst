SuperMimic
==========

SuperMimic 训练人形机器人全身动作跟踪 policy 的方式，是让想象出来的回报沿着一个可微的学习型世界模型反向传播，而不是像 PPO 那样给似然比加权。RLinf 在 MuJoCo Warp 上为 Unitree G1 实现了论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*\ （Liao 等，arXiv:2610.28480，`项目主页 <https://supermimic.github.io>`_）中的方法。本页讲算法本身，以及实现和论文不同的地方；安装、数据、训练和评测命令见 :doc:`../../examples/embodied/supermimic`。

代码分三处：``rlinf/algorithms/supermimic/``\ （目标函数、网络、replay、按回报重置采样），``rlinf/envs/sim/g1_mimic/``\ （参考动作、运动学、奖励、观测、MuJoCo Warp 适配），``rlinf/workers/supermimic/``\ （训练循环、评测、检查点）。

训练循环
--------

算法 A.1 的每次迭代分三步，每张 GPU 上一个进程完成全部工作：

1. **采集。** Policy 在仿真里执行带高斯噪声 :math:`\epsilon_t \sim \mathcal{N}(0, 0.1^2)` 的动作，replay 存的是实际执行的带噪动作。每个 episode 从按回报采样的重置点开始。
2. **拟合世界模型**，用从 replay 里采的 ``k = 8`` 步窗口。
3. **改进 policy**：在冻结的世界模型里想象 ``H = 32`` 步做反向传播，然后用想象轨迹上的 GAE 目标拟合 critic。

``algorithm.updates_per_env_step = 0.4`` 决定更新与数据之比：每 2.5 个向量步做一次更新（一次 10 个批次的世界模型拟合、一次 actor 更新、5 个 epoch 各 4 个小批的 critic 更新）。

世界模型
--------

动力学网络的输入是 71 维状态 :math:`s` 的朝向无关特征（根高度、去掉偏航后的根朝向 Rot6D、朝向坐标系下的线速度、根坐标系下的角速度、关节角和关节速度）、实际执行的动作 :math:`a`，以及每个 episode 的仿真上下文 :math:`\xi`\ （脚底摩擦、关节零位偏差、躯干质心偏移），输出 70 维残差：

.. math::

   \hat s_{t+1} = \hat s_t \oplus f_\omega(\phi(\hat s_t), a_t, \xi)

其中 :math:`\oplus` 在朝向坐标系里叠加位置和速度残差，旋转残差按旋转向量复合。最后一层初始化为零，所以初始模型预测"状态不变"。训练时窗口从记录的状态出发，按记录的动作自回归展开（Eq.1）：

.. math::

   \mathcal{L}_{WM}(\omega) = \mathbb{E}_{\mathcal{B}} \sum_{j=1}^{k}
   \Big[ \| \hat s_j - s_j \|^2_{\Sigma} + \mathrm{BCE}(\hat d_j, d_j) \Big]

状态误差按分块缩放，旋转用相对旋转 :math:`\log(R_{pred} R_{target}^\top)`。奖励头和观测头是解析的：奖励由预测状态经 G1 运动学树的可微正运动学算出，观测由状态、上一动作和 :math:`\xi` 算出。终止头是一个学习型 MLP，输入是相对参考的高度误差、倾斜误差及其绝对值；BCE 标签只有跟踪失败才是 1。

Actor 目标
----------

Policy :math:`a_t = \pi_\theta(o_{t-9..t} + \eta_t, g_t)` 以带噪声的观测历史和 Goal 为输入。用软存活权重 :math:`c_0 = 1`、:math:`c_{t+1} = c_t (1 - \hat d_{t+1})`，想象目标为（Eq.2）

.. math::

   \hat J(\theta) = \mathbb{E}\Big[ \sum_{t=0}^{H-1} \gamma^t \hat c_t \hat r_{t+1}
   + \gamma^H \hat c_H V_\psi(\hat s_H, g_H, \xi) \Big]

梯度通过时间反向传播求得，有三条路径：经过奖励、经过 :math:`\hat c_t`\ （终止头可微）、经过 bootstrap 价值。这一步里世界模型和 critic 都冻结。梯度先按全局范数裁剪到 1.0，再逐元素裁剪到 0.1。

Critic 与 GAE
-------------

Critic :math:`V_\psi(s, g, \xi)`\ （另加一个只给 critic 用的 4 维根误差）拟合在同一条想象轨迹上从后往前算出的 GAE 回报（Eq.3），起点 :math:`\hat R_H = V_H`：

.. math::

   \hat R_t = \hat r_{t+1} + \gamma (1 - \hat d_{t+1})
   \big[ (1 - \lambda) V_{t+1} + \lambda \hat R_{t+1} \big],
   \qquad \mathcal{L}_V = \mathbb{E} \sum_{t=0}^{H-1} (V_t - \hat R_t)^2

其中 :math:`\gamma = 0.99`、:math:`\lambda = 0.95`。拟合 critic 时目标保持不变。

终止、超时与片段结束
--------------------

动作任务在骨盆高度误差超过 0.25 m，或倾斜误差（机器人骨盆系与参考骨盆系下重力方向 z 分量之差）超过 0.8 时判为失败。只有失败才是世界模型的正样本，也只有失败算评测失败。10 s 的训练时长上限和参考片段结束都会结束 episode，但不算失败；想象过程不处理这两种情况，片段结束后保持最后一帧参考。重置采样用的回报在超时处用动作 critic 做 bootstrap，失败和片段结束不做 bootstrap。

时间对齐
--------

在机器人时间 :math:`t` 构造的所有条件量都从参考帧 :math:`m_t` 读起，转移 :math:`s_t \to s_{t+1}` 用 :math:`m_{t+1}` 打分：

* 680 维 Goal :math:`g_t` 包含 :math:`m_t .. m_{t+9}`；
* 9 维锚点、critic 根误差、99 维观测里的相对朝向，比较的都是 :math:`s_t` 和 :math:`m_t`；
* 奖励 :math:`r_{t+1}` 和终止 :math:`d_{t+1}` 比较 :math:`s_{t+1}` 和 :math:`m_{t+1}`；
* 想象的第 :math:`t` 步用 :math:`\tau_0 + t` 时刻的 Goal，用 :math:`m_{\tau_0 + t + 1}` 给 :math:`\hat s_{t+1}` 打分。

论文把 Goal 描述为"10 个未来帧"，CuboiLeo 的实现用 :math:`m_{t+1} .. m_{t+10}`，锚点和 critic 误差也对 :math:`m_{t+1}`。680 维窗口从当前帧开始是本实现的选择：两种约定各自自洽，只是 policy 少看一帧未来。730 维 Goal 沿用 zita-ch 的 :math:`m_{t+1} .. m_{t+10}`。检查点在 ``checkpoint_meta.json`` 里记录窗口，加载到窗口不同的配置时会被拒绝。

奖励
----

动作奖励是 BeyondMimic 的跟踪奖励，去掉自碰撞惩罚，关节限位权重降到 -2，全部乘以 :math:`\Delta t = 0.02` s（表 A.2）：14 个跟踪身体的位置、朝向、线速度、角速度，根位置和根朝向，动作平滑，关节限位。和 BeyondMimic 一样，根项在世界系里比较 ``torso_link``；身体位姿项先把参考平移、旋转到机器人躯干的水平位置和偏航上（``update_relative_body_poses``），所以整体漂移只体现在根项里。身体速度在世界系里比较。

按回报重置采样
--------------

附录 D.1：动作时间轴切成 1 s 的分箱，每个分箱维护从该箱出发的折扣回报的 EMA :math:`G_b`；重置时 50% 在分箱上均匀抽，40% 按 :math:`\mathrm{softmax}_b(-25\, G_b / G_{max})` 抽，10% 按这个 softmax 的慢速 EMA 抽。:math:`G_{max} = 0.1 / (1 - \gamma) = 10` 是动作任务能拿到的最大回报。温度 25、EMA 系数 0.05 和 0.01 是社区取值，论文没有给出。只有动作跟踪 episode 会更新 :math:`G_b`；速度 episode 的回报取决于它的命令而不是参考点，所以速度环境在分箱上均匀重置。

多目标接口与速度跟踪
--------------------

附录 B 让多个目标接口共用一个世界模型和一个解码器 :math:`a = \mathrm{dec}_\theta(\mathrm{enc}_{\theta,i}(g), o)`。速度接口（附录 G 的 RL 编码器）把命令 :math:`(v_x, v_y, \omega_z)` 和观测历史映射到 32 维潜变量。一半环境、每个想象批次的一半用这个接口，使用表 A.6 的奖励、软跌倒终止和自己的 critic。速度更新时解码器权重冻结；在用参考命令的 rollout 上，一个权重 0.1 的 MSE 把速度潜变量拉向（停止梯度的）动作潜变量。附录 G 的蒸馏版编码器没有实现。

论文、社区实现与本实现的选择
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 18 20

   * - 项
     - 论文
     - CuboiLeo
     - zita-ch
     - 本实现
   * - 状态与残差
     - 未规定
     - 71 维状态，缩放残差，显式积分
     - 71 维，坐标系不同
     - 71 维状态，未缩放的朝向系残差
   * - 世界模型状态误差
     - :math:`\|\hat s - s\|^2`
     - 分块缩放
     - --
     - 沿用 CuboiLeo 的缩放，旋转用 SO(3) 误差
   * - 680 维 Goal 窗口
     - 10 个未来帧
     - :math:`m_{t+1}..m_{t+10}`
     - --
     - :math:`m_t..m_{t+9}`
   * - 730 维 Goal
     - --
     - --
     - :math:`m_{t+1}..m_{t+10}`
     - 同 zita-ch
   * - 9 维锚点的身体
     - --
     - torso
     - --
     - torso
   * - 身体重锚
     - BeyondMimic
     - torso
     - pelvis
     - torso
   * - 终止头
     - 学习或解析均可
     - 学习
     - 解析 sigmoid
     - 学习；``termination_head: analytic`` 为实验选项
   * - 输入归一化
     - 未规定
     - 固定尺度
     - 滑动归一化
     - 无
   * - 参考速度
     - 未规定
     - 中心差分
     - 中心差分
     - 中心差分
   * - 重置采样常数
     - 分箱、50/40/10
     - 温度 25，EMA 0.05/0.01
     - --
     - 同 CuboiLeo
   * - 速度阶跃测试
     - G.2，672 条命令 × 4 次试验
     - --
     - --
     - G.2；试验之间仿真上下文和关节噪声不同

CuboiLeo（`CuboiLeo/SuperMimic-Reproduced <https://github.com/CuboiLeo/SuperMimic-Reproduced>`_）和 zita-ch（`zita-ch/supermimic_vibed <https://github.com/zita-ch/supermimic_vibed>`_）是社区复现，本实现只把它们作为设计参考，没有复制它们的代码。上表描述的是开发时查阅的 ``main`` 分支，当时没有记录具体 commit；这两个仓库之后若有改动，以仓库当前代码为准。

已知限制
--------

* **终止头冷启动。** 用 PyTorch 默认初始化时终止头输出约 0.5，32 步存活权重从 :math:`10^{-9}` 量级开始，在终止头经过几百个世界模型更新步之前，actor 只能从最前面几步想象拿到梯度。失败只占记录转移的约 0.3%；在短时间训练里，终止头先学到基础率，并且给大误差的概率反而比小误差低。``algorithm.termination_bias_init``\ （例如 -4.6，对应 1% 先验）和 ``algorithm.termination_head: analytic`` 都能消除冷启动，在本机短时间训练里也得到了方向正确的终止头。两者都是实验选项，默认值是论文设定。
* **网络没有输入归一化。** 观测里关节速度可达 20 左右，:math:`\xi` 里的零位偏差只有 0.01 左右；论文没有规定归一化，本实现也没有加。
* **BPTT 梯度尖峰。** 裁剪前的 actor 梯度范数会突然升高两个数量级；论文的裁剪方式限制了每一步的幅度。
* **没有实现：** 附录 G 的蒸馏编码器、推力恢复实验、BONES-SEED 发布格式的转换器。
* **尚未证明：** 论文规模下学会跟踪的 policy、多卡 NCCL 训练、多节点运行。哪些已经跑过、哪些没有，见使用指南。
