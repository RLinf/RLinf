SuperMimic
==========

SuperMimic 用可微世界模型做人形全身跟踪。策略梯度沿 32 步想象轨迹反传，不使用 PPO 的似然比。

启动
----

唯一入口是 ``examples/embodiment/train_supermimic.py``。

.. code-block:: bash

   cd examples/embodiment
   # 4060 8GB 本机冒烟
   bash run_supermimic.sh supermimic_smoke
   # 论文对照：观测 93 维，目标 680 维。需要更大的显卡和 mjlab。
   export SUPERMIMIC_MOTION_PATH=/path/to/lafan1_g1_clips
   bash run_supermimic.sh supermimic_lafan1
   # 目标改为 689 维（整段目标末尾追加 9 维锚点误差）
   bash run_supermimic.sh supermimic_lafan1_anchor
   # 730 维目标，观测仍是 93 维
   bash run_supermimic.sh supermimic_lafan1_goal730
   # 99 维观测，目标仍是 680 维
   bash run_supermimic.sh supermimic_lafan1_obs99
   # 动作跟踪和速度跟踪各一半环境
   bash run_supermimic.sh supermimic_velocity
   # G1 重定向的 BONES-SEED。没有数据时会报错
   export SUPERMIMIC_BONES_PATH=/path/to/bones_seed
   bash run_supermimic.sh supermimic_bones_seed

依赖
----

``supermimic_smoke`` 只缩小环境数和批量。它使用和正式配置相同的 mjlab /
MuJoCo Warp 后端、``k=8``、``H=32`` 和网络宽度。``SUPERMIMIC_MOTION_PATH``
为空会报错。解析积分只留给 CPU 单测，不能代替这次 smoke。
Ray 的对象存储在这份配置里限制为 512 MiB，因为回放不经过对象存储。
内存监测保持 Ray 自己的阈值。只有 ``run_supermimic.sh supermimic_smoke``
会在用户没有设置时把 ``RAY_memory_usage_threshold`` 设为 0.99。
其它配置不会改这个值。本机验证过的栈是 PyTorch ``2.14.0+cu126``、
Warp ``1.18.0+cu12``、mjlab commit ``8bb2bc9``。这些钉在
``requirements/embodied/envs/supermimic.txt``，不改其它算法的 PyTorch 版本。
LAFAN1 和 BONES-SEED 都不进仓库。每条 ``clip_*.pt`` 要带 50 Hz（或可重采样的
30 Hz）、29 个 G1 关节名、默认姿态 ``q0`` 和 ``[T, 71]`` 状态；缺文件或
格式不对就报错，不用合成动作代替。
动力学网络隐藏层是论文表 A.4 的 2048, 2048, 2048。
重置按 1 秒分箱做 50% / 40% / 10% 混合采样。短回合在累计回报前按有效长度
掩掉奖励缓存，结束时清掉这段缓存。
``only_eval`` 按批从第 0 帧评完全部指定片段，每条一次。归一化回报的分母是
该片段完整预定步数乘每步最大正奖励 0.1，提前失败不会把没走到的部分从分母里拿掉。
``fail_step`` 是从 0 开始的步号，``fail_time_s`` 和 ``episode_length_s`` 是秒。
奖励根是 ``torso_link``，失败看骨盆，评测另记自由根（G1 上就是骨盆）的位置和朝向。
锚点的 9 维每一步用当前状态和参考重算。
每次更新用 CUDA 同步后记录仿真、回放采样、世界模型、32 步 BPTT、critic、
峰值显存和主机 RSS。这些数字来自 8 个环境，不是 4096 的性能结论。
恢复后重新预热采集，不用旧的 ``sim_vector_steps`` 判断新回放已经热好。

同一次想象生成 ``s_0..s_H``、``g_0..g_H``、奖励和终止概率。
critic 用这条轨迹回归 ``V(s_t, g_t)`` 到 ``R_t``，``t = 0..H-1``。

检查点
------

``save_interval <= 0`` 时不做周期性保存，但最后一次更新一定会写检查点。
同一次更新只保存一次。检查点含世界模型、策略、critic、优化器、学习率调度和计数。
速度跟踪打开时再加它自己的编码器和 critic。不保存回放、仿真状态和重置分箱。
恢复会新建训练器，重新预热，不恢复这些运行时状态。

观测噪声
--------

表 A.1 是均匀噪声，不是高斯噪声。关节位置 ``U[-0.01, 0.01]``，关节速度
``U[-0.5, 0.5]``，基座角速度 ``U[-0.2, 0.2]``，投影重力 ``U[-0.05, 0.05]``。
上一动作不加传感器噪声。执行器上的高斯动作噪声是另一回事。
每个控制步只给新的一帧采样一次。历史里已经写下的帧不再抽样。
重置时用这一帧重复填满 10 帧，所以补上的前缀共用同一次噪声。这是本实现的约定，
论文没有规定唯一的历史噪声做法。评测关闭传感器噪声。想象里新噪声是外生的，
加在观测上，不切断状态到策略的梯度。

世界模型把根旋转的误差写成相对旋转的 ``Log(R_pred R_target^T)``，
不再把两个绝对旋转向量相减。其它状态分量的尺度和原来一样。

730 维目标、99 维观测、速度和 BONES-SEED
------------------------------------------

730 维目标按 zita-ch 的实现：未来第 ``t+1`` 到 ``t+10`` 帧，末帧保持。
每一帧是航向系相对根位置 3 维、相对旋转 6 维、该帧根系线速度 3 维、
根系角速度 3 维、原始关节角 29 维、关节速度乘 0.1 的 29 维。
它不接在 680 维后面。99 维观测是 93 维后面加上
``Rot6D(R_robot^T R_reference)``，这 6 维默认无噪声。
速度接口是论文附录 G 的 RL，不是蒸馏。环境数和想象批量各一半给动作跟踪、
一半给速度跟踪。共享解码器在速度更新时冻结权重，梯度仍回到速度编码器。
对齐损失 ``0.1 ||z_vel - stopgrad(z_motion)||^2`` 只加在参考命令上。
BONES-SEED 按附录 E 做名字过滤和可行性过滤。动作库留在 CPU，
查询时只拷贝取到的那些帧，初始化不会把全部片段搬上 GPU。
没有真实数据包时，加载会报错，不能把合成样本说成数据集已经验收。
附录 E 写明的词已经实现；SONIC 的完整私有词表没有，也不把公开规则说成原作者的完整列表。

T 帧参考从第 0 帧评到时间变成 T 的那一步，所以预定控制步数是 T，不是 T-1。
提前失败时分母仍用这 T 步。``fail_step`` 是从 0 开始的步号，
``fail_time_s`` 和 ``episode_length_s`` 是秒。

本机 ``jrwsdjango`` 里还有和 SuperMimic 无关的依赖冲突，例如 TaH 希望
``torch==2.6.0``，而这里是 ``2.14.0+cu126``。requirements 把 numpy 限制在
``<2.4``，当前环境装的是 ``2.4.6``。这些不通过改其它项目的锁来处理。

终止与截断
----------

失败终止（骨盆高度或倾斜超阈值）是世界模型的正样本，评测记为失败。
10 秒超时和参考片段结束不是失败；前者在真实回合回报上用 critic 补全，后者不补全。
8 步窗口不跨越任何回合边界。
