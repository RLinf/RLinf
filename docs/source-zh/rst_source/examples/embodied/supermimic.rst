SuperMimic
==========

使用 **SuperMimic**\ （`arXiv:2610.28480 <https://arxiv.org/abs/2610.28480>`__，`项目主页 <https://supermimic.github.io>`__）在 MuJoCo Warp 中训练 29 自由度 Unitree G1 的全身动作跟踪 policy。SuperMimic 从仿真轨迹中学习可微动力学世界模型，在 32 步想象轨迹上将跟踪回报梯度直接反向传播给 policy，并配合 critic 做末端价值估计与按回报重采样。在 40 条 LAFAN1 片段上，4 卡 NVIDIA A800 按默认 GenCoord 配置训练 6000 次更新耗时约 23 分钟，归一化回报达到 ``0.856``，全部零摔倒。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      SuperMimic（可微世界模型 + Actor-Critic）

   .. grid-item-card:: 模型
      :text-align: center

      动力学世界模型 · 动作 Policy · Critic

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      :doc:`G1 动作跟踪环境 <g1_mimic>` · 40 条 LAFAN1 片段

   .. grid-item-card:: 训练
      :text-align: center

      1–4 张 NVIDIA GPU（FSDP2）· :ref:`AMD ROCm（实验性） <supermimic-hardware>`

| **你将完成：** 安装并准备 LAFAN1 → 启动训练 → 查看 TensorBoard → 评测 checkpoint。
| **前置条件：** :doc:`安装说明 </rst_source/start/installation>` · 一张或多张 NVIDIA GPU（或 AMD ROCm）· 可访问 Hugging Face 下载 LAFAN1 数据。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 28 24 26

   * - 方案
     - 配置名
     - 状态表示
     - 任务
   * - GenCoord（默认）
     - ``supermimic_lafan1_gencoord``
     - 广义坐标（68 维）
     - ``g1_mimic`` 上的 40 条 LAFAN1 片段
   * - MaxCoord
     - ``supermimic_lafan1_maxcoord``
     - 跟踪部位位姿与速度（214 维）
     - ``g1_mimic`` 上的 40 条 LAFAN1 片段

观测与动作
~~~~~~~~~~

14 个跟踪部位、PD 控制器参数、奖励权重及域随机化详见 :doc:`G1 动作跟踪环境 <g1_mimic>`。

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 摘要
   * - 观测
     - 当前 G1 状态编码（GenCoord 或 MaxCoord），加上间隔 5 个控制步（0.1 秒）的 10 帧未来参考窗口。
   * - 动作
     - 50 Hz 下的 29 维目标关节角度（训练叠加标准差 ``0.1`` 的高斯噪声，评测确定性执行）。
   * - 奖励
     - 6 项跟踪核加动作变化和关节软限位惩罚，乘以 ``0.02 s``\ （单步上限 ``0.1``）。
   * - 提示词
     - 不使用。

SuperMimic 算法原理
-------------------

每次全局更新在多卡间同步执行三个阶段：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/supermimic_loop_zh.svg
   :align: center
   :width: 100%
   :alt: SuperMimic 三阶段训练循环：MuJoCo Warp 仿真采集、世界模型拟合、想象轨迹与 policy 及 critic 优化

   RLinf 中的 SuperMimic 三阶段训练循环。

1. **仿真采集与按回报重置。** 并行 G1 环境在 MuJoCo Warp 中带探索噪声执行动作，并将状态转移写入各卡本地 replay buffer。回合结束时，采样器优先从回报偏低的动作片段区间重置，让训练集中攻克高难度动作。
2. **拟合世界模型。** 动力学网络从 replay 状态出发，按记录动作自回归展开 ``k = 8`` 步，预测速度增量并积分到下一时刻位姿，用 Bryson 通道加权 MSE 损失优化。
3. **优化 policy 与 critic。** 从 replay 采样起点，让 policy 在冻结的世界模型中推演 ``H = 32`` 步。每一步用可微正运动学计算跟踪奖励，并用平滑终止概率累加存活权重，将一阶回报梯度直接反向传播给 policy；随后在同一批想象轨迹上计算 GAE 目标拟合 critic。

通过 ``actor.model.representation`` 可在 ``gencoord``\ （默认，68 维广义坐标配合可微正运动学）和 ``maxcoord``\ （214 维跟踪部位位姿速度配合链式逆运动学）之间切换；checkpoint 只能加载到训练时对应的表示中。完整状态布局、数学公式、网络维度、优化器设置、FSDP2 分片以及与论文和官方代码的差异见 :doc:`../../reference/algorithms/supermimic`。

.. _supermimic-hardware:

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. 安装**

      在仓库根目录创建独立虚拟环境（固定特定版本的 PyTorch、MuJoCo Warp 和 ``warp-lang``，不要与其他 RLinf 虚拟环境混用）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
         source .venv-supermimic/bin/activate

      **2. 准备 LAFAN1 数据**

      下载、校验并转换 40 条 Unitree G1 LAFAN1 动作片段（来自 `lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_，源自 Ubisoft La Forge `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_\ ，采用 CC BY-NC-ND 4.0 许可，传 ``--accept-license`` 前请先查阅条款）：

      .. code-block:: bash

         python toolkits/supermimic/prepare_lafan1.py --out ./data/lafan1_g1 --accept-license

      转换成功后最后一行输出 ``[lafan1] wrote 40 clips, 2.45 h, manifest data/lafan1_g1/manifest.json``。如需离线转换或制作自定义 ``clip_*.pt`` 片段，见 :doc:`supermimic_motion`。

      **3. 启动训练**

      设置动作数据目录并关闭 MuJoCo 的 OpenGL 渲染器，在 0 号 GPU 上启动 6000 次更新的 GenCoord 训练（默认开启 CUDA 图回放与 bf16 混合精度）：

      .. code-block:: bash

         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/lafan1_g1
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord

      首次启动会有约 2–3 分钟的 ``torch.compile`` 编译预热，前 3 次预热更新结束后日志输出 ``Recorded the SuperMimic update as a CUDA graph.``。每 100 次更新打印一次以 ``Global Step: <N>/6000`` 开头的指标表；每 2000 次更新及第 6000 次更新后在 ``logs/<timestamp>-supermimic_lafan1_gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_<N>/`` 保存 checkpoint。

   .. tab-item:: AMD ROCm（实验性）
      :sync: rocm

      AMD 后端用 ROCm Warp 运行 MuJoCo Warp，面向 MI300X（``gfx942``）。前置条件：Linux x86_64、含 ``hipcc`` 的 ROCm 7.2 SDK、C++ 编译工具链（参考 `ROCm Warp gfx942 Dockerfile <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_）以及支持 flat wheel 索引的新版 ``uv``。

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export ROCM_PATH=/opt/rocm
         bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
             embodied --env supermimic --venv .venv-supermimic
         source .venv-supermimic/bin/activate

      安装脚本从 AMD 索引获取 PyTorch 2.9.1，编译 ROCm Warp ``1.13.0+rocm.0``，并安装 mjlab 1.2.0、MuJoCo 3.8.1 与 MuJoCo Warp 3.8.1。如需自定义源码路径或目标架构，可设置 ``SUPERMIMIC_WARP_PATH``\ （默认 ``../warp-rocm``）和 ``SUPERMIMIC_HIP_ARCH``\ （默认 ``gfx942``）。

      **2. 准备 LAFAN1 数据**

      .. code-block:: bash

         python toolkits/supermimic/prepare_lafan1.py --out ./data/lafan1_g1 --accept-license

      **3. 启动训练**

      .. code-block:: bash

         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/lafan1_g1
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord

      .. note::

         AMD ROCm 上尚未验证完整训练收敛。

.. dropdown:: 多卡训练

   环境数、世界模型批次和想象 rollout 数均为跨卡总量，多卡时自动均分，学习率与总更新次数保持不变：

   .. list-table::
      :header-rows: 1
      :widths: 40 20 20 20

      * - 每次更新
        - 全局总量
        - 两卡每卡
        - 四卡每卡
      * - 并行环境数
        - 4096
        - 2048
        - 1024
      * - 世界模型窗口（共 10 批）
        - 2048
        - 1024
        - 512
      * - 想象 rollout 条数
        - 4096
        - 2048
        - 1024

   两卡训练示例（四卡将 ``0-1`` 改为 ``0-3``）：

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord \
          '~cluster.component_placement' \
          '+cluster.component_placement={actor:0-1,env:0-1,rollout:0-1}'

.. dropdown:: 断点续训

   训练中断后，在相同 GPU 数量下传入原输出目录与 ``runner.resume_dir=auto`` 即可续训（成功加载时日志会打印 ``Resuming from <path>``）：

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord \
          runner.logger.log_path=./logs/<timestamp>-supermimic_lafan1_gencoord \
          runner.resume_dir=auto

.. dropdown:: 使用 MaxCoord 训练

   将配置名切换为 ``supermimic_lafan1_maxcoord``：

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh supermimic_lafan1_maxcoord

.. dropdown:: 评测 Checkpoint

   在 40 条 LAFAN1 片段上运行动作跟踪评测（MaxCoord checkpoint 需追加 ``rollout.model.representation=maxcoord``；完整的站立推力恢复测试与指标定义见 :doc:`../../evaluations/guides/supermimic`）：

   .. code-block:: bash

      bash evaluations/run_eval.sh supermimic_lafan1_eval \
          runner.ckpt_path=./logs/<timestamp>-supermimic_lafan1_gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
          runner.eval_export.output_dir=./eval_results/supermimic_6000

.. dropdown:: 核心算法配置项

   默认配置位于 ``examples/embodiment/config/supermimic_lafan1_gencoord.yaml``\ （``supermimic_lafan1_maxcoord.yaml`` 继承此配置）。

   .. list-table::
      :header-rows: 1
      :widths: 36 16 48

      * - 配置项
        - 默认值
        - 说明
      * - ``actor.model.representation``
        - ``gencoord``
        - 状态表示：``gencoord`` 或 ``maxcoord``
      * - ``algorithm.k`` / ``algorithm.H``
        - ``8`` / ``32``
        - 世界模型展开窗口长度与 policy 想象步长
      * - ``algorithm.replay_ratio``
        - ``16``
        - 每条状态转移被世界模型抽样的平均次数（4096 环境下每步 0.4 次更新）
      * - ``algorithm.wm_batches`` / ``wm_window_batch``
        - ``10`` / ``2048``
        - 每次更新的世界模型批次数与全局每批窗口数
      * - ``algorithm.imagine_batch``
        - ``4096``
        - 每次 policy 更新的全局想象 rollout 条数
      * - ``actor.cuda_graph``
        - ``True``
        - 将每次全局更新录成单张 CUDA 图（设为 ``False`` 可关闭）
      * - ``actor.fsdp_config.mixed_precision.param_dtype``
        - ``bf16``
        - 每阶段开始时将网络权重转为 bf16，主权重、优化器状态、梯度归约、运动学与损失保持 float32（设为 ``null`` 即回退 TF32）

可视化与结果
------------

使用 TensorBoard（``tensorboard --logdir ./logs``）监控训练进度，通用指标含义见 :doc:`../../reference/metrics`。核心监控指标（横轴为更新次数）：

* ``env/sim/reward_mean``：仿真中单步平均跟踪奖励，收敛趋向上限 ``0.1``。
* ``env/sim/terminal_rate``：每步摔倒的环境比例，收敛趋向 ``0``。
* ``train/world_model/state_loss``：Bryson 加权世界模型预测损失。

动作跟踪结果（LAFAN1 40 片段）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 4 张 NVIDIA A800 上按默认 GenCoord 配置（bf16、seed 0）训练 6000 次更新，总耗时约 23 分钟，评测结果如下：

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 更新次数
     - ``normalized_return``
     - ``fail_rate``
     - ``body_pos_err``\ （m）
   * - 6000
     - 0.856
     - 0.000
     - 0.023

40 条片段全部无摔倒完成（论文在这 40 条片段上报告的归一化回报为 ``0.82``）。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/supermimic_gencoord_curves.png
   :alt: SuperMimic GenCoord 训练曲线
   :width: 100%

   NVIDIA A800 上 6000 次更新的 GenCoord 训练曲线。仿真奖励上限为每秒 5.0（50 Hz 下每步 0.1）；右上角想象回报包含末端 critic 估值，因此会高于 5。

吞吐对比
~~~~~~~~

下表对比 RLinf 与官方单卡实现在独占 NVIDIA A800 上运行 GenCoord 每次全局更新的耗时（预热 200 次后取 400 次更新的实际耗时均值；4 卡 bf16 在 1000 次更新窗口下同样为 0.227 秒）。RLinf 的 bf16 行使用默认配置（开启 CUDA 图，采用 ``no_shard``）；TF32 行将 ``actor.fsdp_config.mixed_precision.param_dtype`` 设为 ``null`` 并将 ``rollout.amp_autocast.enabled`` 设为 ``False``：

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 实现
     - GPU 数
     - 矩阵乘精度
     - 秒 / 次更新
   * - 官方代码
     - 1
     - TF32
     - 0.515
   * - RLinf
     - 1
     - TF32
     - 0.490
   * - RLinf
     - 1
     - bf16
     - 0.383
   * - RLinf
     - 2
     - TF32
     - 0.341
   * - RLinf
     - 2
     - bf16
     - 0.285
   * - RLinf
     - 4
     - TF32
     - 0.263
   * - RLinf
     - 4
     - bf16
     - 0.227

单卡 TF32 下 RLinf 每次更新比单进程官方代码快约 5%（0.490 秒对比 0.515 秒）。使用默认 bf16 混合精度时，单卡为 0.383 秒（速度提升至 1.34 倍），扩展到 4 卡为 0.227 秒（速度提升至 2.27 倍，完整 6000 次更新耗时约 23 分钟）。
