G1 动作跟踪环境
===============

在 MuJoCo Warp（mjlab）中运行 GPU 并行仿真，为 29 自由度 Unitree G1 人形机器人训练和评测全身动作跟踪 policy（``env_type: g1_mimic``）。环境支持 LAFAN1 或自定义动作跟踪以及站立推力恢复任务，配套 :doc:`SuperMimic <supermimic>` 训练示例。

.. toctree::
   :hidden:

   动作数据 <supermimic_motion>

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      SuperMimic Policy（Unitree G1）

   .. grid-item-card:: 算法
      :text-align: center

      SuperMimic

   .. grid-item-card:: 任务
      :text-align: center

      LAFAN1 动作跟踪 · 站立推力恢复

   .. grid-item-card:: 硬件
      :text-align: center

      1–4 张 NVIDIA GPU · :ref:`AMD ROCm（实验性） <supermimic-hardware>`

| **你将完成：** 了解 G1 任务、观测、奖励与环境配置 → 使用 :doc:`SuperMimic <supermimic>` 启动训练或评测。
| **前置条件：** :doc:`安装说明 </rst_source/start/installation>` · 按 :doc:`supermimic` 完成虚拟环境安装和 LAFAN1 数据准备。

任务
~~~~

环境支持两类任务模式，由训练或评测配置文件选择：

.. list-table::
   :header-rows: 1
   :widths: 20 26 24 30

   * - 任务
     - 参考输入
     - 回合上限与失败条件
     - 默认配置
   * - 动作跟踪（``tracking``）
     - 40 条 LAFAN1 片段（或自定义 ``clip_*.pt`` 片段）
     - 训练上限 10 秒（评测播完全片）；片段结束或摔倒时终止
     - ``supermimic_lafan1_gencoord`` / ``supermimic_lafan1_eval``
   * - 站立推力恢复（``push_recovery``）
     - 内置 mjlab 屈膝站立姿态（无需动作片段）
     - 1 秒稳定 + 水平速度推力（0.5–4.0 m/s）后最多 5 秒
     - ``supermimic_push_recovery``

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测
     - 机器人当前状态，加上从当前参考帧起、每隔 5 个控制步（0.1 秒）取一帧、共 10 帧的未来参考窗口（超出片段末尾时保持最后一帧）。状态支持广义坐标编码（``gencoord``，68 维）和骨盆坐标系下跟踪部位位姿速度编码（``maxcoord``，214 维）。
   * - 动作
     - 50 Hz 下的 29 维关节目标角度，由 PD 控制器在 4 个 5 ms 物理子步（200 Hz 物理）中跟踪。训练时叠加高斯噪声（``action_noise_std: 0.1``），评测时确定性执行 policy 输出。
   * - 奖励
     - 骨盆与 14 个部位的跟踪奖励减去动作变化和关节软限位惩罚（单步上限 ``0.1``，详见下文 `奖励与终止条件`_）。
   * - 提示词
     - 不使用（仅低维本体感觉控制）。

环境说明
--------

跟踪部位与域随机化
~~~~~~~~~~~~~~~~~~

环境共跟踪 14 个身体部位：

* **根节点与躯干：** ``pelvis``、``torso_link``
* **双腿：** ``left/right_hip_roll_link``、``left/right_knee_link``、``left/right_ankle_roll_link``
* **双臂：** ``left/right_shoulder_roll_link``、``left/right_elbow_link``、``left/right_wrist_yaw_link``

训练初始化时，每个并行环境独立采样域随机化参数（足底摩擦系数 ``[0.3, 1.2]``、躯干质心偏移及 29 个关节编码器偏置 ``[-0.01, 0.01]`` rad），每次重置还在采样参考帧上叠加初始位姿和速度扰动。评测模式关闭全部域随机化、重置噪声和动作噪声。

奖励与终止条件
~~~~~~~~~~~~~~

计算 14 个部位的位置和朝向误差前，环境先把参考姿态对齐到机器人骨盆当前的水平位置（xy）和偏航角（yaw），用来衡量局部动作形态；骨盆自身的位置和朝向直接与参考骨盆对比。

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - 奖励项
     - 权重
     - 含义
   * - 根节点位置与朝向
     - 0.5、0.5
     - 骨盆在世界系下的位置与测地线旋转误差
   * - 身体部位位置与朝向
     - 1.0、1.0
     - 14 个跟踪部位在骨盆重锚后的平均位置与朝向误差
   * - 身体部位线速度与角速度
     - 1.0、1.0
     - 14 个跟踪部位在世界系下的平均线速度与角速度误差
   * - 动作变化惩罚
     - -0.1
     - 相邻两步原始动作差值的平方和
   * - 关节限位惩罚
     - -2.0
     - 关节角超出软限位的总偏移量

上述各项求和后乘以 ``0.02 s``，单步跟踪奖励上限为 ``0.1``。骨盆高度误差绝对值超过 ``0.25 m``，或骨盆倾斜误差（重力方向在机器人与参考骨盆坐标系下的 z 分量之差）绝对值超过 ``0.8`` 时，回合立即按\ **摔倒**\ 终止。达到 10 秒训练上限或播完片段末尾均正常截断，不计为摔倒。

关键环境配置项
~~~~~~~~~~~~~~

默认环境配置位于 ``examples/embodiment/config/env/g1_mimic.yaml``，常用配置项如下：

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - 配置项
     - 默认值
     - 说明
   * - ``motion_path``
     - ``${oc.env:SUPERMIMIC_MOTION_PATH,""}``
     - 存放转换后 ``clip_*.pt`` 文件的目录
   * - ``motion_format``
     - ``lafan1``
     - 动作库加载格式（``lafan1``）
   * - ``num_clips``
     - ``null``\ （默认训练配置为 ``40``）
     - 加载片段数量；``null`` 加载目录下全部片段
   * - ``motion_on_cpu``
     - ``False``
     - 将动作库留在 CPU 内存，仅把查询帧传到 GPU
   * - ``max_episode_seconds``
     - ``10.0``
     - 训练单回合最长持续时间（秒）
   * - ``action_noise_std``
     - ``0.1``
     - 训练时的动作高斯噪声标准差

安装与运行
----------

请先按 :doc:`SuperMimic <supermimic>` 页面的 :ref:`安装与运行 <supermimic-hardware>` 完成 ``.venv-supermimic`` 环境安装（支持 NVIDIA CUDA 与 AMD ROCm）并将 40 条 LAFAN1 片段准备到 ``./data/lafan1_g1``\ （离线转换或自定义片段见 :doc:`supermimic_motion`）。

.. tab-set::

   .. tab-item:: SuperMimic（GenCoord）

      在仓库根目录激活 ``.venv-supermimic``，启动 ``g1_mimic`` 单卡训练（在 0 号 GPU 上并行 4096 个 G1 环境训练 6000 次更新，每 2000 次更新保存一个 checkpoint）：

      .. code-block:: bash

         source .venv-supermimic/bin/activate
         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/lafan1_g1
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord

可视化与结果
------------

使用 TensorBoard（``tensorboard --logdir ./logs``）查看训练曲线，LAFAN1 40 片段动作跟踪结果与多卡吞吐对比见 :doc:`SuperMimic <supermimic>`。

下一步
------

* **SuperMimic 算法详解、多卡训练、MaxCoord 与实验结果：** :doc:`supermimic`
* **独立评测（动作跟踪与站立推力恢复）：** :doc:`../../evaluations/guides/supermimic`
* **自定义动作数据与离线转换：** :doc:`supermimic_motion`
* **算法公式与实现参考：** :doc:`../../reference/algorithms/supermimic`
