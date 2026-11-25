Go2 高层运动控制
================

Gymnasium 环境 ``Go2DriveEnv-v1`` 默认从 ``rt/lowstate`` 读取 Go2 EDU 的关节和 IMU 数据。检查运动方向时，RLinf 通过宇树 ``SportClient`` 发送机身速度指令，机载控制器负责步态和关节控制。

概览
----

.. grid:: 2 4 4 4

   .. grid-item-card:: 模型

      手动速度指令。

   .. grid-item-card:: 算法

      无训练流程。

   .. grid-item-card:: 任务

      状态接收与六方向检查。

   .. grid-item-card:: 硬件

      Go2 EDU 与 Linux 控制电脑。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 检查项
     - 请求的动作
   * - 只读
     - 接收状态，不发送运动指令。
   * - ``forward`` / ``backward``
     - 以 0.3 m/s 前进或后退。
   * - ``left`` / ``right``
     - 以 0.3 m/s 向左或向右平移。
   * - ``turn_left`` / ``turn_right``
     - 以 0.5 rad/s 向左或向右转动。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 约定
   * - 观测
     - 姿态、角速度、12 关节位置与速度、电池电压、四个原始足力值。
   * - 动作
     - 依次为前向速度、左向速度和左转角速度，范围分别为 ±0.3 m/s、±0.3 m/s 和 ±0.5 rad/s。
   * - 奖励
     - 固定为零，此检查不训练 policy。

安装与运行
----------

**前置条件：** 按照 :doc:`安装指南 </rst_source/start/installation>` 准备 Linux 控制电脑，并用独立网卡连接 Go2 EDU。

.. tab-set::

   .. tab-item:: Go2 EDU

      **1. 安装 SDK**

      在 Linux 控制电脑上按 `宇树 SDK 安装说明 <https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206>`_ 安装 CycloneDDS 0.10.x 原生库，并导出 ``CYCLONEDDS_HOME``。Python 3.11 的绑定需要这个库才能编译。在同一终端进入 RLinf 仓库根目录，安装 SDK：

      .. code-block:: bash

         bash requirements/install.sh embodied --env go2_sport
         source .venv/bin/activate

      按 SDK 说明配置机器人网卡。控制电脑无需 GPU 或 policy 权重。

      **2. 检查状态与运动**

      将 ``<interface>`` 替换为连接 Go2 的网卡名称。先预览指令，此步骤不建立 SDK 通道：

      .. code-block:: bash

         python -m toolkits.realworld_check.check_go2_sport --interface <interface> --direction forward --dry-run

      预览结果包含 ``hardware_access: false``。随后单独检查状态接收：

      .. code-block:: bash

         python -m toolkits.realworld_check.check_go2_sport --interface <interface> --readonly

      十次状态检查结果写入 ``outputs/go2_sport/``。如需从 ``rt/lowstate`` 读取一帧通过校验的关节和 IMU 数据，在新进程中启动只读订阅程序。

      .. code-block:: bash

         python -m toolkits.realworld_check.check_go2_lowstate --interface <interface>

      运动测试须保持原厂控制器启用，现场确认机器人已站立、场地空旷、遥控器可用，并关闭其他运动程序，再运行单个方向：

      .. code-block:: bash

         python -m toolkits.realworld_check.check_go2_sport --interface <interface> --direction forward

      每次选上表中的一个方向，完成现场检查后输入 ``GO``。程序默认以 10 Hz 发送 1 秒速度指令，随后请求停止。用 ``--seconds 2`` 可延长至 2 秒，用 ``--output <path>`` 可指定新的报告文件。

      .. warning::

         停止请求和 SDK 低速估计均不能证明物理停稳。请直接观察机器人，运动未停止就使用遥控器。网络断开或进程故障可能导致软件无法请求停止。

可视化与结果
------------

报告记录接口错误和发出停止请求后 SDK 估计的速度。请另行记录在现场观察到的运动和停稳情况。如果应用需要行进距离、转角或停止距离，请单独测量。
