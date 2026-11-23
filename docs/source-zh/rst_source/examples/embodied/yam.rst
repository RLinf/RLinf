YAM PICO 数据采集
=================

在单台 YAM 主机上使用 PICO 双手柄和 3 台 RealSense RGB 相机采集双臂 YAM 关节空间示教数据，并将成功分片合并为可用于 :doc:`yam_openpi` 的 LeRobot 数据集。

概览
----

本示例通过 PICO 双手柄驱动两台配备 ``flexible_4310`` 夹爪的 YAM 从臂（按臂做逆运动学求解），将采集数据导出为未压缩 LeRobot shard，供后续 OpenPI SFT 或 HG-DAgger 训练使用。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      2× YAM · ``flexible_4310`` 夹爪

   .. grid-item-card:: 遥操作与相机
      :text-align: center

      PICO 双手柄 · 3 台 RealSense RGB 相机

   .. grid-item-card:: 工作流
      :text-align: center

      PICO 逆解数据采集 → 过滤合并 LeRobot

   .. grid-item-card:: 硬件
      :text-align: center

      1 台 YAM 主机 · 2 路 CAN（1 Mbit/s）· PICO publisher

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 观测
     - 顶部、左腕与右腕 3 路 RealSense RGB 图像，加 14 维绝对关节位置向量 ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``\ （弧度；夹爪 ``0=关闭，1=打开``）。
   * - 动作
     - 同顺序 14 维绝对关节与夹爪目标，由每条臂的 YAM 逆运动学（``override_cfg.ik``）从手柄位姿求得。
   * - 采集控制
     - 按住左 / 右手 ``grip`` 接管对应机械臂；右菜单键开始或结束一条录制，左菜单键丢弃当前未完成录制（``yam_episode``）。

硬件与安全准备
--------------

将两台配备 ``flexible_4310`` 夹爪的 YAM 从臂接入两路 1 Mbit/s CAN 接口（配置见 `i2rt CAN 说明 <https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/README.md#can-bus-setup>`_），并按机身实物标签区分顶部、左腕、右腕 3 台 RealSense 相机。按 :ref:`PICO 软件准备 <pico-software-setup>` 与 :ref:`PICO publisher 配置 <pico-publisher-setup>` 准备好 XRoboToolkit、PC Service 与 ZeroMQ publisher。开始前先释放其他进程占用的 CAN 与相机。

.. warning::

   采集器在第一次 reset 时连接两条从臂 CAN 和全部 3 台相机。请先让双臂处于有支撑且已确认安全的姿态，清空运动范围，并安排操作员守在急停旁。示例默认关闭自动复位与停放——结束会话前，操作员必须确认关闭电机输出后双臂仍能安全稳定地受到支撑。

.. dropdown:: 硬件标定细则（可选）

   启用运动前，核对机器人模型、工具中心点、关节限位、运动缩放与操作者到机械臂基座的 yaw：

   * **关节限位与 TCP 校验：** 打开下方 ``i2rt`` 输出的机械臂与夹爪 XML，比对 ``joint1`` 至 ``joint6`` 的 ``range``\ （弧度）与 ``examples/embodiment/config/env/realworld_dual_yam_joint.yaml`` 中两臂的 ``joint_limit_min`` / ``joint_limit_max``\ （任务限位须落在模型与真机允许范围内）。核对夹爪 XML 中的 ``grasp_site`` 与实物工具及 ``env.eval.override_cfg.ik.site_name`` 一致。
   * **缩放与单步限幅：** ``env.eval.pico.position_scale`` 与 ``rotation_scale`` 默认均为 ``0.3``\ （手柄移动 1 cm 对应 3 mm TCP 目标位移）；``env.eval.override_cfg.max_joint_delta`` 限制每步关节目标相对当前反馈的最大弧度变化。
   * **publisher 地址：** 在 publisher PC 上用 ``ip -brief address`` 查出 YAM 节点可达的 IP，端口取 ``configs/vr_bridge.yaml`` 中的 ``zmq.ipc_addr``。将 ``YAM_PICO_ZMQ_ADDR`` 设为 ``tcp://<publisher-PC-address>:<port>``\ （切勿填头显 IP 或 ``0.0.0.0``；同机运行也可用一致的 IPC 地址）。
   * **分臂测量 yaw：** 左右臂需分别测量。校准时头显面向的水平方向为操作者 ``+X``\ （左侧 ``+Y``，上方 ``+Z``）；按模型与厂家坐标图将操作者前方表示为各臂基座水平面内的 ``(dx, dy)``，设 yaw 为 ``atan2(dy, dx)``\ （弧度，正值绕基座 ``+Z`` 按右手定则旋转；角度乘 ``π/180`` 换算；仅水平轴同向时填 ``0``，yaw 不能补偿基座倾斜）。启动时面向实测方向并松开两侧 grip，操作朝向改变后需重新测量校准。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 双臂 YAM + PICO

      **1. 安装 YAM 运行环境**\ （包含固定版本的 ``i2rt`` 后端与 PICO 传输依赖）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env yam && source .venv/bin/activate

      **2. 查询 CAN 接口、相机序列号与模型 XML**\ （只读查询，不会启用电机或图像流）：

      .. code-block:: bash

         ip -details link show type can
         python -c 'import pyrealsense2 as rs; print([device.get_info(rs.camera_info.serial_number) for device in rs.context().devices])'
         python - <<'PY'
         from i2rt.robots.utils import ArmType, GripperType

         print(ArmType.YAM.get_xml_path(), GripperType.FLEXIBLE_4310.get_xml_path(), sep="\n")
         PY

      确认两路 CAN 处于 ``UP``、bitrate 为 1 Mbit/s 且无 ``BUS-OFF`` 故障，按实物接线区分左右 CAN，并按机身标签匹配顶部、左腕、右腕序列号（XML 与 yaw 核对方法见上方“硬件标定细则”）。

      **3. 导出环境变量、验流并启动采集**\ （在 **启动采集前** 导出，供硬件枚举与 env worker 读取；也可将前三个硬件字段直接写入 ``examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml`` 中的 ``left_can_channel``、``right_can_channel`` 与 ``camera_serials``）：

      .. code-block:: bash

         export LEFT_CAN_CHANNEL='can0'                      # 替换为左臂 CAN 接口
         export RIGHT_CAN_CHANNEL='can1'                     # 替换为右臂 CAN 接口
         export CAMERA_SERIALS='12345678,23456789,34567890'  # 替换为顶、左、右相机序列号
         export YAM_PICO_ZMQ_ADDR='tcp://192.168.1.100:5555' # 替换为 publisher IP:端口
         export YAM_LEFT_OPERATOR_TO_ROBOT_YAW='0.0'         # 替换为实测 yaw（弧度）
         export YAM_RIGHT_OPERATOR_TO_ROBOT_YAW='0.0'        # 替换为实测 yaw（弧度）

         # 检查头显、双手柄位姿与按键更新（按 Ctrl-C 退出）
         python -m toolkits.realworld_check.test_pico_data --zmq-addr "$YAM_PICO_ZMQ_ADDR"

         # 启动采集
         bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

      首次接管请逐臂小幅移动，核对方向与缩放；方向不符立即松开两侧 grip、停止采集并修正 yaw。任一手柄掉线时双臂会保持不动，直到两侧 grip 均松开。故障停机后须由现场操作员确认状态再恢复。

      **4. 合并成功分片**\ （未成功或超时录制也会以 ``is_success: false`` 落盘；将 ``20261121-120000`` 替换为本次日志目录时间戳）：

      .. code-block:: bash

         python -m toolkits.lerobot.merge_lerobot_datasets \
             --source-dir "logs/20261121-120000-realworld_dual_yam_collect_data_pico/lerobot_dataset" \
             --output-dir logs/yam_merged \
             --only-success  # 将时间戳替换为实际采集目录

结果与常见问题
--------------

每条录制按未压缩 PNG 写入 ``logs/<timestamp>-realworld_dual_yam_collect_data_pico/lerobot_dataset/rank_0/``\ （640×480 单视角每帧约 0.9 MB，三路相机每分钟约占 5 GB）。训练前务必人工检查三路 RGB 图像——``is_success`` 不能代替任务结果核验。导出格式说明见 :doc:`../../guides/data_collection`，Pi0.5 训练与评估见 :doc:`yam_openpi`。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理方法
   * - 接管时机械臂运动方向反了
     - 立即松开两侧 grip 并停止采集，按 ``atan2(dy, dx)`` 重新测量 ``YAM_LEFT_OPERATOR_TO_ROBOT_YAW`` / ``YAM_RIGHT_OPERATOR_TO_ROBOT_YAW``，重新导出后再次启动。
   * - CAN 接口出现 ``BUS-OFF`` 或打开失败
     - 释放占用 CAN 的其他进程，按 ``i2rt`` CAN 文档重设 1 Mbit/s 速率，并用 ``ip -details link show type can`` 确认 ``UP`` 状态。
   * - ``test_pico_data.py`` 无输出更新
     - 确认 XRoboToolkit PC Service 与 ``vr_data_publisher`` 正在运行，并将 ``YAM_PICO_ZMQ_ADDR`` 设为可达的 ``tcp://<publisher-ip>:<port>``\ （切勿填头显 IP 或 ``0.0.0.0``）。
   * - 左右腕相机画面颠倒
     - 按相机机身标签而非枚举顺序核对序列号，在启动前按 ``top,left,right`` 顺序更新 ``CAMERA_SERIALS``。
   * - 原始分片中混入失败或超时回合
     - 合并时为 ``merge_lerobot_datasets.py`` 传入 ``--only-success``，并在 SFT 前抽查合并后的 RGB 画面。
