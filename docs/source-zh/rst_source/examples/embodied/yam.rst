使用 PICO 采集双臂 YAM 示教数据
===============================

使用 PICO 双手柄和 3 台 RealSense RGB 相机采集双臂 YAM 示教数据。关节状态与动作均为 14 维绝对位置向量，顺序为 ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``；夹爪用 ``0=关闭，1=打开``。

准备设备
--------

准备两台配备 ``flexible_4310`` 夹爪的 YAM 从臂、3 台 RealSense 相机、PICO 头显与双手柄，以及 ZeroMQ PICO publisher。按 :ref:`PICO 软件准备 <pico-software-setup>` 准备 XRoboToolkit、PC Service 和 publisher，再安装 YAM 运行环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

安装脚本包含固定版本的 ``i2rt`` backend 和 PICO 传输依赖。启用运动前，核对已安装的机器人模型、工具中心点、关节限位、操作者坐标系到机械臂基座坐标系的 yaw 和运动缩放参数是否与真机一致。

.. warning::

   采集器在第一次 reset 时连接两条从臂 CAN 和全部 3 台相机。请先让双臂处于有支撑且已确认安全的姿态，清空运动范围，并安排操作员守在急停旁。示例默认不自动复位或停放；结束会话前，操作员需确认关闭电机输出后双臂仍能安全稳定地受到支撑。

硬件检查
--------

启动 Ray 前，在 YAM 节点的仓库根目录、已激活的环境中查询 CAN 接口、相机序列号和模型文件；以下命令不启用电机或启动图像流：

.. code-block:: bash

   ip -details link show type can
   python -c 'import pyrealsense2 as rs; print([device.get_info(rs.camera_info.serial_number) for device in rs.context().devices])'
   python - <<'PY'
   from i2rt.robots.utils import ArmType, GripperType

   print(ArmType.YAM.get_xml_path(), GripperType.FLEXIBLE_4310.get_xml_path(), sep="\n")
   PY

按实物接线区分左、右 CAN，核对接口为 ``UP``、bitrate 为 SDK 要求的 1 Mbit/s，且无 ``BUS-OFF`` 等故障；配置方法见 `i2rt CAN 说明 <https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/README.md#can-bus-setup>`_。按机身标签确认顶部、左腕、右腕相机序列号；不要按枚举顺序猜测左右关系，并先释放其他任务占用的 CAN 和相机。

确认实物为 YAM 与 ``flexible_4310`` 夹爪。打开输出的两个 XML，将机械臂 ``joint1`` 至 ``joint6`` 的 ``range`` 与 ``examples/embodiment/config/env/realworld_dual_yam_joint.yaml`` 中两臂的 ``joint_limit_min/max`` 比对，单位为 rad，任务限位应在模型与真机允许范围内。夹爪 XML 的 ``grasp_site`` 定义 TCP，核对其位置、方向与实物工具一致，并与采集配置的 ``env.eval.override_cfg.ik.site_name`` 对应；更换夹爪或工具后须先验证模型。

采集配置的 ``env.eval.pico.position_scale`` 和 ``rotation_scale`` 分别控制手柄平移与转动的目标比例，示例均为 ``0.3``；手柄移动 1 cm 对应 3 mm TCP 目标位移。``env.eval.override_cfg.max_joint_delta`` 限制每步关节目标相对当前反馈的差值，单位为 rad，不限制整段运动总幅度。

publisher 地址按 :ref:`PICO publisher 配置 <pico-publisher-setup>` 获取：在 publisher PC 用 ``ip -brief address`` 找到 YAM 节点可访问的地址，端口取 ``configs/vr_bridge.yaml`` 的 ``zmq.ipc_addr``。consumer 填 ``tcp://<publisher-PC-address>:<port>``，不是头显 IP 或 ``0.0.0.0``；同机也可使用一致的 IPC 地址。

yaw 需分别测量左右臂：校准时头显面向的水平方向为操作者 +X，左侧为 +Y，上方为 +Z。按模型与厂家坐标图，将操作者前方表示为对应基座水平面内的 ``(dx, dy)``，填 ``atan2(dy, dx)``，单位为 rad，正值绕基座 +Z 按右手定则旋转；用度测量时乘 π/180 换算。仅在水平坐标轴同向时填 0，不能用 yaw 补偿基座倾斜。启动时面向测量时的方向并松开两侧 grip；操作朝向改变后重新测量和校准。

配置并采集
----------

配置两条从臂 CAN、按顶部、左侧、右侧顺序排列的 3 个相机序列号、PICO publisher 地址，以及左右臂各自实测的操作者坐标系到机器人坐标系的 yaw。在 YAM 所在节点 **启动 Ray 前** 导出这些值，供硬件枚举和环境 worker 读取：

.. code-block:: bash

   export LEFT_CAN_CHANNEL='<left-can-interface>'
   export RIGHT_CAN_CHANNEL='<right-can-interface>'
   export CAMERA_SERIALS='<top-serial>,<left-serial>,<right-serial>'
   export YAM_PICO_ZMQ_ADDR='tcp://<publisher-host>:<publisher-port>'
   export YAM_LEFT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'
   export YAM_RIGHT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'

前三个变量填入 ``DualYam`` 硬件配置中未设置的字段，也可以直接在 YAML 里写 ``left_can_channel``、``right_can_channel`` 和 ``camera_serials``。将占位符替换为当前工作站核对过的值。示例配置位于 ``examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml``，将环境 worker 和硬件放在同一个 Ray 节点。

启动采集前，先在 YAM 节点检查 PICO 数据流：

.. code-block:: bash

   python toolkits/realworld_check/test_pico_data.py \
       --zmq-addr "$YAM_PICO_ZMQ_ADDR"

确认头显、双手柄位姿和按键随操作更新，按 Ctrl-C 退出检查。新启动的单机环境可按下面的命令运行：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

首次接管先逐臂小幅移动，核对方向与缩放；方向不符时松开两侧 grip、停止采集后修正配置，故障停机后由现场操作员确认状态再恢复。

按住某侧手柄的 grip 可接管对应机械臂，松开即结束接管。示例为每条手臂各配置一项 ``pico``，各自经 YAM 模型的逆运动学（``override_cfg.ik``）把手柄位姿换算为关节目标。任一手柄掉线时，两条手臂都会保持不动，直到两侧 grip 都松开。``yam_episode`` 段将右菜单键映射为开始或结束一条录制，将左菜单键映射为丢弃当前录制。正式采集前，请在当前工作站的 PICO 设备上确认按键事件能送达。丢弃只会移除当前未完成的 episode；故障或超时的录制仍可能保存为未成功的 episode。

启动脚本默认在 ``logs/<timestamp>-realworld_dual_yam_collect_data_pico/`` 下保存日志。该配置设置 ``export_format: lerobot_dataset``。每条录制单独流式写入该目录 ``lerobot_dataset/rank_0/`` 下的一个 shard，未成功的录制也会保存，``is_success`` 为假。供 OpenPI SFT 或 HG-DAgger 使用前，把这些 shard 合并成一个 dataset，并去掉未成功的录制。将下面路径中的 ``<timestamp>`` 替换为本次采集目录的时间戳：

.. code-block:: bash

   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "logs/<timestamp>-realworld_dual_yam_collect_data_pico/lerobot_dataset" \
       --output-dir logs/yam_merged \
       --only-success

帧以未压缩 PNG 保存，640x480 的一个视角每帧约 0.9 MB，一分钟的录制约占 5 GB。训练前请检查三路 RGB 图像；``is_success`` 不能代替操作员对任务结果的判断。三种格式的用途见 :doc:`../../guides/data_collection`。
