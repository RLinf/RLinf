使用 PICO 采集双臂 YAM 示教数据
===============================

使用 PICO 双手柄和 3 台 RealSense RGB 相机采集双臂 YAM 示教数据。关节状态与动作均为 14 维绝对值，顺序为 ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``；夹爪用 ``0=关闭，1=打开``。

准备设备
--------

准备两台配备 ``flexible_4310`` 夹爪的 YAM 从臂、3 台 RealSense 相机、PICO 头显与双手柄，以及 ZeroMQ PICO publisher。按 :doc:`Franka VR 配置 <franka_vr>` 准备 XRoboToolkit、PC Service 和 publisher，再安装 YAM 运行环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

安装脚本包含固定版本的 ``i2rt`` backend 和 PICO 传输依赖。启用运动前，核对已安装的机器人模型、工具中心点、关节限位、手柄到基座的 yaw 和运动缩放参数是否与真机一致。

.. warning::

   采集器在第一次 reset 时连接两条从臂 CAN 和全部 3 台相机。请先让双臂处于有支撑且已确认安全的姿态，清空运动范围，并安排操作员守在急停旁。示例默认不自动复位或泊车；结束会话前，操作员需确认关闭电机输出后双臂仍能安全稳定地受到支撑。

配置并采集
----------

配置两条从臂 CAN、按顶部、左侧、右侧顺序排列的 3 个相机序列号、PICO publisher 地址，以及左右臂各自实测的操作者坐标系到机器人坐标系的 yaw。在 YAM 所在节点 **启动 Ray 前** 导出这些值，供硬件枚举和环境 worker 读取：

.. code-block:: bash

   export LEFT_CAN_CHANNEL='<left-can-interface>'
   export RIGHT_CAN_CHANNEL='<right-can-interface>'
   export CAMERA_SERIALS='<top-serial>,<left-serial>,<right-serial>'
   export YAM_PICO_ZMQ_ADDR='tcp://<publisher-host>:5555'
   export YAM_LEFT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'
   export YAM_RIGHT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'

前三个变量填入 ``DualYam`` 硬件配置中未设置的字段，也可以直接在 YAML 里写 ``left_can_channel``、``right_can_channel`` 和 ``camera_serials``。将占位符替换为本站确认过的值。示例配置位于 ``examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml``，将环境 worker 和硬件放在同一个 Ray 节点。新启动的单机环境可按下面的命令运行：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

按住某侧手柄的 grip 可接管对应机械臂，松开即结束接管。示例为每条手臂各配置一项 ``pico``，各自经 YAM 模型的逆运动学（``override_cfg.ik``）把手柄位姿换算为关节目标。任一手柄掉线时，两条手臂都会保持不动，直到两侧 grip 都松开。``yam_episode`` 段将右菜单键映射为开始或结束一条录制，将左菜单键映射为丢弃当前录制。正式采集前，请在本站 PICO 设备上确认按键事件能送达。丢弃只会移除当前未完成的 episode；故障或超时的录制仍可能保存为未成功的 episode。

启动脚本默认在 ``logs/<timestamp>-realworld_dual_yam_collect_data_pico/`` 下保存日志。该配方设置 ``export_format: lerobot_dataset``，因此不会写 ``demo_buffer/``。每条录制单独流式写入该目录 ``lerobot_dataset/rank_0/`` 下的一个 shard，未成功的录制也会保存，``is_success`` 为假。供 OpenPI SFT 或 HG-DAgger 使用前，把这些 shard 合并成一个 dataset，并去掉未成功的录制：

.. code-block:: bash

   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir logs/<timestamp>-realworld_dual_yam_collect_data_pico/lerobot_dataset \
       --output-dir logs/yam_merged \
       --only-success

帧以未压缩 PNG 保存，640x480 的一个视角每帧约 0.9 MB，一分钟的录制约占 5 GB。训练前请检查三路 RGB 图像；``is_success`` 不能代替操作员对任务结果的判断。三种格式的用途见 :doc:`../../guides/data_collection`。
