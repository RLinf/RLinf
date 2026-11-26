Franka ZED 与 Robotiq
=====================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robotiq_zed.jpeg
   :align: center
   :width: 80%

   接入 Stereolabs ZED 相机和 Robotiq 2F 夹爪的 Franka 数据采集配置。

概览
----

在双节点 Franka 部署中，将默认的 RealSense 相机和 Franka 夹爪替换为 Stereolabs ZED 相机与 Robotiq 2F 夹爪。节点 0（GPU 服务器）负责采集 ZED 图像流，节点 1（NUC 控制节点）通过网口与 USB-RS485 控制 Franka 机械臂和 Robotiq 夹爪。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · ZED 相机 · Robotiq 2F

   .. grid-item-card:: 连接方式
      :text-align: center

      USB 3.0\ （ZED）· USB-RS485 Modbus RTU

   .. grid-item-card:: 用途
      :text-align: center

      双节点数据采集与下游 RL/SFT

   .. grid-item-card:: 前置依赖
      :text-align: center

      :doc:`franka` · ZED SDK 4.2.x · 双节点

硬件与安全准备
--------------

该方案采用 **双节点** 部署（``num_nodes: 2``）：

- **节点 0（GPU 服务器）**：通过 USB 3.0 连接 ZED 相机，确认相机指示灯亮起且 ``lsusb`` 能看到 ``Stereolabs`` 设备。
- **节点 1（NUC 控制节点）**：网线直连 Franka 机械臂，并通过 USB 转 RS485 适配器连接 Robotiq 2F 夹爪（``ls /dev/ttyUSB*``，通常为 ``/dev/ttyUSB0``）。执行 ``sudo chmod 666 /dev/ttyUSB0`` 或 ``sudo usermod -aG dialout $USER``\ （需注销重登）赋予串口权限。

.. warning::

   测试夹爪或启动采集前，将 Franka 急停按钮放在伸手可及处，清空工作空间并让机械臂处于安全初始姿态。请安装 **ZED SDK 4.2.x**\ （如 v4.2.5）：ZED SDK 5.0+ 自带的 NumPy 与 Franka 跨节点序列化所需的 ``numpy<2`` 约束冲突。

.. _franka-zed-robotiq-data-collection:

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. 安装与硬件检查

      **节点 0（GPU 服务器）—— 安装 ZED SDK 4.2.x 并记录相机序列号**

      从 `Stereolabs 下载页面 <https://www.stereolabs.com/developers/release>`_ 下载匹配操作系统与 CUDA 版本的 ZED SDK 4.2.x（详见 `ZED Python API 官方安装指南 <https://www.stereolabs.com/docs/development/python/install>`_），运行安装程序并在提示安装 Python API 时按 **Y**：

      .. code-block:: bash

         # 以 CUDA 12.x + Ubuntu 22.04 为例
         chmod +x ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run
         ./ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run

      如需将 ``pyzed`` 安装到 RLinf 虚拟环境，在仓库根目录激活环境后运行自带脚本或直接安装 wheel：

      .. code-block:: bash

         source .venv/bin/activate
         python3 /usr/local/zed/get_python_api.py
         # 或直接安装 wheel：
         # uv pip install --reinstall /usr/local/zed/pyzed-*.whl

      查询已连接的 ZED 相机并记录序列号：

      .. code-block:: bash

         python -c "
         import pyzed.sl as sl
         for dev in sl.Camera.get_device_list():
             print(f'Serial: {dev.serial_number}  Model: {dev.camera_model}')
         "

      **节点 1（NUC）—— 安装 RLinf Franka 环境并验证 Robotiq 夹爪**

      ``franka`` 环境会自动安装 Modbus RTU 通信所需的 ``pymodbus``：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      赋予串口权限并在 NUC 上测试夹爪开合：

      .. code-block:: bash

         sudo chmod 666 /dev/ttyUSB0
         python -c "
         from rlinf.robotics.parts.end_effectors.grippers.robotiq import RobotiqGripper
         g = RobotiqGripper(port='/dev/ttyUSB0')
         g.connect()
         print(f'Position: {g.position:.4f} m, Ready: {g.is_ready()}')
         g.open()
         import time; time.sleep(1)
         g.close()
         g.disconnect()
         "

   .. tab-item:: 2. 数据采集与运行

      修改 ``examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml`` 中的必填硬件与目标位姿字段：

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: franka
               node_ranks: 0-1
               hardware:
                 type: Franka
                 configs:
                   - robot_ip: ROBOT_IP           # 机器人 IP 地址
                     camera_serials:
                       - "ZED_SERIAL_1"           # ZED 相机序列号
                     camera_type: zed
                     gripper_type: robotiq
                     gripper_connection: "/dev/ttyUSB0"
                     controller_node_rank: 1

         env:
           eval:
             override_cfg:
               target_ee_pose: TARGET_EE_POSE     # 目标末端执行器位姿

      在 **两个节点** 的 RLinf 仓库根目录（保持 ``.venv`` 已激活），导出节点环境变量并启动 Ray：

      .. code-block:: bash

         # 节点 0（GPU 服务器，Ray 头节点；按实际网络替换 IP 与网卡）
         export RLINF_NODE_RANK=0
         export RLINF_COMM_NET_DEVICES=eth0  # 可选，单网卡可省略
         ray start --head --port=6379 --node-ip-address=192.168.1.10

         # 节点 1（NUC 控制节点）
         export RLINF_NODE_RANK=1
         export RLINF_COMM_NET_DEVICES=eth0  # 可选，单网卡可省略
         ray start --address=192.168.1.10:6379

      在 **节点 0（头节点）** 启动采集并使用 SpaceMouse 遥操作：

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_collect_data_zed_robotiq

      若复制了配置文件，将参数替换为新配置名即可。数据保存在 ``logs/<时间戳>-<配置名>/`` 下的 ``demo_buffer/`` 与 ``lerobot_dataset/``，后续训练时把该路径填到 ``algorithm.demo_buffer.load_path``。

.. dropdown:: 完整双节点 YAML 与硬件/采集字段参考

   节点 0 采集 ZED 图像、节点 1 控制机械臂与夹爪的完整集群配置示例：

   .. code-block:: yaml

      cluster:
        num_nodes: 2
        component_placement:
          actor:
            node_group: gpu
            placement: 0
          env:
            node_group: franka
            placement: 0
          rollout:
            node_group: gpu
            placement: 0
        node_groups:
          - label: gpu
            node_ranks: 0
          - label: franka
            node_ranks: 0-1
            hardware:
              type: Franka
              configs:
                - robot_ip: <ROBOT_IP>
                  node_rank: 0
                  camera_serials:
                    - "<ZED_SERIAL_1>"
                    - "<ZED_SERIAL_2>"
                    - "<ZED_SERIAL_3>"
                  camera_type: zed            # "realsense" 或 "zed"
                  gripper_type: robotiq       # "franka" 或 "robotiq"
                  gripper_connection: "/dev/ttyUSB0"
                  controller_node_rank: 1     # 机械臂控制运行在 NUC 上
                  disable_validate: false

   **硬件配置字段**

   .. list-table::
      :header-rows: 1
      :widths: 25 15 60

      * - 字段
        - 默认值
        - 说明
      * - ``camera_type``
        - ``"realsense"``
        - 相机后端，使用 ZED 相机时设为 ``"zed"``。
      * - ``gripper_type``
        - ``"franka"``
        - 夹爪后端，使用 Robotiq 夹爪时设为 ``"robotiq"``。
      * - ``gripper_connection``
        - ``null``
        - Robotiq 串口路径（如 ``"/dev/ttyUSB0"``），``gripper_type`` 为 ``"franka"`` 时忽略。
      * - ``controller_node_rank``
        - ``null``
        - 运行机械臂控制的节点 rank。为 ``null`` 时与 env worker 同机部署；机械臂与相机分机时设为 ``1``。

   **数据采集字段**

   ``realworld_collect_data_zed_robotiq.yaml`` 中的主要采集字段：

   .. list-table::
      :header-rows: 1
      :widths: 30 18 52

      * - 字段
        - 默认值
        - 说明
      * - ``runner.num_data_episodes``
        - ``20``
        - 采集目标 episode 数，达到后自动退出。
      * - ``env.eval.teleop``
        - ``spacemouse``
        - 采集时驱动机械臂的遥操作设备。
      * - ``env.eval.data_collection.save_dir``
        - ``${runner.logger.log_path}``
        - 导出根目录（``{save_dir}/lerobot_dataset`` 与 ``{save_dir}/demo_buffer``）。
      * - ``env.eval.data_collection.export_format``
        - ``[replay_buffer, lerobot_dataset]``
        - 真机采集支持 ``replay_buffer``、``pickle_episode`` 与 ``lerobot_dataset``。
      * - ``env.eval.data_collection.only_success``
        - ``True``
        - 仅保存成功的 episode。

.. dropdown:: 集群配置注意事项

   - **节点 0（GPU 服务器）**：在执行 ``ray start`` 之前，先在虚拟环境中装好 ZED SDK 4.2.x 和 ``pyzed``。
   - **节点 1（NUC）**：在执行 ``ray start`` 之前，确认 ``pymodbus`` 已安装且 Robotiq 串口可读写。

   .. warning::

      Ray 会在 ``ray start`` 时捕获 Python 解释器和环境变量。启动 Ray 之后安装的 SDK 或依赖库对 Ray worker 不可见。务必先装完所有依赖再启动 Ray。

   多节点 Ray 集群完整配置见 :doc:`franka` 与 :doc:`../../guides/hetero`。

常见问题
--------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 解决方法
   * - 未检测到 ZED 相机
     - 检查 USB 3.0 数据线与相机指示灯，运行 ``lsusb`` 确认存在 ``Stereolabs`` 设备，并核对 ZED SDK 与 CUDA 版本是否匹配。
   * - 安装 ZED SDK 后出现 NumPy 冲突
     - 改用 **ZED SDK 4.2.x**\ （如 v4.2.5），避免 5.0+ 自带 NumPy 与 Franka 的 ``numpy<2`` 约束冲突。
   * - Robotiq 夹爪无响应
     - 用 ``ls /dev/ttyUSB*`` 确认串口存在，执行 ``sudo chmod 666 /dev/ttyUSB0`` 授权，检查 ``pymodbus`` 版本在 ``>=3.0,<4.0``，通信不稳定时可尝试将波特率降为 ``9600``。
   * - 控制器节点 rank 不匹配
     - 机械臂接在 NUC\ （node 1）而相机接在 GPU 服务器\ （node 0）时，必须在硬件配置中设置 ``controller_node_rank: 1``，否则控制器会在无机械臂连接的 node 0 上启动。
