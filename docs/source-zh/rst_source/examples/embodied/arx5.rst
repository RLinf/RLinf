ARX X5
====================================

.. figure:: https://arx-global.us/assets/img/x5.webp
   :align: center
   :width: 80%
   :alt: ARX X5 机械臂

   ARX X5 机械臂。

本指南帮助你使用两只 PICO 手柄和三路 V4L2 RGB 相机采集双臂 ARX X5 示教数据。你将安装运行环境，检查 CAN 与相机，配置工位参数，完成 episode 采集并检查导出的 LeRobot dataset。

概览
----

ARX X5 数采配置在一台 Ubuntu 24.04 主机上运行双臂控制、三路相机和 PICO 遥操。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 任务
      :text-align: center

      双臂 PICO 操作

   .. grid-item-card:: 硬件
      :text-align: center

      2× ARX X5 · PICO · 3× RGB 相机

| **工作概览：** 安装 ARX X5 运行环境 → 检查硬件 → 配置工位 → 采集 episode → 检查 dataset。
| **前置条件：** Ubuntu 24.04 · 两台 ARX X5 及 CAN 适配器 · 三台 V4L2 相机 · PICO 头显与双手柄。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 内容
   * - Observation
     - 一路 224×224 RGB 主视角、两路 224×224 RGB 腕部视角、按 ``[xyz, rot6d]`` 排列的 18 维双臂 TCP 位姿，以及两个归一化夹爪位置。
   * - Action
     - 20 维绝对目标，顺序为 ``[left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]``。
   * - Reward
     - 按 PICO ``X`` 以 reward 1 结束 episode；按 PICO ``A`` 以 reward 0 结束。
   * - Prompt
     - ``env.eval.override_cfg.task_description`` 设置任务文本，LeRobot 导出会将其写入 episode metadata。

安装运行环境
------------

克隆 RLinf 仓库并进入仓库目录：

.. code-block:: bash

   # 如需使用镜像，可用下一行替换官方克隆命令：
   # git clone https://gh-proxy.com/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

在同一个 Python 环境中安装 RLinf 的 ARX X5、相机和 LeRobot 依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --env arx5
   source .venv/bin/activate

该命令安装 ``requirements/embodied/envs/arx5.txt`` 中列出的依赖。当前 Python SDK 需要从 `ARXroboticsX/ARX5_beta <https://github.com/ARXroboticsX/ARX5_beta>`_ 单独安装，其仓库根目录直接包含 RLinf 使用的 ``bimanual`` package。每次使用 ARX X5 前，先激活 Python venv，再把克隆的 ``ARX5_beta`` 根目录加入 ``PYTHONPATH``：

.. code-block:: bash

   export PYTHONPATH="/path/to/ARX5_beta${PYTHONPATH:+:$PYTHONPATH}"

准备 ARX X5 硬件
-----------------

启用 collector 前，按 `ARX5_beta 说明 <https://github.com/ARXroboticsX/ARX5_beta/blob/main/readme-CN.md>`_ 运行 ``ARX_CAN_doctor``，将左右臂分别配置为 ``can1`` 和 ``can3``，然后启动两路 CAN。

查找三路相机的稳定 V4L2 路径：

.. code-block:: bash

   ls -l /dev/v4l/by-path/

打开 ``examples/embodiment/config/collect_data_dual_arx5.yaml``，按实际接线填写左右臂的 CAN 接口和每路视角相机的 V4L2 路径。

.. code-block:: yaml

   left_can_port: can1  # 左臂 CAN 接口
   right_can_port: can3  # 右臂 CAN 接口
   arm_type: 0  # X5-2023 款
   camera_fps: 30  # 选相机支持帧率
   base_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.4:1.0-video-index0']  # 主视角
   left_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7:1.0-video-index0']  # 左腕
   right_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.3:1.0-video-index0']  # 右腕

如果 OpenCV 报错 ``Could not open V4L2 camera``，检查设备属主和当前用户的附加组：

.. code-block:: bash

   ls -l /dev/video*
   id -nG
   sudo usermod -aG video "$USER"

退出并重新登录后，确认 ``id -nG`` 的输出包含 ``video``。如果已启动 Ray，请在新会话中重启，使 worker 继承新的组权限。

准备 PICO 头显
--------------

请按 :ref:`PICO / XRoboToolkit 配置 <pico-software-setup>` 中的头显安装步骤配置 APK，并确认两只手柄都能正常跟踪。

开始采集
--------

启动前，在 ``examples/embodiment/config/collect_data_dual_arx5.yaml`` 中修改以下字段

.. code-block:: yaml

   runner:
     num_data_episodes: 50  # 需要采集的成功示教数
   env:
     eval:
       override_cfg:
         step_frequency: 30  # 控制频率，单位 Hz
         task_description: Pick and place the objects with both arms.  # 填写要写入数据集的任务指令
       data_collection:
         fps: 30  # 数据集帧率，与控制频率和相机帧率对齐

激活 Python 环境并设置好 SDK 的 ``PYTHONPATH`` 后，在仓库根目录执行以下命令：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   bash examples/embodiment/collect_data.sh collect_data_dual_arx5

机械臂复位后，按 **B** 开始录制，成功时按 **X** 结束并保存本次示教，失败时按 **A** 结束并丢弃。

检查 Dataset
------------

成功 episode 会以 LeRobot shard 保存到：

.. code-block:: text

   logs/<timestamp>-collect_data_dual_arx5/
   └── collected_data/
       └── lerobot_dataset/
           └── rank_0/
               └── id_0/

合并当次运行中的所有 shard，再导出逐帧图片、文本 metadata 和 MP4 预览：

.. code-block:: bash

   RUN=logs/<timestamp>-collect_data_dual_arx5

   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "$RUN/collected_data" \
       --output-dir "$RUN/lerobot_merged"

   python toolkits/lerobot/visualize_lerobot_dataset.py \
       --dataset-path "$RUN/lerobot_merged" \
       --output-dir "$RUN/formatted_samples" \
       --export-mp4 --mp4-fps 30

合并后的 dataset 位于 ``$RUN/lerobot_merged``，图片、文本与 MP4 预览位于 ``$RUN/formatted_samples``。
