ARX X5 数据采集
===============

.. figure:: https://arx-global.us/assets/img/x5.webp
   :align: center
   :width: 80%
   :alt: ARX X5 机械臂

   ARX X5 机械臂。

在一台 Ubuntu 24.04 主机上用两只 PICO 手柄和三路 V4L2 RGB 相机采集双臂 ARX X5 示教数据，并合并与预览导出的 LeRobot 数据集。

概览
----

本示例在单台 Ubuntu 24.04 主机上同时运行双臂 ARX X5 控制、三路 V4L2 RGB 相机与 PICO 遥操作，将成功回合直接保存为 LeRobot shard。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      2× ARX X5（``ARX5_beta`` SDK）

   .. grid-item-card:: 遥操作与相机
      :text-align: center

      PICO 双手柄 · 3 路 V4L2 RGB 相机

   .. grid-item-card:: 工作流
      :text-align: center

      PICO 数据采集 → 合并 LeRobot → 可视化预览

   .. grid-item-card:: 硬件
      :text-align: center

      1 台 Ubuntu 24.04 主机 · 2 个 CAN 适配器

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 观测
     - 1 路 224×224 RGB 主视角、2 路 224×224 RGB 腕部视角、按 ``[xyz, rot6d]`` 排列的 18 维双臂 TCP 位姿，以及 2 个归一化夹爪位置。
   * - 动作
     - 20 维绝对目标，顺序为 ``[left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]``。
   * - 奖励
     - 按 PICO ``X`` 以 reward ``1`` 结束并保存回合；按 PICO ``A`` 以 reward ``0`` 结束并丢弃。
   * - 提示词
     - ``env.eval.override_cfg.task_description`` 写入 LeRobot 回合元数据。

硬件与安全准备
--------------

将两台 ARX X5 通过 CAN 适配器接入主机，连接顶部与双腕三路 V4L2 RGB 相机，并按 :ref:`PICO / XRoboToolkit 配置 <pico-software-setup>` 安装头显 APK、确认两只手柄追踪正常。

.. warning::

   启用机械臂前，把急停开关放在触手可及的位置并清空双臂工作空间。确认双臂处于安全起始姿态、线束在全行程内无拉扯，并确保复位路径无障碍物。

启动 RLinf 前，按 `ARX5_beta 说明 <https://github.com/ARXroboticsX/ARX5_beta/blob/main/readme-CN.md>`_ 运行 ``ARX_CAN_doctor``，将左右臂分别映射为 ``can1`` 与 ``can3`` 并拉起两路 CAN 接口。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. 安装与硬件检查

      **安装 RLinf 与 ARX5_beta SDK**\ ：安装 ``arx5`` 运行环境（依赖清单见 ``requirements/embodied/envs/arx5.txt``），单独克隆 `ARXroboticsX/ARX5_beta <https://github.com/ARXroboticsX/ARX5_beta>`_\ （其仓库根目录提供 ``bimanual`` 包），并在每次会话中将其加入 ``PYTHONPATH``：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env arx5 && source .venv/bin/activate
         export PYTHONPATH="/path/to/ARX5_beta${PYTHONPATH:+:$PYTHONPATH}"

      **查询稳定的 V4L2 相机路径**\ ，并在 ``examples/embodiment/config/collect_data_dual_arx5.yaml`` 中填写 CAN 接口、三路相机路径以及按视角顺时针旋转角度 ``camera_rotations``\ （``0`` / ``90`` / ``180`` / ``270``）：

      .. code-block:: bash

         ls -l /dev/v4l/by-path/

      .. code-block:: yaml

         left_can_port: can1  # 左臂 CAN 接口
         right_can_port: can3  # 右臂 CAN 接口
         arm_type: 0  # X5-2023 款
         camera_fps: 30  # 三路相机均支持的帧率
         base_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.4:1.0-video-index0']
         left_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7:1.0-video-index0']
         right_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.3:1.0-video-index0']
         camera_rotations: {base_0_rgb: 180}

   .. tab-item:: 2. 采集与合并数据集

      **设置目标回合数、频率与任务描述**\ （编辑 ``examples/embodiment/config/collect_data_dual_arx5.yaml``）：

      .. code-block:: yaml

         runner:
           num_data_episodes: 50
         env:
           eval:
             override_cfg:
               step_frequency: 30
               task_description: Pick and place the objects with both arms.
             data_collection:
               fps: 30

      **启动数据采集**\ （激活 ``.venv`` 且配置好 ``ARX5_beta`` 的 ``PYTHONPATH`` 后，在仓库根目录执行）：

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh collect_data_dual_arx5

      **PICO 手柄按键：** 双臂复位后，按 **B** 开始录制，成功时按 **X** 结束并保存，失败时按 **A** 结束并丢弃。

      **合并分片并导出 MP4 预览**\ （原始分片位于 ``logs/<timestamp>-collect_data_dual_arx5/collected_data/lerobot_dataset/rank_0/id_0/``）：

      .. code-block:: bash

         RUN=$(ls -td logs/*-collect_data_dual_arx5 | head -n 1)

         python -m toolkits.lerobot.merge_lerobot_datasets \
             --source-dir "$RUN/collected_data" \
             --output-dir "$RUN/lerobot_merged"

         python -m toolkits.lerobot.visualize_lerobot_dataset \
             --dataset-path "$RUN/lerobot_merged" \
             --output-dir "$RUN/formatted_samples" \
             --export-mp4 --mp4-fps 30

结果与常见问题
--------------

合并后的数据集保存在 ``$RUN/lerobot_merged``，逐帧图像、元数据与 MP4 预览保存在 ``$RUN/formatted_samples``。训练前先抽查导出的 MP4 视频，确认相机朝向与示教质量。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理方法
   * - OpenCV 报错 ``Could not open V4L2 camera``
     - 检查 ``ls -l /dev/video*`` 与 ``id -nG``，执行 ``sudo usermod -aG video "$USER"``，退出并重新登录后在新终端中启动，使 worker 继承 ``video`` 组权限。
   * - 报错找不到 ``bimanual`` 模块
     - 在执行 ``collect_data.sh`` 前导出 ``PYTHONPATH="/path/to/ARX5_beta${PYTHONPATH:+:$PYTHONPATH}"``。
   * - 录制画面倒置或侧偏
     - 在 ``collect_data_dual_arx5.yaml`` 中为对应视角设置顺时针 ``camera_rotations``\ （``0`` / ``90`` / ``180`` / ``270``）。
   * - CAN 接口打开失败或机械臂无响应
     - 重新运行 ``ARX5_beta`` 的 ``ARX_CAN_doctor``，确认 ``can1`` 与 ``can3`` 已启动且与配置中的 ``left_can_port`` / ``right_can_port`` 一致。

下一步：训练 OpenPI SFT
-----------------------

抽查导出的 MP4 预览确认画面和轨迹正常后，可按 :doc:`sft_openpi` 中的 **自定义 LeRobot 数据集** 标签页在 ``$RUN/lerobot_merged`` 上训练 OpenPI policy：

1. 按三路 RGB 视角和 20 维双臂 ``[xyz, rot6d, gripper]`` 状态与动作空间调整 ``pi0_custom``\ （或注册专用 ARX X5 dataconfig）。
2. 使用相同的 dataconfig 名称和 ``$RUN/lerobot_merged`` 运行 ``toolkits/lerobot/calculate_norm_stats.py`` 生成 ``norm_stats.json``。
3. 参照 :doc:`sft_openpi` 启动 ``examples/sft/run_vla_sft.sh custom_sft_openpi``，并通过命令行覆盖 ``data.train_data_paths``、``actor.model.model_path`` 与 ``actor.model.openpi_data.norm_stats_path``。
