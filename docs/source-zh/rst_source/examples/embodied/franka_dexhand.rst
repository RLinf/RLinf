Franka 灵巧手
=============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dexhand.jpg
   :align: center
   :width: 80%

   搭配睿研五指灵巧手用于真机操作的 Franka 机械臂。

概览
----

将 Franka 真机 RLPD 与奖励模型流程适配到睿研五指灵巧手。本方案沿用相同的双机集群与奖励模型流程，仅替换末端执行器、SpaceMouse + 数据手套遥操作输入、动作空间和灵巧手任务配置。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      Franka Emika Panda + 睿研五指灵巧手

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      RealSense · SpaceMouse · 数据手套

   .. grid-item-card:: 流程
      :text-align: center

      数据采集 → 奖励模型 → 在线 RLPD

   .. grid-item-card:: 硬件
      :text-align: center

      控制节点 + NVIDIA GPU 节点

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - 阶段
     - 配置
     - 说明
   * - 示教采集
     - ``realworld_collect_dexhand_data``
     - 使用 SpaceMouse 与数据手套采集灵巧手抓放示教。
   * - 奖励模型
     - :doc:`franka_reward_model`
     - 预处理原始轨迹并训练用于灵巧操作的 ResNet 奖励模型。
   * - 强化学习
     - ``realworld_dexpnp_rlpd_cnn_async``
     - 结合奖励模型异步训练机械臂 + 灵巧手 CNN policy。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - Franka 相机帧（默认为 ``wrist_1``），以及配置后的灵巧手与数据手套状态。
   * - 动作
     - 6 维 Franka 机械臂笛卡尔动作 + 6 维睿研灵巧手手指控制指令。
   * - 奖励
     - 奖励模型预测（读取 ``env.main_image_key`` 视角）或目标位姿判定。
   * - 提示词
     - ``examples/embodiment/config/env/realworld_dex_pnp.yaml`` 中的任务文本。

硬件与安全准备
--------------

准备装配睿研五指灵巧手（通过串口如 ``/dev/ttyUSB0`` 连接）的 Franka 机械臂、配套数据手套、SpaceMouse、Intel RealSense 相机、Franka 控制节点与 NVIDIA GPU 训练节点。先按 :doc:`franka` 完成固件检查、实时内核配置与 FCI 激活，并参考 :doc:`franka_reward_model` 了解奖励模型流程。

.. warning::

   每次真机运行均须由操作员全程监控并保持急停按钮触手可及。开始遥操作或自主 RL 前清空工作区域，确认当前用户对灵巧手与数据手套串口（``/dev/ttyUSB*``）有读写权限，并检查手指闭合限位。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Franka + 灵巧手

      **1. 安装依赖**

      在 Franka 控制节点（及 GPU 训练节点）安装 ``franka`` 环境；该环境已内置 ``RLinf-dexterous-hands``，提供睿研灵巧手与数据手套驱动：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. 遥操作与目标位姿配置**

      遥操作时，SpaceMouse 控制机械臂 6 维位姿，数据手套控制 6 维手指动作，按住 SpaceMouse 左键启用相对手套控制。

      将 Franka 切换至引导模式并手动移到目标位姿，在 Franka Desk 中启用 FCI，运行控制器检查脚本并输入 ``getpos_euler`` 读取目标位姿：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_franka_controller \
           --robot-ip 192.168.1.10 \
           --end-effector-type ruiyan_hand \
           --hand-port /dev/ttyUSB0

      在 ``examples/embodiment/config/realworld_collect_dexhand_data.yaml`` 与 ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` 中填写 ``robot_ip``、``target_ee_pose``、``end_effector_config`` 与 ``glove_config``\ （串口参数），并在 RL 配置中补充 policy ``model_path`` 和奖励模型 ``model.model_path``。

      **3. 数据采集、训练奖励模型并启动 RL**

      在 Franka 控制节点先采集专家示教，再用独立日志目录和更大的 ``env.eval.override_cfg.success_hold_steps`` 采集奖励模型原始轨迹：

      .. code-block:: bash

         # 采集专家示教
         bash examples/embodiment/collect_data.sh realworld_collect_dexhand_data

         # 采集奖励模型原始轨迹（增大 success_hold_steps 并使用独立 RLINF_LOG_DIR）
         RLINF_LOG_DIR="$PWD/logs/dexhand-reward-raw" \
           bash examples/embodiment/collect_data.sh realworld_collect_dexhand_data \
           "env.eval.override_cfg.success_hold_steps=20"

      将奖励原始数据复制到训练节点（或提前写入共享存储），按 :doc:`franka_reward_model` 使用 ``examples/reward/preprocess_reward_dataset.py`` 预处理并用 ``examples/reward/run_reward_training.sh`` 训练奖励模型。

      ``realworld_dexpnp_rlpd_cnn_async.yaml`` 默认使用 **2 节点**\ （``cluster.num_nodes: 2``：GPU 训练节点为 head，Franka 控制节点为 worker）。按 :doc:`franka` 启动 2 节点 Ray 集群后，在训练节点启动异步 RLPD：

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async

      .. dropdown:: 进阶配置（可选）

         **自带配置文件**

         .. list-table::
            :header-rows: 1
            :widths: 52 48

            * - 配置文件
              - 用途
            * - ``examples/embodiment/config/realworld_collect_dexhand_data.yaml``
              - 数据采集配置，含 ``end_effector_type: "ruiyan_hand"``、手套遥操作参数，以及同时导出 ``replay_buffer``\ （``demo_buffer/``）和 ``pickle_episode``\ （``pickle_episode/``）。
            * - ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml``
              - 异步 RLPD CNN 训练配置，已启用 ``reward`` 段。
            * - ``examples/embodiment/config/env/realworld_dex_pnp.yaml``
              - 灵巧手抓放环境默认参数（``env.main_image_key`` 默认为 ``wrist_1``）。

         **相机命名与裁剪区域**

         未配置 ``camera_names`` 时，相机按 ``camera_serials`` 列表顺序命名（依次为 ``wrist_1``、``wrist_2``，不按序列号排序）。如需在 ``override_cfg`` 中自定义命名或裁剪区域：

         .. code-block:: yaml

            camera_names:
              "SERIAL1": global
              "SERIAL2": wrist_1
            camera_crop_regions:
              "SERIAL1": [0.4, 0.3, 0.85, 0.7]

         如果将主视角命名为 ``global``，需同步将任务 YAML 中的 ``main_image_key`` 改为 ``global``，以便奖励模型读取同一视角。

结果与常见问题
--------------

在 TensorBoard（``tensorboard --logdir ./logs``）中监控 ``env/success_once``、奖励模型预测与 SAC loss；完整日志选项见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 现象
     - 解决方法
   * - 打开 ``/dev/ttyUSB0`` 报权限不足
     - 将当前用户加入 ``dialout`` 组（``sudo usermod -aG dialout "$USER"``）并重新登录，同时核对 ``end_effector_config`` / ``glove_config`` 中的串口路径。
   * - 灵巧手手指不跟随数据手套运动
     - 按住 SpaceMouse 左键启用相对手套控制，并确认数据手套串口与 ``glove_config`` 一致。
   * - 奖励模型读取了错误的相机视角
     - 确认 ``env.main_image_key`` 与相机名称一致（在 ``camera_names`` 中改名为 ``global`` 后须同步修改 ``main_image_key``）。
