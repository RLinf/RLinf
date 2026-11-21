在 Franka 上使用灵巧手
================================================
.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dexhand.jpg
   :align: center
   :width: 80%

   搭配睿研五指灵巧手进行真机操作训练的 Franka 机械臂。

将 Franka 真机流程适配到睿研五指灵巧手。你将保留相同的集群与 reward-model 流程，并替换末端执行器、遥操作输入、动作布局和灵巧手配置。

概览
----------------------------------------

使用灵巧手末端执行器运行 Franka 真机流程。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN policy · reward model

   .. grid-item-card:: 算法
      :text-align: center

      SAC/RLPD · reward-model assist

   .. grid-item-card:: 任务
      :text-align: center

      Dexterous pick-and-place

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · Ruiyan dex hand · glove

| **你将完成:** 安装 Franka 依赖 → 安装灵巧手依赖 → 配置 glove/hand → 采集数据 → 训练.
| **前置条件:** :doc:`franka` · :doc:`franka_reward_model` · Ruiyan hand driver · glove device.

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - 任务
     - 配置 / 入口
     - 说明
   * - Collection
     - ``realworld_collect_dexhand_data``
     - 采集灵巧手示教。
   * - Training
     - ``realworld_dexpnp_rlpd_cnn_async``
     - 用灵巧手动作训练 CNN policy。
   * - Reward model
     - Franka reward-model workflow
     - 复用 reward-model 路径完成灵巧操作。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - 字段
     - 说明
   * - Observation
     - Franka 相机帧，以及配置中的灵巧手/glove 状态。
   * - Action
     - 机械臂动作加灵巧手关节或命令向量。
   * - Reward
     - 任务完成信号或 reward-model 预测。
   * - Prompt
     - 来自真机 env config 的任务文本。

安装
----------------------------------------

基础 Franka 控制器与实时内核准备参见 :doc:`franka`。

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Franka 控制节点

      ``franka`` 环境已包含 ``RLinf-dexterous-hands``，提供睿研灵巧手与数据手套驱动：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

遥操作
----------------------------------------

灵巧手遥操作使用：

- SpaceMouse 控制机械臂 6 维位姿
- 数据手套控制 6 维手指动作
- SpaceMouse 左键用于启用相对手套控制

奖励模型
------------

reward model 流程与 :doc:`franka_reward_model` 一致。reward 图像默认沿用 ``env.main_image_key``\ （在 ``examples/embodiment/config/env/realworld_dex_pnp.yaml`` 中默认为 ``wrist_1``），``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` 通过 ``reward`` 字段接入 reward model。

配置文件
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - 配置文件
     - 用途
   * - ``examples/embodiment/config/realworld_collect_dexhand_data.yaml``
     - 数据采集，包含 ``end_effector_type: "ruiyan_hand"``、数据手套遥操作参数，以及同时写出 ``replay_buffer``\ （``demo_buffer/``）和 ``pickle_episode``\ （``pickle_episode/``）的 ``data_collection`` 配置。
   * - ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml``
     - 异步 RLPD CNN 训练，启用 ``reward`` 配置。
   * - ``examples/embodiment/config/env/realworld_dex_pnp.yaml``
     - 灵巧手抓放环境默认配置（``main_image_key`` 默认为 ``wrist_1``）。

启动 ``realworld_dexpnp_rlpd_cnn_async.yaml`` 前，需要填写 ``robot_ip``、``target_ee_pose``、policy ``model_path``、reward ``model.model_path``，以及 ``end_effector_config`` 和 ``glove_config`` 中的串口参数。

需要自定义相机命名或 crop 时，直接在 ``override_cfg`` 中按序列号填写。未配置 ``camera_names`` 时，默认命名按 ``camera_serials`` 列表顺序分配：第一个序列号为 ``wrist_1``，第二个为 ``wrist_2``，不按序列号排序。例如：

.. code-block:: yaml

   camera_names:
     "SERIAL1": global
     "SERIAL2": wrist_1
   camera_crop_regions:
     "SERIAL1": [0.4, 0.3, 0.85, 0.7]

如果将某个相机命名为 ``global``，同时把任务 YAML 中的 ``main_image_key`` 改为 ``global``。

运行
----------------------------------------

1. 在 Franka 控制节点激活上方安装的 ``franka`` 环境（已包含睿研灵巧手与数据手套驱动 ``RLinf-dexterous-hands``）。
2. 将 Franka 机器人切换到可编程模式，手动移动到任务目标位姿，然后在 Franka 控制节点运行脚本获取目标末端位姿：

   .. code-block:: bash

      python -m toolkits.realworld_check.test_franka_controller \
        --robot-ip <FRANKA_IP> \
        --end-effector-type ruiyan_hand \
        --hand-port /dev/ttyUSB0

   脚本启动后输入 ``getpos_euler``，记录输出的欧拉角位姿，并填入配置中的 ``target_ee_pose``。
3. 在 Franka 控制节点配好采集任务参数，包括 ``robot_ip``、``target_ee_pose``、``end_effector_config``、``glove_config`` 等。
4. 在 Franka 控制节点采集专家 demo：

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh realworld_collect_dexhand_data

5. 在 Franka 控制节点使用同一个入口再次采集 reward 原始 episode；这一轮建议调大 ``env.eval.override_cfg.success_hold_steps``，并使用单独的日志目录。
6. 将 reward 原始数据从 Franka 控制节点传到训练节点，或者提前写入共享存储。
7. 在训练节点按照 :doc:`franka_reward_model` 中的方法，用 ``examples/reward/preprocess_reward_dataset.py`` 生成 reward dataset。
8. 在训练节点使用 ``examples/reward/run_reward_training.sh`` 训练 reward model。
9. 在启动 RL 之前，按照 :doc:`franka` 的集群配置说明，启动由训练节点和 Franka 控制节点组成的双机 Ray 集群。
10. 在训练节点启动 RL：

   .. code-block:: bash

      bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async
