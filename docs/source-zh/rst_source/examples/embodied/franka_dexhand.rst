Franka + 灵巧手真机强化学习
================================

本文档说明 Franka 机械臂接入睿研灵巧手时需要关注的配置差异。
完整的真机强化学习与 reward model 工作流请参考 :doc:`franka` 和 :doc:`franka_reward_model`。

.. contents:: 目录
   :local:
   :depth: 2

总览
----

灵巧手方案沿用与 Franka 相同的真机强化学习和 reward model 工作流，主要差异集中在末端执行器、遥操作方式和动作空间：

- 动作空间为 12 维
- 前 6 维控制机械臂位姿增量
- 后 6 维控制灵巧手关节
- ``RuiyanHand`` 负责灵巧手硬件控制
- ``DexHandIntervention`` 将 SpaceMouse 和数据手套输入组合为专家动作

遥操作
------

灵巧手遥操作使用：

- SpaceMouse 控制机械臂 6 维位姿
- 数据手套控制 6 维手指动作
- SpaceMouse 左键用于启用相对手套控制

Reward Model
------------

reward model 侧与 :doc:`franka_reward_model` 中的 Franka 真机流程一致。

对当前灵巧手抓放环境：

- reward 图像默认沿用 ``env.main_image_key``
- ``examples/embodiment/config/env/realworld_dex_pnp.yaml`` 中的 ``main_image_key`` 默认为 ``wrist_1``
- ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` 通过 ``reward`` 段接入 reward model

配置文件
--------

数据采集使用 ``examples/embodiment/config/realworld_collect_dexhand_data.yaml``。
该配置包含：

- ``end_effector_type: "ruiyan_hand"``
- 数据手套遥操作参数
- ``data_collection``，用于以 ``pickle`` 格式导出原始 episode

RL 训练使用 ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml``。
启动前需要填写：

- ``robot_ip``
- ``target_ee_pose``
- 策略 ``model_path``
- reward ``model.model_path``
- ``end_effector_config`` 与 ``glove_config`` 中的串口参数

如果需要自定义相机命名或 crop，请直接在 ``override_cfg`` 中按序列号 serial
填写；本 PR 默认不提交任何特定 serial 的配置，避免影响其他项目。不配置
``camera_names`` 时，默认命名会按照 ``camera_serials`` 列表顺序分配：第一个
serial 是 ``wrist_1``，第二个 serial 是 ``wrist_2``，不会按序列号排序。例如：

.. code-block:: yaml

   camera_names:
     "SERIAL1": global
     "SERIAL2": wrist_1
   camera_crop_regions:
     "SERIAL1": [0.4, 0.3, 0.85, 0.7]

如果你把某个相机命名成 ``global``，记得同时把任务 YAML 中的
``main_image_key`` 改成 ``global``。

工作流
------

1. 在 Franka 控制节点安装 Franka DexHand 环境：

   .. code-block:: bash

      bash requirements/install.sh embodied --env franka-dexhand

   该命令会安装 Franka 基础依赖和 ``RLinf-dexterous-hands``，后者包含睿研灵巧手与数据手套驱动。
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
