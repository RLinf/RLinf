数据采集 (真实世界)
====================

本文档介绍如何在 RLinf 中进行真机数据采集.
真机数据采集通常用于 RLPD (Reinforcement Learning from Prior Data) 实验，
通过人工干预 (如 SpaceMouse) 获取初始成功轨迹，为后续训练提供先验数据。

概述
----

真机数据采集系统通过 examples/embodiment/collect_data.sh 脚本启动，
与仿真环境的大规模并行采集不同，真机采集通常在单台控制节点上运行，
支持人类通过 SpaceMouse 进行实时干预。该系统的核心特性是：
支持设定目标成功次数，脚本会持续运行直到采集到指定数量的成功 Episode。

配置
----

真机采集通过 examples/embodiment/config/realworld_collect_data.yaml 进行配置：

配置参数
~~~~~~~~

.. list-table:: 真机数据采集核心配置
   :header-rows: 1
   :widths: 25 15 60

   * - 参数
     - 默认值
     - 说明
   * - ``runner.num_data_episodes``
     - ``20``
     - 需要采集的 **成功** 轨迹数量。脚本达到该数量后停止。
   * - ``cluster.node_groups.hardware.configs.robot_ip``
     - 无
     - Franka 机器人的 IP 地址。
   * - ``env.eval.use_spacemouse``
     - ``True``
     - 是否启用 SpaceMouse 进行人工干预。
   * - ``env.eval.override_cfg.target_ee_pose``
     - 无
     - 任务的目标末端位姿 [x, y, z, rx, ry, rz]。
   * - ``env.eval.override_cfg.success_hold_steps``
     - ``1``
     - 被判定为一次成功轨迹所需要保持的到达目标末端位姿态的step数。

数据格式
--------

文件命名
~~~~~~~~

一次数据采集的会生成一个 pickle (``.pkl``) 文件。
里面将本次数据采集的所有 episode 依次拼接并储存。
文件存放在 logs 中，通常为：``data.pkl``

数据结构
~~~~~~~~

真机采集的 pickle 文件包含一个列表。列表中的每一个元素代表一个时间步 (Step) 的数据，
pickle 文件包含一个字典：

.. code-block:: python

  {
    "transitions": 
    {
      "obs": 
      {
        "states"        # 机器人状态 (shape=[19], 包含位姿、力矩等)
        "main_images"   # 腕部摄像头图像 (shape=[128, 128, 3])
      },
      "next_obs": 
      {
        "states"        # 下一步的机器人状态
        "main_images"   # 下一步的图像观测
      }
    },
    "action"            # 当前步采取的动作 (shape=[6], 对应控制模式输出)
    "rewards"           # 奖励值 (shape=[1], 成功通常为 1.0)
    "dones"             # 是否结束 (shape=[1], bool)
    "terminations"      # 任务是否终止 (shape=[1], bool)
    "truncations"       # 是否达到最大步数截断 (shape=[1], bool)
  }

使用方法
--------

1. 环境准备
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
在控制器节点上，需要 source 虚拟环境以及相关的 ROS/Robot 控制脚本：

.. code-block:: bash

  source <path_to_your_venv>/bin/activate
  source <your_catkin_ws>/devel/setup.bash

2. 配置机器人与任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
修改 `examples/embodiment/config/realworld_collect_data.yaml`。
确保 `robot_ip` 正确，设置好目标位姿 `target_ee_pose`。
`target_ee_pose` 可以通过 `toolkits/realworld_check/test_controller.py` 获取：

3. 启动采集脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
执行采集脚本：

.. code-block:: bash

  bash examples/embodiment/collect_data.sh

4. 人工干预
~~~~~~~~~~~~~~~~
脚本启动后，机器人会复位。当屏幕中出现摄像头画面的时候，你可以通过 SpaceMouse 操控机器人完成任务。
当机器人检测到成功（进入目标区域并维持一定步数）时，成功计数器会增加。

注意：在检测到成功后，在机器人复位期间，摄像头画面停止更新是正常现象。

5. 获取数据
~~~~~~~~~~~~~~~~
当成功次数达到 `num_data_episodes` 后，数据会自动保存到：
``logs/[running-timestamp]/data.pkl``

6. 上传数据
~~~~~~~~~~~~~~~~
采集完成后，将 data.pkl 上传至 GPU 训练服务器，并在训练配置中指向该路径以启动 RLPD 训练。
