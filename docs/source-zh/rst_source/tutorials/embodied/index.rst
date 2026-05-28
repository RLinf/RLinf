具身智能
=========

本节专注于使用 RLinf 进行具身强化学习训练，涵盖支持的环境与模型、
真机部署、数据管理和奖励模型工作流。

- :doc:`supported_envs`
   支持的模拟器（ManiSkill、LIBERO、IsaacLab 等）、
   真机平台（Franka、XSquare Turtle2 等）和
   VLA/WAM 模型（OpenVLA、π₀、GR00T 等）概览。

- :doc:`realworld_robot`
   将多台 Franka 机器人和 GPU 训练节点连接到同一 Ray 集群，
   配置 YAML 并启动真机 RL 训练。

- :doc:`cloud_edge`
   使用 EasyTier 构建云边训练环境，将云端和边缘节点连接在
   同一覆盖网络上，并在其上运行 RLinf。

- :doc:`replay_buffer`
   TrajectoryReplayBuffer 的使用方式、采样流程和存储实践。

- :doc:`data_collection`
   数据采集的配置、输出格式，以及在仿真和真机场景下的使用方法。

- :doc:`reward_model`
   Reward Model 完整使用指南，涵盖仿真和真机工作流：
   数据采集、预处理、训练、RL 推理以及带实时 reward 反馈的遥操作。


.. toctree::
   :hidden:
   :maxdepth: 1

   supported_envs
   realworld_robot
   cloud_edge
   replay_buffer
   data_collection
   reward_model
