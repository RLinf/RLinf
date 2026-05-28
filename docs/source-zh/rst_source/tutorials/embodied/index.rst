具身智能
=========

本节专注于使用 RLinf 进行具身强化学习训练，涵盖数据管理、
真机部署、云边训练和奖励模型工作流。

- :doc:`replay_buffer`
   TrajectoryReplayBuffer 的使用方式、采样流程和存储实践。

- :doc:`data_collection`
   数据采集的配置、输出格式，以及在仿真和真机场景下的使用方法。

- :doc:`realworld_robot`
   将多台 Franka 机器人和 GPU 训练节点连接到同一 Ray 集群，
   配置 YAML 并启动真机 RL 训练。

- :doc:`cloud_edge`
   使用 EasyTier 构建云边训练环境，将云端和边缘节点连接在
   同一覆盖网络上，并在其上运行 RLinf。

- :doc:`reward_model`
   RLinf 仿真环境中完整的奖励模型工作流，
   包括数据采集、离线预处理、训练和 RL 推理。

- :doc:`reward_model_realworld`
   真机奖励模型数据采集及其与 Franka 流水线的集成，
   涵盖遥操作、在线推理以及使用学习奖励进行 RL 训练。


.. toctree::
   :hidden:
   :maxdepth: 1

   replay_buffer
   data_collection
   realworld_robot
   cloud_edge
   reward_model
   reward_model_realworld
