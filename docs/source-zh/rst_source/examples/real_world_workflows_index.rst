通用流程
========

这些指南覆盖所有真机平台共用的基础流程：搭建真机控制端与 GPU 训练端的跨机 Ray 集群、配置遥操作设备，以及采集示范数据与 Replay Buffer。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 真机多机启动
      :link: ../guides/realworld_robot
      :link-type: doc

      把真机控制机与 GPU 训练机接入同一个 Ray 集群并启动运行。

   .. grid-item-card:: 遥操作设备配置
      :link: ../guides/teleoperation
      :link-type: doc

      配置 SpaceMouse、GELLO、主手或 VR 控制器，用于人工控制与在线介入。

   .. grid-item-card:: 真机数据采集
      :link: ../guides/data_collection
      :link-type: doc

      在遥操作或策略 Rollout 过程中录制 Replay Buffer 与 LeRobot 数据集。

.. toctree::
   :hidden:

   真机多机启动 <../guides/realworld_robot>
   遥操作设备配置 <../guides/teleoperation>
   真机数据采集 <../guides/data_collection>
