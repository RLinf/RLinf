Robotics 架构
=============

了解 RLinf 如何抽象真机硬件、管理连接生命周期与远程 Placement，以及如何在机器人之上封装任务环境。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Robotics 接口
      :link: ../robotics
      :link-type: doc

      通过命名部件树读取观测并下发控制动作。

   .. grid-item-card:: Robotics 内部架构
      :link: ../robotics_architecture
      :link-type: doc

      了解共享连接、组件生命周期与跨机远程 Placement。

   .. grid-item-card:: 真机任务与环境
      :link: ../realworld_envs
      :link-type: doc

      了解任务、遥操作与环境 Wrapper 如何围绕机器人分层组装。

.. toctree::
   :hidden:

   Robotics 接口 <../robotics>
   Robotics 内部架构 <../robotics_architecture>
   真机任务与环境 <../realworld_envs>
