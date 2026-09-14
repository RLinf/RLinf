概念
====

概念页面介绍 RLinf 如何执行任务、调度组件和组织机器人硬件。请从与当前问题最相关的主题开始阅读；只有在扩展功能或排查底层问题时，才需要继续了解对应的架构细节。

选择概念主题
------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 执行
      :link: execution-model/index
      :link-type: doc

      了解任务执行流程，以及 worker、cluster、channel 与 collective 之间的关系。

   .. grid-item-card:: 调度
      :link: scheduling-model/index
      :link-type: doc

      了解 placement、执行模式与 replay buffer 的工作机制。

   .. grid-item-card:: 机器人接口
      :link: robotics
      :link-type: doc

      通过具名零部件读取和控制机器人。

   .. grid-item-card:: 机器人架构
      :link: robotics_architecture
      :link-type: doc

      了解共享连接、资源生命周期和远程部署机制。

   .. grid-item-card:: 真机任务与环境
      :link: realworld_envs
      :link-type: doc

      了解任务、遥操作和 wrapper 与机器人的组合关系。

.. toctree::
   :hidden:

   执行 <execution-model/index>
   调度 <scheduling-model/index>
   机器人接口 <robotics>
   机器人架构 <robotics_architecture>
   真机任务与环境 <realworld_envs>
