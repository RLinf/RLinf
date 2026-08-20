概念
====

请根据当前问题选择对应的概念页面，例如训练执行流程、组件部署位置或机器人组织方式。各页面先介绍使用相关功能所需的核心模型，并在需要扩展或排查底层问题时链接至实现细节。

选择概念主题
------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 执行模型
      :link: execution-model/index
      :link-type: doc

      了解任务执行流程，以及 worker、cluster、channel 与 collective 之间的关系。

   .. grid-item-card:: 调度模型
      :link: scheduling-model/index
      :link-type: doc

      了解 placement、执行模式与 replay buffer 的工作机制。

   .. grid-item-card:: 机器人模型
      :link: robotics
      :link-type: doc

      了解部件名称与观测、动作路径之间的对应关系。

   .. grid-item-card:: 机器人架构
      :link: robotics_architecture
      :link-type: doc

      了解共享连接、资源生命周期和远程部署机制。

   .. grid-item-card:: 真机环境模型
      :link: realworld_envs
      :link-type: doc

      了解任务、遥操作和 wrapper 与机器人的组合关系。

.. toctree::
   :hidden:

   执行模型 <execution-model/index>
   调度模型 <scheduling-model/index>
   机器人模型 <robotics>
   机器人架构 <robotics_architecture>
   真机环境模型 <realworld_envs>
