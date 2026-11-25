新增 Env 与真机
===============

按改动面从小到大切入：如果目标真机已在支持列表中，直接定义真机任务即可；如需接入新仿真 Env 或新真机硬件，再按对应指南开发。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 仿真 Env
      :link: ../new_env
      :link-type: doc

      封装仿真器后端、注册 Env 类型并完成动作格式映射。

   .. grid-item-card:: 真机硬件
      :link: ../new_robot
      :link-type: doc

      从本地单个硬件部件起步，逐步组合整机并配置跨机远程 Placement。

   .. grid-item-card:: 真机任务
      :link: ../new_task
      :link-type: doc

      在 RLinf 已支持的真机硬件上定义新任务，无需改动驱动层。

.. toctree::
   :hidden:

   仿真 Env <../new_env>
   真机硬件 <../new_robot>
   真机任务 <../new_task>
