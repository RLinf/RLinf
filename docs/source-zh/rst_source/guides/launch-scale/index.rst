扩展到多机集群
==============

单卡或单机算力不够时，按场景将训练扩展到多节点、异构集群、云边协同，或启用自动 Placement、5D 并行与动态调度。

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: 多节点训练
      :link: ../multi_node
      :link-type: doc

      搭建多机 Ray 集群，配置节点编号与通信网卡。

   .. grid-item-card:: 异构集群部署
      :link: ../hetero
      :link-type: doc

      在同一个集群中混部不同型号 GPU 或独立 CPU 仿真节点。

   .. grid-item-card:: 云边协同训练
      :link: ../cloud_edge
      :link-type: doc

      通过组网打通云端 GPU 训练机与处于 NAT 后的边缘机器。

   .. grid-item-card:: 自动 Placement
      :link: ../auto_placement
      :link-type: doc

      自动搜索 Actor、Rollout 与 Env Worker 的最优硬件排布。

   .. grid-item-card:: 5D 并行策略
      :link: ../5D
      :link-type: doc

      面向大模型组合张量、流水线、上下文、数据与专家并行。

   .. grid-item-card:: 动态弹性调度
      :link: ../dynamic_scheduling
      :link-type: doc

      在动态集群资源下实现弹性扩缩与 Worker 重调度。

.. toctree::
   :hidden:

   多节点训练 <../multi_node>
   异构集群部署 <../hetero>
   云边协同训练 <../cloud_edge>
   自动 Placement <../auto_placement>
   5D 并行策略 <../5D>
   动态弹性调度 <../dynamic_scheduling>
