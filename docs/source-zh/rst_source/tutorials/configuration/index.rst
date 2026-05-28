配置
=====

本节涵盖 RLinf 训练工作负载配置的各个方面。
学习如何编写 YAML 配置文件、设置集群和硬件、管理检查点以及可视化训练指标。

- :doc:`yaml`
   RLinf 中所有 YAML 配置参数的全面指南。
   学习如何结构化配置文件，以实现清晰性、灵活性和可复现性。

- :doc:`cluster`
   介绍全局唯一的 *Cluster* 对象，负责协调分布式训练作业中
   所有角色、进程和跨节点通信。

- :doc:`hetero`
   配置异构软件和硬件集群，以高效利用不同的计算资源和设备。

- :doc:`resume`
   介绍如何从保存的检查点恢复训练，确保长时间运行或中断的训练作业
   具备容错能力和无缝续训。

- :doc:`logger`
   介绍如何在训练过程中可视化和跟踪关键指标。
   目前支持 TensorBoard、Weights & Biases (wandb) 和 SwanLab 三种后端。


.. toctree::
   :hidden:
   :maxdepth: 1

   yaml
   cluster
   hetero
   resume
   logger
