多机与异构集群
================

本章介绍如何在多台机器上搭建 Ray 集群并运行 RLinf，以及如何启动真机强化学习训练任务，如何配置异构软硬件与云边协同场景。

- :doc:`multi_node`
   说明如何在多台机器上启动 Ray 集群、配置环境变量与代码同步，并通过 Ray 集群启动 RLinf 训练任务。

- :doc:`realworld_robot`
   说明如何将多台 Franka 真机与 GPU 训练节点接入同一 Ray 集群、配置 YAML，并启动真机强化学习训练。

- :doc:`hetero`
   介绍如何配置和使用异构软硬件集群，以充分利用不同类型的计算资源和硬件设备。

- :doc:`cloud-edge`
   展示如何使用 EasyTier 搭建云边协同训练环境，把云端与边缘节点接入同一个 overlay 网络，并在该网络之上运行 RLinf。

.. toctree::
   :hidden:
   :maxdepth: 1

   multi_node
   realworld_robot
   hetero
   cloud-edge
