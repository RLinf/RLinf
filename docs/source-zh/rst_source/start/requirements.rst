环境要求
========

以下是经过充分测试的配置。

硬件
----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 配置
   * - GPU
     - 每个节点 8 块 H100
   * - CPU
     - 每个节点 192 核心
   * - 内存
     - 每个节点 1.8TB
   * - 网络
     - NVLink + RoCE / IB，带宽 3.2 Tbps
   * - 存储
     - | 单节点实验使用 1TB 本地存储
       | 分布式实验使用 10TB 共享存储（NAS）

软件
----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 版本
   * - 操作系统
     - Ubuntu 22.04
   * - NVIDIA 驱动
     - 535.183.06
   * - CUDA
     - 12.4
   * - Docker
     - 26.0.0
   * - NVIDIA Container Toolkit
     - 1.17.8
