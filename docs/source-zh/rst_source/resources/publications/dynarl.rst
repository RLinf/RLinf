DynaRL: Flexible and Dynamic Scheduling of Large-Scale Reinforcement Learning Training
=======================================================================================

**论文：** `OSDI 2026 <https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing>`__ | `PDF <https://www.usenix.org/system/files/osdi26-wang-yuanqing.pdf>`__

概述
----

.. image:: https://github.com/RLinf/misc/raw/main/pic/dynarl/overview.png
   :alt: DynaRL 系统架构
   :align: center

面向大语言模型、长程推理与智能体系统的现代 RL 工作负载具有极强动态性：重尾
rollout、不规则的多轮工具交互，以及随时间变化的瓶颈，会使静态 GPU 划分大量空闲并
拉长训练时间。

**DynaRL** 是首个在异构 RL 组件之间动态重分配计算、内存与通信资源的 RL 系统。它将
整条 RL 流水线建模为动态超图，作为集中式、持续演化的控制平面；在统一的资源迁移
接口与上下文感知数据路由支持下，调度器通过多级调度与细粒度迁移，将 GPU 从过度
供给的组件转移到当前瓶颈。

DynaRL 基于 RLinf 实现，并与现有推理引擎（如 SGLang）与训练后端（如 Megatron-LM）
集成，对应 RLinf 的 :doc:`动态调度 <../../guides/dynamic_scheduling>` 功能。

结果
----

在数学推理与多轮智能体 RL（模型规模 1.5B–32B）上的评测表明：

- **数学推理 RL：** 相对最先进系统（verl、RLinf），端到端吞吐量最高提升
  **1.98×**。
- **智能体 RL：** 相对 RLinf 达到 **1.06×–1.53×**；叠加优先级感知请求调度后达到
  **1.27×–1.64×**。
- **开销：** 调度方案可在 200 ms 内生成，每次重分配约 0.5–5 s，在线调度开销低于
  **1%**。

数学推理 RL
~~~~~~~~~~~

数学推理 RL 包含 rollout、推理与训练三个阶段。基线系统采用静态顺序分配（同一时刻
全部 GPU 只服务一个阶段）。DynaRL 检测低利用率阶段，并将容量重分配给当前瓶颈。

.. image:: https://github.com/RLinf/misc/raw/main/pic/dynarl/math_throughput.png
   :alt: 数学推理 RL 端到端吞吐量
   :align: center

不同集群规模与模型规模下，数学推理 RL 的端到端吞吐量。

.. list-table:: 相对静态基线的加速（数学推理 RL）
   :header-rows: 1
   :widths: 18 28 28 26
   :align: left

   * - 设置
     - vs. verl / RLinf（64 GPU）
     - vs. verl / RLinf（128 GPU）
     - 说明
   * - 1.5B / 7B
     - 1.43×–1.55×
     - 1.40×–1.52×
     - 消除 rollout 欠利用
   * - 32B
     - 相对 RLinf 1.27×
     - 1.98× / 1.40×
     - 64 GPU 上 verl OOM
   * - vs. RLHFuse
     - 1.21×–1.42×
     - 1.21×–1.42×
     - 联合调度全部三个阶段

多轮智能体 RL
~~~~~~~~~~~~~

智能体 RL 含多轮 rollout 与工具调用，负载更难预测。DynaRL 将动态 GPU 分配与优先级
感知请求调度结合（优先推进已完成更多工具调用的请求，以降低长尾并提升 KV cache
复用）。

.. image:: https://github.com/RLinf/misc/raw/main/pic/dynarl/agent_throughput.png
   :alt: 多轮智能体 RL 端到端吞吐量
   :align: center

不同集群规模与模型规模下，多轮智能体 RL 的端到端吞吐量。

.. list-table:: 相对 RLinf 的加速（多轮智能体 RL）
   :header-rows: 1
   :widths: 20 35 45
   :align: left

   * - 设置
     - 仅动态分配
     - + 优先级感知请求调度
   * - 64 GPU（1.5B / 7B）
     - 1.06×–1.38×
     - 1.51×–1.53×
   * - 64 GPU（32B）
     - —
     - 1.27×
   * - 128 GPU
     - —
     - 1.40× / 1.64× / 1.58×（1.5B / 7B / 32B）

GPU 分配时间线
~~~~~~~~~~~~~~

下图展示在三种模式下，64 个 GPU 随时间分配给 Trainer、Rollout 与 Inference 的
过程：静态分配、动态分配，以及带优先级感知请求调度的动态分配。

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/allocation_static.png" alt="静态分配" width="700"/>
         <br/><strong>静态分配</strong>
       </td>
     </tr>
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/allocation_dynamic.png" alt="动态分配" width="700"/>
         <br/><strong>动态分配</strong>
       </td>
     </tr>
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/allocation_priority.png" alt="动态分配 + 优先级感知请求调度" width="700"/>
         <br/><strong>动态分配 + 优先级感知请求调度</strong>
       </td>
     </tr>
   </table>
   </div>

调度开销
~~~~~~~~

Trainer 迁移开销主要随模型规模增长（1.5B 亚毫秒级到 32B 数秒级），即便在 128 GPU
上的 32B 训练中也低于端到端延迟的 **0.5%**。Rollout 迁移可在数秒内完成。调度决策
在 **200 ms** 内完成；每次迭代的调度总开销低于 **0.5%**。

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/migration_trainer.png" alt="Trainer 迁移开销" width="320"/>
         <br/><strong>Trainer 迁移开销</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/migration_rollout.png" alt="Rollout 迁移开销" width="320"/>
         <br/><strong>Rollout 迁移开销</strong>
       </td>
     </tr>
   </table>
   </div>

快速开始
--------

- **教程：** :doc:`../../guides/dynamic_scheduling`

引用
----

.. code-block:: bibtex

   @inproceedings{wang2026dynarl,
     author = {Yuanqing Wang and Hao Lin and Junhao Hu and Chunyang Zhu and Quanlu Zhang and Zhen Guo and Yuchen Zhang and Xu Fu and Si Xu and Bo Dai and Zixiao Huang and Chao Yu and Boxun Li and Guohao Dai and Zhi Yang and Yu Wang},
     title = {{DynaRL}: Flexible and Dynamic Scheduling of {Large-Scale} Reinforcement Learning Training},
     booktitle = {20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26)},
     year = {2026},
     isbn = {978-1-939133-55-7},
     address = {Seattle, WA},
     pages = {847--862},
     url = {https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing},
     publisher = {USENIX Association},
     month = jul
   }
