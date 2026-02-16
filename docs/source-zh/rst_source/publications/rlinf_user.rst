RLinf-USER: Unified System for Real-world Online Policy Learning
=================================================================

**Paper:** `arXiv:2509.15965 <https://arxiv.org/abs/2602.07837>`__ 

概述
----

.. image:: https://github.com/RLinf/misc/raw/main/pic/USER/USER-HEAD.png
   :alt: RLinf-USER 概述
   :width: 800px
   :align: center

RLinf-USER 是面向真机在线策略学习的统一可扩展系统，提供奖励、算法与策略的可扩展抽象，支持 CNN/MLP、生成式（流）策略与大规模 VLA 模型在统一流程中进行在线模仿或强化学习。

任务
----

- **插 peg：** 将 peg 对准并插入孔中。
- **充电器任务：** 将充电器插入插座。
- **抓放：** 抓取并移动随机摆放的物体（如橡胶鸭）到目标容器。
- **拧瓶盖：** 将瓶盖旋转拧紧到指定姿态。
- **桌面清理：** 将桌面上杂物清理到指定盒子并盖上盖子。

算法
----

- **SAC (Soft Actor-Critic)：** 经典真机 RL 算法。
- **RLPD (RL with Prior Data)：** 结合先验示教数据与高 update-to-data 比。
- **SAC Flow：** 流匹配策略的样本高效 RL。
- **HG-DAgger：** 交互式模仿学习。

硬件配置
--------

.. list-table:: 推荐硬件
   :header-rows: 1
   :widths: 25 50
   :align: left

   * - 组件
     - 规格
   * - 机械臂
     - Franka Emika Panda
   * - 相机
     - Intel RealSense (RGB)
   * - 计算
     - RTX 4090（CNN/Flow），A100 × 4（π₀）
   * - 机器人控制器
     - NUC（无 GPU）
   * - 遥操作
     - 3D Connection SpaceMouse Compact

结果
----

真机表现
~~~~~~~~

RLinf-USER 支持多种学习范式。下方为多种 RL 算法（RLPD、SAC、SAC-Flow）在多任务上的训练曲线，以及 VLA（π₀）经在线微调后的提升。

.. image:: https://github.com/RLinf/misc/raw/main/pic/USER/USER-main_rl.jpg
   :alt: RL 训练曲线
   :width: 800px
   :align: center

**多任务与多算法的 RL 训练曲线**

**VLA（π₀）与 HG-DAgger：** RLinf-USER 在少量干预下显著提升基础 VLA 模型在真实世界中的成功率。

.. list-table:: π₀ 在线训练提升
   :header-rows: 1
   :widths: 25 30 30
   :align: left

   * - 任务
     - 在线训练前
     - 在线训练后
   * - Pick-and-Place
     - 39/60 (65%)
     - **58/60 (96.7%)**
   * - Table Clean-up
     - 9/20 (45%)
     - **16/20 (80%)**

系统效率：异步 vs 同步
~~~~~~~~~~~~~~~~~~~~~~

RLinf-USER 采用完全异步流水线，将数据生成、训练与权重同步解耦，尤其对大模型优于同步流水线。

.. list-table:: 性能：生成与训练吞吐
   :header-rows: 1
   :widths: 28 22 22 22
   :align: left

   * - 模型 + 算法
     - 流水线模式
     - 生成 (s/回合) ↓
     - 训练 (s/更新) ↓
   * - π₀ + HG-DAgger
     - 同步
     - 45.07
     - 45.01
   * - π₀ + HG-DAgger
     - **异步 (RLinf-USER)**
     - **37.54**
     - **7.90**
   * - π₀ + HG-DAgger
     - *加速*
     - 1.20×
     - 5.70×
   * - CNN + SAC
     - 同步
     - 20.29
     - 0.64
   * - CNN + SAC
     - **异步 (RLinf-USER)**
     - **13.11**
     - **0.14**
   * - CNN + SAC
     - *加速*
     - 1.55×
     - 4.61×

多机与异构支持
~~~~~~~~~~~~~~

在统一硬件抽象下，RLinf-USER 将机器人视为一等资源：

- **并行训练：** 多机同时训练（如 2× Franka）多任务以扩大数据采集。
- **异构训练：** 在不同本体（如 Franka 7-DoF + ARX 6-DoF）上训练统一策略。

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/USER/USER-multi.jpg" alt="Multi-Robot" width="370"/><br/>
         <strong>并行训练 (2× Franka)</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/USER/USER-hetero.jpg" alt="Heterogeneous" width="300"/><br/>
         <strong>异构 (Franka + ARX)</strong>
       </td>
     </tr>
   </table>
   </div>

在多机与异构配置下，RLinf-USER 在可比时间内实现策略完全收敛。

快速开始
--------

- :doc:`../examples/embodied/franka`

可视化
------

启动 TensorBoard 监控训练：

.. code-block:: bash

   tensorboard --logdir ./logs

引用
----

RLinf-USER 及 RLinf 真实世界 RL 请引用 RLinf 主论文：

.. code-block:: bibtex

   @article{yu2025rlinf,
     title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
     author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
     journal={arXiv preprint arXiv:2509.15965},
     year={2025}
   }
