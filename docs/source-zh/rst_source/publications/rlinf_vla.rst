RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training
=================================================================

**论文：** `arXiv:2510.06710 <https://arxiv.org/abs/2510.06710>`__ | **模型：** `RLinf-OpenVLA <https://huggingface.co/collections/RLinf/openvla>`__ | `RLinf-OpenVLAOFT <https://huggingface.co/collections/RLinf/openvla-oft>`__

概述
----

.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf-vla/rlinf_vla_overview.png
   :alt: RLinf-VLA 概述
   :align: center

RLinf-VLA 是面向 VLA 模型可扩展 RL 训练的统一高效框架，通过统一接口整合多种 VLA 架构、多种 RL 算法与异构仿真器；采用灵活的资源分配架构，针对 GPU 并行仿真器引入混合细粒度流水线策略，带来约 1.61×–1.88× 训练加速。在 LIBERO、ManiSkill、RoboTwin 等基准上，RLinf-VLA 训练得到的模型均有约 20–85% 的性能提升。

结果
----

训练曲线（ManiSkill）
~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-vla/mani_openvla.png" alt="mani_openvla" width="350"/>
         <br/><strong>OpenVLA</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-vla/mani_openvlaoft.png" alt="mani_openvlaoft" width="350"/>
         <br/><strong>OpenVLA-OFT</strong>
       </td>
     </tr>
   </table>
   </div>

ManiSkill “PutOnPlateInScene25Mani-v3” 上使用 OpenVLA 与 OpenVLA-OFT、PPO 与 GRPO 的训练曲线。PPO 持续优于 GRPO 且更稳定。

ManiSkill 评估
~~~~~~~~~~~~~~

.. list-table:: ManiSkill 评估结果（成功率 %）
   :header-rows: 1
   :widths: 22 14 12 12 14 12
   :align: left

   * - 模型
     - 分布内
     - Vision
     - Semantic
     - Execution
     - 平均
   * - OpenVLA (Base)
     - 53.91
     - 38.75
     - 35.94
     - 42.11
     - 39.10
   * - `RL4VLA (PPO) <https://huggingface.co/gen-robot/openvla-7b-rlvla-rl>`__
     - 93.75
     - 80.47
     - 75.00
     - 81.77
     - 79.15
   * - `OpenVLA (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood>`__
     - 84.38
     - 74.69
     - 72.99
     - 77.86
     - 75.15
   * - `OpenVLA (RLinf-PPO) <https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood>`__
     - **96.09**
     - 82.03
     - **78.35**
     - **85.42**
     - **81.93**
   * -
     -
     -
     -
     -
     -
   * - OpenVLA-OFT (Base)
     - 28.13
     - 27.73
     - 12.95
     - 11.72
     - 18.29
   * - `OpenVLA-OFT (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood>`__
     - 94.14
     - 84.69
     - 45.54
     - 44.66
     - 60.64
   * - `OpenVLA-OFT (RLinf-PPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood>`__
     - **97.66**
     - **92.11**
     - 64.84
     - 73.57
     - 77.05

LIBERO（统一模型，五类任务）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 五类 LIBERO 任务组上统一模型评估结果（%）
   :header-rows: 1
   :widths: 28 12 10 10 10 8 10
   :align: left

   * - 模型
     - Spatial
     - Object
     - Goal
     - Long
     - 90
     - 平均
   * - `OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`__
     - 72.18
     - 71.48
     - 64.06
     - 48.44
     - 70.97
     - 65.43
   * - `OpenVLA-OFT (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130>`__
     - **99.40**
     - **99.80**
     - **98.79**
     - **93.95**
     - **98.59**
     - **98.11**
   * - Δ 提升
     - +27.22
     - +28.32
     - +34.73
     - +45.51
     - +27.62
     - +32.68

RoboTwin（六项任务）
~~~~~~~~~~~~~~~~~~~~

.. list-table:: OpenVLA-OFT 在六项 RoboTwin 任务上的评估结果（%）
   :header-rows: 1
   :widths: 22 12 12 12 10 10 10 8 8
   :align: left

   * - 模型
     - beat_block_hammer
     - pick_dual_bottles
     - place_empty_cup
     - move_can_pot
     - lift_pot
     - handover_block
     - 平均
     - Δ 平均
   * - OpenVLA-OFT (SFT)
     - 10.15
     - 20.31
     - 75.78
     - 9.37
     - 3.13
     - 28.13
     - 24.48
     - —
   * - OpenVLA-OFT (RLinf-GRPO)
     - **96.09**
     - **92.96**
     - **94.53**
     - **83.59**
     - **70.31**
     - **70.31**
     - **84.63**
     - **+60.15**

“Base”与“SFT”指 RL 训练前的监督微调模型。

快速开始
--------

- **ManiSkill：** :doc:`../examples/embodied/maniskill`
- **LIBERO：** :doc:`../examples/embodied/libero`
- **RoboTwin：** :doc:`../examples/embodied/robotwin`
- **更多示例：** :doc:`../examples/embodied/index`

引用
----

.. code-block:: bibtex

   @article{zang2025rlinf,
     title={RLinf-VLA: A unified and efficient framework for VLA+ RL training},
     author={Zang, Hongzhi and Wei, Mingjie and Xu, Si and Wu, Yongji and Guo, Zhen and Wang, Yuanqing and Lin, Hao and Shi, Liangzhi and Xie, Yuqing and Xu, Zhexuan and others},
     journal={arXiv preprint arXiv:2510.06710},
     year={2025}
   }
