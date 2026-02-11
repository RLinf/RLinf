RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training
=================================================================

**Paper:** `arXiv:2510.06710 <https://arxiv.org/abs/2510.06710>`__ | **Models:** `RLinf-OpenVLA <https://huggingface.co/collections/RLinf/openvla>`__ | `RLinf-OpenVLAOFT <https://huggingface.co/collections/RLinf/openvla-oft>`__

Overview
--------

.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf-vla/rlinf_vla_overview.png
   :alt: RLinf-VLA overview
   :align: center

RLinf-VLA is a unified and efficient framework for scalable RL training of VLA models. It provides a unified interface that standardizes the integration of diverse VLA architectures, multiple RL algorithms, and heterogeneous simulators. The system uses a flexible resource allocation architecture for rendering, inference, and training; for GPU-parallelized simulators it introduces a hybrid fine-grained pipeline allocation strategy, yielding a 1.61×–1.88× training speedup. Models trained with RLinf-VLA show consistent improvements of approximately 20–85% across LIBERO, ManiSkill, and RoboTwin.

Results
-------

Training curves (ManiSkill)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Training curves on ManiSkill “PutOnPlateInScene25Mani-v3” with OpenVLA and OpenVLA-OFT, using PPO and GRPO. PPO consistently outperforms GRPO and is more stable.

ManiSkill evaluation
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Evaluation results on ManiSkill (success rates %)
   :header-rows: 1
   :widths: 22 14 12 12 14 12
   :align: left

   * - Model
     - In-Dist.
     - Vision
     - Semantic
     - Execution
     - Avg.
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

LIBERO (unified model, five task groups)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Evaluation results of the unified model on the five LIBERO task groups (%)
   :header-rows: 1
   :widths: 28 12 10 10 10 8 10
   :align: left

   * - Model
     - Spatial
     - Object
     - Goal
     - Long
     - 90
     - Avg.
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
   * - Δ Improvement
     - +27.22
     - +28.32
     - +34.73
     - +45.51
     - +27.62
     - +32.68

RoboTwin (six tasks)
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Evaluation results of OpenVLA-OFT on six RoboTwin tasks (%)
   :header-rows: 1
   :widths: 22 12 12 12 10 10 10 8 8
   :align: left

   * - Model
     - beat_block_hammer
     - pick_dual_bottles
     - place_empty_cup
     - move_can_pot
     - lift_pot
     - handover_block
     - Average
     - Δ Avg.
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

"Base" and "SFT" refer to supervised fine-tuned models before RL training.

Quickstart
----------

- **ManiSkill:** :doc:`../examples/embodied/maniskill`
- **LIBERO:** :doc:`../examples/embodied/libero`
- **RoboTwin:** :doc:`../examples/embodied/robotwin`
- **More examples:** :doc:`../examples/embodied/index`

Citation
--------

.. code-block:: bibtex

   @article{zang2025rlinf,
     title={RLinf-VLA: A unified and efficient framework for VLA+ RL training},
     author={Zang, Hongzhi and Wei, Mingjie and Xu, Si and Wu, Yongji and Guo, Zhen and Wang, Yuanqing and Lin, Hao and Shi, Liangzhi and Xie, Yuqing and Xu, Zhexuan and others},
     journal={arXiv preprint arXiv:2510.06710},
     year={2025}
   }
