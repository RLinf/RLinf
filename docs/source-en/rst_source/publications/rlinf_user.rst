RLinf-USER: Unified System for Real-world Online Policy Learning
=================================================================

**Paper:** `arXiv:2509.15965 <https://arxiv.org/abs/2602.07837>`__ 

Overview
--------

.. image:: https://github.com/RLinf/misc/raw/main/pic/USER/USER-HEAD.png
   :alt: RLinf-USER overview
   :width: 800px
   :align: center

RLinf-USER is a unified and extensible system for real-world online policy learning. It provides extensible abstractions for rewards, algorithms, and policies, supporting online imitation or reinforcement learning of CNN/MLP, generative (flow) policies, and large vision–language–action (VLA) models within a unified pipeline.

Tasks
-----

- **Peg-Insertion:** Aligning and inserting a peg into a hole.
- **Charger Task:** Plugging a charger into a socket.
- **Pick-and-Place:** Grasping and transporting a randomly initialized object (e.g. rubber duck) to a target container.
- **Cap Tightening:** Rotating and tightening a bottle cap to a specified pose.
- **Table Clean-up:** Cleaning cluttered objects from the tabletop into a designated box, then closing the lid.

Algorithms
----------

- **SAC (Soft Actor-Critic):** Classical algorithm for real-world RL.
- **RLPD (RL with Prior Data):** Incorporates prior demonstration data with high update-to-data ratios.
- **SAC Flow:** Sample-efficient flow-based policy RL.
- **HG-DAgger:** Interactive imitation learning.

Hardware setup
--------------

.. list-table:: Recommended hardware
   :header-rows: 1
   :widths: 25 50
   :align: left

   * - Component
     - Specification
   * - Robotic Arm
     - Franka Emika Panda
   * - Cameras
     - Intel RealSense (RGB)
   * - Computing
     - RTX 4090 (CNN/Flow), A100 × 4 (π₀)
   * - Robot Controller
     - NUC (no GPU)
   * - Teleop
     - 3D Connection SpaceMouse Compact

Results
-------

Robust real-world performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf-USER supports diverse learning paradigms. Below are training curves for RL algorithms (RLPD, SAC, SAC-Flow) on several tasks, and the gain for VLA (π₀) after online fine-tuning.

.. image:: https://github.com/RLinf/misc/raw/main/pic/USER/USER-main_rl.jpg
   :alt: RL training curves
   :width: 800px
   :align: center

**RL Training Curves of Diverse Tasks & Algorithms**

**VLA (π₀) with HG-DAgger:** RLinf-USER significantly improves success rate of foundation VLA models in real-world settings with minimal interventions.

.. list-table:: Online training improvement for π₀
   :header-rows: 1
   :widths: 25 30 30
   :align: left

   * - Task
     - Before Online Training
     - After Online Training
   * - Pick-and-Place
     - 39/60 (65%)
     - **58/60 (96.7%)**
   * - Table Clean-up
     - 9/20 (45%)
     - **16/20 (80%)**

System efficiency: asynchronous vs synchronous
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf-USER uses a fully asynchronous pipeline that decouples data generation, training, and weight synchronization, outperforming synchronous pipelines especially for large models.

.. list-table:: Profiling: generation & training throughput
   :header-rows: 1
   :widths: 28 22 22 22
   :align: left

   * - Model + Algorithm
     - Pipeline Mode
     - Generation (s/episode) ↓
     - Training (s/update) ↓
   * - π₀ + HG-DAgger
     - Synchronous
     - 45.07
     - 45.01
   * - π₀ + HG-DAgger
     - **Asynchronous (RLinf-USER)**
     - **37.54**
     - **7.90**
   * - π₀ + HG-DAgger
     - *Speed up*
     - 1.20×
     - 5.70×
   * - CNN + SAC
     - Synchronous
     - 20.29
     - 0.64
   * - CNN + SAC
     - **Asynchronous (RLinf-USER)**
     - **13.11**
     - **0.14**
   * - CNN + SAC
     - *Speed up*
     - 1.55×
     - 4.61×

Multi-robot and heterogeneous support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the unified hardware abstraction, RLinf-USER treats robots as first-class resources:

- **Parallel training:** Train on multiple robots at once (e.g. 2× Franka) in a multi-task setting to scale data collection.
- **Heterogeneous training:** Train a unified policy across different embodiments (e.g. Franka 7-DoF + ARX 6-DoF).

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/USER/USER-multi.jpg" alt="Multi-Robot" width="370"/><br/>
         <strong>Parallel Training (2× Franka)</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/USER/USER-hetero.jpg" alt="Heterogeneous" width="300"/><br/>
         <strong>Heterogeneous (Franka + ARX)</strong>
       </td>
     </tr>
   </table>
   </div>

Under multi-robot and heterogeneous settings, RLinf-USER achieves full policy convergence within comparable time.

Quickstart
----------

- :doc:`../examples/embodied/franka`

Visualization
-------------

Launch TensorBoard to monitor training:

.. code-block:: bash

   tensorboard --logdir ./logs

Citation
--------

For RLinf-USER and real-world RL with RLinf, cite the main RLinf paper:

.. code-block:: bibtex

   @article{yu2025rlinf,
     title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
     author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
     journal={arXiv preprint arXiv:2509.15965},
     year={2025}
   }
