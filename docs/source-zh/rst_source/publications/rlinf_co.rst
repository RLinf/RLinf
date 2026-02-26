Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models
===================================================================================

**论文：** `arXiv:2602.12628 <https://arxiv.org/abs/2602.12628>`__

概述
----


.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf-co/overview.png
   :alt: RLinf-Co 概述
   :align: center

所提出的两阶段仿真-真实协同训练框架概览。我们建立数字孪生设置，尽管存在视觉差异，:math:`T_{\text{sim}}` 仍可作为 :math:`T_{\text{real}}` 的数字近亲。在 **阶段 I** 中，我们将真实与仿真数据按比例 :math:`\alpha` 混合进行监督训练以初始化 VLA 策略。这一步可快速注入真实世界知识，并为后续仿真交互做好准备。在 **阶段 II** 中，我们在仿真器中进行 RL 微调以探索并提升性能，同时引入真实世界 SFT 损失作为正则项，防止模型遗忘真实世界行为。

结果
----

主要结果
~~~~~~~~

.. list-table:: 不同训练范式下的真实世界成功率对比
   :header-rows: 1
   :widths: 16 20 13 12 12 12 15
   :align: left

   * - VLA 模型
     - 实验设置
     - Pick and Place
     - Push Cube
     - Open Drawer
     - Close Drawer
     - 平均
   * - **OpenVLA**
     - 仅真实数据训练
     - 6.3 ± 0.0
     - 20.0 ± 13.3
     - 0.0 ± 0.0
     - 10.0 ± 10.0
     - 16.5 ± 13.3
   * -
     - SFT 协同训练
     - 23.4 ± 4.7
     - 51.7 ± 5.0
     - 0.0 ± 0.0
     - 85.0 ± 5.0
     - 40.0 ± 3.7
   * -
     - **RL-Co（我们的方法）**
     - **58.8 ± 10.0**
     - **68.3 ± 11.7**
     - **35.0 ± 15.0**
     - **95.0 ± 5.0**
     - **64.0 ± 0.7**
   * - **π₀.₅**
     - 仅真实数据训练
     - 71.9 ± 9.4
     - 0.0 ± 0.0
     - 0.0 ± 0.0
     - 35.0 ± 15.0
     - 26.7 ± 1.4
   * -
     - SFT 协同训练
     - 68.8 ± 9.4
     - 10.0 ± 3.3
     - 10.0 ± 0.0
     - 95.0 ± 5.0
     - 45.9 ± 4.4
   * -
     - **RL-Co（我们的方法）**
     - **81.3 ± 9.4**
     - **18.4 ± 1.7**
     - **65.0 ± 5.0**
     - **100.0 ± 0.0**
     - **66.2 ± 4.0**

消融实验
~~~~~~~~

.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf-co/success_rate.png
   :alt: 仿真 SFT 初始化消融实验
   :align: center

仿真 SFT 初始化消融实验。我们报告了在是否使用仿真 SFT 初始化条件下模型在 RL 训练过程中的仿真成功率。每次 RL 训练均使用三个独立随机种子，结果以平均成功率展示，阴影区域表示标准差。

数据效率
~~~~~~~~

.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf-co/data_efficiency.png
   :alt: 真实世界演示数量的影响
   :align: center

真实世界演示数量的影响。我们改变 ``Open Drawer`` 任务中的真实世界演示数量，并使用 :math:`\pi_{0.5}` 模型评估所有训练范式。性能以成功率衡量，阴影区域表示标准差。

快速开始
--------

- **说明：** :doc:`../examples/embodied/co_training`

引用
----

.. code-block:: bibtex

    @article{shi2026rlinf,
      title={Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models},
      author={Shi, Liangzhi and Chen, Shuaihang and Gao, Feng and Chen, Yinuo and Chen, Kang and Zhang, Tonghe and Zhang, Hongzhi and Zhang, Weinan and Yu, Chao and Wang, Yu},
      journal={arXiv preprint arXiv:2602.12628},
      year={2026}
    }
