WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL
=================================================================

**论文：** `arXiv:2602.13977 <https://arxiv.org/abs/2602.13977>`__ | **世界模型：** `WoVR <https://huggingface.co/collections/RLinf/wovr>`__

概述
----

.. image:: https://github.com/RLinf/misc/raw/main/pic/wovr_overview.png
   :alt: WoVR 框架概述
   :align: center

WoVR 是基于世界模型构建的、面向 VLA 模型强化学习微调的可靠训练框架。它首先将世界模型强化为一个可控、稳定的生成式仿真器，实现动作响应与长时滚动稳定生成；在此基础上，通过 Keyframe-Initialized Rollouts（KIR）与掩码 GRPO 构建可靠的想象交互机制，降低有效误差深度并避免在幻觉成功上优化；最后，通过 PACE 策略实现策略与世界模型的协同进化，对齐不断演化的策略分布，缓解分布偏移并维持仿真器可靠性。

LIBERO（四类任务）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 四类 LIBERO 任务组上评估结果（%）
   :header-rows: 1
   :widths: 28 12 10 10 10 10
   :align: left

   * - 模型
     - Spatial
     - Object
     - Goal
     - Long
     - 平均
   * - OpenVLA-OFT (Base) 
     - 61.5
     - 36.3
     - 48.2
     - 13.7
     - 39.9
   * - OpenVLA-OFT （Wan 作为世界模型的 RLinf-GRPO）
     - **81.5**
     - **82.0**
     - **77.5**
     - **35.8**
     - **69.2**
   * - Δ 提升
     - +20.0
     - +45.7
     - +29.3
     - +17.9
     - +29.3


“Base” 指 RL 训练前的监督微调模型。

快速开始
--------

- **LIBERO：** :doc:`../examples/embodied/wan`
- **更多示例：** :doc:`../examples/embodied/index`

引用
----

.. code-block:: bibtex

   @misc{jiang2026wovr,
      title={WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL}, 
      author={Jiang, Zhennan and Zhou, Shangqing and Jiang, Yutong and Huang, Zefang and Wei, Mingjie and Chen, Yuhui and Zhou, Tianxing and Guo, Zhen and Lin, Hao and Zhang, Quanlu and Wang, Yu and Li, Haoran and Yu, Chao and Zhao, Dongbin},
      year={2026},
      journal={arXiv preprint arXiv:2602.13977},
    }
