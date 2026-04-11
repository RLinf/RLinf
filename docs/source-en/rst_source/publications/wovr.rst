WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL
=================================================================================

**Paper:** `arXiv:2602.13977 <https://arxiv.org/abs/2602.13977>`__ | **World Model:** `WoVR <https://huggingface.co/collections/RLinf/wovr>`__

Overview
--------

.. image:: https://github.com/RLinf/misc/raw/main/pic/wovr_overview.png
   :alt: WoVR framework overview
   :align: center

WoVR is a reliable training framework built on world models for RL fine-tuning of VLA policies. It first strengthens the world model into a controllable and stable generative simulator that supports action-conditioned generation and long-horizon rollout stability. On top of this, it introduces Keyframe-Initialized Rollouts (KIR) and masked GRPO to build a reliable imagination-based interaction mechanism, reducing effective error depth and avoiding optimization on hallucinated successes. Finally, through the PACE strategy, it achieves co-evolution of the policy and the world model, aligning the evolving policy distribution, mitigating distribution shift, and maintaining simulator reliability.

LIBERO (four task suites)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Evaluation results on four LIBERO task suites (%)
   :header-rows: 1
   :widths: 28 12 10 10 10 10
   :align: left

   * - Model
     - Spatial
     - Object
     - Goal
     - Long
     - Avg
   * - OpenVLA-OFT (Base)
     - 61.5
     - 36.3
     - 48.2
     - 13.7
     - 39.9
   * - OpenVLA-OFT (RLinf-GRPO with Wan as world model)
     - **81.5**
     - **82.0**
     - **77.5**
     - **35.8**
     - **69.2**
   * - Δ Improvement
     - +20.0
     - +45.7
     - +29.3
     - +17.9
     - +29.3


"Base" refers to the supervised fine-tuned model before RL training.

Quick Start
-----------

- **LIBERO:** :doc:`../examples/embodied/wan`
- **More examples:** :doc:`../examples/embodied/index`

Citation
--------

.. code-block:: bibtex

   @misc{jiang2026wovr,
      title={WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL}, 
      author={Jiang, Zhennan and Zhou, Shangqing and Jiang, Yutong and Huang, Zefang and Wei, Mingjie and Chen, Yuhui and Zhou, Tianxing and Guo, Zhen and Lin, Hao and Zhang, Quanlu and Wang, Yu and Li, Haoran and Yu, Chao and Zhao, Dongbin},
      year={2026},
      journal={arXiv preprint arXiv:2602.13977},
    }
