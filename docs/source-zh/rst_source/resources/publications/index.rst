论文
====

本页汇总基于或提出 **RLinf** 的论文与技术报告，并链接到对应的结果详情页、论文原文与可复现配方。

论文与配方一览
--------------

.. list-table::
   :header-rows: 1
   :widths: 15 8 15 28 16 18

   * - 论文
     - 年份
     - 领域
     - 核心主题
     - 论文链接
     - 快速开始 / 配方
   * - :doc:`CEL <cel>`
     - 2026
     - 真机集群
     - 汇集部署中的机器人集群经验，改进共享 policy。
     - `arXiv:2610.25736 <https://arxiv.org/abs/2610.25736>`__
     - :doc:`真机机器人 <../../examples/real_world_index>`
   * - :doc:`DynaRL <dynarl>`
     - 2026
     - RL 系统
     - 大规模强化学习训练的灵活动态调度。
     - `OSDI 2026 <https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing>`__
     - :doc:`动态调度 <../../guides/dynamic_scheduling>`
   * - :doc:`πRL <pi_rl>`
     - 2025
     - 流匹配 VLA
     - 面向流匹配 VLA 模型的在线 RL 微调。
     - `arXiv:2510.25889 <https://arxiv.org/abs/2510.25889>`__
     - :doc:`π₀ / π₀.₅ <../../examples/embodied/pi0>`
   * - :doc:`RLinf <rlinf_system>`
     - 2026
     - RL 系统
     - 基于宏观到微观流转换的灵活高效 RL 系统。
     - `OSDI 2026 <https://www.usenix.org/conference/osdi26/presentation/yu-chao>`__
     - :doc:`安装指南 <../../start/installation>`
   * - :doc:`RLinf-Co <rlinf_co>`
     - 2026
     - 仿真实协同训练
     - 基于强化学习的仿真-真实协同训练 VLA 模型。
     - `arXiv:2602.12628 <https://arxiv.org/abs/2602.12628>`__
     - :doc:`协同训练 <../../examples/embodied/co_training>`
   * - :doc:`RLinf-VLA <rlinf_vla>`
     - 2026
     - 具身 VLA
     - 跨模拟器的 VLA+RL 统一训练框架。
     - `RSS 2026 <https://roboticsproceedings.org/rss22/p089.html>`__
     - :doc:`LIBERO <../../examples/embodied/libero>`
   * - :doc:`STEAM <steam>`
     - 2026
     - 离线 / 真机
     - 面向真实机器人学习的自监督时序集成优势建模。
     - `arXiv:2606.29834 <https://arxiv.org/abs/2606.29834>`__
     - :doc:`STEAM 配方 <../../examples/embodied/steam>`
   * - :doc:`USER <rlinf_user>`
     - 2026
     - 真机 RL
     - 统一真机在线 policy 学习系统。
     - `RSS 2026 <https://roboticsproceedings.org/rss22/p037.html>`__
     - :doc:`Franka 真机 RL <../../examples/embodied/franka>`
   * - :doc:`WideSeek-R1 <wideseek_r1>`
     - 2026
     - 多智能体 RL
     - 通过多智能体强化学习探索广泛信息检索的宽度扩展。
     - `arXiv:2602.04634 <https://arxiv.org/abs/2602.04634>`__
     - :doc:`WideSeek-R1 <../../examples/agentic/wideseek_r1/index>`
   * - :doc:`WoVR <wovr>`
     - 2026
     - 世界模型 RL
     - 基于世界模型的 VLA 强化学习微调。
     - `arXiv:2602.13977 <https://arxiv.org/abs/2602.13977>`__
     - :doc:`Wan 世界模型 <../../examples/embodied/wan>`

引用 RLinf
----------

如果 RLinf 对你的研究有所帮助，请引用 RLinf 系统论文：

.. code-block:: bibtex

   @inproceedings{yu2026rlinf,
     author = {Chao Yu and Yuanqing Wang and Zhen Guo and Hao Lin and Si Xu and Hongzhi Zang and Quanlu Zhang and Yongji Wu and Chunyang Zhu and Junhao Hu and Zixiao Huang and Mingjie Wei and Yuqing Xie and Ke Yang and Bo Dai and Zhexuan Xu and Jiakun Du and Xiangyuan Wang and Xu Fu and Letong Shi and Zhihao Liu and Kang Chen and Weilin Liu and Gang Liu and Boxun Li and Jianlei Yang and Zhi Yang and Guohao Dai and Yu Wang},
     title = {{RLinf}: Flexible and Efficient {Large-Scale} Reinforcement Learning via {Macro-to-Micro} Flow Transformation},
     booktitle = {20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26)},
     year = {2026},
     isbn = {978-1-939133-55-7},
     address = {Seattle, WA},
     pages = {829--846},
     url = {https://www.usenix.org/conference/osdi26/presentation/yu-chao},
     publisher = {USENIX Association},
     month = jul
   }

.. toctree::
   :hidden:
   :maxdepth: 2

   CEL <cel>
   DynaRL <dynarl>
   πRL <pi_rl>
   RLinf <rlinf_system>
   RLinf-Co <rlinf_co>
   RLinf-VLA <rlinf_vla>
   STEAM <steam>
   USER <rlinf_user>
   WideSeek-R1 <wideseek_r1>
   WoVR <wovr>
