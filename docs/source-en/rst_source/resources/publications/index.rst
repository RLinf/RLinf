Publications
============

Papers and technical reports built on or introducing **RLinf**. Each entry links to its detailed results page, paper venue, and matching runnable recipe or guide.

Papers and Recipes
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 8 15 28 16 18

   * - Publication
     - Year
     - Topic
     - Focus
     - Paper
     - Quickstart / Recipe
   * - :doc:`CEL <cel>`
     - 2026
     - Real-World Fleet
     - Collective experience learning from deployed robot fleets.
     - `arXiv:2610.25736 <https://arxiv.org/abs/2610.25736>`__
     - :doc:`Real-World Robots <../../examples/real_world_index>`
   * - :doc:`DynaRL <dynarl>`
     - 2026
     - RL Systems
     - Flexible and dynamic scheduling for large-scale RL training.
     - `OSDI 2026 <https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing>`__
     - :doc:`Dynamic Scheduling <../../guides/dynamic_scheduling>`
   * - :doc:`πRL <pi_rl>`
     - 2025
     - Flow-Based VLA
     - Online RL fine-tuning for flow-based VLA models.
     - `arXiv:2510.25889 <https://arxiv.org/abs/2510.25889>`__
     - :doc:`π₀ / π₀.₅ <../../examples/embodied/pi0>`
   * - :doc:`RLinf <rlinf_system>`
     - 2026
     - RL Systems
     - Flexible and efficient RL system via macro-to-micro flow transformation.
     - `OSDI 2026 <https://www.usenix.org/conference/osdi26/presentation/yu-chao>`__
     - :doc:`Installation <../../start/installation>`
   * - :doc:`RLinf-Co <rlinf_co>`
     - 2026
     - Sim-Real Co-Training
     - Reinforcement learning-based sim-real co-training for VLA models.
     - `arXiv:2602.12628 <https://arxiv.org/abs/2602.12628>`__
     - :doc:`Co-Training <../../examples/embodied/co_training>`
   * - :doc:`RLinf-VLA <rlinf_vla>`
     - 2026
     - Embodied VLA
     - Unified framework for VLA+RL training across simulators.
     - `RSS 2026 <https://roboticsproceedings.org/rss22/p089.html>`__
     - :doc:`LIBERO <../../examples/embodied/libero>`
   * - :doc:`STEAM <steam>`
     - 2026
     - Offline / Real-World
     - Self-supervised temporal ensemble advantage modeling for real-world robot learning.
     - `arXiv:2606.29834 <https://arxiv.org/abs/2606.29834>`__
     - :doc:`STEAM Recipe <../../examples/embodied/steam>`
   * - :doc:`USER <rlinf_user>`
     - 2026
     - Real-World RL
     - Unified system for real-world online policy learning.
     - `RSS 2026 <https://roboticsproceedings.org/rss22/p037.html>`__
     - :doc:`Franka Real-World RL <../../examples/embodied/franka>`
   * - :doc:`WideSeek-R1 <wideseek_r1>`
     - 2026
     - Multi-Agent RL
     - Exploring width scaling for broad information seeking via multi-agent RL.
     - `arXiv:2602.04634 <https://arxiv.org/abs/2602.04634>`__
     - :doc:`WideSeek-R1 <../../examples/agentic/wideseek_r1/index>`
   * - :doc:`WoVR <wovr>`
     - 2026
     - World Model RL
     - World model-based RL fine-tuning for VLA policies.
     - `arXiv:2602.13977 <https://arxiv.org/abs/2602.13977>`__
     - :doc:`Wan World Model <../../examples/embodied/wan>`

Cite RLinf
----------

If RLinf contributes to your research, cite the RLinf system paper:

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
