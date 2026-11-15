Learning from Fleet Experience: Collective Experience Learning Across 96 Robots
===============================================================================

**Paper:** `arXiv:2610.25736 <https://arxiv.org/abs/2610.25736>`__

Overview
--------

Robot fleets can improve a shared policy using experience from their own
executions. Collective Experience Learning (CEL) pools this experience and
returns updated policies to the fleet. After expert-supervised initialization,
CEL learns without additional expert demonstrations or corrective actions;
operators supply binary task-outcome labels and reset environments.

Failure-guided Experience Retrieval Fine-Tuning (FERT) uses failure contexts to
retrieve relevant segments from successful trajectories. It prioritizes these
segments during fine-tuning, using the robots' recorded actions as supervision.
The Fault-tolerant Asynchronous Training System (FATS), built on RLinf,
coordinates robot execution, cloud inference, and policy updates. Local fault
recovery lets available robots keep collecting experience during interruptions.

.. figure:: https://arxiv.org/html/2610.25736v3/overview.png
   :alt: CEL fleet experience sharing, FERT learning, and FATS system overview
   :width: 800px
   :align: center

   CEL combines failure-guided learning from successful executions with
   asynchronous fleet operation.

Results
-------

Experiments cover five manipulation tasks on dual-arm AgiBot G2 robots. Each
policy is evaluated over 120 trials on two robots excluded from training data
collection. The table compares the (M)HG-DAgger initialization with FERT at the
largest evaluated fleet size for each task.

.. list-table:: Task success rates before and after learning from fleet experience
   :header-rows: 1
   :widths: 34 22 22 22
   :align: left

   * - Task
     - Training Robots
     - (M)HG-DAgger (%)
     - FERT (%)
   * - Flower Arrangement
     - 16
     - 98.3
     - **99.2**
   * - Clothes Folding
     - 16
     - 85.8
     - **98.3**
   * - Phone Charging
     - 16
     - 60.0
     - **97.5**
   * - Stool Assembly
     - 32
     - 81.7
     - **93.3**
   * - Phone Packaging
     - 96
     - 79.2
     - **98.3**
   * - Mean
     - —
     - 81.0
     - **97.3**

Quickstart
----------

CEL code is pending release. The project page provides method details and
robot demonstrations.

- `CEL Project Page <https://cel-robot-learning.github.io/>`__

Citation
--------

.. code-block:: bibtex

   @misc{yu2026learningfleetexperiencecollective,
     title         = {Learning from Fleet Experience: Collective Experience Learning Across 96 Robots},
     author        = {Chao Yu and Shu'ang Yu and Hao Lin and Hongzhi Zang
                      and Yuxiang Yan and Chenxi Jiang and Zhen Guo
                      and Chuankang Li and Yihao Chen and Xin Xu and Xu Fu
                      and Jiakai Zhou and Liangzhi Shi and Hongyi Ren
                      and Yiwei Liu and Jing Wu and Yuanqing Wang and Chen Wu
                      and Bo Dai and Chunyu Yang and Hongliang Huang
                      and Shuoyu Yue and Weiqi Yin and Yinuo Chen
                      and Ruihua Chen and Chunyang Zhu and Han Xiao
                      and Yanghao Li and Ruipu Wu and Zhikai Cai and Xiu Chen
                      and Xu Tan and Si Liu and Yuchen Yan and Jialin Zhang
                      and Wen Zeng and Yiming Zhou and Zixiao Huang
                      and Xinlei Chen and Guanghui Ren and Quanlu Zhang
                      and Boxun Li and Maoqing Yao and Yu Wang},
     year          = {2026},
     eprint        = {2610.25736},
     archivePrefix = {arXiv},
     primaryClass  = {cs.RO},
     url           = {https://arxiv.org/abs/2610.25736}
   }
