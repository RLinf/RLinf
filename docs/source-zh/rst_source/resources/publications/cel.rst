Learning from Fleet Experience: Collective Experience Learning Across 96 Robots
===============================================================================

**论文：** `arXiv:2610.25736 <https://arxiv.org/abs/2610.25736>`__

概述
----

机器人集群在执行任务时积累经验，可以用来改进共享的控制模型（policy）。Collective Experience Learning（CEL，集体经验学习）汇集这些经验，训练后将更新的 policy 发回集群。完成专家监督的初始化后，后续学习无需新增专家示教或纠错动作，操作人员只需标注任务成败并复位环境。

CEL 根据失败时的执行情况确定学习重点。Failure-guided Experience Retrieval Fine-Tuning（FERT）从成功轨迹中检索相关片段，微调时优先采样这些片段，用机器人自身的动作作为监督信号。基于 RLinf 的容错异步训练系统 FATS 协调真机执行、云端推理和 policy 更新。发生中断时，系统在局部执行故障恢复，其他可用机器人继续采集经验。

.. figure:: https://arxiv.org/html/2610.25736v3/overview.png
   :alt: CEL 集群经验共享、FERT 学习与 FATS 系统概览
   :width: 800px
   :align: center

   CEL 根据失败经验确定学习重点，利用成功轨迹训练 policy，并支持机器人集群异步运行。

结果
----

实验使用双臂 AgiBot G2 机器人，包含五项操作任务。每项任务的每个 policy 都在两台未参与训练数据采集的机器人上测试，共 120 次。下表对比（M）HG-DAgger 初始化与 FERT 学习后的成功率，FERT 采用各任务测试过的最大训练集群规模。

.. list-table:: 集群经验学习前后的任务成功率
   :header-rows: 1
   :widths: 34 22 22 22
   :align: left

   * - 任务
     - 训练机器人数量
     - （M）HG-DAgger（%）
     - FERT（%）
   * - 插花
     - 16
     - 98.3
     - **99.2**
   * - 叠衣
     - 16
     - 85.8
     - **98.3**
   * - 手机充电
     - 16
     - 60.0
     - **97.5**
   * - 凳子组装
     - 32
     - 81.7
     - **93.3**
   * - 手机包装
     - 96
     - 79.2
     - **98.3**
   * - 平均
     - —
     - 81.0
     - **97.3**

快速开始
--------

CEL 代码尚未发布，方法介绍与真机演示见项目主页。

- `CEL 项目主页 <https://cel-robot-learning.github.io/>`__

引用
----

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
