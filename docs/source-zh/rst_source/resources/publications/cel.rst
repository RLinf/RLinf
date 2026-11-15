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

真机任务表现
~~~~~~~~~~~~

实验使用双臂 AgiBot G2 机器人，包含五项操作任务。每项任务的每个 policy 都在两台未参与训练数据采集的机器人上测试，共 120 次。RLT、RECAP 和 FERT 从同一份选定的任务专用（M）HG-DAgger checkpoint 开始训练。

.. figure:: https://arxiv.org/html/2610.25736v3/all.png
   :alt: AgiBot G2 执行插花、叠衣、手机充电、凳子组装和手机包装
   :width: 800px
   :align: center

   五项真机操作任务。

FERT 将初始化后的平均成功率从 81.0% 提高到 97.3%，平均失败率下降 86.0%。在论文的实验配置下，FERT 的平均成功率比 RLT 和 RECAP 分别高 12.3、11.5 个百分点。对比图还列出了 SFT 与人工遥操作的结果。

.. figure:: https://arxiv.org/html/2610.25736v3/all_tasks_success_by_method_grid.svg
   :alt: 人工遥操作、SFT、多机器人 HG-DAgger、RLT、RECAP 和 FERT 在五项任务上的成功率
   :width: 800px
   :align: center

   不同方法的任务成功率，误差条为 95% Wilson 置信区间。FERT 采用各任务测试过的最大训练集群规模，具体数量见下表。

.. list-table:: 集群经验学习前后的任务成功率
   :header-rows: 1
   :widths: 34 22 22 22
   :align: left

   * - 任务
     - FERT 训练机器人数量
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

机器人集群规模与成功率
~~~~~~~~~~~~~~~~~~~~~~

实验固定每台机器人的采集时间，集群越大，收集的总经验越多。手机包装任务中，FERT 使用 16 台机器人训练后成功率为 86.7%，使用 96 台后达到 98.3%。4 台机器人训练后的成功率为 50.8%，低于初始 policy 的 79.2%。

.. figure:: https://arxiv.org/html/2610.25736v3/all_tasks_success_by_robots_combined.svg
   :alt: HG-DAgger 和 FERT 的任务成功率随训练机器人数量变化的曲线
   :width: 800px
   :align: center

   训练集群规模与任务成功率。HG-DAgger 从 SFT 开始，FERT 从选定的（M）HG-DAgger checkpoint 开始。虚线标出各自的初始成功率，误差条为 95% Wilson 置信区间。

消融实验
~~~~~~~~

消融实验使用 16 台机器人训练手机包装任务。完整 FERT 的成功率为 86.7%；去掉失败引导检索后降至 64.2%，去掉检索片段前后的上下文扩展后降至 57.5%。消融实验还比较了 policy 同步间隔和采样优先级保留设置，各子图的误差条为 95% Wilson 置信区间。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://arxiv.org/html/2610.25736v3/rft_ablation_fr.svg
         :alt: 保留和移除失败引导检索的 FERT 成功率
         :width: 100%

         失败引导检索

   .. grid-item::

      .. figure:: https://arxiv.org/html/2610.25736v3/rft_ablation_interval.svg
         :alt: 不同 policy 同步间隔下的 FERT 成功率
         :width: 100%

         Policy 同步间隔

   .. grid-item::

      .. figure:: https://arxiv.org/html/2610.25736v3/rft_ablation_extend.svg
         :alt: 不同检索片段上下文扩展长度下的 FERT 成功率
         :width: 100%

         检索片段上下文

   .. grid-item::

      .. figure:: https://arxiv.org/html/2610.25736v3/rft_ablation_decay.svg
         :alt: 不同采样优先级保留设置下的 FERT 成功率
         :width: 100%

         采样优先级保留

系统可靠性与采集效率
~~~~~~~~~~~~~~~~~~~~

96 台机器人在本次运行中共采集 9,834 个 episode，期间 22 台机器人发生 30 次故障。故障造成的时间损失占机器人总运行时间的 1.3%。午休以外，平均有 65 台机器人同时执行任务，集群每分钟完成 24 个 episode。

论文回放各台机器人的实测运行记录，估算其他系统设计下的采集量。同步采集的 episode 数量为实测值的 33%；若每次故障后全系统重启耗时为 2、5、10 分钟，采集量分别为实测值的 92%、86%、74%。这些替代方案的数字来自轨迹回放估计，比较基准是 FATS 的实测采集结果。

.. figure:: https://arxiv.org/html/2610.25736v3/fats_trace.svg
   :alt: FATS 采集速率时间曲线、故障记录、轨迹回放对比和各机器人采集的 episode 数量
   :width: 800px
   :align: center

   96 台机器人运行期间的实测采集与故障记录，以及同步采集、全系统重启方案的轨迹回放估计。

端云通信
~~~~~~~~

机器人现场与云端服务器使用专用 30 Gbps 链路。下图给出通道操作的耗时统计，包括等待数据或计算的时间。机器人约每四分钟完成一个 episode，超过 99% 的经验传输在两分钟内完成。

.. figure:: https://arxiv.org/html/2610.25736v3/channel_time_20260813_bars.svg
   :alt: 观测、动作和经验传输通道操作的耗时分布
   :width: 800px
   :align: center

   观测、动作与经验通道的实测操作耗时分布和操作次数。

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
