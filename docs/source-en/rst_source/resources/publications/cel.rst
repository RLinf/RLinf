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

Real-World Task Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Experiments cover five manipulation tasks on dual-arm AgiBot G2 robots. Each
policy is evaluated over 120 trials on two robots excluded from training data
collection. RLT, RECAP, and FERT start from the same selected task-specific
(M)HG-DAgger checkpoint.

.. figure:: https://arxiv.org/html/2610.25736v3/all.png
   :alt: AgiBot G2 performing flower arrangement, clothes folding, phone charging, stool assembly, and phone packaging
   :width: 800px
   :align: center

   The five real-world manipulation tasks.

FERT raises mean success from 81.0% after initialization to 97.3%, an 86.0%
reduction in mean failure rate. In the evaluated configurations, its mean success
exceeds RLT and RECAP by 12.3 and 11.5 percentage points, respectively. The
comparison also includes SFT and human teleoperation as references.

.. figure:: https://arxiv.org/html/2610.25736v3/all_tasks_success_by_method_grid.svg
   :alt: Success rates for human teleoperation, SFT, multi-robot HG-DAgger, RLT, RECAP, and FERT on five tasks
   :width: 800px
   :align: center

   Task success across methods, with 95% Wilson confidence intervals.
   FERT uses the largest evaluated fleet for each task, listed below.

.. list-table:: Task success rates before and after learning from fleet experience
   :header-rows: 1
   :widths: 34 22 22 22
   :align: left

   * - Task
     - FERT Training Robots
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

Scaling with Fleet Size
~~~~~~~~~~~~~~~~~~~~~~~

With collection time per robot held fixed, larger fleets supply more experience.
On Phone Packaging, FERT success increases from 86.7% with 16 training robots to
98.3% with 96. The four-robot policy reaches 50.8%, below the initial policy's
79.2%.

.. figure:: https://arxiv.org/html/2610.25736v3/all_tasks_success_by_robots_combined.svg
   :alt: Task success as a function of training fleet size for HG-DAgger and FERT
   :width: 800px
   :align: center

   Success versus training fleet size. HG-DAgger starts from SFT;
   FERT starts from the selected (M)HG-DAgger checkpoint. Dashed lines mark
   these different initial policies; error bars show 95% Wilson intervals.

Ablation Studies
~~~~~~~~~~~~~~~~

On Phone Packaging with 16 robots, full FERT reaches 86.7% success. Removing
failure-guided retrieval lowers success to 64.2%; removing the context extensions
around retrieved segments lowers it to 57.5%. The ablation plots also compare policy
synchronization intervals and priority retention settings.
All panels report 95% Wilson confidence intervals.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://arxiv.org/html/2610.25736v3/rft_ablation_fr.svg
         :alt: FERT success with and without failure-guided retrieval
         :width: 100%

         Failure-Guided Retrieval

   .. grid-item::

      .. figure:: https://arxiv.org/html/2610.25736v3/rft_ablation_interval.svg
         :alt: FERT success at different policy synchronization intervals
         :width: 100%

         Policy Synchronization Interval

   .. grid-item::

      .. figure:: https://arxiv.org/html/2610.25736v3/rft_ablation_extend.svg
         :alt: FERT success with different context extensions around retrieved segments
         :width: 100%

         Retrieved-Segment Context

   .. grid-item::

      .. figure:: https://arxiv.org/html/2610.25736v3/rft_ablation_decay.svg
         :alt: FERT success at different priority retention settings
         :width: 100%

         Priority Retention

System Reliability and Collection Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 96-robot run collects 9,834 episodes despite 30 fault incidents affecting
22 robots. Faults consume 1.3% of total robot time. Outside the lunch break,
the fleet completes 24 episodes per minute, with 65 robots executing at a time
on average.

Replaying the recorded per-robot traces under synchronous collection retains
33% of the measured episode count. Whole-system restarts lasting 2, 5, or
10 minutes retain 92%, 86%, or 74%, respectively. These alternative-design
results are trace-replay estimates; the FATS run supplies the measured baseline.

.. figure:: https://arxiv.org/html/2610.25736v3/fats_trace.svg
   :alt: FATS collection rate over time, fault incidents, trace-replay comparisons, and episodes collected per robot
   :width: 800px
   :align: center

   Measured collection and faults in the 96-robot run, together with
   trace-replay estimates for synchronous collection and whole-system restarts.

Edge-Cloud Communication
~~~~~~~~~~~~~~~~~~~~~~~~

The robot site and cloud servers share a dedicated 30 Gbps link. The following
measurements cover channel operations, including waits for data or computation.
Robots complete an episode about every four minutes. More than 99% of
experience transfers finish within two minutes.

.. figure:: https://arxiv.org/html/2610.25736v3/channel_time_20260813_bars.svg
   :alt: Duration distributions for observation, action, and experience-transfer channel operations
   :width: 800px
   :align: center

   Measured duration distributions and operation counts for the
   observation, action, and experience channels.

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
