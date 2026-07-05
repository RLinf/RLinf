STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning
=========================================================================================

**论文：** `arXiv:2606.29834 <https://arxiv.org/abs/2606.29834>`__

概述
----

真实机器人学习越来越依赖异构数据，但演示与 rollout 常常把有效进展与停滞、
纠正和次优行为混在一起。因此，有效的策略学习需要帧级优势，以区分可靠的局部
进展与失败和倒退。

**STEAM**\ （Self-supervised Temporal Ensemble Advantage Modeling，自监督时序
集成优势建模）是一种无需标注的方法，可从专家演示中学习这类优势，无需人工标注
或手工设计奖励。STEAM 在专家轨迹的帧对上训练一组时序偏移预测器，以两帧之间
归一化的时序偏移作为自监督信号；每个预测器将帧对映射为时序偏移上的分布，并
转换为标量优势；随后 STEAM 取集成中的 **最小优势**\ （worst-of-N），对混合质量
的 rollout 数据进行保守评分，抑制单一预测器对分布外数据的优势高估。

得到的优势可用于专家数据、人工纠正和策略 rollout。与 CFGRL（与 RECAP 相同的
classifier-free guidance 训练）结合后，STEAM 在多项真机任务上显著提升策略
性能。

结果
----

在真实世界的双臂叠毛巾、芯片结账、可乐补货以及单臂抓取放置任务上，STEAM 能够
识别停滞、失败与恢复；与 CFGRL 结合后，相较 RECAP 基线进一步提升下游策略性能。

快速开始
--------

- **教程：** :doc:`../../examples/embodied/steam`

引用
----

.. code-block:: bibtex

   @misc{liu2026steam,
     title         = {STEAM: Self-Supervised Temporal Ensemble Advantage
                      Modeling for Real-World Robot Learning},
     author        = {Liu, Zhihao and Gu, Qiuyi and Wang, Yitao and Qiao, Dongming
                      and Zhang, Yixian and Chen, Shuaihang and Shi, Liangzhi
                      and Zhou, Tianxing and Huang, Zefang and Chen, Kang
                      and Guo, Zhen and Zhang, Quanlu and Yu, Jincheng
                      and Liang, Xiaodan and Fan, Guoliang and Wang, Yu
                      and Gao, Feng and Chen, Xinlei and Yu, Chao},
     year          = {2026},
     eprint        = {2606.29834},
     archivePrefix = {arXiv},
     primaryClass  = {cs.RO}
   }
