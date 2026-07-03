STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning
=========================================================================================

**Paper:** `arXiv:2606.29834 <https://arxiv.org/abs/2606.29834>`__

Overview
--------

Real-world robot learning increasingly relies on heterogeneous data, but
demonstrations and rollouts often mix useful progress with stalls, corrections,
and suboptimal behavior. Effective policy learning therefore needs frame-level
advantages that distinguish reliable local progress from failures and
regressions.

**STEAM** (Self-supervised Temporal Ensemble Advantage Modeling) is a label-free
method that learns such advantages from expert demonstrations, without manual
annotations or hand-crafted rewards. STEAM trains an ensemble of temporal-offset
predictors on frame pairs drawn from expert trajectories, using the normalized
temporal offset between two frames as a self-supervised signal. Each predictor
maps a frame pair to a distribution over temporal offsets, which is converted
into a scalar advantage; STEAM then takes the **minimum advantage across the
ensemble** (worst-of-N) to score mixed-quality rollout data conservatively and
suppress the over-estimation a single predictor would assign to
out-of-distribution data.

The resulting advantages apply to expert data, human corrections, and policy
rollouts. When combined with CFGRL — the same classifier-free guidance training
used by RECAP — STEAM substantially improves policy performance on real-world
tasks.

Results
-------

Across real-world bimanual towel folding, chip checkout, cola restocking, and
single-arm pick-and-place tasks, STEAM identifies stalls, failures, and
recoveries; combined with CFGRL it improves downstream policy performance over
the RECAP baseline.

Quick Start
-----------

- **Instruction:** :doc:`../../examples/embodied/steam`

Citation
--------

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
