πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models
========================================================================

**论文：** `arXiv:2510.25889 <https://arxiv.org/abs/2510.25889>`__ 

概述
----

.. image:: https://github.com/RLinf/misc/raw/main/pic/pi_rl_teaser.png
   :alt: πRL 概览
   :align: center

πRL 在 **RLinf** 框架内为基于流的视觉-语言-动作（VLA）模型 π₀ 和 π₀.₅ 提供在线强化学习微调。通过将 PPO/GRPO 与流匹配策略相结合,该方法使少样本 SFT 模型能够通过环境反馈实现强大的操作性能。它支持 LIBERO、ManiSkill3、MetaWorld 和 CALVIN 基准测试,通过强化学习联合优化视觉理解、语言理解和连续动作生成。

结果
----

π₀ 模型
~~~~~~~

.. list-table:: π₀ 模型评估结果
   :header-rows: 1
   :widths: 15 25 15 20 20
   :align: left

   * - 环境
     - 任务
     - SFT
     - Flow-SDE
     - Flow-Noise
   * - LIBERO
     - Spatial, Object, Goal
     - `SFT <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT>`__
     - —
     - —
   * - LIBERO
     - Long
     - `SFT <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Long-SFT>`__
     - —
     - —
   * - ManiSkill3
     - Multi-task
     - 38.4%
     - `78.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowSDE>`__
     - `77.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowNoise>`__
   * - MetaWorld
     - MT50
     - 50.8%
     - `78.1% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowSDE>`__
     - `85.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise>`__
   * - CALVIN
     - ABC-D
     - 57.5%
     - `61.7% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowSDE>`__
     - `59.9% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowNoise>`__

π₀.₅ 模型
~~~~~~~~~

.. list-table:: π₀.₅ 模型评估结果
   :header-rows: 1
   :widths: 15 30 15 20 20
   :align: left

   * - 环境
     - 任务
     - SFT
     - Flow-SDE
     - Flow-Noise
   * - LIBERO
     - Spatial, Object, Goal, Long
     - `SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__
     - —
     - —
   * - ManiSkill3
     - Multi-task
     - 40.1%
     - `90.9% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE>`__
     - `89.7% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise>`__
   * - MetaWorld
     - MT50
     - 43.8%
     - `70.7% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE>`__
     - `66.1% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowNoise>`__
   * - CALVIN
     - ABC-D
     - 61.3%
     - `87.0% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE>`__
     - `84.5% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowNoise>`__

快速开始
--------

**完整指南：** :doc:`../examples/embodied/pi0`

**运行：** ``bash examples/embodiment/run_embodiment.sh <CONFIG_NAME>`` （配置文件位于 ``examples/embodiment/config/``）

**模型选择：**

- **π₀：** 名称中**不含** ``_pi05`` 的配置
- **π₀.₅：** 名称中**包含** ``_pi05`` 的配置（例如 ``*_openpi_pi05.yaml``）

**基准测试：**

- **LIBERO：** :doc:`../examples/embodied/libero`
- **ManiSkill3：** :doc:`../examples/embodied/maniskill`
- **MetaWorld：** :doc:`../examples/embodied/metaworld`
- **CALVIN：** :doc:`../examples/embodied/calvin`
- **Real2Sim2Real (GSEnv)：** :doc:`../examples/embodied/gsenv`

引用
----

.. code-block:: bibtex

   @article{chen2025pi_rl,
     title={$$\backslash$pi\_$\backslash$texttt $\{$RL$\}$ $: Online RL Fine-tuning for Flow-based Vision-Language-Action Models},
     author={Chen, Kang and Liu, Zhihao and Zhang, Tonghe and Guo, Zhen and Xu, Si and Lin, Hao and Zang, Hongzhi and Li, Xiang and Zhang, Quanlu and Yu, Zhaofei and others},
     journal={arXiv preprint arXiv:2510.25889},
     year={2025}
   }
