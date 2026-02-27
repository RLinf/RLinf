RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation
==========================================================================================================

**论文：** `arXiv:2509.15965 <https://arxiv.org/abs/2509.15965>`__

概述
----

.. raw:: html

  <div align="center">
    <table border="0">
      <tr>
        <td align="center">
          <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/rlinf_arch.jpg" alt="mani_openvla" width="450"/>
        </td>
      </tr>
    </table>
    </div>


RLinf 是面向基础模型后训练的灵活可扩展开源强化学习基础设施。它支持 **推理 RL** （如使用 GRPO 的数学推理）、**具身 RL** （如在仿真器中训练 VLA）等多种场景。RLinf 基于宏到微流转换（M2Flow）范式，将逻辑工作流编程与执行规划解耦，并利用弹性流水线、上下文切换和基于性能分析的调度来最大化吞吐量。评测表明，RLinf 实现了 **1.07×–2.43×** 的端到端训练加速：推理 RL 场景最高可达 **1.7×**，具身 RL 场景最高可达 **2.43×**。

结果
----

我们在数学推理和具身 RL 工作负载上对 RLinf 进行了全面评估，涵盖四种不同规模的模型（即 Qwen2.5、Qwen3-MoE、Open-VLA、OpenVLA-OFT）、两种 RL 算法（GRPO 和 PPO）以及多种集群规模。

数学训练性能
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 在多种数学推理 RL 设置下，吞吐量始终优于最先进的 RL 系统 veRL 和 Slime，提升幅度为 1.07×∼1.70×。结果还表明，不同的 RL 设置适合不同的执行模式。

稠密模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_verl_all.jpg" alt="mani_openvla" width="350"/>
         <br/><strong>吞吐量（GRPO）</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/latency_breakdown_7B.jpg" alt="mani_openvlaoft" width="350"/>
         <br/><strong>耗时占比（GRPO，7B）</strong>
       </td>
     </tr>
   </table>
   </div>


下图展示了 PPO 算法上的性能表现。

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_verl_ppo.jpg" alt="mani_openvla" width="350"/>
         <br/><strong>吞吐量（PPO）</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_verl_breakdown.jpg" alt="mani_openvlaoft" width="350"/>
         <br/><strong>耗时占比（PPO，7B，32 GPUs）</strong>
       </td>
     </tr>
   </table>
   </div>


MoE 模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于 MoE 模型，我们在 32、64 和 128 GPU 上评估了 Qwen3-30B-A3B，rollout batch size 为 1536，序列长度为 20480。下图展示了 GRPO 算法上的性能和耗时占比。


.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_slime_all.jpg" alt="mani_openvla" width="250"/>

         <br/><strong>吞吐量</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_slime_breakdown.jpg" alt="mani_openvlaoft" width="350"/>
         <br/><strong>耗时占比（32 GPUs）</strong>
       </td>
     </tr>
   </table>
   </div>


具身训练性能
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ManiSkill 和 LIBERO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们分别在 ManiSkill 和 LIBERO 上评估了 OpenVLA 和 OpenVLA-OFT。在 LIBERO 上，我们将 RLinf 与 SimpleVLA-RL（commit d001d，基于 veRL 构建）进行对比。在 ManiSkill 上，由于没有分布式 RL 基线，我们比较了 RLinf 的不同执行模式。训练速度以 **steps\/sec** 报告，即环境步数总量除以迭代时间。

.. raw:: html

  <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/eval_embody_all.jpg" alt="mani_openvla" width="450"/>

         <br/><strong>吞吐量</strong>
       </td>
     </tr>
   </table>
   </div>

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/latency_breakdown_libero_maniskill.jpg" alt="mani_openvla" width="500"/>

         <br/><strong>耗时占比</strong>
       </td>
     </tr>
   </table>
   </div>


模型评估性能
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下表报告了使用 RLinf（及基线模型）在数学基准上训练的模型评估性能。RLinf-math 模型使用 RLinf 训练，并在 AIME 24、AIME 25 和 GPQA-diamond 上进行评估。

1.5B 模型结果
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 1.5B 模型结果
   :header-rows: 1
   :widths: 35 12 12 15 12
   :align: left

   * - 模型
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - 平均
   * - `DeepSeek-R1-Distill-Qwen-1.5B（基座） <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`__
     - 28.33
     - 24.90
     - 27.45
     - 26.89
   * - `DeepMath-1.5B <https://huggingface.co/zwhe99/DeepMath-1.5B>`__
     - 37.80
     - 30.42
     - 32.11
     - 33.44
   * - `DeepScaleR-1.5B-Preview <https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview>`__
     - 40.41
     - 30.93
     - 27.54
     - 32.96
   * - `AReaL-1.5B-Preview-Stage-3 <https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-3>`__
     - 40.73
     - 31.56
     - 28.10
     - 33.46
   * - AReaL-1.5B-retrain\*
     - 44.42
     - 34.27
     - 33.81
     - 37.50
   * - `FastCuRL-1.5B-V3 <https://huggingface.co/Nickyang/FastCuRL-1.5B-V3>`__
     - 43.65
     - 32.49
     - 35.00
     - 37.05
   * - **RLinf-math-1.5B** （`HuggingFace <https://huggingface.co/RLinf/RLinf-math-1.5B>`__）
     - **48.44**
     - **35.63**
     - **38.46**
     - **40.84**

\* 使用默认设置重训 600 步。

7B 模型结果
~~~~~~~~~~~~~~~~

.. list-table:: 7B 模型结果
   :header-rows: 1
   :widths: 35 12 12 15 12
   :align: left

   * - 模型
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - 平均
   * - `DeepSeek-R1-Distill-Qwen-7B（基座） <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`__
     - 54.90
     - 40.20
     - 45.48
     - 46.86
   * - `AReaL-boba-RL-7B <https://huggingface.co/inclusionAI/AReaL-boba-RL-7B>`__
     - 61.66
     - 49.38
     - 46.93
     - 52.66
   * - `Skywork-OR1-7B <https://huggingface.co/Skywork/Skywork-OR1-7B>`__
     - 66.87
     - 52.49
     - 44.43
     - 54.60
   * - `Polaris-7B-Preview <https://huggingface.co/POLARIS-Project/Polaris-7B-Preview>`__
     - **68.55**
     - 51.24
     - 43.88
     - 54.56
   * - `AceMath-RL-Nemotron-7B <https://huggingface.co/nvidia/AceMath-RL-Nemotron-7B>`__
     - 67.30
     - **55.00**
     - 45.57
     - 55.96
   * - **RLinf-math-7B** （`HuggingFace <https://huggingface.co/RLinf/RLinf-math-7B>`__）
     - 68.33
     - 52.19
     - **48.18**
     - **56.23**

RLinf 在数学推理任务上达到当前最优水平，在 1.5B 与 7B 规模下于 AIME 24、AIME 25、GPQA-diamond 等基准上均优于已有模型。

快速开始
--------

- **安装：** :doc:`../start/installation`
- **数学（推理）训练：** :doc:`../start/llm`
- **具身训练：** :doc:`../start/vla`

引用
----

.. code-block:: bibtex

   @article{yu2025rlinf,
     title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
     author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
     journal={arXiv preprint arXiv:2509.15965},
     year={2025}
   }
