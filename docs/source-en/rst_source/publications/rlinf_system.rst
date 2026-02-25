RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation
==========================================================================================================

**Paper:** `arXiv:2509.15965 <https://arxiv.org/abs/2509.15965>`__

Overview
--------

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


RLinf is a flexible and scalable open-source infrastructure for post-training foundation models via reinforcement learning. It supports **reasoning RL** (e.g., math with GRPO), **embodied RL** (e.g., VLAs in simulators), and other scenarios. Built on the macro-to-micro flow transformation (M2Flow) paradigm, RLinf decouples logical workflow programming from execution planning and uses elastic pipelining, context switching, and profiling-guided scheduling to maximize throughput. Evaluations show **1.07×–2.43×** end-to-end training speedup over state-of-the-art systems: up to **1.7×** in reasoning RL and up to **2.43×** in embodied RL.

Results
-------

We extensively evaluate RLinf across math-reasoning and embodied RL workloads, covering four different models of different sizes (i.e., Qwen2.5, Qwen3-MoE, Open-VLA, OpenVLA-OFT), two RL algorithms (GRPO and PPO), and multiple cluster scales.

Math training performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf consistently outperforms state-of-the-art RL systems veRL and Slime by 1.07×∼1.70× on a variety of math-reasoning RL settings. The results also show that different RL settings favor different execution modes.

Dense models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_verl_all.jpg" alt="mani_openvla" width="350"/>
         <br/><strong>Throughput (GRPO)</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/latency_breakdown_7B.jpg" alt="mani_openvlaoft" width="350"/>
         <br/><strong>Latency breakdown (GRPO on 7B)</strong>
       </td>
     </tr>
   </table>
   </div>


The following figures shows the performance on PPO algorithm.

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_verl_ppo.jpg" alt="mani_openvla" width="350"/>
         <br/><strong>Throughput (PPO)</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_verl_breakdown.jpg" alt="mani_openvlaoft" width="350"/>
         <br/><strong>Latency breakdown (PPO on 7B, 32 GPUs)</strong>
       </td>
     </tr>
   </table>
   </div>


MoE models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For MoE models, we evaluate the Qwen3-30B-A3B on 32, 64, and 128 GPUs with a rollout batch size of 1536 and sequence length 20480. The following figures show the performance and latency breakdown on GRPO algorithm.


.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_slime_all.jpg" alt="mani_openvla" width="250"/>

         <br/><strong>Throughput</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/rlinf_vs_slime_breakdown.jpg" alt="mani_openvlaoft" width="350"/>
         <br/><strong>Latency breakdown (32 GPUs)</strong>
       </td>
     </tr>
   </table>
   </div>


Embodied training performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ManiSkill and LIBERO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We evaluate on OpenVLA and OpenVLA-OFT on ManiSkill and LIBERO, respectively. On LIBERO, we compare RLinf with SimpleVLA-RL (commit d001d), which is built on veRL. On ManiSkill, no distributed RL baseline exists, so we compare different execution modes of RLinf. Training speed is reported in **steps\/sec**, computed as the total number of environment steps divided by the iteration time.

.. raw:: html

  <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/eval_embody_all.jpg" alt="mani_openvla" width="450"/>

         <br/><strong>Throughput</strong>
       </td>
     </tr>
   </table>
   </div>

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-system/evaluation/latency_breakdown_libero_maniskill.jpg" alt="mani_openvla" width="500"/>

         <br/><strong>Latency breakdown</strong>
       </td>
     </tr>
   </table>
   </div>


Model evaluation performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tables report evaluation performance of models trained with RLinf (and baselines) on math benchmarks. RLinf-math models are trained with RLinf and evaluated on AIME 24, AIME 25, and GPQA-diamond.

1.5B model results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 1.5B model results
   :header-rows: 1
   :widths: 35 12 12 15 12
   :align: left

   * - Model
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - Average
   * - `DeepSeek-R1-Distill-Qwen-1.5B (base) <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`__
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
   * - **RLinf-math-1.5B** (`HuggingFace <https://huggingface.co/RLinf/RLinf-math-1.5B>`__)
     - **48.44**
     - **35.63**
     - **38.46**
     - **40.84**

\* Retrained using default settings for 600 steps.

7B model results
~~~~~~~~~~~~~~~~

.. list-table:: 7B model results
   :header-rows: 1
   :widths: 35 12 12 15 12
   :align: left

   * - Model
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - Average
   * - `DeepSeek-R1-Distill-Qwen-7B (base) <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`__
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
   * - **RLinf-math-7B** (`HuggingFace <https://huggingface.co/RLinf/RLinf-math-7B>`__)
     - 68.33
     - 52.19
     - **48.18**
     - **56.23**

RLinf achieves state-of-the-art performance on math reasoning tasks, consistently outperforming existing models across AIME 24, AIME 25, and GPQA-diamond for both 1.5B and 7B model sizes.

Quickstart
----------

- **Installation:** :doc:`../start/installation`
- **Math (reasoning) training:** :doc:`../start/llm`
- **Embodied training:** :doc:`../start/vla`

Citation
--------

.. code-block:: bibtex

   @article{yu2025rlinf,
     title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
     author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
     journal={arXiv preprint arXiv:2509.15965},
     year={2025}
   }
