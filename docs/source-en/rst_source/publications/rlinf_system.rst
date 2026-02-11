RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation
==========================================================================================================

**Paper:** `arXiv:2509.15965 <https://arxiv.org/abs/2509.15965>`__ 

Overview
--------

RLinf is a flexible and scalable open-source infrastructure for post-training foundation models via reinforcement learning. This page summarizes the **reasoning scenario**: using RL to train large language models (LLMs) for mathematical reasoning. Compared to supervised fine-tuning (SFT), RL encourages models to explore diverse reasoning paths while prioritizing correct final answers.

Results
-------

1.5B model results
~~~~~~~~~~~~~~~~~~

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
- **Run training:** :doc:`../start/llm`

Citation
--------

.. code-block:: bibtex

   @article{yu2025rlinf,
     title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
     author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
     journal={arXiv preprint arXiv:2509.15965},
     year={2025}
   }
