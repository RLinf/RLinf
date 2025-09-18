Reasoning RL-LLM
=================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document introduces how we train large language models (LLMs) for mathematical reasoning using reinforcement learning (RL) in the RLinf framework.
Compared with supervised fine-tuning (SFT), RL encourages the model to explore diverse reasoning paths while prioritizing correct final answers.

Our goal is to improve the model's ability to solve challenging math problems by optimizing both its reasoning process and its final answers.


Dataset
-------------

We use the dataset from `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`_.  
This dataset integrates data from DeepScaleR, Open-Reasoner-Zero, Light-R1, DAPO, NuminaMath (AoPS/Olympiad subsets), and ZebraLogic.  
Overly simple problems are filtered out to ensure dataset quality and effectiveness.

An example training sample looks like:

.. code-block:: json

   {
      "prompt": "<｜User｜>\nProblem description... Please reason step by step, and put your final answer within \\boxed{}.<｜Assistant｜><think>\n",
      "task": "math",
      "query_id": "xx",
      "solutions": ["\\boxed{x}"]
   }

Algorithm
---------

We adopt GRPO (Group Relative Policy Optimization) with the following modifications:

- Token-level loss: Instead of averaging loss over the entire response sequence, we compute the average over tokens, as in DAPO.  
  This prevents excessively long responses from dominating training and reduces their gradient impact.

- Minibatch early-stop: If the importance ratio within a minibatch becomes too large, we discard that minibatch to stabilize training.

Reward function:

- +5 if the final boxed/numeric answer is correct;
- -5 if incorrect.

Running the Script
---------------------

**1. Key Parameters Configuration**

Before launching, check the configuration file. Key fields include:

- Cluster setup: ``cluster.num_nodes`` (number of nodes), ``cluster.num_gpus_per_node`` (GPUs per node).  
- Paths: ``runner.output_dir`` (the path to save training logs & checkpoints), ``rollout.model_dir`` (the path that saves base model), ``data.train_data_paths`` (the path that save training data), etc.  

**2. Configuration File**

Recommended configurations can be found in:

- ``examples/math/config/qwen2.5-1.5b-grpo-megatron.yaml``  
- ``examples/math/config/qwen2.5-7b-grpo-megatron.yaml``  

**3. Launch Command**

Run the following commands to start the Ray cluster and begin training:

.. code-block:: bash

   cd /path_to_RLinf/ray_utils;
   rm /path_to_RLinf/ray_utils/ray_head_ip.txt;
   export TOKENIZERS_PARALLELISM=false
   bash start_ray.sh;
   if [ "$RANK" -eq 0 ]; then
       bash check_ray.sh 128; # set to cluster.num_nodes*cluster.num_gpus_per_node
       cd /path_to_RLinf;
       bash examples/math/qwen2.5/run_main_math_grpo_megatron.sh grpo-1.5b-megatron # change config file
   else
     if [ "$RANK" -eq 1 ]; then
         sleep 3m
     fi
     sleep 10d
   fi

   sleep 10d

Results
-------

We trained both 1.5B and 7B models based on DeepSeek-R1-Distill-Qwen.  

After successfully launched your training, you can monitor the metrics with:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

Key metrics to track:

- ``rollout/rewards``: Accuracy of model responses on training data. Higher scores normally suggest stronger reasoning ability.  
- ``rollout/response_length``: Average response length for the training dataset. RL often causes verbosity, and DAPO-like strategies mitigate this problem.  
- ``train/entropy_loss``: Representing the exploration ability of the model. Entropy should decrease and slowly converge.  

Training Curve
~~~~~~~~~~~~~~

The following plots show training curves.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/1.5b-loss-curve.jpg" style="width: 100%;"/>
       <p><em>MATH 1.5B</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/7b-loss-curve.jpg" style="width: 100%;"/>
       <p><em>MATH 7B</em></p>
     </div>
   </div>


Final Performance
~~~~~~~~~~~~~~~~~

We provide an evaluation `toolkit <https://github.com/RLinf/LLMEvalKit>`_ and corresponding :doc:`evaluation documentation <../start/llm-eval>`.

Measured performance on AIME24, AIME25, and GPQA-diamond shows RLinf achieves SOTA performance.

.. list-table:: **1.5 B model results**
   :header-rows: 1
   :widths: 45 15 15 25 15

   * - Model
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - Average
   * - |huggingface| `DeepSeek-R1-Distill-Qwen-1.5B (base model) <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_
     - 28.33
     - 24.90
     - 27.45
     - 26.89
   * - |huggingface| `DeepMath-1.5B <https://huggingface.co/zwhe99/DeepMath-1.5B>`_
     - 37.80
     - 30.42
     - 32.11
     - 33.44
   * - |huggingface| `DeepScaleR-1.5B-Preview <https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview>`_
     - 40.41
     - 30.93
     - 27.54
     - 32.96
   * - |huggingface| `AReaL-1.5B-Preview-Stage-3 <https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-3>`_
     - 40.73
     - 31.56
     - 28.10
     - 33.46
   * - AReaL-1.5B-retrain\*
     - 44.42
     - 34.27
     - 33.81
     - 37.50
   * - |huggingface| `FastCuRL-1.5B-V3 <https://huggingface.co/Nickyang/FastCuRL-1.5B-V3>`_
     - 43.65
     - 32.49
     - 35.00
     - 37.05
   * - |huggingface| `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`_
     - **48.44**
     - **35.63**
     - **38.46**
     - **40.84**

\* We retrain the model using the default settings for 600 steps.

.. list-table:: **7 B model results**
   :header-rows: 1
   :widths: 45 15 15 25 15

   * - Model
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - Average
   * - |huggingface| `DeepSeek-R1-Distill-Qwen-7B (base model) <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`_
     - 54.90
     - 40.20
     - 45.48
     - 46.86
   * - |huggingface| `AReaL-boba-RL-7B <https://huggingface.co/inclusionAI/AReaL-boba-RL-7B>`_
     - 61.66
     - 49.38
     - 46.93
     - 52.66
   * - |huggingface| `Skywork-OR1-7B <https://huggingface.co/Skywork/Skywork-OR1-7B>`_
     - 66.87
     - 52.49
     - 44.43
     - 54.60
   * - |huggingface| `Polaris-7B-Preview <https://huggingface.co/POLARIS-Project/Polaris-7B-Preview>`_
     - **68.55**
     - 51.24
     - 43.88
     - 54.56
   * - |huggingface| `AceMath-RL-Nemotron-7B <https://huggingface.co/nvidia/AceMath-RL-Nemotron-7B>`_
     - 67.30
     - **55.00**
     - 45.57
     - 55.96
   * - |huggingface| `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`_
     - 68.33
     - 52.19
     - **48.18**
     - **56.23**


Public Checkpoints
------------------

We release trained models on Hugging Face for public use:

- `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`_  
- `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`_  
