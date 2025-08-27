Reasoning RL-LLM
=================

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

The following plots show training curves, consistent with those reported in AReaL:

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/user-attachments/assets/66b79012-f2e7-4b1d-9785-5d8f4f7d9025" style="width: 100%;"/>
       <p><em>MATH 1.5b</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/user-attachments/assets/37afd9f1-c503-49ec-8657-1f8f883a85c5" style="width: 100%;"/>
       <p><em>MATH 7b</em></p>
     </div>
   </div>


Final Performance
~~~~~~~~~~~~~~~~~

We provide an evaluation `toolkit <https://github.com/RLinf/LLMEvalKit>`_ and corresponding :doc:`evaluation documentation <../start/eval>`.

Measured performance on AIME24, AIME25, and GPQA-diamond shows RLinf achieves results comparable to or better than AReaL.

.. **1.5B model results**:

.. +---------------------------------------------------------------+--------+--------+--------------+
.. | Model                                                         | AIME24 | AIME25 | GPQA-diamond |
.. +===============================================================+========+========+==============+
.. | DeepSeek-R1-Distill-Qwen-1.5B                                 | 29.06  | 22.60  | 27.00        |
.. +---------------------------------------------------------------+--------+--------+--------------+
.. | AReaL-1.5B                                                    | 43.55  | 35.00  | 34.73        |
.. +---------------------------------------------------------------+--------+--------+--------------+
.. | `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`_ | 48.44  | 35.63  | 38.46        |
.. +---------------------------------------------------------------+--------+--------+--------------+

.. **7B model results**:

.. +-------------------------------------------------------------+--------+--------+--------------+
.. | Model                                                       | AIME24 | AIME25 | GPQA-diamond |
.. +=============================================================+========+========+==============+
.. | DeepSeek-R1-Distill-Qwen-7B                                 | 54.90  | 40.20  | 45.48        |
.. +-------------------------------------------------------------+--------+--------+--------------+
.. | AReaL-7B                                                    | 62.82  | 47.29  | 46.54        |
.. +-------------------------------------------------------------+--------+--------+--------------+
.. | `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`_ | 68.33  | 52.19  | 48.18        |
.. +-------------------------------------------------------------+--------+--------+--------------+

.. list-table:: **1.5 B model results**
   :header-rows: 1
   :widths: 45 15 15 25

   * - Model
     - AIME 24
     - AIME 25
     - GPQA-diamond
   * - DeepSeek-R1-Distill-Qwen-1.5B
     - 29.06
     - 22.60
     - 27.00
   * - AReaL-1.5B
     - 43.55
     - 35.00
     - 34.73
   * - RLinf-math-1.5B
     - 48.44
     - 35.63
     - 38.46

.. list-table:: **7 B model results**
   :header-rows: 1
   :widths: 45 15 15 25

   * - Model
     - AIME 24
     - AIME 25
     - GPQA-diamond
   * - DeepSeek-R1-Distill-Qwen-7B
     - 54.90
     - 40.20
     - 45.48
   * - AReaL-7B
     - 62.82
     - 47.29
     - 46.54
   * - RLinf-math-7B 
     - 68.33
     - 52.19
     - 48.18



Public Checkpoints
------------------

We release trained models on Hugging Face for public use:

- `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`_  
- `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`_  
