Quickstart 2: GRPO Training of LLMs on MATH
==============================================

This quick-start walks you through training
`DeepSeek-R1-Distill-Qwen-1.5B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_
on the
`AReaL-boba <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data>`_
math-reasoning dataset with **RLinf**.  
For maximum simplicity, you can run the following scripts within a single GPU.

Dataset Introduction
--------------------

*AReaL-boba* covers a broad spectrum of mathematical and logical
problems. A example is shown below.

.. code-block:: text

   Question
   --------
   What is the unit digit of the product
   \[
     (5+1)\,(5^{3}+1)\,(5^{6}+1)\,(5^{12}+1)
   \]?
   (a) 0   (b) 1   (c) 2   (d) 5   (e) 6
   Please reason step-by-step and put your final answer within \boxed{}.

   Answer
   ------
   [ "\\boxed{e}" ]

Launch Training
-----------------

**Step 1: Download the model and the datasets:**

.. code-block:: bash

   # model
   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --local-dir /model/DeepSeek-R1-Distill-Qwen-1.5B

   # dataset
   hf download inclusionAI/AReaL-boba-Data --repo-type=dataset \
   --local-dir /dataset/boba

**Step 2: Execute the provided launch script:**

For user convenience, our configuration file is set up to run with a single GPU by default.  
However, if you have multiple GPUs and wish to accelerate the quickstart process,  
we highly recommend updating the following configuration option in  
``./examples/math/config/qwen2.5-1.5b-single-gpu.yaml``:  
``cluster.num_gpus_per_node``.


You can dynamically set it to **1, 2, 4, or 8** depending on your available resources.

.. code-block:: yaml

   cluster:
     num_nodes: 1
     num_gpus_per_node: 1
     component_placement:
        actor,rollout: all


.. code-block:: bash

   bash examples/math/run_main_math_grpo_megatron.sh qwen2.5-1.5b-single-gpu

**Step 3: View the results:**

* Final checkpoints & metrics: ``../results``

* TensorBoard summaries: ``../results/grpo-1.5b/tensorboard/``  
  Launch with:

  .. code-block:: bash

     tensorboard --logdir ../results/grpo-1.5b/tensorboard/ --port 6006


Open TensorBoard, and you should see an interface similar to the one below.  
Key metrics to pay attention to include  
``rollout/response_length`` and ``rollout/reward_scores``.  

.. raw:: html

   <img src="https://github.com/user-attachments/assets/818b013d-18e9-4edb-ba0b-db0e58b53536" width="800"/>


