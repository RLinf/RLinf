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


**Step 1: Download the model and the datasets:**

.. code-block:: bash

   # model
   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --local-dir /workspace/model/DeepSeek-R1-Distill-Qwen-1.5B

   # dataset
   hf download inclusionAI/AReaL-boba-Data --repo-type=dataset \
   --local-dir /workspace/data/boba

**Step 2: Execute the provided launch script:**

.. code-block:: bash

   bash xxx

**Step 3: View the results:**

* Final checkpoints & metrics: ``/workspace/RLinf/results``

* TensorBoard summaries: ``/workspace/RLinf/results/tensorboard``  
  Launch with:

  .. code-block:: bash

     tensorboard --logdir ../results/grpo-1.5b/tensorboard/ --port 6006
