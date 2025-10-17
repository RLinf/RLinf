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
   --local-dir /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

   # dataset
   hf download inclusionAI/AReaL-boba-Data --repo-type=dataset \
   --local-dir /path/to/dataset/boba

**Step 2: Execute the provided launch script:**

For user convenience, our configuration file is set up to run with a single GPU by default.  
However, if you have multiple GPUs and wish to accelerate the quickstart process,  
we highly recommend updating the following configuration option in  
``./examples/math/config/qwen2.5-1.5b-single-gpu.yaml``:  
``cluster.component_placement``.


You can set it to **0-1**, **0-3** or  **0-7** to use 2/4/8 GPUs depending on your available resources.
Refer to :doc:`../tutorials/user/yaml` for a more detailed explanation of the placement configuration.

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
        actor,rollout: 0

Finally, before running the script, you need to modify the corresponding configuration options in the YAML file according to the download paths of the model and dataset. Specifically, update:

- ``rollout.model_dir``
- ``data.train_data_paths``
- ``data.val_data_paths``
- ``actor.tokenizer.tokenizer_model``

After these modifications, launch the following script to start training!


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

   <img src="https://github.com/RLinf/misc/raw/main/pic/math-quickstart-metric.jpg" width="800"/>


