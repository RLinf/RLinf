Quickstart 2: GRPO Training of LLMs on MATH
==============================================

This quick-start tutorial will guide you through training the
`DeepSeek-R1-Distill-Qwen-1.5B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_ model
on the math reasoning dataset
`AReaL-boba <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data>`_
using **RLinf**.

To simplify the process, you can directly run the following scripts on a single GPU to complete the training.

Dataset Introduction
--------------------

*AReaL-boba* covers a variety of mathematical and logical reasoning problems. Below is an example:

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
--------------------

**Step 1: Download the model and dataset**

.. code-block:: bash

   # model
   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --local-dir /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

   # dataset
   hf download inclusionAI/AReaL-boba-Data --repo-type=dataset \
   --local-dir /path/to/dataset/boba

**Step 2: Modify the configuration file**


Before running the script, please modify the ``./examples/reasoning/config/math/qwen2.5-1.5b-single-gpu.yaml`` file
according to your model and dataset download paths.

Specifically, set the model configuration to the path where the ``DeepSeek-R1-Distill-Qwen-1.5B`` checkpoint is located, and set the data configuration to the path where the ``AReaL-boba-106k.jsonl`` dataset is located.

- ``rollout.model.model_path``  
- ``data.train_data_paths``
- ``data.val_data_paths``
- ``actor.tokenizer.tokenizer_model``

**Step 3: Launch training**

After completing the above modifications, run the following script to launch training:

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-single-gpu


View Training Results
--------------------------------

- Final model and metrics files are located at: ``../results``  
- TensorBoard logs are located at: ``../results/grpo-1.5b/tensorboard/``  
  Launch as follows:

  .. code-block:: bash

     tensorboard --host 0.0.0.0 --logdir ../results/grpo-1.5b/tensorboard/

After opening TensorBoard, you will see the following interface:  
Recommended key metrics to focus on include:

- ``rollout/response_length``  
- ``rollout/reward_scores``  

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/math-quickstart-metric.jpg" width="800"/>

.. note::
   For user convenience, the configuration file we provide supports single GPU training by default.  
   If you have multiple GPUs and wish to speed up the training process,  
   we recommend that you modify the parameter ``cluster.component_placement`` in the configuration file.

   You can set this item to **0-1**, **0-3** or **0-7** to use 2/4/8 GPUs depending on your actual resources.
   See :doc:`../tutorials/user/yaml` for more detailed instructions on Placement configuration.

   .. code-block:: yaml

      cluster:
      num_nodes: 1
      component_placement:
         actor,rollout,reward: 0-3
