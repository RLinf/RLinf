快速上手 2：使用 GRPO 训练 LLM 进行 MATH 推理
==============================================

本快速教程将带你使用 **RLinf** 在数学推理数据集  
`AReaL-boba <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data>`_  
上训练  
`DeepSeek-R1-Distill-Qwen-1.5B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_ 模型。

为简化流程，你可以在单卡 GPU 上直接运行以下脚本完成训练。

数据集简介
--------------------

*AReaL-boba* 涵盖了多种数学与逻辑推理问题。以下是一个示例：

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

开始训练
--------------------

**步骤 1：下载模型和数据集**

.. code-block:: bash

   # 下载模型
   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --local-dir /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

   # 下载数据集
   hf download inclusionAI/AReaL-boba-Data --repo-type=dataset \
   --local-dir /path/to/dataset/boba

**步骤 2：运行官方提供的训练脚本**

为方便用户，我们提供的配置文件默认支持单卡训练。  
如果你拥有多张 GPU 并希望加快训练过程，  
我们推荐你修改配置文件  
``./examples/reasoning/config/math/qwen2.5-1.5b-single-gpu.yaml`` 中的参数 ``cluster.component_placement``。

你可以根据实际资源将该项设置为 **0-1**， **0-3** 或 **0-7** 来使用 2/4/8 张 GPU。
查看 :doc:`../tutorials/user/yaml` 以获取有关 Placement 配置的更详细说明。

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
        actor,rollout: 0

在运行脚本之前，请根据你的模型和数据集下载路径，  
在 YAML 配置文件中修改以下字段：

- ``rollout.model_dir``  
- ``data.train_data_paths``  
- ``data.val_data_paths``  
- ``actor.tokenizer.tokenizer_model``

完成以上修改后，运行以下脚本即可启动训练：

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-single-gpu

**步骤 3：查看训练结果**

- 最终模型与指标文件位于：``../results``  
- TensorBoard 日志位于：``../results/grpo-1.5b/tensorboard/``  
  启动方式如下：

  .. code-block:: bash

     tensorboard --logdir ../results/grpo-1.5b/tensorboard/ --port 6006

打开 TensorBoard 后，你会看到如下界面：  
推荐关注的关键指标包括：

- ``rollout/response_length``  
- ``rollout/reward_scores``  

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/math-quickstart-metric.jpg" width="800"/>
