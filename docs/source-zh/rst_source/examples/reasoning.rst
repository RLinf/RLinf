Math推理的强化学习训练
=======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍了如何在 RLinf 框架下，使用强化学习（RL）来训练大语言模型（LLM）以进行数学推理。  
与监督微调（SFT）相比，RL 能够鼓励模型探索多样化的推理路径，同时优先保证最终答案的正确性。  

我们的目标是提升模型解决复杂数学问题的能力，同时优化推理过程和最终答案。

数据集
-------------

我们使用 `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`_ 数据集。  
该数据集整合了 DeepScaleR、Open-Reasoner-Zero、Light-R1、DAPO、NuminaMath（AoPS/Olympiad 子集）和 ZebraLogic。  
过于简单的问题会被过滤，以保证数据集质量和有效性。  

一个训练样例如下：

.. code-block:: json

   {
      "prompt": "<｜User｜>\nProblem description... Please reason step by step, and put your final answer within \\boxed{}.<｜Assistant｜><think>\n",
      "task": "math",
      "query_id": "xx",
      "solutions": ["\\boxed{x}"]
   }

.. note::

  请确认数据集格式是按照上述结构配置。
  否则，请仔细阅读下方的配置指南，使用 RLinf 适配您的数据集。

我们支持导入其他类型结构的数据集。
如需导入不同的数据集并作出特殊处理，您可根据需求调整配置。

- **Prompt key 和 answer key 配置**

  默认配置要求数据集使用 `prompt` 和 `solutions` 键分别用于获取提示词信息和答案信息。

  但不同数据集可能使用不同的键名或结构，您可自定义配置以匹配数据集格式。
  在配置 yaml 文件中修改 `prompt_key` 和 `answer_key` 的值，使其指向数据集中对应的字段即可。

  比如说，如果您的数据集使用如下所示的 `prompt` 和 `label` 作为键名，您需要设置：

  .. code-block:: yaml

      prompt_key: "prompt"
      answer_key: "label"

- **apply_chat_template 配置**

  部分数据集的提示词信息可能需要使用 tokenizer 中的 chat template 进行特殊处理。
  若需此功能，需在配置中启用 `apply_chat_template` 选项。

  .. code-block:: yaml

      apply_chat_template: true

  比如说，如果您的数据集使用如下所示的特定结构对话消息，则需启用该选项以正确格式化提示词信息：

  .. code-block:: json

      {
          "prompt": [{"content": "<str>", "role": "<str>"},],
          "label": "<str>",
      }

  启用该选项后，原始数据集将通过 `tokenizer.apply_chat_template()` 方法处理，按照使用模型的 tokenizer 中对话模板对提示词信息进行格式化。
  处理完成后，提示词信息将转换为字符串格式，用于模型输入。

算法
---------

我们采用 GRPO（Group Relative Policy Optimization），并做了如下改进：  

- **Token 级别的损失**：不是在整个响应序列上平均损失，而是在 token 级别上平均（类似 DAPO）。  
  这样可以避免过长的回答主导训练，减少它们对梯度的影响。  

- **小批次提前停止**：如果一个 minibatch 中的重要性比率过大，则丢弃该批次，以稳定训练。  

奖励函数：  

- 最终 boxed/数值答案正确：+5  
- 错误：-5  

运行脚本
---------------------

**1. 关键参数配置**

在启动前，检查配置文件。主要字段包括：  

- 集群设置：``cluster.num_nodes`` （节点数）。  
- 路径：``runner.output_dir`` （保存训练日志与检查点的路径）、``rollout.model_dir`` （基础模型保存路径）、``data.train_data_paths`` （训练数据路径）等。  

**2. 配置文件**

推荐配置示例：  

- ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml``  
- ``examples/reasoning/config/math/qwen2.5-7b-grpo-megatron.yaml``  

**3. 启动命令**

运行以下命令以启动 Ray 集群并开始训练：  

.. code-block:: bash

   cd /path_to_RLinf/ray_utils;
   rm /path_to_RLinf/ray_utils/ray_head_ip.txt;
   export TOKENIZERS_PARALLELISM=false
   bash start_ray.sh;
   if [ "$RANK" -eq 0 ]; then
       bash check_ray.sh 128;
       cd /path_to_RLinf;
       bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron # 修改配置文件
   else
     if [ "$RANK" -eq 1 ]; then
         sleep 3m
     fi
     sleep 10d
   fi

   sleep 10d

结果
-------

我们基于 DeepSeek-R1-Distill-Qwen 训练了 1.5B 和 7B 模型。  

启动训练后，你可以通过以下命令监控指标：  

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

关键监控指标：  

- ``rollout/rewards``：模型在训练数据上的准确率。更高的分数通常意味着更强的推理能力。  
- ``rollout/response_length``：训练数据集上的平均响应长度。RL 往往会导致回答过长，DAPO 类似的方法可以缓解此问题。  
- ``train/entropy_loss``：表示模型的探索能力。熵值应逐渐降低并收敛。  

训练曲线
~~~~~~~~~~~~~~

下面展示训练曲线。

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

最终性能
~~~~~~~~~~~~~~~~~

我们提供了一个评估 `工具包 <https://github.com/RLinf/LLMEvalKit>`_ 以及相应的 :doc:`评估文档 <../start/llm-eval>`。  

在 AIME24、AIME25 和 GPQA-diamond 上的评测结果表明，RLinf 达到了 SOTA 性能。  

.. list-table:: **1.5 B 模型结果**
   :header-rows: 1
   :widths: 45 15 15 25 15

   * - 模型
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - 平均值
   * - |huggingface| `DeepSeek-R1-Distill-Qwen-1.5B (基础模型) <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_
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

\* 我们使用默认配置对模型进行了 600 步重训。  

.. list-table:: **7 B 模型结果**
   :header-rows: 1
   :widths: 45 15 15 25 15

   * - 模型
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - 平均值
   * - |huggingface| `DeepSeek-R1-Distill-Qwen-7B (基础模型) <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`_
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

公开检查点
------------------

我们在 Hugging Face 上发布了训练好的模型，供大家使用：  

- `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`_  
- `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`_  
