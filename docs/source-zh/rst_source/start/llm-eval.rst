评估 2：数学推理场景
=================================

简介
------------

我们提供了一套集成评估工具，用于长链式（CoT）数学推理任务。  
该工具包 `toolkit <https://github.com/RLinf/LLMEvalKit>`_ 同时包含了代码和数据集，  
便于研究人员对训练后的大语言模型在数学推理方面进行评估。

**致谢：** 本评估工具改编自 `Qwen2.5-Math <https://github.com/QwenLM/Qwen2.5-Math>`_ 项目。

环境准备
-----------------

首先，克隆该仓库：

.. code-block:: bash

   git clone https://github.com/RLinf/LLMEvalKit.git 

安装依赖：

.. code-block:: bash

   pip install -r requirements.txt 

如果你正在使用我们的 Docker 镜像，仅需额外安装：

.. code-block:: bash

   pip install Pebble
   pip install timeout-decorator

快速开始
-----------------

模型转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^
在训练过程中，模型以 Megatron 格式被存储下来。 你可以使用位于 ``toolkits/ckpt_convertor/`` 的转换脚本将其转换为 Huggingface 格式。

你有两种方式使用脚本：

**方式一：编辑脚本文件**

手动打开 ``mg2hf_7b.sh`` 或 ``mg2hf_1.5b.sh``，将以下变量设置为你想要的路径。

1. ``CKPT_PATH_MG`` （Megatron checkpoint路径，例如 ``results/run_name/checkpoints/global_step_xx/actor/``）， 
2. ``CKPT_PATH_HF`` （Huggingface目标路径，任意路径），以及
3. ``CKPT_PATH_ORIGINAL_HF`` （初始化训练的基模checkpoint，例如 ``/path/to/DeepSeek-R1-Distill-Qwen-1.5B``） 

**方式二：命令行参数**

更灵活的方式是直接通过命令行参数传入路径。
.. code-block:: bash

   # 对于1.5B模型
   bash mg2hf_1.5b.sh /path/to/megatron_checkpoint /target/path/to/huggingface_checkpoint /path/to/base_model_checkpoint

   # 对于7B模型
   bash mg2hf_7b.sh /path/to/megatron_checkpoint /target/path/to/huggingface_checkpoint /path/to/base_model_checkpoint

运行评测脚本
^^^^^^^^^^^^^^^^^^^^^^

如果你想在 **单个数据集** 上运行评估，可以执行如下命令：

.. code-block:: bash

   MODEL_NAME_OR_PATH=/model/path  # 替换为你的模型路径
   OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval
   SPLIT="test"
   NUM_TEST_SAMPLE=-1
   export CUDA_VISIBLE_DEVICES="0"

   DATA_NAME="aime24"  # 可选项包括：aime24, aime25, gpqa_diamond
   PROMPT_TYPE="r1-distilled-qwen"
   # 注意：
   # 如果是 aime24 或 aime25，请使用 PROMPT_TYPE="r1-distilled-qwen"
   # 如果是 gpqa_diamond，请使用 PROMPT_TYPE="r1-distilled-qwen-gpqa"

   TOKENIZERS_PARALLELISM=false \
   python3 -u math_eval.py \
       --model_name_or_path ${MODEL_NAME_OR_PATH} \
       --data_name ${DATA_NAME} \
       --output_dir ${OUTPUT_DIR} \
       --split ${SPLIT} \
       --prompt_type ${PROMPT_TYPE} \
       --num_test_sample ${NUM_TEST_SAMPLE} \
       --use_vllm \
       --save_outputs

若进行 **批量评估**，可运行``main_eval.sh``脚本。该脚本将依次在 AIME24、AIME25 和 GPQA-diamond 数据集上评估模型。

.. code-block:: bash

   bash main_eval.sh /path/to/model_checkpoint

你可以在脚本中指定``CUDA_VISIBLE_DEVICES``，进行更灵活的GPU管理。  


评估结果
-----------------

结果会被打印在终端，并保存在 ``OUTPUT_DIR`` 中。  
结果内容包括：

1. 元信息（``xx_metrics.json``）：统计摘要  
2. 完整模型输出（``xx.jsonl``）：包含完整推理过程和预测结果  

元信息示例：

.. code-block:: javascript

   {
       "num_samples": 30,
       "num_scores": 960,
       "timeout_samples": 0,
       "empty_samples": 0,
       "acc": 42.39375,
       "time_use_in_second": 3726.008672475815,
       "time_use_in_minite": "62:06"
   }

字段 ``acc`` 表示 **所有采样回答的平均准确率**，是主要评估指标。

模型输出示例：

.. code-block:: javascript

   {
      "idx": 0, 
      "question": "Find the number of...", 
      "gt_cot": "None", 
      "gt": "204", // 标准答案
      "solution": "... . Thus, we have the equation $(240-t)(s) = 540$ ..., ", // 标准解法
      "answer": "204", // 标准答案
      "code": ["Alright, so I need to figure out ... . Thus, the number of ... is \\(\\boxed{204}\\)."], // 模型生成的推理链
      "pred": ["204"], // 从推理链中提取的最终答案
      "report": [null], 
      "score": [true] // 是否预测正确
   }

支持数据集
-----------------

该工具目前支持以下评估数据集：

.. list-table:: 支持的数据集
   :header-rows: 1
   :widths: 20 80

   * - 数据集
     - 简介
   * - ``aime24``
     - 来自 **AIME 2024** （美国数学邀请赛）的题目，主要关注高中奥数级别的数学推理。
   * - ``aime25``
     - 来自 **AIME 2025**，与 AIME24 格式一致但测试集不同。
   * - ``gpqa_diamond``
     - **GPQA（研究生级别 Google-Proof 问答）** 中难度最高的子集（Diamond 分支），  
       包含跨学科问题（如数学、物理、计算机），要求具备深度推理能力而非记忆。

参数配置
-----------------

主要可配置参数如下：

.. list-table:: 配置参数说明
   :header-rows: 1
   :widths: 20 80

   * - 参数名
     - 说明
   * - ``data_name``
     - 要评估的数据集，支持：``aime24``、``aime25``、``gpqa_diamond``
   * - ``prompt_type``
     - 所用提示词模板。AIME 数据集用 ``r1-distilled-qwen``，GPQA 用 ``r1-distilled-qwen-gpqa``
   * - ``temperature``
     - 采样温度。推荐值：1.5B 模型用 ``0.6``，7B 模型用 ``1.0``
   * - ``top_p``
     - nucleus sampling 的参数，默认值为 ``0.95``
   * - ``n_sampling``
     - 每道题采样回答的数量，用于计算平均准确率，默认值为 ``32``
   * - ``max_tokens_per_call``
     - 每次生成的最大 token 数，默认值为 ``32768``
   * - ``output_dir``
     - 保存结果的输出目录，默认是 ``./outputs``
