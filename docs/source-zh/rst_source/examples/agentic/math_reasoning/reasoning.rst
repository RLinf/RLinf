Qwen2.5 数学推理（GRPO）
========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

使用本配方在数学数据上用 GRPO 训练 Qwen 系列推理模型。相比监督微调，RL 鼓励模型探索更多样的推理路径，同时优化最终答案正确性。

概述
----------------------------------------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-1.5B 与 Qwen2.5-7B

   .. grid-item-card:: 算法
      :text-align: center

      GRPO，使用 token-level loss 与 minibatch early-stop

   .. grid-item-card:: 数据
      :text-align: center

      AReaL-boba 数学推理数据

   .. grid-item-card:: 硬件
      :text-align: center

      多节点 Megatron 训练

数据集
----------------------------------------

我们使用 `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`_ 数据集，整合了 DeepScaleR、Open-Reasoner-Zero、Light-R1、DAPO、NuminaMath（AoPS/Olympiad 子集）和 ZebraLogic，并过滤了过于简单的问题。

训练样例如下：

.. code-block:: json

   {
      "prompt": "<｜User｜>\nProblem description... Please reason step by step, and put your final answer within \\boxed{}.<｜Assistant｜><think>\n",
      "task": "math",
      "query_id": "xx",
      "solutions": ["\\boxed{x}"]
   }

.. note::

   使用默认配置时，数据集须符合上述结构；其他格式请按下方说明调整配置。

如需导入其他格式的数据集，可调整以下配置项：

- **Prompt key 和 answer key 配置**

  默认配置读取数据集中的 ``prompt`` 和 ``solutions`` 字段。若数据集键名不同，修改 YAML 中的 ``prompt_key`` 和 ``answer_key``。例如数据集使用 ``prompt`` 和 ``label`` 时：

  .. code-block:: yaml

      prompt_key: "prompt"
      answer_key: "label"

- **apply_chat_template 配置**

  若提示词需要通过 tokenizer 的 chat template 格式化，在配置中开启 ``apply_chat_template``：

  .. code-block:: yaml

      apply_chat_template: true

  例如数据集以对话消息列表形式存储提示词：

  .. code-block:: json

      {
          "prompt": [{"content": "<str>", "role": "<str>"},],
          "label": "<str>",
      }

  开启后，RLinf 会调用 ``tokenizer.apply_chat_template()`` 将提示词格式化为输入字符串。

GRPO 工作方式
----------------------------------------

我们采用 GRPO（Group Relative Policy Optimization），并做了如下改进：

- **Token 级别的损失**：在 token 级别而非整条响应序列上平均损失（类似 DAPO），避免过长回答主导梯度。

- **小批次提前停止**：若一个 minibatch 中的重要性比率过大，则丢弃该批次以稳定训练。

奖励函数：

- 最终 boxed/数值答案正确：+5
- 错误：-5

一键上手
----------------------------------------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。通用路径、集群和 runner 字段见 :doc:`训练配置 <../../../reference/configuration>`。

.. tab-set::

   .. tab-item:: DeepSeek-R1-Distill-Qwen-1.5B

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. 启动训练**

      在 ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml`` 中设置 ``cluster.num_nodes``、``runner.output_dir``、``rollout.model.model_path`` 和 ``data.train_data_paths``，然后启动 Ray 集群并在 head 节点启动训练：

      .. code-block:: bash

         cd /path_to_RLinf/ray_utils;
         rm /path_to_RLinf/ray_utils/ray_head_ip.txt;
         export TOKENIZERS_PARALLELISM=false
         bash start_ray.sh;
         if [ "$RANK" -eq 0 ]; then
             bash check_ray.sh 128;
             cd /path_to_RLinf;
             bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron
         else
           if [ "$RANK" -eq 1 ]; then
               sleep 3m
           fi
           sleep 10d
         fi

         sleep 10d

   .. tab-item:: DeepSeek-R1-Distill-Qwen-7B

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ./DeepSeek-R1-Distill-Qwen-7B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. 启动训练**

      在 ``examples/reasoning/config/math/qwen2.5-7b-grpo-megatron.yaml`` 中设置 ``cluster.num_nodes``、``runner.output_dir``、``rollout.model.model_path`` 和 ``data.train_data_paths``，然后启动 Ray 集群并在 head 节点启动训练：

      .. code-block:: bash

         cd /path_to_RLinf/ray_utils;
         rm /path_to_RLinf/ray_utils/ray_head_ip.txt;
         export TOKENIZERS_PARALLELISM=false
         bash start_ray.sh;
         if [ "$RANK" -eq 0 ]; then
             bash check_ray.sh 128;
             cd /path_to_RLinf;
             bash examples/reasoning/run_main_grpo_math.sh qwen2.5-7b-grpo-megatron
         else
           if [ "$RANK" -eq 1 ]; then
               sleep 3m
           fi
           sleep 10d
         fi

         sleep 10d

可视化与结果
----------------------------------------

我们基于 DeepSeek-R1-Distill-Qwen 训练了 1.5B 和 7B 模型。

启动 TensorBoard 监控训练指标：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

通用指标含义见 :doc:`训练指标 <../../../reference/metrics>`。下面展示训练曲线。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/1.5b-loss-curve.jpg" style="width: 100%;"/>
       <p><em>MATH 1.5B</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/7b-loss-curve.jpg" style="width: 100%;"/>
       <p><em>MATH 7B</em></p>
     </div>
   </div>

最终性能
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们提供了评估 `工具包 <https://github.com/RLinf/LLMEvalKit>`_。

AIME 24、AIME 25 和 GPQA-diamond 上的评测结果如下：

.. list-table:: **1.5 B 模型结果**
   :header-rows: 1
   :widths: 45 15 15 25 15

   * - 模型
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - 平均值
   * - |huggingface| `DeepSeek-R1-Distill-Qwen-1.5B（基础模型） <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_
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

\* 使用默认配置重训了 600 步。

.. list-table:: **7 B 模型结果**
   :header-rows: 1
   :widths: 45 15 15 25 15

   * - 模型
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - 平均值
   * - |huggingface| `DeepSeek-R1-Distill-Qwen-7B（基础模型） <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`_
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
----------------------------------------

我们在 Hugging Face 上发布了训练好的模型，供大家使用：

- `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`_
- `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`_
