Qwen2.5 数学推理（GRPO）
========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

使用 :doc:`GRPO </rst_source/reference/algorithms/grpo>` 在 AReaL-boba 数学数据上训练 DeepSeek-R1-Distill-Qwen 1.5B 或 7B 模型；在 AIME 24、AIME 25 与 GPQA-diamond 上的平均分分别达到 40.84（1.5B）和 56.23（7B）。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      DeepSeek-R1-Distill-Qwen-1.5B · 7B

   .. grid-item-card:: 算法
      :text-align: center

      GRPO（token 级 Loss，Megatron + SGLang）

   .. grid-item-card:: 任务
      :text-align: center

      数学推理（AReaL-boba）

   .. grid-item-card:: 硬件
      :text-align: center

      默认 16 节点（单机快速启动 ≥ 2 / 4 张 GPU）

| **你将完成：** 安装 ``agentic`` 依赖 → 下载模型与 AReaL-boba 数据集 → 运行 ``run_main_grpo_math.sh`` → 观察训练曲线并评测。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · 单机快速启动需至少 2 张 GPU（1.5B）或 4 张 GPU（7B），默认配置为 16 节点。

安装与运行
----------

中国大陆用户可在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: DeepSeek-R1-Distill-Qwen-1.5B

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载基础模型和 `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`__：

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. 启动训练**

      .. warning::

         ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml`` 默认面向 **16 节点**\ （``cluster.num_nodes: 16``）。单机快速启动需覆盖 ``cluster.num_nodes=1``\ （TP=2 至少需要 2 张 GPU）。

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron \
           cluster.num_nodes=1 \
           rollout.model.model_path=./DeepSeek-R1-Distill-Qwen-1.5B \
           actor.tokenizer.tokenizer_model=./DeepSeek-R1-Distill-Qwen-1.5B \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

   .. tab-item:: DeepSeek-R1-Distill-Qwen-7B

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载基础模型和 `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`__：

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ./DeepSeek-R1-Distill-Qwen-7B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. 启动训练**

      .. warning::

         ``examples/reasoning/config/math/qwen2.5-7b-grpo-megatron.yaml`` 默认面向 **16 节点**\ （``cluster.num_nodes: 16``）。单机快速启动需覆盖 ``cluster.num_nodes=1``\ （TP=4 至少需要 4 张 GPU）。

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-7b-grpo-megatron \
           cluster.num_nodes=1 \
           rollout.model.model_path=./DeepSeek-R1-Distill-Qwen-7B \
           actor.tokenizer.tokenizer_model=./DeepSeek-R1-Distill-Qwen-7B \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

.. dropdown:: 进阶配置：多节点、vLLM 与自定义数据集（可选）

   - **完整 16 节点训练**：按 :doc:`/rst_source/guides/multi_node` 启动 Ray 集群后，去掉 ``cluster.num_nodes=1`` 并将模型与数据指向共享存储路径，在 head 节点运行启动命令。

   - **切换 vLLM rollout 后端**：执行 ``bash requirements/install.sh agentic --engine vllm`` 安装环境，启动时追加 ``rollout.rollout_backend=vllm``。详见 :doc:`Rollout 引擎版本 </rst_source/guides/version>` 与 :doc:`训练配置 </rst_source/reference/configuration>`。

   - **自定义数据集格式**：默认配置读取 ``prompt`` 与 ``solutions`` 字段（boxed 答案正确得 +5 Reward，错误得 -5）。使用其他数据集时可覆盖 ``data.prompt_key`` 与 ``data.answer_key``；若提示词为对话消息列表，开启 ``data.apply_chat_template=True``。

可视化与结果
------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看训练曲线；各指标含义及 W&B / SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

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

使用 `LLMEvalKit <https://github.com/RLinf/LLMEvalKit>`__ 在 AIME 24、AIME 25 与 GPQA-diamond 上评测。训练好的 checkpoint 已发布在 `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`__ 与 `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`__。

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

\* 使用默认配置重训 600 步。

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
