Qwen3-VL GRPO 视觉推理
======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 框架下，使用强化学习（RL）训练视觉语言模型（VLM）完成几何推理。

一键上手
----------------------

Qwen3-VL 系列模型需要 ``torch >= 2.8.0``、``sglang == 0.5.4`` 和 ``transformers == 4.57.1``。中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: Qwen3-VL-2B-Instruct（FSDP2 + SGLang）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export MEGATRON_PATH=/path/to/Megatron-LM
         bash requirements/install.sh agentic \
             --torch 2.8.0 \
             --sglang 0.5.4 \
             --transformers 4.57.1 \
             --no-apex && source .venv/bin/activate

      .. note::

         ``MEGATRON_PATH`` 需要指向一个已有的 Megatron-LM 仓库克隆（推荐 ``core_r0.13.0`` 分支）；如果该目录不存在，安装脚本会提前退出，不执行后续的 flash-attn 和 apex 安装步骤。本示例使用 FSDP2 作为训练后端，通过 ``--no-apex`` 跳过 apex 编译。

      **2. 下载模型与数据**

      下载基础模型和 `geo3K 数据集 <https://huggingface.co/datasets/CAIR-HKISI/geo3k>`__：

      .. code-block:: bash

         hf download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen3-VL-2B-Instruct
         hf download --repo-type dataset CAIR-HKISI/geo3k --local-dir ./geo3k

      **3. 启动训练**

      在 ``examples/reasoning/config/vqa/qwen3-vl-2b-grpo-fsdp-geo3k.yaml`` 中填写 ``rollout.model.model_path`` 和 ``data.train_data_paths``\ （保持 ``rollout.model.model_type: qwen3_vl``），然后启动训练：

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_vqa.sh qwen3-vl-2b-grpo-fsdp-geo3k

      多机扩展按 :doc:`/rst_source/guides/multi_node` 启动 Ray 集群，再在 head 节点运行上述命令。

数据集格式
-------------

geo3K 数据集的训练样例如下：

.. code-block:: text

   {
      "problem": "<image>\nProblem description",
      "images": An numpy.ndarray of image bytes,
      "answer": "\\boxed{x}"
   }

.. note::

   geo3k 数据集有多种保存形式，如果从其他地址下载，图片数据可能通过 base64 或者 PIL.Image 格式保存，需要相应适配。

我们支持多种数据集格式配置：

- **Prompt key 和 answer key 配置**

  默认配置要求数据集使用 ``problem`` 和 ``answer`` 键分别获取提示词信息和答案信息。在配置 YAML 文件中修改 ``prompt_key`` 和 ``answer_key`` 的值，使其指向数据集中对应的字段即可。

  .. code-block:: yaml

      prompt_key: "problem"
      answer_key: "answer"

- **图像数据配置**

  对于视觉语言任务，需要配置 ``image_keys`` 参数指定图像字段名称：

  .. code-block:: yaml

      image_keys: ["images"]

- **图像占位符解析**

  数据集支持在 prompt 中使用 ``<image>`` 占位符来标记图像位置。框架会自动解析占位符并将文本与图像交叉排列。如果 prompt 中不包含占位符，则图像会被放置在文本之前。

- **system_prompt 配置**

  如果需要在数据集原有的 prompt 之前添加统一的提示词，可以使用 ``system_prompt`` 配置：

  .. code-block:: yaml

      system_prompt: 'Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer}.'


算法
-------

我们采用标准的 GRPO（Group Relative Policy Optimization），并且通过配置项打开了\ **TIS**\ （截断重要性采样）以稳定训练：

  .. code-block:: yaml

      importance_sampling_fix: True
      importance_sampling_clip: 2

实验证明，打开 TIS 可以避免熵的无序增加和序列长度的过快增长，从而获得更稳定的训练。

奖励函数：

- 正确：+1（可通过 ``reward_max_val`` 配置）
- 错误：0（可通过 ``reward_min_val`` 配置）

技术细节
------------

**序列打包（Packing）**

为了提高训练效率，框架支持动态序列打包功能。启用 ``enable_dynamic_batch_size: True`` 后，多个短序列会被打包成一个长序列训练。可以通过以下参数控制打包行为：

- ``max_tokens_per_mbs``：每个 micro-batch 的最大 token 数
- ``variable_seq_lengths``：是否允许变长序列，设为 ``False`` 时，还会将打包后的长序列 pad 到固定长度，这有可能在某些场景下避免重复 compile 等问题。本例中我们设为 ``True``

**FSDP2 训练**

Qwen3-VL 训练使用 FSDP2 作为训练后端。关键配置包括：

.. code-block:: yaml

   actor:
     fsdp_config:
       strategy: "fsdp2"
       gradient_checkpointing: True
       gradient_checkpointing_use_reentrant: False

**Rollout 与训练的 KL 监控**

框架新增 ``actor/rollout_train_kl`` 指标，用于监控 rollout 阶段和训练阶段 logprobs 之间的差异，帮助诊断训练稳定性问题。打开 ``recompute_logprobs`` 和 ``return_logprobs`` 后会自动显示该曲线。

结果
-------

我们使用 Qwen3-VL-2B-Instruct 完成实验，训练曲线如下：

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/qwen3vl_grpo_reward.jpeg" style="width: 50%;"/>
       <p><em>reward with TIS</em></p>
     </div>
   </div>
   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/qwen3vl_grpo_entropy.jpeg" style="width: 50%;"/>
       <p><em>entropy with TIS</em></p>
     </div>
   </div>

训练 750 步后，reward 仍有上涨趋势，entropy 趋于收敛。

如果关闭 TIS，则有很大的可能性导致一定步数后 reward 曲线崩溃，且 entropy 较不稳定：

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/qwen3vl_grpo_reward_wo_tis.jpeg" style="width: 50%;"/>
       <p><em>reward without TIS</em></p>
     </div>
   </div>
   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/qwen3vl_grpo_entropy_wo_tis.jpeg" style="width: 50%;"/>
       <p><em>entropy without TIS</em></p>
     </div>
   </div>
