Qwen-VL 监督微调
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/release_0.2/qwen2_5_sft_vlm.png
   :align: center
   :width: 85%

   Qwen2.5-VL 在 Robo2VLM 视觉问答数据集上的监督微调。

使用 RLinf 微调视觉语言模型：在 Robo2VLM 视觉问答数据上对 Qwen2.5-VL、Qwen3-VL 和 Qwen3-VL-MoE 做\ **全量 SFT** 与评估，或在 RPent 多轮规划数据上对 Qwen3.8-27B 做\ **全量与 LoRA** SFT，并将训练后的 checkpoint 导出为 HuggingFace 格式。

概览
----------------------------------------

两个流程均使用 FSDP 并通过 ``examples/sft/run_vlm_sft.sh`` 启动，按下方标签页完成环境配置与训练。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-VL · Qwen3-VL · Qwen3-VL-MoE · Qwen3.8-27B

   .. grid-item-card:: 方法
      :text-align: center

      全量 SFT · LoRA

   .. grid-item-card:: 数据
      :text-align: center

      Robo2VLM（视觉问答）· RPent

   .. grid-item-card:: 硬件
      :text-align: center

      1–2 节点 · GPU（27B LoRA 需 4 × 80 GB 起）

| **你将完成：**\ 安装环境 → 准备模型权重与数据（Robo2VLM 或 RPent JSONL）→ 启动 ``run_vlm_sft.sh`` → 观察 loss 与指标并导出 HuggingFace 权重。
| **前置条件：**\ :doc:`安装 </rst_source/start/installation>` · Qwen-VL 或 Qwen3.8-27B 权重 · Robo2VLM 或 RPent 数据集。

.. _rpent-planner-sft:

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Qwen2.5-VL（Robo2VLM）

      **1. 安装并下载模型与数据**

      安装 agentic/VLM 环境，并下载 `Qwen2.5-VL-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>`_ 权重与 `Robo2VLM-1 <https://huggingface.co/datasets/keplerccc/Robo2VLM-1>`_ 数据集：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct
         hf download keplerccc/Robo2VLM-1 --repo-type dataset --local-dir ./Robo2VLM-1

      .. warning::

         Robo2VLM 下载后会把 train 与 test parquet 放在同一目录（如 ``train-00000-of-00262.parquet`` 与 ``test-0000X-of-00003.parquet``）。启动前请将它们分到 ``train_data`` 和 ``test_data`` 两个文件夹，否则 RLinf 会直接读取整个数据集。

      .. note::

         如需训练 **qwen3_vl** 或 **qwen3_vl_moe**，确保 ``transformers >= 4.57.1``。

      **2. 配置路径**

      在 ``examples/sft/config/qwen2_5_vl_sft_vlm.yaml`` 中填写数据集与模型路径：

      .. code-block:: yaml

         data:
           train_data_paths: "./Robo2VLM-1/train_data"
           val_data_paths: "./Robo2VLM-1/test_data"

         actor:
           model:
             model_path: "./Qwen2.5-VL-3B-Instruct"

      **3. 启动训练**

      在仓库根目录执行（日志写入 ``logs/<timestamp>/``；传入其他配置名可运行自定义 YAML，将 ``data.train_data_paths`` 设为 ``null`` 即可进入仅评估模式）：

      .. code-block:: bash

         bash examples/sft/run_vlm_sft.sh qwen2_5_vl_sft_vlm

   .. tab-item:: Qwen3.8-27B（RPent）

      使用 ``examples/sft/config/qwen3_8_27b_rpent_sft.yaml`` 在 `RPent <https://github.com/RLinf/RPent>`_ 多轮规划数据上微调 Qwen3.8-27B（``qwen3_5`` 架构），仅对目标 assistant 回复与工具调用计算 loss。LoRA 微调已在 4 × A800 80 GB GPU 上验证，27B 全量微调需要更多 GPU。

      **1. 安装环境**

      安装默认 ``agentic --engine sglang`` 环境（已固定 ``transformers==5.6.1``，直接支持 ``qwen3_5``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate

      **2. 下载权重并准备数据**

      从 HuggingFace 下载 Qwen3.8-27B 权重：

      .. code-block:: bash

         hf download Qwen/Qwen3.8-27B --local-dir ./Qwen3.8-27B

      准备 RPent JSONL 文件或目录（相对图片路径以所在 JSONL 文件目录为准）：

      .. list-table::
         :header-rows: 1
         :widths: 22 78

         * - 字段
           - 说明
         * - ``episode_id``、``turn_id``
           - episode 与轮次标识。
         * - ``tools``
           - 可用工具定义，每项包含 ``name``、``description`` 与 ``input_schema``。
         * - ``messages``
           - 对话历史，每条消息包含 ``role``、``content`` 内容块（``text`` 或 ``image``）、可选 ``tool_calls`` 与 ``tool_call_id``。
         * - ``target``
           - 目标 assistant 轮次，包含 ``role: "assistant"``、``content`` 与可选 ``tool_calls``。

      单条 JSONL 记录最小示例（为便于阅读已按多行排版）：

      .. code-block:: json

         {
           "episode_id": "ep_001",
           "turn_id": "P1",
           "tools": [{"name": "look", "description": "Inspect view", "input_schema": {"type": "object"}}],
           "messages": [
             {"role": "user", "content": [{"type": "text", "text": "Find the cup."}]},
             {"role": "assistant", "content": [], "tool_calls": [{"id": "c1", "name": "look", "arguments": {"view": 1}}]},
             {"role": "tool", "tool_call_id": "c1", "content": [
               {"type": "image", "artifact": {"path": "images/view1.png"}},
               {"type": "text", "text": "Cup on left"}
             ]}
           ],
           "target": {"role": "assistant", "content": [{"type": "text", "text": "Picking up the cup."}]}
         }

      **3. 启动训练**

      将 ``RPENT_SAMPLES_PATH`` 设为 JSONL 文件或目录，将 ``QWEN3_8_MODEL_PATH`` 设为本地权重路径，然后在仓库根目录启动全量或 rank-16 LoRA 微调：

      .. code-block:: bash

         export RPENT_SAMPLES_PATH=./rpent_samples.jsonl
         export QWEN3_8_MODEL_PATH=./Qwen3.8-27B

         # 全量监督微调（默认）
         bash examples/sft/run_vlm_sft.sh qwen3_8_27b_rpent_sft

         # LoRA 监督微调（语言模型注意力与 MLP 层，rank 16）
         bash examples/sft/run_vlm_sft.sh qwen3_8_27b_rpent_sft actor.model.is_lora=true

      训练完成后，按下方“检查点转换（FSDP 转 HuggingFace）”导出或合并权重，再到 RPent / RoboTwin 环境中评测规划效果。

.. dropdown:: 进阶配置与字段说明

   仓库自带的 ``examples/sft/config/qwen2_5_vl_sft_vlm.yaml`` 已设置好数据列名、batch size、优化器和 FSDP 默认项：

   .. code-block:: yaml

      cluster:
        num_nodes: 1
        component_placement:
          actor: all

      runner:
        task_type: sft
        max_epochs: -1
        max_steps: 6000
        val_check_interval: 1000
        save_interval: 1000

      data:
        type: vlm
        dataset_name: "robo2vlmsft"
        prompt_key: "question"
        choice_key: "choices"
        answer_key: "correct_answer"
        image_keys: ["image"]
        apply_chat_template: True
        use_chat_template: True
        max_prompt_length: 1024
        lazy_loading: true
        num_workers: 16

      actor:
        group_name: "ActorGroup"
        training_backend: "fsdp"
        micro_batch_size: 4
        eval_batch_size: 4
        global_batch_size: 256
        seed: 42
        model:
          model_type: "qwen2.5_vl"
          precision: fp32
          is_lora: False
        optim:
          lr: 1e-5
          adam_beta1: 0.9
          adam_beta2: 0.999
          adam_eps: 1.0e-08
          weight_decay: 0.01
          clip_grad: 1.0
          lr_scheduler: "cosine"
          total_training_steps: ${runner.max_steps}
          lr_warmup_steps: 200
        fsdp_config:
          strategy: "fsdp"
          sharding_strategy: "full_shard"
          use_orig_params: False
          gradient_checkpointing: False
          mixed_precision:
            param_dtype: bf16
            reduce_dtype: fp32
            buffer_dtype: bf16

   **常用字段：**

   - ``micro_batch_size``：单卡一次前向/反向的样本数。
   - ``global_batch_size``：全局 batch（需满足可整除关系）。
   - ``max_epochs``：按数据集完整遍历的轮数。
   - ``save_interval``：每多少 step 保存一次 checkpoint。
   - ``model_path``：本地模型目录（必须存在）。
   - ``train_data_paths`` / ``val_data_paths``：数据目录或文件路径（``train_data_paths: null`` 时仅运行评估）。

.. dropdown:: 检查点转换（FSDP 转 HuggingFace）

   FSDP SFT 默认将完整权重保存为 ``full_weights.pt``。在仓库根目录传入本次训练保存的 ``config.yaml``，即可导出 HuggingFace 权重：

   .. code-block:: bash

      export REPO_PATH=$(pwd)
      python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
          +convertor.train_config_path=./logs/<timestamp>/tensorboard/config.yaml \
          convertor.ckpt_path=./logs/<timestamp>/<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt \
          convertor.save_path=./hf_converted_model

   将 ``<timestamp>``、``<experiment_name>`` 和 ``<N>`` 替换为实际训练目录中的值。转换 LoRA checkpoint 时，默认将 adapter 合并回基座权重；传入 ``convertor.merge_lora_weighs=False`` 则仅将 adapter 导出到 ``lora_adapter/`` 目录。详见 :doc:`检查点转换 <../../guides/convertor>`。

.. dropdown:: 常见报错与排查

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - 问题
        - 处理
      * - 找不到模型路径
        - 检查 ``actor.model.model_path`` 是否正确且有读取权限。
      * - 数据字段不匹配
        - 检查 ``prompt_key``、``choice_key``、``answer_key`` 与 ``image_keys`` 是否与数据列名一致。
      * - 显存不足（OOM）
        - 先降低 ``micro_batch_size``，再减少 ``num_workers``；仍不足时缩小模型或降低输入长度。
      * - 快速跑通流程
        - 使用小数据子集，设 ``max_epochs: 1``，并调小 ``save_interval`` 以便观察。

可视化与结果
----------------------------------------

正常训练中 **loss** 下降、**评估准确率** 上升。运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``logs/<timestamp>`` 中的曲线，指标含义见 :doc:`训练指标 <../../reference/metrics>`。

不同规模模型的参考结果（Robo2VLM）：

.. list-table::
   :header-rows: 1
   :widths: 34 20 14 32

   * - 模型
     - 硬件
     - 迭代数
     - 评估准确率（训练前 → 训练后）
   * - Qwen2.5-VL-3B
     - 8 × H100
     - 6000
     - — → 89.96%
   * - Qwen3-VL-4B
     - 4 × H100
     - 6000
     - — → 96.9%
   * - Qwen3-VL-30B-A3B（MoE）
     - 2 × 8 × A100
     - 1000
     - 58.4% → 91.3%

**Qwen3.8-27B（RPent）**\ ：训练时在 TensorBoard 中观察 ``train/loss``。在 4 × A800 80 GB GPU 上以 ``actor.model.lora_rank=592`` 训练 48 步 LoRA 后，最终 ``train/loss`` 降至 0.199；合并权重后，RoboTwin 规划任务的成功 episode 数从基座模型的 8/50 提升到 14/50。

**Qwen2.5-VL-3B** —— 每 1000 次迭代的评估准确率、grad_norm 与 loss 曲线：

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sft_vlm_eval_accuracy.png
   :alt: Qwen2.5-VL-3B VLM SFT eval accuracy
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sft_vlm_eval_grad_norm.png
   :alt: Qwen2.5-VL-3B VLM SFT grad norm
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sft_vlm_eval_loss.png
   :alt: Qwen2.5-VL-3B VLM SFT loss
   :width: 85%
   :align: center

**Qwen3-VL-4B** —— 每 1000 次迭代的评估准确率、grad_norm 与 loss 曲线：

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_sft_vlm_eval_accuracy.png
   :alt: Qwen3-VL-4B VLM SFT eval accuracy
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_sft_vlm_eval_grad_norm.png
   :alt: Qwen3-VL-4B VLM SFT grad norm
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_sft_vlm_eval_loss.png
   :alt: Qwen3-VL-4B VLM SFT loss
   :width: 85%
   :align: center

**Qwen3-VL-30B-A3B（MoE）** —— 1000 次迭代的 grad_norm 与 loss 曲线：

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_moe_sft_vlm_eval_grad_norm.png
   :alt: Qwen3-VL-30B-A3B MoE VLM SFT grad norm
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_moe_sft_vlm_eval_loss.png
   :alt: Qwen3-VL-30B-A3B MoE VLM SFT loss
   :width: 85%
   :align: center
