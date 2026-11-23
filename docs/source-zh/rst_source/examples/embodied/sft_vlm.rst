Qwen-VL 监督微调
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/release_0.2/qwen2_5_sft_vlm.png
   :align: center
   :width: 85%

   Qwen2.5-VL 在 Robo2VLM 视觉问答数据集上的监督微调。

使用 RLinf 对视觉-语言模型（Qwen2.5-VL、Qwen3-VL、Qwen3-VL-MoE）做 **全量监督微调（Full-parameter SFT）**——完成训练、评估，并将检查点转换为 HuggingFace 格式。

概览
----------------------------------------

在 Robo2VLM 视觉问答数据集上对 Qwen-VL 系列模型做全量 SFT，使用 FSDP 并内置评估。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-VL · Qwen3-VL · Qwen3-VL-MoE

   .. grid-item-card:: 方法
      :text-align: center

      Full-parameter SFT

   .. grid-item-card:: 数据
      :text-align: center

      Robo2VLM (visual QA)

   .. grid-item-card:: 硬件
      :text-align: center

      1–2 节点 · GPU

| **你将完成：** 安装环境 → 下载模型与 Robo2VLM → 修改配置 → 启动 ``run_vlm_sft.sh`` → 观察 loss 与评估准确率。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Qwen-VL 权重 · Robo2VLM 数据集。

本示例围绕启动脚本 ``examples/sft/run_vlm_sft.sh`` 和训练配置 ``examples/sft/config/qwen2_5_vl_sft_vlm.yaml`` 展开。

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

   FSDP SFT 默认保存 ``full_weights.pt``。如需导出 HuggingFace 格式，在内置的 ``fsdp_model_convertor`` 配置中填写：

   - ``convertor.ckpt_path``：指向 ``full_weights.pt``
   - ``convertor.save_path``：输出 HF 权重目录
   - ``model.model_path``：原始基座模型路径
   - ``model.model_type``：模型类型（``qwen2.5_vl``、``qwen3_vl`` 或 ``qwen3_vl_moe``）

   然后运行内置转换工具：

   .. code-block:: bash

      python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
          --config-path rlinf/utils/ckpt_convertor/fsdp_convertor/config \
          --config-name fsdp_model_convertor

   详见 :doc:`检查点转换 <../../guides/convertor>`。

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

不同规模模型的参考结果：

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
