LLM On-Policy 蒸馏
===================

本示例介绍如何用冻结 Teacher 的评分，在 Student 自己生成的数学推理回答上训练语言模型。从同步 FSDP 配置出发，先准备兼容的模型和独立的 Teacher GPU，再选择分布损失或采样 token 目标并启动训练。

概览
----

Student 生成回答，Teacher 对相同上下文打分，Actor 根据评分更新 Student。训练既可以只使用蒸馏信号，也可以结合任务奖励，同时优化强化学习目标。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5 Student 与 tokenizer 兼容的 Teacher

   .. grid-item-card:: 算法
      :text-align: center

      On-policy 蒸馏，可结合任务奖励训练

   .. grid-item-card:: 数据
      :text-align: center

      AReaL-boba 数学题目

   .. grid-item-card:: 硬件
      :text-align: center

      示例使用六张 GPU：Student 四张，Teacher 两张

.. list-table:: 任务
   :header-rows: 1
   :widths: 25 40 35

   * - 任务
     - 配置
     - 训练方式
   * - 数学推理
     - ``qwen2.5-1.5b-opd-fsdp``
     - FSDP Actor，vLLM rollout 与 Teacher，同步更新

蒸馏流程
--------

每轮先用当前 Student 生成回答，再获取 Teacher 评分，完成 Student 更新后才开始下一轮 rollout。Teacher 在整个训练过程中保持初始权重。将 ``runner.execution_mode`` 设为 ``sync``，并配置一个 Teacher 模型；同一模型可以启动多个副本。

分布目标的评分范围由 ``topk_source`` 决定。设为 ``teacher`` 时，使用 Teacher 概率最高的 token；设为 ``student`` 时，rollout 直接记录各生成位置的 Student top-k ID，再交给 Teacher 评分。Actor 在正常训练前向中重新计算这些固定候选的当前概率。两条路径都使用回答中各位置的原始上下文。

Teacher 和 Student 的概率均按各自完整的模型输出词表归一化，词表大小以模型配置为准，即使 tokenizer 的 ID 数量更少也保持这一范围。Top-k 从该分布中选取条目并保留其原始概率质量。超出模型词表、仅用于 tensor parallel 对齐的补齐行不参与归一化。

安装
----

按推理后端安装 agentic 环境。示例的 rollout 和 Teacher 都使用 vLLM：

.. code-block:: bash

   bash requirements/install.sh agentic --engine vllm --venv .venv-opd
   source .venv-opd/bin/activate

命令会创建并激活训练环境。如果使用 SGLang，安装时传入 ``--engine sglang``，并将 ``rollout.rollout_backend`` 和 ``distillation.teacher_models.default.inference.backend`` 都设为 ``sglang``。系统与平台依赖见 :doc:`安装指南 <../../../start/installation>`。

使用 SGLang Teacher 的 Student top-k 模式时，需安装带逐位置评分补丁的 SGLang ``0.5.12.post1``。``requirements/install.sh`` 会为该版本自动应用补丁。更新已有环境时，先激活环境，再运行：

.. code-block:: bash

   bash requirements/patches/apply_sglang.sh

补丁支持 Teacher 用一个 prefill-only 请求对整段序列评分，在各位置提取不同的候选 ID，并处理 prefill 与 logits 分块，返回的概率保留完整模型词表的归一化结果。其他 SGLang 版本需有对应补丁才能使用 Student top-k 评分。Teacher top-k 通过引擎原生的 prompt 评分接口，在一个请求中获取结果；vLLM Teacher 对 Student 指定的候选保留逐前缀的兼容评分路径。两种 rollout 后端均在生成时记录 Student top-k ID。

下载模型
--------

准备 Student、Teacher 的本地 checkpoint，以及 :doc:`数学推理 <reasoning>` 中介绍的 AReaL-boba JSONL 数据。Student 需同时兼容 Actor 与 rollout 后端，Teacher 需兼容所选推理引擎。Actor 和 rollout 均使用 Student 的 tokenizer。

两个模型的 token 到 ID 映射、added tokens 和特殊 token 设置必须一致。同一模型家族或相同词表大小并不能保证兼容；初始化时会检查 tokenizer fingerprint，不匹配则报错。每个模型的词表都必须覆盖 tokenizer 的全部有效 ID。Teacher 的上下文长度至少要比最长 prompt 加回答的总长度多一个 token。

分配 Teacher 设备
-----------------

为 Teacher 单独预留 GPU，并显式指定 Student 的设备范围。示例把 GPU rank 0–3 分配给 Actor、rollout 和 reward，把 rank 4–5 分配给一个 tensor parallel 大小为 2 的 Teacher 副本：

.. code-block:: yaml

   cluster:
     component_placement:
       actor,rollout,reward: 0-3
       teacher_default: 4-5
   distillation:
     enabled: true
     teacher_models:
       default:
         model_path: /path/to/teacher
         num_replicas: 1
         inference:
           backend: vllm
           tensor_parallel_size: 2
           pipeline_parallel_size: 1
           max_model_len: 32768

placement 的 key 为 ``teacher_<model key>``。所需 GPU 总数等于 ``num_replicas × tensor_parallel_size × pipeline_parallel_size``，每个副本的设备必须位于同一节点。Teacher 设备不能与任何 Student 组件重叠，因此应使用明确的 rank 范围代替 ``all``。如果配置了 ``cluster.auto_scheduler``，将其设为 ``false``。设备分配语法见 :doc:`组件放置 <../../../concepts/placement>`。

选择训练目标
------------

可以先使用示例默认的 ``forward_kl_topk`` 和 ``topk: 64``。在 ``distillation.distillation_loss`` 下选择以下目标：

.. list-table:: 支持的目标
   :header-rows: 1
   :widths: 35 65

   * - ``loss_mode``
     - 配置要求
   * - ``forward_kl_topk``、``forward_kl_topk_tail``
     - 正数 ``topk`` 可选择 Teacher 或 Student 的 token 集合；``topk: -1`` 使用完整词表。
   * - ``reverse_kl_topk``、``reverse_kl_topk_tail``
     - 正数 ``topk`` 要求 ``topk_source: student``；``topk: -1`` 使用完整词表。
   * - ``k1`` （``kl``）
     - 采样 token 目标，要求 ``use_policy_gradient: true``。
   * - ``k2`` （``mse``）、``k3`` （``low_var_kl``）、``abs``
     - 采样 token 目标，可直接优化，也可使用 policy gradient。

带 tail 的目标还会计算所选集合之外的概率质量。完整词表目标可能占用大量主机内存和传输带宽，``inference.max_payload_bytes`` 用于限制单个 Teacher 请求的大小。reverse、tail 和完整词表目标要求 ``log_prob_min_clamp: null``。所有目标当前均使用 ``policy_loss_mode: vanilla``，输入限于文本。

设置 ``use_policy_gradient: true`` 后，蒸馏信号用于 policy-gradient 目标。此时采样参数要求 ``do_sample: true``、``temperature: 1.0``、``top_p: 1.0``、``top_k: -1`` （或 ``0``）以及 ``repetition_penalty: 1.0``。使用 SGLang rollout 时，还需检查 ``rollout.sampling_params`` 中的覆盖值。Teacher 始终对原始模型概率评分。

Policy-gradient 训练还需获取旧 Student 的 logprob。设置 ``rollout.return_logprobs: true`` 可在生成时收集，设置 ``algorithm.recompute_logprobs: true`` 可在 Actor 更新前重新计算，两者至少开启一项。示例默认采用直接蒸馏，因此关闭了这两项。

联合训练需设置 ``use_task_rewards: true``，并按 :doc:`reasoning` 配置任务奖励、advantage 和 PPO 参数。总目标为任务损失加上 ``distillation_loss_coef`` 倍的蒸馏损失；纯蒸馏模式的系数固定为 1。任务与 Teacher 的 mask 分别归一化。开启 ``algorithm.use_valid_token_scale`` 后，entropy 和 reference-KL 正则项也按任务的有效 token 加权。如果拒绝采样导致任一 Actor rank 的序列全部被过滤，所有 Actor rank 会共同跳过本轮更新，并记录 ``rollout/skipped_empty_batch``。

运行
----

替换下列模型和数据路径后，从仓库根目录启动。YAML 已配置纯蒸馏模式和前述六张 GPU 的分配方式：

.. code-block:: bash

   PYTHONPATH="$PWD:$PYTHONPATH" python examples/reasoning/main_grpo.py \
     --config-path config/math --config-name qwen2.5-1.5b-opd-fsdp \
     actor.model.model_path=/path/to/student \
     actor.tokenizer.tokenizer_model=/path/to/student \
     rollout.model.model_path=/path/to/student \
     distillation.teacher_models.default.model_path=/path/to/teacher \
     'data.train_data_paths=[/path/to/AReaL-boba-106k.jsonl]' \
     'data.val_data_paths=[/path/to/AReaL-boba-106k.jsonl]'

该命令启动冻结的 Teacher 服务、生成 Student 回答，再更新 FSDP Actor。序列长度、batch size 和设备分配均可在启动前修改 YAML。若使用 Megatron Actor，从对应的 GRPO 配置出发，添加相同的 ``distillation`` 配置和独立的 Teacher 设备分配。

可视化与结果
------------

在配置的 TensorBoard 日志目录中查看 ``actor/training/distillation/loss`` 和 ``actor/training/distillation/token_count``：前者是所选蒸馏目标，后者用于确认参与训练的回答 token 数量。请在相同评测设置下，对比训练结果与初始 Student checkpoint 的验证集任务奖励；本示例不预设提升幅度。通用指标说明见 :doc:`训练指标 <../../../reference/metrics>`。
