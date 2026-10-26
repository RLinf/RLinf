Search-R1 的同步 PPO 训练
=============================

使用同步 PPO 和独立 critic 训练 Qwen2.5-3B-Instruct。先按 :doc:`index` 配置检索服务，下载模型和问答数据集，再选择本页的 GPU 布局。GRPO 配置见 :doc:`qwen2.5`。

概述
--------

PPO 复用已有的多轮 agent 和搜索工具接口。每个训练 step 先完成整个 rollout batch，计算固定的预测值和目标，再更新 actor 与 critic，最后开始下一批生成。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      同步 PPO 与 GAE

   .. grid-item-card:: 工具
      :text-align: center

      FAISS 或 Qdrant wiki 检索

   .. grid-item-card:: 硬件
      :text-align: center

      8 张 NVIDIA A100 80 GB GPU；Megatron 和 SGLang

任务
~~~~~~~~

模型可以先检索 wiki 段落，再回答问题。每个完整 episode 在结束时计算一次答案 reward。

.. list-table:: Search-R1 任务
   :header-rows: 1
   :widths: 25 75

   * - 项目
     - 配置
   * - 数据集
     - `RLinf/Search-R1-Data <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__
   * - 训练单元
     - 完整的多轮 episode
   * - PPO batch
     - 1024 个 prompt，每个 prompt 一条轨迹，一个 minibatch

观测与动作
~~~~~~~~~~~~~~

每轮对话追加模型生成的 token 和检索到的段落。只有生成的 action token 参与 PPO loss。

.. list-table:: Agent 接口
   :header-rows: 1
   :widths: 25 75

   * - 组成
     - 含义
   * - 观测
     - 问题、历史对话和检索段落
   * - 动作
     - 推理文本、``<search>...</search>`` 查询或 ``<answer>...</answer>`` 答案
   * - 长度限制
     - 最多两轮、总长度 7096 个 token，每次工具响应最多 500 个 token

安装
--------

按照 :doc:`/rst_source/start/installation` 安装 Megatron 训练和 SGLang rollout 环境：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang
   source .venv/bin/activate

PPO 当前要求 Megatron TP=PP=CP=1，提供的配置使用 SGLang TP=PP=1。检索服务的资源与八张训练 GPU 分开管理，配置步骤见 :doc:`index`。

下载模型
------------

下载 actor/critic 的初始化模型和训练数据集：

.. code-block:: bash

   hf download Qwen/Qwen2.5-3B-Instruct --local-dir /path/to/model/Qwen2.5-3B-Instruct
   hf download RLinf/Search-R1-Data --repo-type dataset --local-dir /path/to/data/searchr1

将 ``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml`` 中的 ``rollout.model.model_path`` 和 ``data.train_data_paths`` 设置为模型目录和训练 JSONL 文件。各布局配置继承这些路径。将 ``tools.search.server_addr`` 设置为检索服务的可访问地址。

运行
--------

向启动脚本传入不带 ``.yaml`` 的配置名，后续参数作为 Hydra 覆盖项。

.. list-table:: 同步 PPO 布局
   :header-rows: 1
   :widths: 43 19 19 19

   * - 配置
     - Rollout GPU
     - Actor GPU
     - Critic GPU
   * - ``train_qwen2.5_ppo_shared``
     - 0–7
     - 0–7
     - 0–7
   * - ``train_qwen2.5_ppo_split``
     - 0–3
     - 4–7
     - 4–7
   * - ``train_qwen2.5_ppo_separate``
     - 0–3
     - 4–5
     - 6–7

例如，运行 shared PPO，最多训练 200 个 batch：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_shared \
     runner.max_steps=200 runner.max_epochs=200

覆盖 epoch 数使 dataloader 能提供足够的 batch。不使用这些覆盖项时，配置默认训练一个数据集 epoch，每 25 个 batch 及最后一个 batch 保存 checkpoint。使用 ``train_qwen2.5_ppo_split`` 或 ``train_qwen2.5_ppo_separate`` 选择其他布局。

Rollout 与训练顺序
~~~~~~~~~~~~~~~~~~~~~~

每个 step 开始时，runner 发布 actor 权重，并等待整个 batch 的 episode 完成。Rollout 与训练共享 GPU 时，在 critic inference 前 offload rollout 引擎。Critic 预测 old values，GAE 再沿 episode 的 action token 计算固定 returns 和全局归一化的 advantages。工具文本与 padding 不参与计算，最终答案 reward 只应用一次。达到轮数或 token 上限也会终止 episode，未来 value 设为零。

Actor 与 critic 共享 GPU 时，runner 先训练 critic，将其状态 offload 后，再执行 actor inference 和训练。Shared 配置对两个角色都开启 ``offload_weight``、``offload_grad`` 和 ``offload_optimizer``，分别控制角色切换时哪些状态移到 CPU。Actor 与 critic 使用独立 GPU 时，两者的 inference 和训练可以并行执行。所有布局都会先等待当前 rollout batch 完成，再开始更新。

Actor 使用 PPO clipping，critic 使用带 value clipping 的 Huber loss。``algorithm.n_minibatches=1`` 时，每个 rollout batch 分别产生一次 actor 和 critic optimizer update，Megatron microbatch 在每次更新内部累积梯度。Actor 学习率为 ``1e-6``；critic 学习率为 ``5e-6``，使用 10 次更新的 warmup。``critic.value_head_init`` 支持独立 value head 的 ``normal`` 或 ``zero`` 初始化。

``actor.pack_traj=true`` 合并可兼容的多轮轨迹，同时保留每个 action token 的目标和 behavior probability。``algorithm.use_valid_token_scale=true`` 补偿 microbatch 和数据并行的平均操作，使 loss 保持为全局 action token 均值。``runner.max_tokens_per_mbs=8192`` 限制每个训练 microbatch 的 token 预算。设置 ``actor.pack_traj=false`` 可关闭 packing。

保留生成 Token
~~~~~~~~~~~~~~~~~~

``agentloop.truncate_response`` 默认为 ``true``，使用模型对应的 tokenizer 在搜索结束标签处截取生成 token 前缀。PPO 配置将其设为 ``false``，保留生成序列及对应的 behavior logprobs：

.. code-block:: yaml

   agentloop:
     truncate_response: false
   algorithm:
     recompute_logprobs: false
   rollout:
     return_logprobs: true

无论截断开关为何值，剩余响应 token 预算始终生效：超出上限的 token 和对应 logprobs 一起删除。保留的 token 不会经过解码再编码；解码只为工具解析和 reward 提供文本。直接使用 rollout logprobs 可以跳过额外的 actor inference。若需重算，设置 ``algorithm.recompute_logprobs=true`` 和 ``rollout.return_logprobs=false``。

恢复与评估
~~~~~~~~~~~~~~

将 ``runner.resume_dir`` 设置为完整的 ``checkpoints/global_step_N`` 目录，恢复 actor、critic、optimizer 和 dataloader 状态。保留原有的 actor 与 critic checkpoint 转换缓存；若修改实验名，将两个 ``megatron.ckpt_convertor.save_path`` 字段指向各自原来的目录。

按照 :doc:`qwen2.5` 中的 Megatron 说明导出 actor checkpoint 并运行评估。Critic 仅用于训练。

训练指标与验证
------------------

在 TensorBoard 中查看 ``rollout/reward_scores_traj``。``Global Step`` 每完成一个 rollout batch 增加一次，训练指标使用 optimizer step 编号。只有一个 minibatch 时，两种计数一致。通用指标说明见 :doc:`/rst_source/reference/metrics`。

CI 配置使用 Qwen2.5-1.5B 验证三种布局。在具备所需模型、数据集和检索资源的机器上，逐个运行：

.. code-block:: bash

   REPO_PATH="$PWD" bash tests/e2e_tests/agent/searchr1/run-train.sh qwen2.5-1.5b-ppo-shared
   REPO_PATH="$PWD" bash tests/e2e_tests/agent/searchr1/run-train.sh qwen2.5-1.5b-ppo-split
   REPO_PATH="$PWD" bash tests/e2e_tests/agent/searchr1/run-train.sh qwen2.5-1.5b-ppo-separate
