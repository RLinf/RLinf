RPent LIBERO planner 的同步 PPO 训练
=========================================

本页用同步 PPO 和独立 critic，把 Qwen3.5-4B 训练成 `RPent <https://github.com/RLinf/RPent>`__ 在 LIBERO-PRO 任务上的 planner。每个 episode 由 RPent 驱动：它向 planner 要下一次工具调用，用冻结的 Pi0.5 policy 和 SAM3 执行，最后给出任务结果。RLinf 用 SGLang 生成 planner 的每一轮回复，用 Megatron 训练 actor 和 critic。

概述
--------

LIBERO 仿真器、Pi0.5 服务和 SAM3 服务放在另一台机器上，RLinf 把 agent worker 也放到这台机器上。每个 worker 为每个 episode 启动一个 RPent 进程，planner 在这个进程里原样运行，每一轮都调用 worker 提供的 OpenAI chat completions 接口。每个训练 step 先跑完一批完整的 episode，计算固定的预测值和目标，再更新 actor 与 critic，然后开始下一批。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen3.5-4B

   .. grid-item-card:: 算法
      :text-align: center

      同步 PPO 与 GAE

   .. grid-item-card:: 环境
      :text-align: center

      RPent 与 LIBERO-PRO，使用 Pi0.5 和 SAM3

   .. grid-item-card:: 硬件
      :text-align: center

      训练用 8 张 NVIDIA A100 80 GB；仿真器用另一台 GPU 机器

任务
--------

每个 episode 是一个从固定初始状态开始的 LIBERO-PRO 任务。只有任务成功时 episode 才得到 1 分，episode 的每一轮通过 GAE 共享这个结果。

.. list-table:: RPent 任务
   :header-rows: 1
   :widths: 25 75

   * - 项目
     - 配置
   * - 训练单元
     - 完整的多轮 episode
   * - PPO batch
     - 128 个 episode，每个任务一条轨迹，2 个 minibatch
   * - 任务文件
     - JSONL；``prompt`` 是任务描述，``answer`` 是把任务编码成 JSON 的字符串，包含 ``suite``、``task``、``seed``，以及可选的 ``vla_seed`` 和冻结记忆 ``memory_entries``

``answer`` 必须是 JSON 编码后的字符串，不能直接写成嵌套对象。不带记忆的一行如下；完整的任务行可以用下文的 ``prepare_tasks.py`` 生成：

.. code-block:: json

   {"prompt": "open the bottom drawer of the cabinet", "answer": "{\"suite\": \"libero_goal_task\", \"task\": 0, \"seed\": 1}"}

观测与动作
------------

planner 以文本形式收到工具结果，物体位置由 SAM3 的 ``segment`` 工具给出，不看图片。每一轮的 prompt 等于上一轮的 prompt 加上采样出的回复，再加上新消息，所以同一个 episode 的各轮共享 token 前缀，可以合成一行训练。行内的工具结果 token 没有 advantage、value 目标和 loss。

.. list-table:: Agent 接口
   :header-rows: 1
   :widths: 25 75

   * - 组成
     - 含义
   * - 观测
     - 任务、冻结记忆，以及之前工具调用的文本结果
   * - 动作
     - 一条带一次 Qwen3.5 工具调用（``<tool_call>`` XML）的回复，或调用 ``finish``
   * - 限制
     - 最多 24 轮，总长 65,536 token，每条回复最多 2,048 token

安装
--------

按 :doc:`/rst_source/start/installation` 安装带 SGLang 的 Megatron 训练环境：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang
   source .venv/bin/activate

在仿真器所在的机器上同样安装这套 RLinf 环境，agent worker 运行在这台机器上，两台机器的 Python 和 Ray 版本必须一致。这台机器还要按 `RPent 快速开始 <https://rpent.readthedocs.io/en/latest/rst_source/get_started/quickstart.html>`__ 在另一个环境里安装带 LIBERO-PRO 依赖的 RPent：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git && cd RPent
   pip install -e ".[libero-pro]"

两台机器组成一个 Ray 集群（见 :doc:`/rst_source/guides/multi_node`）。agent worker 从 ``rollout.model.model_path`` 加载 tokenizer，所以仿真器所在的机器上也要能访问这个路径。在配置里把 ``tools.rpent.python`` 设为 RPent 环境的 Python 解释器。

下载模型
--------

下载 planner、Pi0.5 LIBERO policy 和 SAM3，再提取 Megatron 训练用的语言模型：

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir /path/to/Qwen3.5-4B
   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT --exclude optimizer.pt \
     --local-dir /path/to/RLinf-Pi05-LIBERO-130-fullshot-SFT
   modelscope download facebook/sam3 --local-dir /path/to/sam3
   python examples/agent/rpent/checkpoint.py extract \
     --vlm /path/to/Qwen3.5-4B --output /path/to/Qwen3.5-4B-text

``extract`` 从 SGLang 部署的模型中取出语言模型部分，作为 actor 和 critic 的初始纯文本模型。在 ``examples/agent/rpent/config/train_qwen3.5_4b_ppo.yaml`` 里，把 ``rollout.model.model_path`` 和两处 ``tokenizer_model`` 设为部署的模型，两处 ``hf_model_path`` 设为提取出的模型。

原版 Qwen3.5 checkpoint 可能还包含 MTP 辅助预测权重。``extract`` 会一并排除视觉塔和 MTP 权重，PPO 只训练语言模型。

准备任务
--------

每条任务都指明它使用的冻结记忆文件。先下载记忆库，再在装有 LIBERO-PRO 的 RPent 环境里生成任务文件：

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --revision 47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1 \
     --include 'libero/GPT_5.5_xhigh/*' --local-dir /path/to/rpent-memory
   python examples/agent/rpent/prepare_tasks.py \
     --memory-dir /path/to/rpent-memory/libero/GPT_5.5_xhigh \
     --output-dir /path/to/rpent

脚本保留 50 个初始状态齐全、记忆文件也齐全的 72 个任务，并从每个 suite 留出一个任务。训练配置读取的 ``train.jsonl`` 用其余 64 个任务的 seed 1-32，每遍历一次数据是 16 个 step。``panel_train.jsonl``、``dev_quick.jsonl`` 和 ``holdout.jsonl`` 分别评估训练用过的 seed、新的 seed 和留出的任务。``manifest.json`` 列出留出和被排除的任务。

运行
--------

在仿真器所在的机器上，用 RPent 环境启动 Pi0.5 服务和 SAM3 服务：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/RLinf-Pi05-LIBERO-130-fullshot-SFT
   python -m rpent.robots.components.pi05_vla_server --embodiment libero \
     --cuda-device 6 --transport http --host 127.0.0.1 --port 8100 &
   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
   python -m rpent.robots.components.sam3_server \
     --cuda-device 7 --transport http --host 127.0.0.1 --port 8101 &

启动 Ray，训练机器作为 node 0，仿真器所在的机器作为 node 1，然后在 node 0 上启动训练：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_ppo

配置把 8 个 agent worker 放在 node 1 上（``cluster.component_placement`` 里的 ``agentloop``）。每个 worker 同时最多跑 ``tools.rpent.episodes_per_worker`` 个 episode，仿真器在 ``tools.rpent.env_gpus`` 上渲染。每个 episode 在自己的进程里新建 LIBERO 环境，episode 结束或者 worker 退出时，这个进程和它的仿真器也随之结束。``tools.rpent.frozen_memory_dir`` 必须是生成任务文件时用的记忆目录；``tools.rpent.keep_episodes``（默认 ``failures``）决定哪些 episode 在 ``tools.rpent.output_dir`` 下保留产物。配置每 10 个 step 保存一次 checkpoint。

Rollout 与训练顺序
------------------

本示例保持 ``rollout.sglang.enable_weights_cpu_backup=true``：SGLang 在释放显存前把完整模型权重备份到主机内存，恢复时保留视觉塔，再接收纯语言 actor 更新后的权重。这会增加主机内存和传输开销，与 Megatron 的优化器 CPU offload 是不同机制。

actor、critic 和 SGLang 共用全部 8 张卡，在各阶段之间 offload。每个 step 中，runner 先在进程内把 actor 的权重同步给 SGLang，等这一批的 episode 全部结束，再推理旧的 logprob 和 value。GAE 的 ``algorithm.gae_lambda`` 设为 1.0，actor 的 advantage 和 critic 的回报都用它计算，所以每个 value 的目标就是这个 episode 的结果。advantage 在这一批所有生成 token 上归一化。

SGLang 和 Megatron 算出的 logprob 有差异，训练用按 token 截断（上限 2）的重要性采样修正，``rollout/tis/*`` 记录差异大小。``algorithm.n_minibatches=2`` 表示每批数据让 actor 和 critic 各更新 2 次。多轮在 agent loop 中合并（``agentloop.merge_prefix_turns``），所以 actor 保持 ``actor.pack_traj=false``。

示例配置项
----------

本示例设置了下面这些配置项。其中 ``agentloop`` 和 ``tools.rpent`` 是 RPent 自己的配置，其余是本示例打开的通用训练选项。

.. list-table:: RPent 示例配置项
   :header-rows: 1
   :widths: 40 60

   * - 配置项
     - 含义
   * - ``agentloop.minimum_output_tokens``
     - 每次请求在 ``runner.seq_length`` 内至少要留出的输出 token 数。历史更长时，这个 episode 以 ``context_limit`` 结束，记为失败。
   * - ``agentloop.history_compaction``
     - planner 的历史压缩：消息数超过 ``max_messages``，或长度超过 ``context_window`` 的 ``max_fraction`` 时，只保留第一条 user 消息和最近的 ``keep_messages`` 条。压缩后的请求会从新的一行训练数据开始。
   * - ``agentloop.max_scored_tokens``
     - 合并后的一行最多包含多少个参与训练的生成 token，默认等于 ``algorithm.sampling_params.max_new_tokens``。调大后行数更少、每行更长，每行的 logits 显存也更多。
   * - ``agentloop.model_request_timeout_s``、``agentloop.episode_timeout_s``
     - planner 对单次模型请求和整个 episode 的时间限制。达到时间限制的 episode 记为失败。
   * - ``agentloop.max_infra_retries``、``agentloop.max_infra_retry_rate``
     - 服务故障时一个 episode 最多重试几次，以及重试的 episode 超过多大比例时停止本次更新。
   * - ``tools.rpent.process_timeout_s``
     - episode 进程运行超过这个时间就会被停止，这个 episode 会重新运行。
   * - ``rollout.replica_affinity``
     - 把每个 agent worker 的请求固定发到同一个 SGLang 副本，让同一个 episode 的各轮复用前缀缓存。
   * - ``actor.response_only_logits``
     - 只在给生成 token 打分的位置计算词表 logits。
   * - ``actor.megatron.empty_cache_after_backward``、``critic.megatron.empty_cache_after_backward``
     - 在梯度归约前释放缓存的激活显存。

恢复与评估
----------

把 ``runner.resume_dir`` 设为 ``checkpoints/global_step_N`` 目录，即可恢复 actor、critic、优化器和 dataloader。``rollout.model.model_path`` 不用改：开启 ``rollout.sglang.enable_weights_cpu_backup=true`` 时，SGLang 从部署模型加载完整权重并保留未参与同步的视觉部分，第一次同步再用恢复后的 actor 权重替换语言部分，所以 ``rollout.validate_weight_first_sync`` 只在从头训练时生效。

要部署训练后的 policy，例如用于评估，先把它写入部署模型的一份副本：

.. code-block:: bash

   python examples/agent/rpent/checkpoint.py export \
     --checkpoint logs/rpent-qwen3.5-4b-ppo/checkpoints/global_step_10 \
     --vlm /path/to/Qwen3.5-4B --output /path/to/exports/step10

``export`` 会保留原版模型的视觉塔和 MTP 权重。部署导出的 policy 时保持投机解码关闭：这组原版 MTP 权重尚未与训练后的语言模型重新对齐。

然后用同一套服务，在固定的任务文件上评估：

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b \
     rollout.model.model_path=/path/to/exports/step10 \
     "data.val_data_paths=[/path/to/rpent/holdout.jsonl]"

每个任务跑 ``algorithm.group_size`` 次。总的和各 suite 的成功率，以及每个 episode 的结果，写在 ``logs/rpent-qwen3.5-4b-eval/eval_results.json``。

训练指标与验证
--------------

在 TensorBoard 中主要看 ``rpent/success_rate``；``rpent/turns`` 和 ``rpent/rows`` 反映 episode 长度，``rollout/tis/*`` 反映采样差异，``rpent/infra_retry_rate`` 反映服务故障。需要重试的 episode 超过 ``agentloop.max_infra_retry_rate`` 时，更新会停止。通用的指标约定见 :doc:`/rst_source/reference/metrics`。

限制
--------

- 训练时 rollout 要求 TP1；actor 和 critic 的 pipeline 并行与 context 并行大小必须为 1。
- CI 只跑单元测试；训练需要 GPU、RPent 和 LIBERO。
