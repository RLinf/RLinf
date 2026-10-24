Search-R1 的同步与异步 PPO 训练
==========================================

使用同步 PPO 或采用 SAO 的异步 PPO 训练 Qwen2.5-3B-Instruct。先按 :doc:`index` 配置检索服务，再选择本页的 GPU 布局和训练配置。已有 GRPO 配置见 :doc:`qwen2.5`。

概述
----

各配置使用相同的问答任务和检索接口。PPO 在已有多轮 agent 流程中加入独立 critic；异步 PPO 在 agent worker 持续生成的同时，从已完成的 episode 中组成训练 batch。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      GRPO、同步 PPO、采用 SAO 的异步 PPO

   .. grid-item-card:: 工具
      :text-align: center

      FAISS 或 Qdrant wiki 检索

   .. grid-item-card:: 硬件
      :text-align: center

      PPO：8 张 NVIDIA A100 80 GB GPU；Megatron 和 SGLang

任务
~~~~

训练数据包含问题和参考答案。模型可以先检索 wiki 段落，再生成最终答案。

.. list-table:: Search-R1 任务
   :header-rows: 1
   :widths: 25 75

   * - 项目
     - 配置
   * - 数据集
     - `RLinf/Search-R1-Data <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__
   * - 检索语料
     - `ASearcher Local Knowledge <https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge>`__
   * - 训练单元
     - 完整的多轮 episode，在结束时计算一次答案 reward
   * - PPO batch
     - 1024 个 prompt，每个 prompt 一条轨迹，一个 actor minibatch

观测与动作
~~~~~~~~~~

每轮对话追加模型生成的 token；调用搜索后，再追加检索到的段落。只有生成的 action token 参与 PPO loss。

.. list-table:: Agent 接口
   :header-rows: 1
   :widths: 25 75

   * - 组成
     - 含义
   * - 观测
     - 问题、历史对话和检索段落
   * - 动作
     - 推理文本、``<search>...</search>`` 查询或 ``<answer>...</answer>`` 答案
   * - Reward
     - Search-R1 答案 reward，在 episode 完成时应用一次
   * - 长度限制
     - 最多两轮、总长度 7096 个 token，每次工具响应最多 500 个 token

安装
----

按照 :doc:`/rst_source/start/installation` 安装 agentic 环境。PPO 配置使用 Megatron 训练和 SGLang rollout：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang --sglang 0.5.12.post1
   source .venv/bin/activate

异步引擎适配针对 ``sglang==0.5.12.post1``，支持带完成确认的 pause/resume、请求 retract 和 token 级权重版本记录。PPO 配置使用 Megatron TP=PP=CP=1 和 SGLang TP=PP=1。

检索服务使用独立环境和资源，保留八张 GPU 用于训练；配置步骤见 :doc:`index`。

下载模型
--------

下载 `Qwen2.5-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>`__ 和训练数据。PPO 的 actor 与 critic 从同一模型初始化，critic 使用独立的 value head 和 optimizer。

.. code-block:: bash

   hf download Qwen/Qwen2.5-3B-Instruct --local-dir /path/to/model/Qwen2.5-3B-Instruct
   hf download RLinf/Search-R1-Data --repo-type dataset --local-dir /path/to/data/searchr1

将 ``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml`` 中的 ``rollout.model.model_path`` 和 ``data.train_data_paths`` 分别设置为模型目录和训练 JSONL 文件。Shared、split 和 SAO 配置继承这些路径。使用 GRPO 时，在 ``train_qwen2.5.yaml`` 或 ``train_qwen2.5_fsdp.yaml`` 中设置对应字段。

运行
----

先启动检索服务，再选择训练配置。各执行模式向模型提供相同的工具接口。

选择训练配置
~~~~~~~~~~~~

重新激活 agentic 环境。向 ``run_train.sh`` 传入不带 ``.yaml`` 的配置名，后续参数作为 Hydra 覆盖项。

.. list-table:: PPO 布局与执行模式
   :header-rows: 1
   :widths: 22 34 22 22

   * - 模式
     - 配置
     - Rollout GPU
     - Actor 和 critic GPU
   * - 同步 shared
     - ``train_qwen2.5_ppo_shared``
     - 0–7
     - 0–7
   * - 异步 shared
     - ``train_qwen2.5_sao``
     - 0–7
     - 0–7
   * - 同步 split
     - ``train_qwen2.5_ppo_split``
     - 0–3
     - 4–7
   * - 异步 split
     - ``train_qwen2.5_sao_split``
     - 0–3
     - 4–7

例如，启动 shared 同步 PPO 或 split 异步 PPO：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_shared
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao_split

在同一台机器上使用相同 GPU 时，分别运行这两条命令。默认训练一个数据集 epoch，每 25 个 learner batch 以及最后一个 batch 保存 checkpoint。若需固定运行 200 个 batch，应允许足够的数据集 epoch：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao \
     runner.max_steps=200 runner.max_epochs=200

同步 PPO 还支持 ``train_qwen2.5_ppo_separate``：rollout 使用 GPU 0–3，actor 使用 4–5，critic 使用 6–7。异步 PPO 要求 actor 和 critic 共用同一组 GPU，rollout 的 GPU 集合必须与训练集合相同或完全分离。

已有 GRPO 训练方式保持可用：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_fsdp

第一条命令选择 Megatron multi-agent actor，第二条选择 FSDP2。PPO 配置使用独立的 Megatron critic。

理解 PPO 更新
~~~~~~~~~~~~~

同步 PPO 等待整批 rollout 完成，计算固定的 old values 和 GAE targets，然后更新 actor 与 critic。Inference 和 training 复用已有 multi-agent worker。工具 observation 和 padding 不参与 action-token loss；自然终止和轮数/token 预算结束都以零 future value 结算 episode。配置关闭 KL reward 和 reference-policy loss。

``actor.pack_traj=true`` 将同一轨迹中兼容的 turn 合并为更少的训练行，同时保留每个 action token 的 targets 和 behavior probability。``algorithm.use_valid_token_scale=true`` 补偿 Megatron 的 microbatch 与 data-parallel 平均，使 loss 保持全局 action-token 均值。``runner.max_tokens_per_mbs=8192`` 限制每个训练 microbatch 的 token 数。

Actor 学习率为 ``1e-6``，不使用 warmup；critic 学习率为 ``5e-6``，使用 10 次 optimizer update 的 warmup。``critic.value_head_init`` 支持 ``normal`` 和 ``zero``；``normal`` 使用 ``critic.model.init_method_std``，默认标准差为 0.02。加载 checkpoint 时恢复保存的 head。

保留生成 Token
~~~~~~~~~~~~~~

``agentloop.truncate_response`` 控制工具边界处的应用侧响应处理。默认值 ``true`` 保留到搜索结束标签的生成 token 前缀，并通过模型对应的 tokenizer 跳过推理块。Token、behavior logprobs 和权重版本同步截取，不重新编码。这沿用当前同步 Search-R1 的 tokenizer 路径，同步 PPO 也可以设置 ``algorithm.recompute_logprobs=false``。

两种 PPO 配置均保留生成 token，直接使用 rollout behavior probabilities：

.. code-block:: yaml

   agentloop:
     truncate_response: false
   algorithm:
     recompute_logprobs: false
   rollout:
     return_logprobs: true

两种模式都执行剩余 response-token 预算限制，超出预算的 token 与对应 logprobs 一起移除。关闭截断后，预算内的 token IDs 和 rollout logprobs 保持原样；解码仅用于工具解析和 reward 计算。此路径跳过额外的 actor logprob inference。同步 PPO 仍可通过 ``algorithm.recompute_logprobs=true`` 启用重算；异步 PPO 要求保留生成 token 的路径。

异步执行
~~~~~~~~

Async runner 从有界 Channel 消费已完成的 episode，不等待固定一组 prompt 全部结束。它复用同步 PPO 的 worker group、packing、checkpoint 工具和权重传输。GPU 角色切换与请求 retract 参考 `Miles <https://github.com/radixark/miles>`__，loss 和更新顺序参考 `SAO <https://arxiv.org/abs/2607.07508>`__ 的第 3.1、3.2 节及附录 A。

Shared 布局下，SGLang 在训练前暂停生成、retract 未完成请求，并释放模型及 cache 显存。Critic 和 actor 训练完成后，runner 发布 actor 权重、恢复 rollout 显存并继续这些请求。已有生成前缀及其原始 behavior logprobs 在切换过程中保留。

Split 布局下，rollout 与 learner 在各自 GPU 上并行运行，发布权重时短暂暂停生成。若显存无法容纳 actor 和 critic 同时驻留，可启用角色切换时的 offload：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao_split \
     actor.offload_weight=true actor.offload_grad=true actor.offload_optimizer=true \
     critic.offload_weight=true critic.offload_grad=true critic.offload_optimizer=true

这六项覆盖也适用于同步 split PPO，shared 配置默认开启。SGLang 在确认暂停前等待已提交 GPU 工作结束，发布权重后重建 KV 状态。已有 token 的 logprobs 保持原样，不使用新权重重新评分。每个生成 token 记录实际生成它的 actor 权重版本。

每个异步 learner batch 按以下顺序执行：

1. 接收完整轨迹，检查版本滞后，并按生成 action token 计算固定 Monte Carlo critic targets。
2. 在同一批数据上更新 critic ``critic.update_epochs`` 次，然后推理新的 token values。
3. 使用 ``lambda = 1 - 1 / (gae_lambda_alpha * L)`` 计算并全局归一化 policy GAE，其中 ``L`` 为整个 episode 的生成 token 数，``gae_lambda_alpha=1.5``。
4. 使用 DIS 更新 actor 一次，发布成功更新后的权重。

DIS 使用记录的 behavior logprobs。默认边界下，只有当前 policy 与 behavior policy 的概率比严格介于 0.7 和 6.0 之间的 token 贡献 policy gradient。概率比做 detach，均值分母仍包含全部 action token。

``critic.loss_type=sao`` 选择不归一化、不裁剪的平方误差回归。设置 ``critic.loss_type=ppo`` 可使用同步 PPO 的 clipped Huber loss，沿用 ``algorithm.value_cliprange`` 和 Huber delta 10000。首次 critic 训练 forward 缓存 detached old values，后续该 batch 的所有更新共用这份旧值。两种 loss 都使用固定 returns；actor advantages 使用 critic 训练后重新推理的 values。

``critic.update_epochs`` 支持任意正整数，当前配置为 ``1``；设置为 ``2`` 可复现 SAO 的两次 critic 更新。在 ``algorithm.n_minibatches=1`` 时，这恰好对应相同次数的完整 batch optimizer step；microbatching 和 packing 不增加 optimizer step。Dense Qwen 配置未包含论文中的 MoE 参数冻结和 value pretraining 阶段，任务沿用 Search-R1 的两轮、7096-token 限制及终止 reward。

控制缓冲数据
~~~~~~~~~~~~

Producer 槽位和完成队列容量限制 rollout 能够领先训练多少数据。默认值由配置校验计算：

.. list-table:: 异步控制项
   :header-rows: 1
   :widths: 37 23 40

   * - 配置
     - 默认值
     - 含义
   * - ``rollout.max_in_flight_rollouts``
     - ``2 * data.rollout_batch_size``
     - 并发 episode producer 数，包含工具调用和结果传输
   * - ``rollout.rollout_buffer_capacity``
     - ``2 * data.rollout_batch_size``
     - Channel 保留的完整 episode 数量
   * - ``rollout.max_weight_staleness``
     - ``null``
     - 关闭版本过滤；非负整数限制允许的版本滞后

容量设为 ``null`` 时自动推导，显式正整数优先。Buffer 至少应容纳一个 learner batch。Batch size 为 1024 时，两项默认容量均为 2048，最多允许 4096 个尚未消费的 prompt。

消费数据时，以最新发布的 rollout 权重版本检查 staleness。一条轨迹中最老的生成 token 决定其版本滞后。过旧轨迹从原 prompt 重新生成，保留原有 prompt 槽位；成功消费后释放槽位，补入新的 prompt。

Buffer 已满或 producer 等待工具响应时，rollout GPU 可能空闲。增大容量前，先检查 ``async/buffer_size``、``async/in_flight`` 和 ``async/waiting_for_buffer``。更大的 buffer 可能增加 staleness；训练成为瓶颈后，增大 buffer 不会提高吞吐量。

恢复与评估
~~~~~~~~~~

将 ``runner.resume_dir`` 设置为完整的 ``checkpoints/global_step_N`` 目录，恢复两个 optimizer。异步 checkpoint 还保存 buffer 轨迹、behavior probabilities 和 prompt 归属。已完成的 buffer 数据会恢复；进程重启后的未完成 episode 从原 prompt 重新生成。运行中发布权重时的 pause/resume 则保留生成前缀。

保留原 actor 和 critic 转换缓存。恢复时若修改 experiment name，将两个 ``megatron.ckpt_convertor.save_path`` 字段设置为原目录。

Checkpoint 导出可参考 :doc:`/rst_source/resources/faq`，或使用 Megatron 转换工具处理保存的 actor checkpoint：

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
     --load-path /path/to/checkpoints/global_step_N/actor \
     --save-path /path/to/intermediate --model qwen_2.5_3b \
     --tp-size 1 --ep-size 1 --pp-size 1 \
     --te-ln-linear-qkv true --te-ln-linear-mlp_fc1 true \
     --te-extra-state-check-none true --use-gpu-num 0 --process-num 16
   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
     --load-path /path/to/intermediate --save-path /path/to/exported-model \
     --model qwen_2.5_3b --use-gpu-num 0 --process-num 16

将原模型的配置和 tokenizer 文件复制到导出目录，保留转换得到的权重。FSDP GRPO 使用 ``rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf``，输入保存的训练配置和 ``model_state_dict/full_weights.pt``。

在 ``examples/agent/searchr1/config/eval_qwen2.5.yaml`` 中设置导出模型、评估数据集和检索服务地址，再运行：

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh

训练指标与验证
--------------

在 TensorBoard 中观察 ``rollout/reward_scores_traj``。两个 PPO runner 都在每个 learner batch 后推进一次 ``Global Step``。异步 critic 的逐次更新使用 ``critic_update/`` 命名空间；``async_trace.jsonl`` 记录样本归属和权重版本。通用指标约定参见 :doc:`/rst_source/reference/metrics`。

配置好模型和数据路径后，可运行受控对照：

.. code-block:: bash

   python tests/e2e_tests/agent/searchr1/run_ppo_benchmarks.py \
     --steps 200 --batch-size 1024 --critic-updates 2 --name searchr1-ppo-comparison
   python tests/e2e_tests/agent/searchr1/plot_ppo_benchmarks.py \
     --steps 200 --name searchr1-ppo-comparison

Launcher 顺序执行四种模式，保持 prompt、学习率、采样、seed、packing、offload 和 checkpoint 周期一致。异步采用 DIS、adaptive GAE 及指定的 critic 更新次数。报告脚本依赖 NumPy、Matplotlib 和 TensorBoard，检查 reward 上升、完整训练耗时、optimizer 次数和样本归属；模型初始化单独计时。只运行 split 时，启动命令加 ``--modes sync-split async-split``，报告命令加 ``--layouts split``。

若要独立检查 SGLang 的前缀保留行为，使用安装时配置的 Megatron 路径：

.. code-block:: bash

   PYTHONPATH="$PWD:$PWD/examples:$MEGATRON_PATH" python \
     tests/e2e_tests/agent/searchr1/verify_async_engine.py \
     --config-path "$PWD/examples/agent/searchr1/config" \
     --config-name train_qwen2.5_sao

该 GPU 测试在生成期间发布三次权重，检查保留的 token IDs、behavior logprobs 和生成权重版本。

参考资料
--------

`Search-R1 <https://github.com/PeterGriffinJin/Search-R1>`__、`SAO <https://arxiv.org/abs/2607.07508>`__、`Miles <https://github.com/radixark/miles>`__ 和 `ASearcher <https://github.com/inclusionAI/ASearcher>`__ 分别提供本配置的任务、异步算法、引擎调度参考和检索语料。
