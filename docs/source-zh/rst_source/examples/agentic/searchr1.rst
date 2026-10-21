Search-R1的强化学习训练
========================================

使用 GRPO、同步 PPO 或完全异步 PPO 训练调用搜索工具回答问题的 Qwen 模型。本页介绍 RLinf 中 `Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning <https://arxiv.org/abs/2503.09516>`__ 的检索服务、训练配置和 checkpoint 评测流程。

概述
----------------------------------------

使用本配方结合本地 wiki 检索服务训练搜索增强推理模型。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      带搜索工具调用的多轮强化学习

   .. grid-item-card:: 工具
      :text-align: center

      FAISS 或 Qdrant 本地 wiki server

   .. grid-item-card:: 硬件
      :text-align: center

      参考运行使用 8×H100

环境
----------------------------------------

RLinf环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 环境配置参照 :doc:`RLinf Installation </rst_source/start/installation>`。

Local Wiki Server运行环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们使用search-R1示例中的local retrieve
server，通过conda安装faiss，详细文档见\ `SearchR1 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__\ ，安装过程参考\ `Search-R1 &
veRL-SGLang <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__\ ，同样使用conda来配置环境

.. code-block:: bash

   conda create -n retriever python=3.10 -y
   conda activate retriever

   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install transformers datasets pyserini huggingface_hub

   #  安装 GPU 版 faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   pip install uvicorn fastapi

Wiki配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们使用Asearcher提供的本地检索文件，下载文件大约 50~60GB

.. code-block:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/agent/searchr1/download.py --save_path $save_path

从huggingface上下载\ `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding模型，并生成index

.. code-block:: bash

   bash examples/agent/tools/search_local_server_faiss/build_index.sh

将之前下载好的wiki文件路径和index路径等写入examples/agent/searchr1/launch_local_server.sh

.. code-block:: bash

   #!/bin/bash

   set -ex

   WIKI2018_WORK_DIR=$save_path

   index_file=$WIKI2018_WORK_DIR/e5.index/e5_Flat.index
   corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
   pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl
   retriever_name=e5
   retriever_path=path/to/intfloat/e5-base-v2

   python3  ./local_retrieval_server.py --index_path $index_file \
                                               --corpus_path $corpus_file \
                                               --pages_path $pages_file \
                                               --topk 3 \
                                               --retriever_name $retriever_name \
                                               --retriever_model $retriever_path \
                                               --faiss_gpu --port 8000

运行launch_local_server.sh启动Local Wiki Server，等待直至输出server ip等信息，代表server启动完成

(Optional) 使用Qdrant作为Wiki Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们也支持使用 qdrant 作为 wiki 服务器。如果你不打算使用 qdrant，可以直接跳到 训练 部分。

使用上一部分中提到的方式准备 Asearcher 提供的本地 wiki corpus 检索文件。

从 huggingface 上下载\ `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding 模型。

下载 `qdrant <https://github.com/qdrant/qdrant/releases>`__ 并按照以下步骤构建 qdrant collection。首先，创建一个文件夹并把下载好的 qdrant 二进制文件放入该文件夹中，方便后续存储 qdrant 程序及其构建的 collection 文件。

在 `examples/agent/tools/search_local_server_qdrant/build_index.sh` 和 `examples/agent/tools/search_local_server_qdrant/launch_local_server.sh` 中，根据之前下载的 wiki corpus, e5-base-v2 和 qdrant 路径更新 `WIKI2018_DIR`、 `retriever_path` 和 `qdrant_path` 的文件路径。

使用以下指令构建 qdrant wiki 服务器的 collection：

.. code-block:: bash

   # 创建 qdrant 存放的文件夹
   mkdir -p /path/to/qdrant
   # 拷贝二进制执行文件
   cp qdrant /path/to/qdrant

   # 启动 qdrant server
   /path/to/qdrant/qdrant &

   # 构建 qdrant collection
   bash examples/agent/tools/search_local_server_qdrant/build_index.sh

运行 launch_local_server.sh 启动 Local Qdrant Wiki Server ，等待直至输出 server ip 等信息，代表 server 启动完成

.. code-block:: bash

   # 启动 qdrant server
   /path/to/qdrant/qdrant &

   # 启动基于 qdrant 的 wiki server
   bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

Qdrant 默认使用 HNSW 图索引算法。关于 HNSW 图索引的优化,请参考 `Qdrant 文档 <https://qdrant.tech/documentation/guides/optimize/>`__。

在8*H100上训练
----------------------------------------

从huggingface上下载\ `训练集 <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__
，并将路径写入 `examples/agent/searchr1/config/train_qwen2.5.yaml`:

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/train.jsonl"]

修改 `train_qwen2.5.yaml` 中 `rollout.model.model_path` 的路径

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
       model_type: qwen2.5

如果使用 `sampling_params.stop` 来控制模型停止节省训练时间，detokenize应当设置为True

.. code-block:: yaml

   rollout:
      ……
      distributed_executor_backend: mp   # ray or mp
      disable_log_stats: False
      detokenize: True

默认的 ``agentloop.truncate_response: true`` 会在工具边界截断生成结果并重新编码文本。此时保持 ``recompute_logprobs: true``，使训练使用保留 token 的 logprob。

.. code-block:: yaml

   algorithm:
      ……
      recompute_logprobs: True
      shuffle_rollout: False

要直接使用 rollout 概率训练，请保留原始生成 token：

.. code-block:: yaml

   agentloop:
      truncate_response: false
   algorithm:
      recompute_logprobs: false
   rollout:
      return_logprobs: true

两种模式都会执行响应长度上限：超出剩余预算的生成 token 及其 logprob 会一并移除。设置 ``truncate_response: false`` 时，上限内的原始 token ID 和 rollout logprob 保持不变，包括停止边界处的 token；仅跳过工具边界的手动截断和重新编码。工具解析和 reward 计算会解码文本，但不会改写这些 token。PPO 会跳过额外的 actor logprob inference；若要保留原始 token 并继续重算 logprob，可设置 ``recompute_logprobs: true``。
运行 `bash examples/agent/searchr1/run_train.sh` 启动训练。

MA-FSDP 训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 同时提供基于 FSDP2 的 Search-R1 多智能体 Actor 实现。使用
`examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml`，按上文设置相同的模型和数据集路径，然后运行：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_fsdp

当 `agentloop.is_dynamic_rollout_batch: True` 时，入口会在
`actor.training_backend: fsdp` 下选择 `MAFSDPActor`。该配置设置了
`actor.pack_traj: True`，会在 FSDP 训练前打包同一轨迹中可兼容的多轮样本，避免重复计算共享前缀。对于提供的 Search-R1 FSDP 配置，应保持该选项开启。

在 8×A100 上运行同步 PPO
-------------------------

PPO 配置使用独立的 Megatron actor 和 critic，并复用现有 SearchR1 模型、数据、搜索工具与奖励。每个 global step 等待全部 rollout 完成，固定 old predictions 与 GAE targets，更新两边模型，再等待 SGLang 安装新 actor 权重后开始下一批。训练不加载 reference model，也不计算 KL reward 或 KL loss。

根据八张 GPU 的分配方式选择配置：

.. list-table:: PPO Placement
   :header-rows: 1
   :widths: 24 20 20 20

   * - 配置后缀
     - Rollout GPU
     - Actor GPU
     - Critic GPU
   * - ``shared``
     - 0–7
     - 0–7
     - 0–7
   * - ``split``
     - 0–3
     - 4–7
     - 4–7
   * - ``separate``
     - 0–3
     - 4–5
     - 6–7

全共享配置每批释放一次 rollout 显存，先执行完整 critic 阶段，再执行完整 actor 阶段；新权重安装完成且 actor 卸载后，恢复 rollout KV/graph。生成训练分离配置默认让全部模型常驻，critic 与 actor 训练阶段串行执行。三者分离配置默认全部常驻，actor/critic 并行 inference，汇合计算一次 GAE，再并行 training。

先在 ``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml`` 中设置现有模型、数据和搜索服务路径，再运行对应命令：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_shared
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_split
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_separate

三条命令使用相同初始模型、每批 1024 个 prompt、每个 prompt 一条轨迹（``algorithm.group_size: 1``），以及一个全局 minibatch，仅 placement 与显存驻留方式不同。Critic 的 optimizer state 独立，value head 默认使用均值为 0、标准差为 0.02 的正态初始化。默认配置为 ``critic.value_head_init=normal``，设为 ``critic.value_head_init=zero`` 可使用零初始化；设置 ``critic.model.init_method_std`` 可调整正态分布的标准差。这些配置用于初始化新 head，加载 checkpoint 时恢复已保存的权重。``runner.synchronous: true`` 选择同步调度；``critic.use_critic_model: true`` 和 ``algorithm.adv_type: actor_gae`` 选择 PPO。原有 ``train_qwen2.5`` 命令仍选择 GRPO。

提供的 PPO 配置运行一个 epoch，每 25 个 global step 及最后一步保存两个模型。设置 ``runner.max_steps=20 runner.save_interval=20`` 可运行 20 步。

配置通过 ``runner.max_tokens_per_mbs`` 将每个训练 microbatch 限制为 8192 个 token，控制 actor 和 critic 共用训练 GPU 时的 activation 显存。调整该预算会改变 microbatch 调度，全局 batch 和全局有效 token 的 loss 权重保持不变。

Runner 负责 rollout 与模型阶段的调度，inference 和 training 复用现有多轮 Megatron actor 与 Megatron critic。各 worker 负责 minibatch 更新、microbatch 调度和 checkpoint 保存。

PPO 和 SAO 从 PPO base YAML 继承 ``agentloop.truncate_response: false`` 和 ``algorithm.recompute_logprobs: false``，``rollout.return_logprobs`` 也随之解析为 ``true``。两者都直接使用 rollout behavior logprobs 训练，并保留响应长度上限内的原始 token。Base 同时将 actor 学习率设为 ``1e-6``、无 warmup，critic 学习率设为 ``5e-6``、warmup 10 次 optimizer update。PPO 每批更新一次 critic；异步训练每批更新 ``critic.update_epochs`` 次。每批更新一次时，critic warmup 对应 10 个 batch；每批更新两次时，对应 5 个 batch。

同步 PPO 也可按上文选择 response 处理方式。启用 logprob 重算时，训练 GPU 共享的两种模式在 actor inference 后保留权重，接着训练。工具 observation 不参与 policy/value loss，最终 reward 每条轨迹只应用一次。SearchR1 在自然终止或轮数/token 预算结束后结算完整 episode，两种边界的 future value 均为零。

使用 ``actor.pack_traj: true``，可在 actor 和 critic 训练前合并同一轨迹中可兼容的多轮样本，避免重复计算共享前缀。Old predictions 和全局归一化后的 GAE targets 在打包前固定；打包合并其动作位置，并保持工具 observation 的 mask。两项 loss 均使用全局有效 token 均值，打包保持其权重语义。设置 ``algorithm.loss_agg_func: token-mean`` 和 ``algorithm.use_valid_token_scale: true``，按每次更新的全局有效 token 数补偿 Megatron 的 microbatch 和 DP 平均。Worker 按 GRPO 的执行顺序，先打包整轮 rollout batch，再在 DP 副本间平衡训练行，最后切分 optimizer minibatch。打包保留整轮 batch 的 action token 和 targets，但在多个 minibatch 下可能改变各次更新包含的 token，因此开关两边的训练曲线可能不同。追加 ``actor.pack_traj=false`` 可按独立 turn 行训练。

要在生成训练分离模式中启用 actor/critic offload 轮换，为两个角色同时开启三项状态卸载：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_split \
     actor.offload_weight=true actor.offload_grad=true actor.offload_optimizer=true \
     critic.offload_weight=true critic.offload_grad=true critic.offload_optimizer=true

开启后，inference 与 training 对每个角色每批执行一次 onload/offload。默认的生成训练分离与三者分离配置保持状态常驻，保存 checkpoint 后也保持常驻。保存时临时加载已卸载的状态，写入完成后按各角色的 ``offload_weight``、``offload_grad`` 和 ``offload_optimizer`` 设置处理。``runner.save_interval`` 为正数时，按该间隔及最后一步保存两个模型、optimizer 和 dataloader state。

TensorBoard 使用与 ``AgentRunner`` 相同的 minibatch 步数。第 ``g`` 个 global step 从 ``(g - 1) * algorithm.n_minibatches`` 开始，在该位置记录一次 reward 和耗时，并逐 minibatch 记录 actor、critic 的训练指标。

在 8×A100 上运行完全异步 PPO
--------------------------------

要让 learner 按轨迹完成顺序训练，使用异步配置。实现采用 `SAO <https://arxiv.org/abs/2607.07508>`__ 第 3.1、3.2 节与附录 A 中的 token-level 方法，持续生成 rollout，并通过有界 buffer 收集完成的轨迹。Runner 扩展同步 PPO runner，复用已有多轮 actor、critic、轨迹打包、token 权重、checkpoint 和权重传输机制；角色切换与请求 retract 参考 `Miles <https://github.com/radixark/miles>`__。

在基础 PPO 配置中设置模型、数据和检索服务路径，然后选择两种支持的布局之一：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao_split

第一条命令让三个模型共享 GPU 0–7。训练前，SGLang retract 未完成请求并释放 rollout 显存；actor 训练后，runner 安装新权重、恢复 KV/graph 显存，再继续这些请求。已生成的 token 前缀及其原始 behavior logprobs 保持不变。第二条命令使用 GPU 0–3 生成，actor 与 critic 共享 GPU 4–7；critic 和 actor 训练期间持续生成，只在发布权重时短暂暂停。训练显存不足时，启用上文列出的六项 actor/critic offload 设置，在角色切换时轮换驻留状态。

两种配置均使用每批 1024 个 prompt、每个 prompt 一条轨迹，并直接使用原始 rollout logprobs。采样配置使用温度 1、无 repetition penalty 的全词表随机采样，模型并行配置为 Megatron TP=PP=CP=1、SGLang TP=PP=1。定向 adapter 支持 ``sglang==0.5.12.post1``：确认 pause/resume 完成、排空重叠 GPU 工作、将等待及部分 prefill 请求移出 cache 管理，再清理旧 KV 状态并恢复请求。每个生成 token 记录其生成时的权重版本，同一请求跨多次权重发布时也保留这些信息。只有全部 rollout worker 确认安装后，published version 才前进；发布失败则停止运行。

恢复的请求重建 KV state，同时保留已记录的输出 logprobs；仅请求输出 logprobs 时，保留的前缀不参与 logprob 重算。

Runner 管理 dataloader，启动时填充 prompt Channel，并在每次消费 batch 后、开始训练前补充 prompt。Agent worker 持续生成轨迹，将完成的结果写入有界 Channel。Runner 直接调用组批函数，按全局完成顺序取数、检查 staleness，再将 targets 发送给各个 critic 训练 rank。Critic 更新并完成 inference 后，runner 通过 Channel 读取新 values，计算全局归一化的 advantages，再将 targets 发送给 actor 训练。以下配置限制待处理工作量，并可选择过滤过旧数据：

.. list-table:: 异步配置
   :header-rows: 1
   :widths: 36 14 50

   * - 配置
     - 默认值
     - 含义
   * - ``rollout.max_in_flight_rollouts``
     - 2048
     - 轨迹 producer 槽位数，包括工具调用和结果传输。
   * - ``rollout.rollout_buffer_capacity``
     - 2048
     - Channel 缓存的已完成轨迹容量。
   * - ``rollout.max_weight_staleness``
     - ``null``
     - 不过滤过旧数据；启用时设置为非负整数。

两个容量字段设为 ``null`` 或省略时，配置校验会按 ``2 * data.rollout_batch_size`` 计算默认值；显式设置的整数值优先。PPO base 配置统一设置 ``data.rollout_batch_size=1024``，同步与异步运行均继承该值。同时生成相当于两个 learner batch 的轨迹，可在慢轨迹尚未结束时填充下一批数据。Staleness 在取数时根据最新 published rollout version 判断；一条轨迹的年龄由其最旧生成 action token 的版本确定。被拒绝的轨迹使用原 prompt 重新生成。Producer 的有限 prompt 预算为 ``runner.max_steps * data.rollout_batch_size``，因此 benchmark 即使完成顺序不同，也会消耗相同的 prompt 集合。

两个容量之和限制全部尚未消费的 prompt，包括生成中的请求、传输队列和 buffer 中的结果。消费一条轨迹后释放其 prompt 槽位；过旧轨迹在重新生成期间保留原槽位。两个容量均为 2048 时，最多有 4096 条尚未消费的 prompt。当 rollout 先完成这些轨迹时，split 布局的 rollout GPU 会空闲，直到 learner 取数释放空间。将 ``async/buffer_size`` 与 ``async/buffer_capacity`` 对照，并检查 ``async/waiting_for_buffer``。``async/in_flight`` 统计尚未完成的轨迹，包括工具调用，但不包含等待 buffer 空位的已完成轨迹。增大 buffer 会让 rollout 提前生成更多数据，并增加权重 staleness；训练成为瓶颈后，增大 buffer 不会提高稳态吞吐量。

每个 learner batch 按以下顺序执行：

1. 使用 ``gamma=1``、critic lambda 1 计算固定 Monte Carlo targets。默认 critic loss 为不裁剪、按全局 action token 平均的平方误差。
2. 在同一个完整 batch 上更新 critic ``critic.update_epochs`` 次，再推理最新 token values。
3. 使用 ``lambda = 1 - 1 / (1.5 * L)`` 计算 policy GAE，其中 ``L`` 是整条 episode 跨轮的生成 action token 数。工具 observation 和 padding 不推进 GAE；advantages 在完整 learner batch 上归一化。
4. 使用 DIS 更新 actor 一次，再发布成功更新后的权重。

通过 ``critic.loss_type`` 选择 critic loss：``sao`` 使用 SAO loss，``ppo`` 复用同步 PPO 的 clipped Huber loss。使用后者训练：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao critic.loss_type=ppo

首次 critic 训练前向传播的 value 预测经过 detach，作为该 batch 所有更新共用的 clipping reference，无需单独进行 value inference。两种 loss 都使用固定 Monte Carlo targets；actor GAE 使用全部 critic 更新完成后重新推理的 values。Huber 分支沿用同步 loss 的 ``huber_delta=10000`` 和 ``algorithm.value_cliprange``。将命令中的配置名换成 ``train_qwen2.5_sao_split`` 即可使用 split 布局。

DIS 根据记录的 rollout logprobs 计算当前 policy 与 behavior policy 的概率比。只有 ``0.7 < ratio < 6.0`` 的 token 贡献 policy gradient。在 score-function loss ``-mean(mask * ratio * advantage * logprob)`` 中，ratio 做 detach，均值分母仍包含全部 action token。这里将论文中的 importance weight 视为求导时的常量，避免对 ``ratio * logprob`` 额外求导，无需执行 old-policy inference。

``algorithm.n_minibatches=1`` 表示 actor 在全部 1024 条轨迹上执行一次 optimizer step。``critic.update_epochs`` 支持任意正整数，控制 critic 的 optimizer step 数，每次都使用同一个完整 batch。当前示例配置为 ``1``；设置 ``critic.update_epochs=2`` 可复现 SAO 的两次更新安排。设置 ``n_minibatches=2`` 会把 batch 切成两部分，含义不同。Packing 与梯度累积的 microbatch 不增加 optimizer step。Actor 学习率为 ``1e-6``；critic 使用 ``5e-6``，并按 10 个 optimizer step warmup：每批更新一次时对应 10 个 learner batch，每批更新两次时对应 5 个。配置显式关闭按比例 warmup。

Dense Qwen2.5 配置训练全部 value-model 参数，不使用 value pretraining 语料。实现保留附录 A 中优于 turn-level ablation 的 token-level 方法，未实现 step-average 与 last-token ablation。Search-R1 沿用已有的两轮、7096-token 限制及终止 reward，未复制论文的 Python/coding 环境与 128k context。

Checkpoint 保存两个 optimizer、携带 behavior probabilities 的 buffer 轨迹及 prompt 归属。将 ``runner.resume_dir`` 指向完整的 ``checkpoints/global_step_N`` 目录即可恢复。已完成的 buffer 数据精确恢复；进程重启后，未完成请求从原 prompt 重新生成。运行中的 pause/resume 则保留已生成前缀。现有 Megatron worker 在恢复 checkpoint 前先从初始转换缓存初始化，因此需要保留原 actor/critic 转换缓存；更改 experiment name 时，将两个 ``megatron.ckpt_convertor.save_path`` 字段指向原目录。

两个 PPO runner 均在每个 learner batch 完成后推进一次 ``Global Step`` 进度条。同步与异步配置均设置 ``algorithm.n_minibatches=1``，因此共用的 TensorBoard 指标具有一致的 step 编号。每个 SAO learner batch 执行 ``critic.update_epochs`` 次完整 batch 的 critic 更新；这些更新的独立指标在 ``critic_update/`` 下按 critic optimizer 调用次数记录。``async_trace.jsonl`` 记录消费的 sample ID、权重版本、成功 optimizer 更新次数、retract 数量及训练期间生成的 token 数。

验证真实 retract 与前缀保留：

.. code-block:: bash

   PYTHONPATH="$PWD:$PWD/examples:/opt/venv/reason/Megatron-LM" python \
     tests/e2e_tests/agent/searchr1/verify_async_engine.py \
     --config-path "$PWD/examples/agent/searchr1/config" \
     --config-name train_qwen2.5_sao \
     runner.experiment_name=searchr1-engine-check

测试在生成期间发布三次权重，完成后逐项检查前缀 token ID、logprobs 和 token version。

运行四组受控对比实验：

.. code-block:: bash

   python tests/e2e_tests/agent/searchr1/run_ppo_benchmarks.py \
     --steps 200 --name searchr1-ppo-bsz1024-200

该命令依次在两种布局下运行同步和异步训练，batch size 为 1024，并统一模型、数据、数据与训练 seed、采样、packing、学习率、角色 offload 及 checkpoint 间隔。SAO 使用不同的 DIS、adaptive GAE、value loss 与 critic 更新频率。启动脚本允许跨多个 dataset epoch 达到指定步数。``benchmark.json`` 计入首次 buffer 填充、训练、权重发布和 checkpoint 耗时，也计入两种 runner 的最后一次权重发布；模型初始化单独记录。两种布局都启用全部六项 actor/critic offload 设置，以预留显存。Reward 保存在各运行的 TensorBoard 目录中。

四组运行完成后，生成 reward 曲线与总耗时对比：

.. code-block:: bash

   python tests/e2e_tests/agent/searchr1/plot_ppo_benchmarks.py \
     --steps 200 --name searchr1-ppo-bsz1024-200

报告脚本需要 Matplotlib、NumPy 和 TensorBoard，在对比实验的日志目录中写入图表、``report.md`` 和 ``results.json``，并检查 reward 增长、总耗时、optimizer 更新次数及异步样本归属。只比较 split 布局时，给启动命令追加 ``--modes sync-split async-split``，给报告命令追加 ``--layouts split``。报告从每次运行中读取 batch size，并要求参与对比的值一致。

Checkpoint 导出与评测
----------------------------------------

checkpoint 的导出方式取决于训练后端。

对于 Megatron，运行以下命令将 checkpoint 转换为 HuggingFace model：

.. code-block:: bash

   CKPT_PATH_MG={your_output_dir}/{exp_name}/checkpoints/global_step_xxx/actor
   CKPT_PATH_HF={path/to/save/huggingface/model}
   CKPT_PATH_ORIGINAL_HF={path/to/model/Qwen2.5-3B-Instruct}
   CKPT_PATH_MF="${CKPT_PATH_HF}_middle_file"

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
       --load-path "${CKPT_PATH_MG}" \
       --save-path "${CKPT_PATH_MF}" \
       --model qwen_2.5_3b \
       --tp-size 1 --ep-size 1 --pp-size 1 \
       --te-ln-linear-qkv true --te-ln-linear-mlp_fc1 true \
       --te-extra-state-check-none true --use-gpu-num 0 --process-num 16

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
       --load-path "${CKPT_PATH_MF}" \
       --save-path "${CKPT_PATH_HF}" \
       --model qwen_2.5_3b \
       --use-gpu-num 0 --process-num 16

   rm -rf "${CKPT_PATH_MF}"
   rm -f "${CKPT_PATH_HF}"/*.done
   shopt -s extglob
   cp "${CKPT_PATH_ORIGINAL_HF}"/!(*model.safetensors.index.json) "${CKPT_PATH_HF}"

对于 FSDP，请使用内置的 FSDP 转换器，而不是上面的 Megatron 转换命令。将保存的 FSDP checkpoint 与对应训练运行生成的解析后配置文件传入转换器，并指定 HuggingFace 模型输出目录：

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
       convertor.train_config_path=/path/to/config.yaml \
       convertor.ckpt_path=/path/to/checkpoints/global_step_xxx/actor/model_state_dict/full_weights.pt \
       convertor.save_path=/path/to/hf_model

`config.yaml` 会随训练运行写出；当它位于 `convertor.ckpt_path` 附近时，转换器也可以自动查找。将导出的 HuggingFace 模型用于独立评测。

将转换得到的 HuggingFace model 路径填入 `examples/agent/searchr1/config/eval_qwen2.5.yaml`

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/eval/model
       model_type: qwen2.5

修改测试数据集路径

.. code-block:: yaml

   data:
     ……
     val_data_paths: ["/path/to/eval.jsonl"]

运行 `bash examples/agent/searchr1/run_eval.sh` 启动测试。

可视化与结果
----------------------------------------

下面展示 reward 曲线和训练时间曲线。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png" style="width: 100%;"/>
       <p><em>Qwen2.5-3B-Instruct in RLinf</em></p>
     </div>
   </div>

相较于原版性能( response length 稳定后，单 step 133s)，我们加速了 55%，同时 reward 曲线和 eval 结果保持一致。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1_orig_impl_time.png" style="width: 35%;"/>
       <p><em>Qwen2.5-3B-Instruct in original implementation at PeterGriffinJin/Search-R1</em></p>
     </div>
   </div>

References
----------

search-r1: https://github.com/PeterGriffinJin/Search-R1

Search-R1 &
veRL-SGLang:
https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md

Asearcher: https://github.com/inclusionAI/ASearcher
