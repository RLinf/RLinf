与 VeRL 的对比
=======================

最后更新：08/04/2025。

本文档提供了针对 VeRL 的完整基准测试指南，包括环境搭建、配置项说明与性能结果。  
VeRL 是一个用于通过强化学习（GRPO、PPO 等）训练大语言模型的高性能框架。  
但目前 VeRL 仅支持 **共享式（collocated）** 模式，因此为公平起见，我们也在 **共享式** 模式下与 RLinf 进行对比。

环境搭建
------------------

为便于部署，推荐使用 Docker 镜像进行训练环境搭建。该方式能确保环境一致性并降低配置复杂度。  
更详细的环境配置及其他安装方式，请参考 `VeRL 文档 <https://verl.readthedocs.io/en/latest/start/install.html>`_。

社区镜像
~~~~~~~~~~~~~~~

VeRL 提供了多种预构建的 Docker 镜像，面向不同的推理后端与训练配置进行了优化：

- vLLM + FSDP + Megatron：``verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1``，带 Deep-EP 支持：``verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1-deepep``。  
- SGLang + FSDP + Megatron：``verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.1`` （需要 vLLM 支持，但可能存在包冲突），带 Deep-EP 支持：``verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.1-deepep``。  
- SGLang + FSDP + Megatron 预览版，CUDA 12.6：``verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.1``  
- SGLang + FSDP + Megatron 预览版，CUDA 12.8：``verlai/verl:app-preview-verl0.5-sglang0.4.8-mcore0.12.1``

Docker 安装与启动
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

按照以下步骤基于 Docker 搭建 VeRL 环境：

**1. 启动容器**

.. code-block:: bash

    # 创建并以 GPU 支持启动容器
    docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" \
        --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
    docker start verl
    docker exec -it verl bash

**2. 安装 VeRL 框架**

对于预构建镜像，只需安装 VeRL 本体（不装依赖）：

.. code-block:: bash

    # 安装 nightly 版本（推荐，功能更新更快）
    git clone https://github.com/volcengine/verl && cd verl
    pip3 install --no-deps -e .

数据集准备
-------------------

VeRL 需要 **Parquet** 格式的数据集，并满足特定 schema。框架期望结构化数据，包含 prompt、标准答案信息以及用于奖励建模的元数据。

**必需的数据格式：**

.. code-block:: python

    data = {
        "data_source": data_source,           # 数据来源标识
        "prompt": [                           # 对话格式的 prompt
            {
                "role": "user",
                "content": question,          # 实际问题 / 提示
            }
        ],
        "ability": "math",                    # 任务类别（如 "math"、"coding"、"reasoning"）
        "reward_model": {                     # 奖励模型配置
            "style": "rule",                  # 奖励计算方式
            "ground_truth": solution          # 期望的正确答案
        },
        "extra_info": {                       # 额外元信息
            "split": split,                   # 数据集划分（train/val/test）
            "index": idx,                     # 样本索引
            "answer": answer_raw,             # 原始答案
            "question": question_raw,         # 原始问题文本
        },
    }

**数据转换提示：**

- 将你已有的数据集转换为上述格式；  
- 确保所有必需字段齐全；  
- 训练前校验字段类型与格式。

配置
-------------

VeRL 与我们的框架在参数配置上有不少差异。下面给出一个示例，并解释部分配置项含义。

Bash 示例
~~~~~~~~~~~~

.. code-block:: bash

    set -x
    export CUDA_DEVICE_MAX_CONNECTIONS=1 

    math_train_path=/path/to/dataset/boba.parquet
    math_test_path=/path/to/dataset/test_mini.parquet

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$math_train_path" \
        data.val_files="$math_test_path" \
        data.train_batch_size=128 \
        data.max_prompt_length=1024 \
        data.max_response_length=27648 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/path/to/models/DeepSeek-R1-Distill-Qwen-7B \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=30000 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30000 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.top_k=1000000 \
        actor_rollout_ref.rollout.top_p=1.0 \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","tensorboard"]' \
        trainer.project_name='verl_grpo_boba' \
        trainer.experiment_name='ds_7b_fsdp_sglang' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=8 \
        trainer.val_before_train=False \
        trainer.save_freq=50 \
        trainer.test_freq=-1 \
        trainer.total_epochs=15000 $@

参数类别与说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

批量大小（Batch Size）配置
^^^^^^^^^^^^^^^^^^^^^^^^

以下参数决定了数据在训练流水线中的流动方式：

- ``data.train_batch_size``：**全局训练批量** —— 单次训练迭代在所有 GPU 上合计处理的 prompt 数量  
- ``actor_rollout_ref.actor.ppo_mini_batch_size``：**PPO mini-batch** —— 单次迭代内、每次梯度更新所用的全局 prompt 数量  
- ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``：**Actor micro-batch** —— 每张 GPU 上单次正反传处理的样本数  
- ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``：**Reference micro-batch** —— 参考模型 log prob 计算的每 GPU 样本数  
- ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``：**Rollout micro-batch** —— rollout 阶段 log prob 计算的每 GPU 样本数  

**动态批量管理：**

- ``actor_rollout_ref.actor.use_dynamic_bsz``：启用 Actor 训练的动态批量  
- ``actor_rollout_ref.actor.ppo_max_token_len_per_gpu``：Actor 训练每 GPU 的最大 token 数  
- ``actor_rollout_ref.ref.log_prob_use_dynamic_bsz``：启用参考模型计算的动态批量  
- ``actor_rollout_ref.ref.log_prob_max_token_len_per_gpu``：参考模型 log prob 每 GPU 的最大 token 数  
- ``actor_rollout_ref.rollout.log_prob_use_dynamic_bsz``：启用 rollout 计算的动态批量  
- ``actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu``：rollout 阶段每 GPU 的最大 token 数

FSDP（Fully Sharded Data Parallel）配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FSDP 通过在多张 GPU 上分片参数，支持大模型训练：

- ``actor_rollout_ref.model.use_remove_padding``：**移除 padding 优化** —— 去除 padding token，提升效率并降低显存占用  
- ``actor_rollout_ref.actor.ulysses_sequence_parallel_size``：**Sequence 并行规模** —— 将序列维度划分到多少张 GPU  
- ``actor_rollout_ref.model.enable_gradient_checkpointing``：**梯度检查点** —— 以计算换显存，反向阶段重算激活

**内存优化选项：**

- ``actor_rollout_ref.ref.fsdp_config.param_offload``：将参考模型参数 offload 到 CPU  
- ``actor_rollout_ref.actor.fsdp_config.param_offload``：将 Actor 模型参数 offload 到 CPU  
- ``actor_rollout_ref.actor.fsdp_config.optimizer_offload``：将优化器状态 offload 到 CPU

模型与算法配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``actor_rollout_ref.model.path``：**基础模型路径** —— HuggingFace 路径或本地预训练模型目录  
- ``actor_rollout_ref.actor.optim.lr``：**学习率**  
- ``algorithm.adv_estimator``：**优势估计器** —— 支持 ``["gae", "grpo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "rloo"]``

**KL 与正则化：**

- ``actor_rollout_ref.actor.use_kl_loss``：启用 KL 损失以约束策略偏移  
- ``actor_rollout_ref.actor.kl_loss_coef``：KL 系数  
- ``actor_rollout_ref.actor.kl_loss_type``：KL 计算方式 ``["kl (k1)", "abs", "mse (k2)", "low_var_kl (k3)", "full"]``  
- ``actor_rollout_ref.actor.entropy_coeff``：探索用熵系数

Rollout 与推理配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``actor_rollout_ref.rollout.name``：**推理后端** —— 可选 ``["hf", "sglang", "vllm]"``  
- ``actor_rollout_ref.rollout.tensor_model_parallel_size``：**张量并行（TP）规模** ——（仅 vLLM 生效）  
- ``actor_rollout_ref.rollout.gpu_memory_utilization``：**GPU 显存占比** —— 推理阶段使用的显存比  
- ``actor_rollout_ref.rollout.n``：**每个 prompt 的采样数** —— rollout 时每个 prompt 生成的响应数量

**生成相关参数：**

- ``actor_rollout_ref.rollout.temperature``：随机性控制  
- ``actor_rollout_ref.rollout.top_k``：Top-k 采样  
- ``actor_rollout_ref.rollout.top_p``：Top-p 采样

训练控制参数
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``trainer.logger``：**日志后端** —— 可选 ``["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "clearml"]``  
- ``trainer.project_name``：实验追踪项目名  
- ``trainer.experiment_name``：具体实验名  
- ``trainer.n_gpus_per_node``：单节点 GPU 数  
- ``trainer.nnodes``：集群节点数  
- ``trainer.total_epochs``：最大训练 epoch 数  
- ``trainer.save_freq``：保存检查点的步频（每 N 步）  
- ``trainer.test_freq``：验证频率（-1 表示关闭周期验证）

多节点训练
-------------------------

对于多机大规模训练，VeRL 使用 Ray 进行分布式协调。本节简述集群初始化与管理。

Ray 集群初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~

**手动启动 Ray：**

1. **启动 Head 节点：**
   
   .. code-block:: bash
   
       ray start --head --dashboard-host=0.0.0.0

2. **启动 Worker 节点：**
   
   .. code-block:: bash
   
       ray start --address=<head_node_ip:port>

更详细的多节点说明，见 `VeRL 多节点部署文档 <https://verl.readthedocs.io/en/latest/start/multinode.html>`_。

自动化 Ray 集群脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用下面的脚本可在多节点上自动初始化集群：

.. code-block:: bash

    #!/bin/bash

    # 参数校验
    if [ -z "$RANK" ]; then
        echo "Error: RANK environment variable not set!"
        exit 1
    fi

    # 配置（按需修改）
    SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    REPO_PATH=$(dirname "$SCRIPT_PATH")
    RAY_HEAD_IP_FILE=$REPO_PATH/ray_utils/ray_head_ip.txt
    RAY_PORT=$MASTER_PORT  # Ray 默认端口，可按需修改

    # Head 节点启动逻辑
    if [ "$RANK" -eq 0 ]; then
        IP_ADDRESS=$(hostname -I | awk '{print $1}')
        echo "Starting Ray head node on rank 0, IP: $IP_ADDRESS"
        # export VLLM_ATTENTION_BACKEND=XFORMERS
        # export VLLM_USE_V1=0
        ray start --head --memory=461708984320 --port=29500
        
        echo "$IP_ADDRESS" > $RAY_HEAD_IP_FILE
        echo "Head node IP written to $RAY_HEAD_IP_FILE"
    else
        echo "Waiting for head node IP file..."
        for i in {1..360}; do
            if [ -f $RAY_HEAD_IP_FILE ]; then
                HEAD_ADDRESS=$(cat $RAY_HEAD_IP_FILE)
                if [ -n "$HEAD_ADDRESS" ]; then
                    break
                fi
            fi
            sleep 1
        done
        
        if [ -z "$HEAD_ADDRESS" ]; then
            echo "Error: Could not get head node address from $RAY_HEAD_IP_FILE"
            exit 1
        fi
        
        echo "Starting Ray worker node connecting to head at $HEAD_ADDRESS"
        # export VLLM_ATTENTION_BACKEND=XFORMERS
        export VLLM_USE_V1=0
        ray start --memory=461708984320 --address="$HEAD_ADDRESS:29500"
    fi

基准结果
-----------------


基于 **DeepSeek-R1-Distill-Qwen-1.5B** 模型、Boba 数学推理数据集，对 **VeRL** 与 **RLinf** 在 **共享式** 模式下进行对比评测（测试日期：2025-08-04）。

两者共同的关键参数如下：

.. list-table:: **共同训练参数**
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 数值
   * - Model
     - DeepSeek-R1-Distill-Qwen-1.5B
   * - Dataset
     - Boba math reasoning dataset
   * - Hardware
     - 1 nodes × 8 H100 GPUs
   * - Tensor Parallelism
     - 2
   * - Data Parallelism
     - 4
   * - Pipeline Parallelism
     - 1
   * - Context Length
     - 28672
   * - MaxPrompt Length
     - 1024
   * - Batch Size Per DP
     - 128
   * - recompute
     - 20 blocks

下面的表格汇总了 **RLinf** 与 **VeRL** 的对比结果。  
VeRL 测试基于 **Commit ID 8fdc4d3（v0.5.0 release）**。

一般而言，时间相关指标越小越好；吞吐相关指标越大越好；响应长度通常没有绝对优劣结论。  
表中，RLinf 相比 VeRL 的 **改进** 用 :red:`红色` 高亮，**回退** 用 :green:`绿色` 高亮。

.. list-table:: **RLinf vs VeRL 对比（共享式模式）**
   :header-rows: 1
   :widths: 27 12 12 15 20

   * - 指标（Metric）
     - RLinf
     - VeRL
     - RLinf 相比 VeRL
     - 单位
   * - response length
     - 13975.00
     - 14254.84
     - \
     - tokens
   * - generation time
     - 266.08
     - 260.92
     - :green:`↑ 1.98%`
     - seconds
   * - prev logprob time
     - 17.78
     - 17.51
     - :green:`↑ 1.54%`
     - seconds
   * - training time
     - 61.12
     - 66.53
     - :red:`↓ 8.13%`
     - seconds
   * - step time
     - 346.33
     - 363.55
     - :red:`↓ 4.74%`
     - seconds
   * - gen throughput
     - 3361.35
     - 3533.27
     - :green:`↓ 4.87%`
     - per-GPU tokens/s
   * - prev logprob throughput
     - 50835.06
     - 52635.84
     - :green:`↓ 3.42%`
     - per-GPU tokens/s
   * - step throughput
     - 19850.13
     - 20022.92
     - :green:`↓ 0.87%`
     - total tokens/s

.. note::
   上述 RLinf 结果未计入 **ref logprob** 时间。

结论：两者整体训练效率接近，但在 **training time** 上，RLinf 相比 VeRL 有明显降低与优势。
