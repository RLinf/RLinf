Compare with Verl
=======================

Last updated: 07/22/2025.

This document provides a comprehensive guide for benchmarking VERL, including environment setup, configuration options, and performance results.
VERL is a high-performance framework for training large language models using reinforcement learning techniques (GRPO, PPO, etc.).

Environment Setup
------------------

For streamlined deployment, we recommend using Docker images for training setup. This approach ensures consistent environments and reduces configuration complexity.
For detailed environment configuration and alternative installation methods, please refer to the `VERL documentation <https://verl.readthedocs.io/en/latest/start/install.html>`_.

Community Image
~~~~~~~~~~~~~~~

VERL provides several pre-built Docker images optimized for different inference backends and training configurations:

- vLLM with FSDP and Megatron: ``verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1``, with Deep-EP support: ``verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1-deepep``.
- SGLang with FSDP and Megatron: ``verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.1`` (need vLLM support, but can have some package conflicts), with Deep-EP support: ``verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.1-deepep``.
- Preview version of SGLang with FSDP and Megatron, CUDA 12.6: ``verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.1``
- Preview version of SGLang with FSDP and Megatron, CUDA 12.8: ``verlai/verl:app-preview-verl0.5-sglang0.4.8-mcore0.12.1``

Docker Installation and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow these steps to set up your VERL environment using Docker:

**1. Launch Docker Container**

.. code-block:: bash

    # Create and start the container with GPU support
    docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" \
        --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
    docker start verl
    docker exec -it verl bash

**2. Install VERL Framework**

For pre-built images, install only VERL without dependencies:

.. code-block:: bash

    # Install the nightly version (recommended for latest features)
    git clone https://github.com/volcengine/verl && cd verl
    pip3 install --no-deps -e .

Dataset Preparation
-------------------

VERL requires datasets in Parquet format with a specific schema. The framework expects structured data that includes prompts, ground truth information, and metadata for reward modeling.

**Required Data Format:**

.. code-block:: python

    data = {
        "data_source": data_source,           # Source identifier for the dataset
        "prompt": [                           # Conversation format prompt
            {
                "role": "user",
                "content": question,          # The actual question/prompt
            }
        ],
        "ability": "math",                    # Task category (e.g., "math", "coding", "reasoning")
        "reward_model": {                     # Reward model configuration
            "style": "rule",                  # Reward calculation method
            "ground_truth": solution          # Expected correct answer
        },
        "extra_info": {                       # Additional metadata
            "split": split,                   # Dataset split (train/val/test)
            "index": idx,                     # Sample index
            "answer": answer_raw,             # Raw answer
            "question": question_raw,         # Original question text
        },
    }

**Data Conversion Tips:**

- Convert your existing datasets to this format
- Ensure all required fields are present
- Validate data types and formats before training

Configuration
-------------

VERL and our framework have many differences in parameter configuration. Here we provide an example and explain the meaning of some configurations.

Bash example
~~~~~~~~~~~~

.. code-block:: bash

    set -x
    export CUDA_DEVICE_MAX_CONNECTIONS=1 

    math_train_path=/mnt/public/wangxiangyuan/dataset/boba.parquet
    math_test_path=/mnt/public/wangxiangyuan/dataset/test_mini.parquet

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$math_train_path" \
        data.val_files="$math_test_path" \
        data.train_batch_size=128 \
        data.max_prompt_length=1024 \
        data.max_response_length=27648 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/mnt/public/hf_models/DeepSeek-R1-Distill-Qwen-7B \
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

Parameter Categories and Explanations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch Size Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

These parameters control how data flows through the training pipeline:

- ``data.train_batch_size``: **Global training batch size** - The global number of prompts processed in one training iteration across all GPUs
- ``actor_rollout_ref.actor.ppo_mini_batch_size``: **PPO mini-batch size** - The global number of prompts used for each gradient update step within a training iteration across all GPUs
- ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``: **Actor micro-batch size** - Batch size of samples for one forward_backward pass per GPU
- ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``: **Reference model micro-batch size** - Batch size of samples for reference model log prob calculations per GPU
- ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``: **Rollout micro-batch size** - Batch size of samples for rollout phase log prob calculations per GPU

**Dynamic Batch Size Management:**

- ``actor_rollout_ref.actor.use_dynamic_bsz``: Enable dynamic batch sizing for actor training
- ``actor_rollout_ref.actor.ppo_max_token_len_per_gpu``: Maximum token count per GPU for actor training
- ``actor_rollout_ref.ref.log_prob_use_dynamic_bsz``: Enable dynamic batch sizing for reference model computations
- ``actor_rollout_ref.ref.log_prob_max_token_len_per_gpu``: Maximum token count per GPU for reference log prob calculations
- ``actor_rollout_ref.rollout.log_prob_use_dynamic_bsz``: Enable dynamic batch sizing for rollout log prob calculations
- ``actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu``: Maximum token count per GPU for rollout phase

FSDP (Fully Sharded Data Parallel) Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FSDP enables training of large models by sharding parameters across multiple GPUs:

- ``actor_rollout_ref.model.use_remove_padding``: **Remove padding optimization** - Eliminates padding tokens to improve computational efficiency and reduce memory usage
- ``actor_rollout_ref.actor.ulysses_sequence_parallel_size``: **Sequence parallelism size** - Number of GPUs to split sequence dimensions across 
- ``actor_rollout_ref.model.enable_gradient_checkpointing``: **Gradient checkpointing** - Trade computation for memory by recomputing activations during backward pass

**Memory Optimization Options:**

- ``actor_rollout_ref.ref.fsdp_config.param_offload``: Offload reference model parameters to CPU memory 
- ``actor_rollout_ref.actor.fsdp_config.param_offload``: Offload actor model parameters to CPU memory
- ``actor_rollout_ref.actor.fsdp_config.optimizer_offload``: Offload optimizer states to CPU memory

Model and Algorithm Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``actor_rollout_ref.model.path``: **Base model path** - HuggingFace model path or local directory containing the pre-trained model
- ``actor_rollout_ref.actor.optim.lr``: **Learning rate** - Learning rate for the optimizer
- ``algorithm.adv_estimator``: **Advantage estimator** - Algorithm type, support ``["gae", "grpo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "rloo"]``

**KL Divergence and Regularization:**

- ``actor_rollout_ref.actor.use_kl_loss``: Enable KL divergence loss to prevent the model from deviating too far from the reference policy
- ``actor_rollout_ref.actor.kl_loss_coef``: KL loss coefficient 
- ``actor_rollout_ref.actor.kl_loss_type``: Type of KL loss computation ``["kl (k1)", "abs", "mse (k2)", "low_var_kl (k3)", "full"]``
- ``actor_rollout_ref.actor.entropy_coeff``: Entropy coefficient for exploration 

Rollout and Inference Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``actor_rollout_ref.rollout.name``: **Inference backend** - Include ``["hf", "sglang", "vllm]"`` 
- ``actor_rollout_ref.rollout.tensor_model_parallel_size``: **Tensor parallelism** - TP size for rollout. Only effective for vllm
- ``actor_rollout_ref.rollout.gpu_memory_utilization``: **GPU memory usage** - Fraction of GPU memory to use for inference 
- ``actor_rollout_ref.rollout.n``: **Samples per prompt** - Number of responses to generate for each prompt during rollout

**Generation Parameters:**

- ``actor_rollout_ref.rollout.temperature``: Controls randomness in generation 
- ``actor_rollout_ref.rollout.top_k``: Top-k sampling parameter 
- ``actor_rollout_ref.rollout.top_p``: Top-p sampling parameter

Training Control Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``trainer.logger``: **Logging backends** - Available options: ``["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "clearml"]``
- ``trainer.project_name``: Project name for experiment tracking
- ``trainer.experiment_name``: Specific experiment identifier
- ``trainer.n_gpus_per_node``: Number of GPUs per compute node
- ``trainer.nnodes``: Number of compute nodes in the cluster
- ``trainer.total_epochs``: Maximum number of training epochs
- ``trainer.save_freq``: Model checkpoint saving frequency (every N steps)
- ``trainer.test_freq``: Validation frequency (-1 disables periodic validation)
  
Multi-Node Training Setup
-------------------------

For large-scale training across multiple nodes, VERL uses Ray for distributed coordination. This section covers cluster setup and management.

Ray Cluster Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Manual Ray Setup:**

1. **Start Head Node:**
   
   .. code-block:: bash
   
       ray start --head --dashboard-host=0.0.0.0

2. **Start Worker Nodes:**
   
   .. code-block:: bash
   
       ray start --address=<head_node_ip:port>

For detailed multi-node setup instructions, refer to the `VERL Multi-node Documentation <https://verl.readthedocs.io/en/latest/start/multinode.html>`_.

Automated Ray Cluster Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this script for automated cluster initialization across multiple nodes:

.. code-block:: bash

    #!/bin/bash

    # Parameter validation
    if [ -z "$RANK" ]; then
        echo "Error: RANK environment variable not set!"
        exit 1
    fi

    # Configuration file path (modify according to actual requirements)
    SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    REPO_PATH=$(dirname "$SCRIPT_PATH")
    RAY_HEAD_IP_FILE=$REPO_PATH/ray_utils/ray_head_ip.txt
    RAY_PORT=$MASTER_PORT  # Ray default port, can be modified as needed

    # Head node startup logic
    if [ "$RANK" -eq 0 ]; then
        # Get local IP address (assuming internal network IP)
        IP_ADDRESS=$(hostname -I | awk '{print $1}')
        # Start Ray head node
        echo "Starting Ray head node on rank 0, IP: $IP_ADDRESS"
        # export VLLM_ATTENTION_BACKEND=XFORMERS
        # export VLLM_USE_V1=0
        ray start --head --memory=461708984320 --port=29500
        
        # Write IP to file
        echo "$IP_ADDRESS" > $RAY_HEAD_IP_FILE
        echo "Head node IP written to $RAY_HEAD_IP_FILE"
    else
        # Worker node startup logic
        echo "Waiting for head node IP file..."
        
        # Wait for file to appear (maximum 360 seconds)
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


Benchmark Results
-----------------

Performance evaluation of VERL using the Boba mathematical reasoning dataset with DeepSeek-R1-Distill-Qwen-7B model. Testing conducted on July 22, 2025, using the `latest VERL <https://github.com/volcengine/verl>`_.

**Test Configuration:**
- **VERL Commit ID**: f252da3
- **Model**: DeepSeek-R1-Distill-Qwen-7B
- **Dataset**: Boba mathematical reasoning dataset
- **Hardware**: 8 nodes × 8 GPUs
- **Algorithm**: GRPO 

================== ============ ========
Metric             Value        Unit    
================== ============ ========
generate_sequences 316.959756   seconds 
reshard            4.191206     seconds 
gen                325.086604   seconds 
reward             5.143515     seconds 
old_log_prob       21.583357    seconds 
ref                20.738621    seconds 
adv                0.465133     seconds 
update_actor       73.008971    seconds 
step               447.358303   seconds 
response_length    10425.773048 tokens  
save_checkpoint    6.603002     seconds 
================== ============ ========



Performance evaluation of VERL using the Boba mathematical reasoning dataset with DeepSeek-R1-Distill-Qwen-1.5B model. Testing conducted on Aug 4, 2025, using `VERL <https://github.com/volcengine/verl>`_.


Used VERL params are as belows:

==================== ===============================
Params               Value
==================== ===============================
VERL Commit ID       8fdc4d3 (v0.5.0 release)
Model                DeepSeek-R1-Distill-Qwen-1.5B
Dataset              Boba math reasoning dataset
Hardware             1 nodes × 8 H100 GPUs 
Sequence Parallelism 2
Data Parallelism     4
Pipeline Parallelism 1
Context Length       28672
MaxPrompt Length     1024
Batch Size Per DP    128
==================== ===============================

VERL benchmark results are as follows:

======================= =============== ====================
Metric                  Value           Unit    
======================= =============== ====================
response length         14254.837890625 tokens
generation time         260.922         seconds 
prev logprob time       17.513          seconds 
training time           61.125          seconds 
step time               363.545         seconds 
gen throughput          6992.96         per-GPU tokens/s
prev logprob throughput 52635.84        per-GPU tokens/s
step throughput         20022.92        total tokens/s
======================= =============== ====================

Used RLinf params are as belows:

==================== ===============================
Params               Value
==================== ===============================
Model                DeepSeek-R1-Distill-Qwen-1.5B
Dataset              Boba math reasoning dataset
Hardware             1 nodes × 8 H100 GPUs 
Tensor Parallelism   2
Data Parallelism     4
Pipeline Parallelism 1
Context Length       28672
MaxPrompt Length     1024
Batch Size Per DP    128
recompute            6 blocks
==================== ===============================


RLinf benchmark results are as follows:

======================= =============== ====================
Metric                  Value           Unit    
======================= =============== ====================
response length         13975.00        tokens
generation time         266.083         seconds 
prev logprob time       17.783          seconds 
training time           61.125          seconds 
step time               346.33          seconds 
gen throughput          6800.64         per-GPU tokens/s
prev logprob throughput 50835.06        per-GPU tokens/s
step throughput         20881.81        total tokens/s
======================= =============== ====================

**Note**: RLinf results below does not count ref logprob time. 