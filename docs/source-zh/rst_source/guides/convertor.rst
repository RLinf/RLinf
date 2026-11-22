Checkpoint 转换
=================

RLinf 提供 Megatron-LM 和 FSDP checkpoint 转换脚本，支持将 ``.distcp`` 或 ``.pt`` 格式的 checkpoint 文件转换为 ``.safetensors`` 格式。

FSDP 转换脚本
----------------------

FSDP Checkpoint 文件结构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSDP checkpoint 文件结构如下：

.. code-block:: text

   experiment_name/checkpoints/
   ├── global_step_10/
   │   └── actor/
   │       ├── dcp_checkpoint/
   │       │   ├── __0_0.distcp
   │       │   ├── __1_0.distcp
   │       │   ├── __2_0.distcp
   │       │   └── __3_0.distcp
   │       └── model_state_dict/
   │           └── full_weights.pt
   └── global_step_20/
       └── …

RLinf 在保存 checkpoint 时会同时保存一份完整的模型权重，例如 ``global_step_10/actor/model_state_dict/full_weights.pt``。该 pt 文件可直接用于评估 checkpoint（将 ``runner.ckpt_path`` 设为该文件路径），或继续转换为 safetensors 格式。


distcp 格式转换到 pt 格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果使用最新代码保存 checkpoint，目录中已包含 ``model_state_dict/full_weights.pt`` 文件，可跳过此步骤。如需将已保存的 ``.distcp`` 文件转换为 ``.pt`` 格式，或旧版本代码仅保存了 ``.distcp`` 文件，可先运行 DCP 转 PT 脚本将 ``.distcp`` 转换为 ``.pt``，再执行后续操作。

.. code-block:: bash

   python convert_dcp_to_pt.py [-h] --dcp_path DCP_PATH --output_path OUTPUT_PATH


其中 ``DCP_PATH`` 是包含 DCP 文件的目录，``OUTPUT_PATH`` 是保存转换后模型 State Dict 文件的路径。



pt 格式转换到 safetensors 格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **修改配置文件**

运行转换脚本前，先修改 FSDP 模型转换配置文件 ``fsdp_model_convertor.yaml``\ （位于 FSDP 转换脚本的 ``config`` 目录），确认以下 7 个参数正确：

    ``defaults``、``convertor.save_path``、``convertor.merge_lora_weighs``、``convertor.ckpt_path``、``model.model_type``、``model.model_path``、``model.is_lora``

.. code-block:: yaml

    defaults:
        - model/openvla_oft@model                                    # 默认的模型参数，根据需要转换的模型类型指定对应的模型
        - override hydra/job_logging: stdout

    convertor:
        save_path: /path/to/save                                     # 转换后的文件保存路径
        merge_lora_weighs: True                                      # 是否保存 merge LoRA 后的权重，如果设置为 False 则仅保存 LoRA 权重
        ckpt_path: /path/to/model_state_dict/full_weights.pt         # 完整的模型权重文件

    # Override the default values in model/openvla_oft
    model:
        model_type: "openvla_oft"                                    # 模型类型，根据需要转换的模型类型指定
        model_path: "/path/to/Openvla-oft-SFT-libero-goal-traj1/"    # 初始模型权重
        is_lora: True                                                # 是否开启 LoRA


2. **（可选）添加自定义保存逻辑**

如果你的模型需要特殊保存逻辑，可在 FSDP 转换工具中注册对应的保存辅助函数。

3. **运行脚本**

在仓库根目录下，可直接传入训练配置（推荐），或使用修改后的 ``fsdp_model_convertor.yaml``：

.. code-block:: bash

   # 推荐：直接传入训练配置
   python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
       convertor.train_config_path=/path/to/train_config.yaml \
       convertor.ckpt_path=/path/to/model_state_dict/full_weights.pt \
       convertor.save_path=/path/to/save

   # 或使用独立的转换配置 YAML
   python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
       --config-path "$(pwd)/rlinf/utils/ckpt_convertor/fsdp_convertor/config" \
       --config-name fsdp_model_convertor

4. **查看保存的 safetensors 文件**

脚本运行结束后，可在 ``convertor.save_path`` 下查看生成的 HuggingFace 格式文件。


Megatron 转换脚本
---------------------

Megatron检查点文件结构如下：

.. code-block:: text

   logs/grpo-1.5b/checkpoints/
   ├── global_step_50/
   │   ├── actor/
   │   │   ├── iter_0000050/
   │   │   │   ├── mp_rank_00/
   │   │   │   │   ├── distrib_optim.pt
   │   │   │   │   └── model_optim_rng.pt
   │   │   │   └── mp_rank_01/                 
   │   │   │       ├── distrib_optim.pt
   │   │   │       └── model_optim_rng.pt
   │   │   └── latest_checkpointed_iteration.txt
   │   └── data/
   │       └── data.pt                         
   └── global_step_100/
       └── …


请直接运行以下命令。先设置：
1. ``CKPT_PATH_MG`` （Megatron checkpoint路径，例如 ``results/run_name/checkpoints/global_step_xx/actor/``），
2. ``CKPT_PATH_HF`` （HuggingFace目标路径），以及
3. ``CKPT_PATH_ORIGINAL_HF`` （初始化训练的基模checkpoint路径，例如 ``/path/to/DeepSeek-R1-Distill-Qwen-1.5B``）。

.. code-block:: bash

    CKPT_PATH_MG=/path/to/megatron_checkpoint
    CKPT_PATH_HF=/target/path/to/huggingface_checkpoint
    CKPT_PATH_ORIGINAL_HF=/path/to/base_model_checkpoint
    CKPT_PATH_MF="${CKPT_PATH_HF}_middle_file"

    # 示例：1.5B
    python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
        --load-path "${CKPT_PATH_MG}" \
        --save-path "${CKPT_PATH_MF}" \
        --model DeepSeek-R1-Distill-Qwen-1.5B \
        --tp-size 2 \
        --ep-size 1 \
        --pp-size 1 \
        --te-ln-linear-qkv true \
        --te-ln-linear-mlp_fc1 true \
        --te-extra-state-check-none true \
        --use-gpu-num 0 \
        --process-num 16

    python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
        --load-path "${CKPT_PATH_MF}" \
        --save-path "${CKPT_PATH_HF}" \
        --model DeepSeek-R1-Distill-Qwen-1.5B \
        --use-gpu-num 0 \
        --process-num 16

    rm -rf "${CKPT_PATH_MF}"
    rm -f "${CKPT_PATH_HF}"/*.done
    shopt -s extglob
    cp "${CKPT_PATH_ORIGINAL_HF}"/!(*model.safetensors.index).json "${CKPT_PATH_HF}"
