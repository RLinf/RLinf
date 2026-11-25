配置 Qwen3.5 权重同步
================================================================================

以下配置用于在已有的 Megatron/SGLang 任务中同步 Qwen3.5 Dense 模型的权重，包括语言模型、视觉编码器、Merger 和可选的多 token 预测（MTP）分支。

准备环境
----------------------------------------------

checkpoint 的配置应声明 ``Qwen3_5ForConditionalGeneration``\ 。actor 和 rollout 都需要读取完整的 checkpoint，包括 tokenizer 和 processor 文件。

在仓库根目录安装 agentic SGLang 环境：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang

这条命令会安装 SGLang 0.5.12.post1、Megatron Bridge 和 FLA。引擎版本的选择方式见 :doc:`version`。

配置模型
------------------------------------------------------

将以下片段合入任务配置，把三处 ``<checkpoint>`` 都替换为同一个本地 checkpoint 路径。

.. code-block:: yaml

   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
       precision: bf16
     optim:
       bf16: true
       fp16: false
     tokenizer:
       tokenizer_model: <checkpoint>
     megatron:
       mbridge: true
       use_hf_ckpt: true
       ckpt_convertor:
         hf_model_path: <checkpoint>
   rollout:
     rollout_backend: sglang
     model:
       model_type: qwen3_5
       model_path: <checkpoint>
       precision: bf16

校验首次同步
------------------------------------------

从初始模型启动任务时，加入以下配置，开启 checkpoint 比对：

.. code-block:: yaml

   rollout:
     validate_weight_first_sync: true

用任务原有的命令启动。SGLang 会将 checkpoint 权重与首次收到的 actor 权重比对。各 rollout rank 的日志中应出现 ``sglang: validate_weight success``\ 。如果不一致，程序会报错并列出有差异的参数。

这项检查比较权重范数，允许一定的数值误差。从训练 checkpoint 恢复时，权重可能已不同于初始模型，RLinf 会跳过检查。新任务的首次校验通过后，可移除 ``rollout.validate_weight_first_sync``\ ，省去同步前从磁盘加载 rollout 权重的开销。

开启 MTP 训练与推理
------------------------------------------------------

需要训练和同步 MTP 时，加入：

.. code-block:: yaml

   actor:
     model:
       load_mtp: true

这个开关还会开启 NEXTN 投机解码，使用同一 checkpoint 中的 MTP 分支作为 draft。不支持外部 draft checkpoint。

训练会对生成的 response token 计算多 token 预测损失，梯度也会更新语言模型主干。``actor/mtp_loss`` 记录缩放前的损失值，``actor.model.mtp_loss_scaling_factor`` 控制它在训练中的权重。开启 ``rollout.validate_weight_first_sync`` 后，也会校验已启用的 draft。

选择视觉分支布局
------------------------------------------------

设置 ``rollout.sglang.mm_enable_dp_encoder: true``\ ，可以在 rollout 各 rank 上保留完整的视觉编码器副本；否则使用张量并行。例如，以下配置对应 attention TP 2、前馈层 TP 4、视觉分支 TP 1：

.. code-block:: yaml

   rollout:
     tensor_parallel_size: 4
     sglang:
       enable_dp_attention: true
       dp_size: 2
       mm_enable_dp_encoder: true

SGLang rollout 需要保持 ``rollout.pipeline_parallel_size: 1``\ 。actor 的张量并行、流水线并行和虚拟流水线并行可以与 rollout 的并行配置不同。Megatron 并行设置见 :doc:`5D`，GPU 部署设置见 :doc:`../concepts/placement`。
