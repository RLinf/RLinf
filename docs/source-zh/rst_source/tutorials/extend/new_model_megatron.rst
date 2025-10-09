使用 Megatron+SGLang 添加新模型
=============================================

本文档基于 Megatron 实现，提供了在 RLinf 框架中添加新的 HuggingFace 模型的完整指南。  

前置条件
-------------

* 熟悉 **Megatron-LM** 分布式训练框架  
* 理解 **RLinf** 框架架构  
* 掌握 **PyTorch** 和分布式训练知识  

逐步实现
---------------------------

1. Megatron 模型配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**文件：** ``rlinf/hybrid_engines/megatron/megatron_model_manager.py``  
修改 RLinf 传递给 Megatron 的 ``model_provider_func``，以支持自定义模型结构。  

.. code-block:: python

   def model_provider_func(self, pre_process, post_process):
       """模型依赖于流水线并行。"""
       use_te = HAVE_TE

       if self.mcore_gpt:
           model = MCoreGPTModel(
               config=self.transformer_config,
               transformer_layer_spec=get_specs(
                   self.spec_name,
                   self.transformer_config,
                   use_te,
               ),
               vocab_size=self._cfg.model.override_vocab_size,
               max_sequence_length=self._cfg.model.max_position_embeddings,
               pre_process=pre_process,
               post_process=post_process,
               parallel_output=True,
               share_embeddings_and_output_weights=self._cfg.model.share_embeddings_and_output_weights,
               position_embedding_type=self._cfg.model.position_embedding_type,
               rotary_percent=self._cfg.model.rotary_percentage,
               seq_len_interpolation_factor=self._cfg.model.seq_len_interpolation_factor,
               rotary_base=self._cfg.model.rotary_base,
           )

       else:
           from megatron.legacy.model.gpt_model import GPTModel
           config = build_config(ModelParallelConfig, self._cfg.model)
           setattr(config, 'hidden_size', self._cfg.model.hidden_size)

           model = GPTModel(
               config=config,
               num_tokentypes=0,
               parallel_output=True,
               pre_process=pre_process,
               post_process=post_process
           )
       return model

2. Megatron 权重转换支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**工具：** ``toolkits/ckpt_convert``  
提供与修改后的模型匹配的权重转换脚本。以下示例将 qwen2.5 的检查点进行转换：  

.. code-block:: bash

   CKPT_PATH_HF=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_hf
   CKPT_PATH_MF=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_middle_file
   CKPT_PATH_MG=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_mg_tp1_pp2

   TP_SIZE=1
   PP_SIZE=2

   # 下载（可选）
   # modelscope download --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' --local_dir "$CKPT_PATH_HF"

   rm -rf $CKPT_PATH_MF
   python convert_hf_to_middle_file.py \
       --load-path $CKPT_PATH_HF \
       --save-path $CKPT_PATH_MF \
       --model 'DeepSeek-R1-Distill-Qwen-1.5B' \
       --use-gpu-num 0 \
       --process-num 16

   rm -rf $CKPT_PATH_MG
   python convert_middle_file_to_mg.py \
       --load-path $CKPT_PATH_MF \
       --save-path $CKPT_PATH_MG \
       --model 'DeepSeek-R1-Distill-Qwen-1.5B' \
       --tp-size $TP_SIZE \
       --tpe-size 1 \
       --ep-size 1 \
       --pp-size $PP_SIZE \
       --use-gpu-num 0 \
       --process-num 16

   rm -rf $CKPT_PATH_MF

3. SGLang 模型配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 仅支持已经被 SGLang 适配过的模型。  
如果需要新模型，需要在 SGLang 中添加适配代码。参考官方指南：  
`SGLang-support-new-model <https://docs.SGLang.ai/supported_models/support_new_models.html>`__  

4. 从 Megatron 同步权重到 SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**文件：** ``rlinf/utils/resharding/utils.py``  
在每次 rollout 前，**MegatronActor** 会将更新后的权重转换为 HuggingFace 格式，并分发到各个 SGLang 实例。  
目前仅实现了 *Qwen-2.5* 系列。  

.. code-block:: python

   @staticmethod
   def convert_mega_qwen2_5_to_hf(model_state_dict: dict, config) -> dict:
       new_statedict = {}
       for name, param in model_state_dict.items():
           transform_type, hf_names = TransformFunc.mega_name_qwen2_5_to_hf(name)
           if transform_type == TransformType.SPLIT_QKV:
               TransformFunc._split_gqa_tensor(param, new_statedict, hf_names, config)
           elif transform_type == TransformType.SPLIT_QKV_BIAS:
               TransformFunc._split_gqa_tensor(param, new_statedict, hf_names, config)
           elif transform_type == TransformType.SPLIT_FC1:
               TransformFunc.split_fc1(param, new_statedict, hf_names, config)
           elif transform_type == TransformType.SPLIT_NONE:
               TransformFunc.split_none(param, new_statedict, hf_names)
           else:
               raise NotImplementedError(f"Transform type {transform_type} not implemented")
       return new_statedict

5. 配置文件
~~~~~~~~~~~~~~~~~~~~~

**文件：** ``examples/math/config/qwen2.5-1.5b-grpo-megatron.yaml``  
设置 RLinf 使用的 Megatron 参数。  

.. code-block:: yaml

   # Megatron 参数
   model:
    precision: fp16
    add_bias_linear: False

    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1

    activation: swiglu
    sequence_parallel: True
    # recompute_method: block
    # recompute_granularity: selective

    recompute_method: block
    recompute_granularity: full
    recompute_num_layers: 20

    seq_length: ${runner.seq_length}
    encoder_seq_length: ${runner.seq_length}

    normalization: rmsnorm

    position_embedding_type: rope

    apply_rope_fusion: True
    bias_dropout_fusion: False
    persist_layer_norm: False
    bias_activation_fusion: False
    attention_softmax_in_fp32: True
    batch_p2p_comm: False
    variable_seq_lengths: True
    gradient_accumulation_fusion: False
    moe_token_dispatcher_type: alltoall
    use_cpu_initialization: False

   megatron:
    ddp_bucket_size: null
    distributed_backend: nccl # 支持 'nccl' 和 'gloo'
    distributed_timeout_minutes: 30
    ckpt_format: torch
    use_dist_ckpt: False
    tp_comm_bootstrap_backend: nccl
    tp_comm_overlap_cfg: null # tp_comm_overlap_cfg.yaml
    use_hf_ckpt: True # 若为 True，会将 hf 模型转换生成 megatron 检查点，并用于训练。

    ckpt_convertor: # ckpt 转换器配置
      model: DeepSeek-R1-Distill-Qwen-1.5B
      model_type: null # 若为 null，会根据 hf 模型配置自动设置
      hf_model_path: ${rollout.model_dir} # hf 模型路径
      save_path: ${runner.output_dir}/${runner.experiment_name}/converted_ckpts/actor
      use_gpu_num : 0
      use_gpu_index: null
      process_num: 16 # 用于检查点转换的进程数
      tensor_model_parallel_size: ${actor.model.tensor_model_parallel_size}
      pipeline_model_parallel_size: ${actor.model.pipeline_model_parallel_size}
