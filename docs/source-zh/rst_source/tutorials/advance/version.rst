切换 SGLang 版本
======================

RLinf 可以将不同的 *generation backends* 接入其强化学习流水线。  
在当前版本中 **支持 SGLang与vLLM**。

.. note::

   RLinf 兼容 **SGLang 0.4.4 → 0.5.2**, **vLLM 0.8.5  → 0.8.5.post1**  
   不需要手动打补丁 —— 框架会自动检测已安装的版本并加载匹配的 shim。  

安装要求
-------------------------

* **CUDA** ≥ 11.8（或与 PyTorch 构建版本匹配的 12.x）  
* **Python** ≥ 3.8  
* 所选模型需要足够的 **GPU 内存**  
* 兼容版本的 **PyTorch** 和 *transformers*  

.. note::

   CUDA / PyTorch 版本不匹配是最常见的安装问题。  
   安装 SGLang 前请先确认二者版本一致。  

通过 pip 安装


.. code-block:: bash

   # 参考版本
   pip install sglang==0.4.4

   # 推荐用于生产
   pip install sglang==0.4.8

   # 最新支持版本
   pip install sglang==0.5.2

   # 安装vLLM
   pip install vllm==0.8.5


从源码安装

.. code-block:: bash

   # 安装 SGLang
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   git checkout v0.4.8          # 选择需要的 tag
   pip install -e "python[all]"

   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git checkout v0.8.5          # 选择需要的 tag
   pip install -e .

.. note::

   从源码构建可能耗时且占用大量磁盘空间；  
   除非需要最新修复，否则推荐使用预编译的 wheels。  

----------------------------

.. code-block:: yaml

    ....
    rollout:
        group_name: "RolloutGroup" # SGLang Generation Group 名称，用于通信

        gpu_memory_utilization: 0.55 # SGLang 参数，决定静态内存池使用的显存比例

        model:
          model_path: /model/path # 模型路径
          model_type: qwen2.5    # 模型架构
        enforce_eager: False   # 若为 False，rollout 引擎会捕获 cuda graph，会增加初始化时间
        distributed_executor_backend: mp   # ray 或 mp
        disable_log_stats: False     # 若为 True，则关闭 sglang 输出日志
        detokenize: False            # 是否反解码输出。在 RL 训练中通常不需要反解码，可设为 True 进行调试
        padding: null                # 若为 null，则使用 tokenizer.pad_token_id；用于过滤 Megatron 的 padding
        eos: null                    # 若为 null，则使用 tokenizer.eos_token_id

        rollout_backend: sglang     # [sglang, vllm] 在这里选择所使用的 rollout 引擎,目前支持SGLang与vLLM

        sglang:
            attention_backend: triton # [flashinfer, triton] SGLang 使用的注意力后端,更多信息见 SGLang 文档
            decode_log_interval: 500000 # SGLang 打印解码时间和统计信息的间隔
            use_torch_compile: False # 是否在 SGLang rollout 中启用 torch_compile
            torch_compile_max_bs: 128 # torch compile 的最大 batch size，超过则不使用

        vllm:
            attention_backend: FLASH_ATTN # [FLASH_ATTN,XFORMERS] VLLM 使用的注意力后端,更多信息见 vLLM 文档
            enable_chunked_prefill: True  # 是否在 vLLM 中启用 chunked_prefill
            enable_prefix_caching: True   # 是否在 vLLM 中启用 prefix_caching
            enable_flash_infer_sampler: True # 是否在 vLLM 中使用flashinfer 代替原有Pytorch实现的采样

        tensor_parallel_size: 1      # tp_size
        pipeline_parallel_size: 1    # pp_size
        
        validate_weight: False       # 是否在开始时发送所有权重用于对比
        validate_save_dir: null      # 保存权重对比文件的目录
        print_outputs: False         # 是否打印 rollout 引擎的输出（token ids, texts 等）

        max_running_requests: 64     # rollout 引擎的最大并发请求数
        cuda_graph_max_bs: 128       # cuda graph 的最大 batch size，超过则不使用 cuda graph

    ...

