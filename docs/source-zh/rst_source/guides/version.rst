切换 SGLang 版本
======================

RLinf 可以将不同的 *generation backends* 接入其强化学习流水线。  
在当前版本中 **支持 SGLang与vLLM**。

.. note::

   RLinf 兼容 **SGLang 0.4.4 → 0.5.13** 与 **vLLM 0.8.5 → 0.23.x**。
   不需要手动打补丁 —— 框架会自动检测已安装的版本并加载匹配的 shim。

支持的引擎版本
-------------------------

RLinf 能安装的每个引擎版本都对应 ``requirements/agentic/`` 下的一个文件，
命名为 ``<engine>_<version>_<cu12|cu13>.txt``。目录中的文件集合*就是*受支持
的版本集合，因此列出该目录即可知道可以安装哪些版本。

============  ==============  =========  =======
引擎          版本            CUDA 分支  torch
============  ==============  =========  =======
SGLang        0.5.12.post1    cu12,cu13  2.11.0
SGLang        0.5.4           cu12       2.8.0
SGLang        0.5.2           cu12       2.8.0
SGLang        0.4.6.post5     cu12       2.6.0
vLLM          0.23.0          cu12,cu13  2.11.0
vLLM          0.8.5           cu12       2.6.0
============  ==============  =========  =======

默认版本是上表中最高的版本——目前是 torch 2.11 上的 SGLang 0.5.12.post1 与
vLLM 0.23.0——该默认值直接由这些文件推导得出，因此新增一个更高版本的文件即可
让默认值随之前进。CUDA 分支跟随 torch wheel 而非驱动，因此没有 cu13 构建的
torch 版本即使在 CUDA 13 主机上也会留在 cu12 分支。

每个 venv 只装一个引擎
-------------------------

一个 venv 只包含一个引擎。SGLang 与 vLLM 会把同一批 kernel 库
（``nvidia-cutlass-dsl``、``flashinfer-python``、``tilelang``、
``tokenspeed-mla``）固定到不同版本，有时 torch 版本也不同；若共用一个 venv，
后安装的那个会把先装的 kernel 降级——而且不会报错，直到真正调用 kernel 时才暴露。
用 ``--engine`` 选择引擎；要两个都装就用不同的 ``--venv`` 各装一次：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang
   bash requirements/install.sh agentic --venv .venv-vllm --engine vllm

reason Docker 镜像同时提供两者：``reason``（SGLang，默认激活）与 ``reason-vllm``。

切换版本
-------------------------

把版本传给安装脚本即可，torch 会随之确定，无需另外指定：

.. code-block:: bash

   # 默认：torch 2.11 上的 SGLang 0.5.12.post1
   bash requirements/install.sh agentic

   # SGLang 0.4.x 分支，torch 2.6
   bash requirements/install.sh agentic --sglang 0.4.6.post5

   # vLLM 0.8.5，torch 2.6
   bash requirements/install.sh agentic --engine vllm --vllm 0.8.5

传入不受支持的版本会立即报错并列出实际存在的版本，而不会解析出一个不可用的环境。

.. note::

   请不要用 ``pip install sglang`` / ``pip install vllm`` 装进已有环境。两者都会
   固定整个 torch 系列，而它们的 CUDA 13 版本会固定 CUDA 13 运行时 wheel，
   并就地覆盖对应的 CUDA 12 版本 —— 这些 requirements 文件正是为了避免这种情况。

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

