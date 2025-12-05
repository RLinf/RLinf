Switch SGLang Versions
======================

RLinf can plug different *generation backends* into its
reinforcement-learning pipeline. For the current release **SGLang and vLLM** is supported;

.. note::

   RLinf is compatible with **SGLang 0.4.4 → 0.5.2**, **vLLM 0.8.5  → 0.8.5.post1**.  
   No manual patching is required – the framework detects the installed
   version and loads the matching shim automatically.

Installation Requirements
-------------------------

* **CUDA** ≥ 11.8 (or 12.x matching your PyTorch build)  
* **Python** ≥ 3.8  
* Sufficient **GPU memory** for the chosen model  
* Compatible versions of **PyTorch** and *transformers*

.. note::

   Mismatched CUDA / PyTorch wheels are the most common installation
   issue.  Verify both before installing SGLang.

Install via pip
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Reference version
   pip install sglang==0.4.4

   # Recommended for production
   pip install sglang==0.4.8

   # Latest supported
   pip install sglang==0.5.2

   # Install vLLM
   pip install vllm==0.8.5

Install from Source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install SGLang
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   git checkout v0.4.8          # pick the tag you need
   pip install -e "python[all]"

   # Install vLLM
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git checkout v0.8.5          # pick the tag you need
   pip install -e .

.. note::

   Building from source can be time-consuming and heavy on disk space;
   prefer the pre-built wheels unless you need bleeding-edge fixes.

----------------------------

.. code-block:: yaml

    ....
    rollout:
        group_name: "RolloutGroup" # SGLang Generation Group Name, used for communication

        gpu_memory_utilization: 0.55 # SGLang's parameter, which decides how much vram is used for static memory pool

        model:
           model_path: /model/path # model path
           model_type: qwen2.5 # model type
        enforce_eager: False         # if False, rollout engine will capture cuda graph, which will take more time to initialize.
        distributed_executor_backend: mp   # ray or mp
        disable_log_stats: False     # if true will log sglang's output
        detokenize: False            # Whether to detokenize the output. During RL we actually don't need to detokenize it. Can be set to True for debugging.
        padding: null               # will be tokenizer.pad_token_id if null. it is used to filter megatron's padding for rollout engine
        eos: null                   # will be tokenizer.eos_token_id if null.

        rollout_backend: sglang     # [sglang, vllm] here to choose which rollout backend to use.

        sglang: # used when rollout_backend is sglang
            attention_backend: triton # [flashinfer, triton] for more, see sglang's doc
            decode_log_interval: 500000 # the interval for SGLang to log the decode time and other stats.
            use_torch_compile: False # enable torch_compile in SGLang for rollout.
            torch_compile_max_bs: 128 # the maximum batch size for torch compile. If the batch size is larger than this, torch compile will not be used.

        vllm: # used when rollout_backend is vllm
            attention_backend: FLASH_ATTN # [FLASH_ATTN,XFORMERS] attention backend used by vLLM, for more info,see vLLM's doc
            enable_chunked_prefill: True  # enable vllm to use chunked_prefill.
            enable_prefix_caching: True  # enable vllm to use prefix_caching.
            enable_flash_infer_sampler: True #  # if True, vllm will use flashinfer to do sampling.

        tensor_parallel_size: 1 # tp_size
        pipeline_parallel_size: 1 # pp_size
        
        validate_weight: False # whether to send all weights at first for weight comparison.
        validate_save_dir: null # the directory to save the weights for comparison. If validate_weight is True, this will be used to save the weights for comparison.
        print_outputs: False         # whether to print the outputs (token ids, texts, etc.) of rollout engine.

        max_running_requests: 64 # the maximum number of running requests in the rollout engine.
        cuda_graph_max_bs: 128 # the maximum batch size for cuda graph. If the batch size is larger than this, cuda graph will not be used.

    ...
