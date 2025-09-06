Switch SGLang Versions
======================

RLinf can plug different *generation backends* into its
reinforcement-learning pipeline. For the current release **only
SGLang** is supported; vLLM integration is under development.

.. note::

   RLinf is compatible with **SGLang 0.4.4 → 0.4.9**.  
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
   pip install sglang==0.4.9

Install from Source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   git checkout v0.4.8          # pick the tag you need
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

        model_dir: /model/path # model path
        model_arch: qwen2.5 # model arch
        enforce_eager: False         # if False, rollout engine will capture cuda graph, which will take more time to initialize.
        distributed_executor_backend: mp   # ray or mp
        disable_log_stats: False     # if true will log sglang's output
        detokenize: False            # Whether to detokenize the output. During RL we actually don't need to detokenize it. Can be set to True for debugging.
        padding: null               # will be tokenizer.pad_token_id if null. it is used to filter megatron's padding for rollout engine
        eos: null                   # will be tokenizer.eos_token_id if null.

        attention_backend: triton # attention backend used by SGLang
        recompute_logprobs: True # whether SGLang will compute log probs

        tensor_parallel_size: 1 # tp_size
        pipeline_parallel_size: 1 # pp_size
        
        validate_weight: False # whether to send all weights at first for weight comparison.
        validate_save_dir: null # the directory to save the weights for comparison. If validate_weight is True, this will be used to save the weights for comparison.
        print_outputs: False         # whether to print the outputs (token ids, texts, etc.) of rollout engine.

        sglang_decode_log_interval: 500000 # the interval for SGLang to log the decode time and other stats.
        max_running_requests: 64 # the maximum number of running requests in the rollout engine.
        cuda_graph_max_bs: 128 # the maximum batch size for cuda graph. If the batch size is larger than this, cuda graph will not be used.

        use_torch_compile: False # enable torch_compile in SGLang for rollout.
        torch_compile_max_bs: 128 # the maximum batch size for torch compile. If the batch size is larger than this, torch compile will not be used.

    ...


Internal Version Routing
------------------------

Directory layout::

   rlinf/hybrid_engines/sglang/
   ├── __init__.py               # Version detection and routing
   ├── sglang_worker.py          # Main worker implementation
   ├── sglang_0_4_4/             # SGLang 0.4.4 specific implementation
   │   ├── __init__.py
   │   ├── io_struct.py          # I/O structures for 0.4.4
   │   ├── sgl_engine.py         # Engine implementation for 0.4.4
   │   ├── sgl_scheduler.py      # Scheduler for 0.4.4
   │   └── tokenizer_manager.py  # Tokenizer management for 0.4.4
   └── sglang_0_4_x/             # Future version implementations
       └── ...

The loader in ``__init__.py`` resolves the installed package:

.. code-block:: python

   from importlib.metadata import PackageNotFoundError, version

   def get_version(pkg):
       try:
           return version(pkg)
       except PackageNotFoundError:
           return None

   package_name = "sglang"
   package_version = get_version(package_name)
   
   if package_version == "0.4.4":
       sglang_version = "0.4.4"
       from .sglang_0_4_4 import io_struct
       from .sglang_0_4_4.sgl_engine import Engine
   else:
       raise ValueError(f"sglang version {package_version} not supported")
