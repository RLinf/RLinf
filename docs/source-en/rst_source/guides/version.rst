Switch SGLang Versions
======================

RLinf can plug different *generation backends* into its
reinforcement-learning pipeline. For the current release **SGLang and vLLM** is supported;

.. note::

   RLinf is compatible with **SGLang 0.4.4 -> 0.5.13** and **vLLM 0.8.5 ->
   0.23.x**.  No manual patching is required - the framework detects the
   installed version and loads the matching shim automatically.

Supported engine builds
-----------------------

Each engine build RLinf can install is described by one file under
``requirements/agentic/``, named ``<engine>_<version>_<cu12|cu13>.txt``.  The
set of files *is* the set of supported versions, so listing the directory
answers "what can I install?".

============  ==============  =========  =======
Engine        Version         CUDA line  torch
============  ==============  =========  =======
SGLang        0.5.12.post1    cu12,cu13  2.11.0
SGLang        0.5.4           cu12       2.8.0
SGLang        0.5.2           cu12       2.8.0
SGLang        0.4.6.post5     cu12       2.6.0
vLLM          0.23.0          cu12,cu13  2.11.0
vLLM          0.8.5           cu12       2.6.0
============  ==============  =========  =======

The default is the highest version listed above -- SGLang 0.5.12.post1 and vLLM
0.23.0 today, both on torch 2.11 -- derived from the files themselves, so adding
a newer one moves the default with it.  The CUDA line follows the torch wheel
rather than the driver, so a torch version with no cu13 build stays on cu12 even
on a CUDA 13 host.

One venv per engine
-------------------

A venv holds exactly one engine.  SGLang and vLLM pin the same kernel libraries
(``nvidia-cutlass-dsl``, ``flashinfer-python``, ``tilelang``,
``tokenspeed-mla``) to different versions, and sometimes a different torch, so
sharing a venv means whichever is installed second downgrades the other's
kernels -- silently, until a kernel actually runs.  ``--engine`` selects the
engine; install twice with different ``--venv`` to get both:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang
   bash requirements/install.sh agentic --venv .venv-vllm --engine vllm

The reason Docker image ships both, in ``reason`` (SGLang, activated by default)
and ``reason-vllm``.

Switching versions
------------------

Pass the version to the installer; torch follows from it, so it need not be
given as well:

.. code-block:: bash

   # default: SGLang 0.5.12.post1 on torch 2.11
   bash requirements/install.sh agentic

   # the SGLang 0.4.x line, on torch 2.6
   bash requirements/install.sh agentic --sglang 0.4.6.post5

   # vLLM 0.8.5, on torch 2.6
   bash requirements/install.sh agentic --engine vllm --vllm 0.8.5

An unsupported version fails immediately and prints the builds that do exist,
rather than resolving into a broken environment.

.. note::

   Avoid ``pip install sglang`` / ``pip install vllm`` into an existing
   environment.  Both pin the whole torch family, and their CUDA 13 releases
   pin CUDA 13 runtime wheels that overwrite their CUDA 12 counterparts in
   place -- the requirements files exist precisely to keep that from happening.

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
