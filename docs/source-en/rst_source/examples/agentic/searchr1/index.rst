Search-R1 RL Training
=====================

Reproduce `Search-R1 <https://arxiv.org/abs/2503.09516>`__ in RLinf, training LLMs with multi-turn reinforcement learning to answer questions by calling a local Wiki-2018 search tool. RLinf runs Qwen2.5-3B 55% faster than the original implementation on 8×H100 and cuts GLM-4.7-Flash step time in half with 2-node asynchronous training.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-3B-Instruct · GLM-4.7-Flash

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO · PPO · Async Multi-Turn RL

   .. grid-item-card:: Tools
      :text-align: center

      Qdrant or FAISS local Wiki-2018 server

   .. grid-item-card:: Hardware
      :text-align: center

      8×H100 · 8×A100 · 2×8 A100

| **What you'll do:** start the local Wiki-2018 retriever → install & download → launch ``run_train.sh`` → monitor reward and step time.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · :doc:`Multi-node guide </rst_source/guides/multi_node>` (2-node runs) · :doc:`Async Agentic RL </rst_source/guides/async_agentic>` (async tuning).

.. _searchr1-setup:
.. _searchr1-retriever:
.. _searchr1-glm:
.. _searchr1-async:

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__. Start the local retriever (Tab 1) before launching any training or evaluation run.

.. tab-set::

   .. tab-item:: 1. Local Wiki Retriever
      :sync: searchr1-retriever

      Both backends expose ``POST /retrieve`` and ``POST /access`` on port ``8000`` using `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__. Prebuilt Qdrant is recommended: Qdrant runs on CPU, so only the encoder uses GPU memory (~1 GB).

      **1. Install & Download (Qdrant, recommended)**

      .. code-block:: bash

         uv pip install qdrant-client==1.16.2
         hf download --repo-type dataset RLinf/Wiki-2018-Corpus --local-dir ./Wiki-2018-Corpus
         hf download intfloat/e5-base-v2 --local-dir ./e5-base-v2

      **2. Start Qdrant & Retrieval Server**

      Start Qdrant in tmux (``cd ./Wiki-2018-Corpus/qdrant && ./qdrant``), then configure ``WIKI2018_DIR``, ``retriever_path``, and ``qdrant_url`` in ``examples/agent/tools/search_local_server_qdrant/launch_local_server.sh`` (collection wiki_collection_m32_cef512, HNSW ef 256) and launch:

      .. code-block:: bash

         bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

      Run the retriever on the training host when possible; if remote, find its IP with ``hostname -I`` and pass ``tools.search.server_addr=<server_ip>:8000``.

      .. dropdown:: Custom Qdrant index and FAISS GPU (ASearcher) option

         **Custom Qdrant index:** download a binary from `Qdrant Releases <https://github.com/qdrant/qdrant/releases>`__, configure ``examples/agent/tools/search_local_server_qdrant/build_index.sh``, and run it (see the `Qdrant optimization guide <https://qdrant.tech/documentation/guides/optimize/>`__).

         **FAISS GPU (ASearcher):** create a ``retriever`` conda environment with PyTorch 2.4.0, ``faiss-gpu=1.8.0``, transformers, datasets, pyserini, uvicorn, and fastapi. Set paths in both scripts under ``examples/agent/tools/search_local_server_faiss/``, then download, index, and serve:

         .. code-block:: bash

            python examples/agent/searchr1/download.py --save_path ./wiki2018
            hf download intfloat/e5-base-v2 --local-dir ./e5-base-v2
            bash examples/agent/tools/search_local_server_faiss/build_index.sh
            bash examples/agent/tools/search_local_server_faiss/launch_local_server.sh

         References: `Search-R1 <https://github.com/PeterGriffinJin/Search-R1>`__, `ASearcher <https://github.com/inclusionAI/ASearcher>`__.

   .. tab-item:: 2. Qwen2.5-3B (GRPO / PPO / FSDP)
      :sync: searchr1-qwen

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download `Qwen2.5-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>`__ and `RLinf/Search-R1-Data <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__:

      .. code-block:: bash

         hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. Launch (Megatron GRPO)**

      Launch ``examples/agent/searchr1/config/train_qwen2.5.yaml`` (set ``rollout.detokenize=True`` when using ``sampling_params.stop``, and keep ``rollout.return_logprobs`` equal to ``not algorithm.recompute_logprobs``):

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_qwen2.5 \
           rollout.model.model_path=./Qwen2.5-3B-Instruct \
           'data.train_data_paths=["./Search-R1-Data/train.jsonl"]'

      .. dropdown:: PPO, FSDP GRPO, checkpoint export, and evaluation

         **Megatron PPO** (``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml``): saves both ``actor`` and ``critic`` under each ``global_step_xxx``; set ``runner.resume_dir`` to resume both.

         **FSDP GRPO** (``examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml``): enables dynamic rollout batches with ``actor.training_backend: fsdp`` and ``actor.pack_traj: True`` to pack compatible turns from the same trajectory and avoid recomputing shared prefixes.

         .. code-block:: bash

            bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo \
              rollout.model.model_path=./Qwen2.5-3B-Instruct 'data.train_data_paths=["./Search-R1-Data/train.jsonl"]'
            bash examples/agent/searchr1/run_train.sh train_qwen2.5_fsdp \
              rollout.model.model_path=./Qwen2.5-3B-Instruct 'data.train_data_paths=["./Search-R1-Data/train.jsonl"]'

         **Export checkpoints to Hugging Face and evaluate** (uses ``examples/agent/searchr1/config/eval_qwen2.5.yaml``):

         .. code-block:: bash

            # Megatron (export actor dir; copy tokenizer JSONs from base model afterward)
            python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
              --load-path "${CKPT_PATH_MG}" --save-path "${CKPT_PATH_MF}" --model qwen_2.5_3b \
              --tp-size 1 --ep-size 1 --pp-size 1 --te-ln-linear-qkv true --te-ln-linear-mlp_fc1 true \
              --te-extra-state-check-none true --use-gpu-num 0 --process-num 16
            python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
              --load-path "${CKPT_PATH_MF}" --save-path "${CKPT_PATH_HF}" --model qwen_2.5_3b --use-gpu-num 0 --process-num 16
            # FSDP
            python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
              convertor.train_config_path=/path/to/config.yaml \
              convertor.ckpt_path=/path/to/checkpoints/global_step_xxx/actor/model_state_dict/full_weights.pt \
              convertor.save_path=/path/to/hf_model
            # Standalone evaluation
            bash examples/agent/searchr1/run_eval.sh eval_qwen2.5 \
              rollout.model.model_path=/path/to/hf_model 'data.val_data_paths=["./Search-R1-Data/eval.jsonl"]'

   .. tab-item:: 3. GLM-4.7-Flash
      :sync: searchr1-glm

      GLM-4.7-Flash (MoE, ~3B active parameters) renders ``prompt`` message lists with GLM's chat template (``data.apply_chat_template: True``). Convert Search-R1-Data prompts from Qwen strings into ``[{"role": "system", ...}, {"role": "user", ...}]`` lists first (keep ``solutions`` unchanged; drop the ``<think>`` instruction when thinking is disabled).

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download zai-org/GLM-4.7-Flash --local-dir ./GLM-4.7-Flash
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. Launch (Single-Node 8×A100-80G)**

      Launch ``examples/agent/searchr1/config/train_glm4.7_flash.yaml`` (loads Hugging Face weights directly via ``actor.megatron.mbridge: True``):

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_glm4.7_flash \
           rollout.model.model_path=./GLM-4.7-Flash \
           'data.train_data_paths=["./Search-R1-Data/searchr1_train_glm.jsonl"]' \
           tools.search.server_addr=localhost:8000

      .. dropdown:: 2-node server mode (2×8 A100) and parallelism notes

         Set up a 2-node Ray cluster (:doc:`Multi-node guide </rst_source/guides/multi_node>`, unique ``RLINF_NODE_RANK`` per node), then run ``examples/agent/searchr1/config/train_glm4.7_flash_2node_server.yaml`` (``cluster.num_nodes: 2``; search tool runs on node 0) from the head node:

         .. code-block:: bash

            bash examples/agent/searchr1/run_train.sh train_glm4.7_flash_2node_server \
              rollout.model.model_path=./GLM-4.7-Flash \
              'data.train_data_paths=["./Search-R1-Data/searchr1_train_glm.jsonl"]' \
              tools.search.server_addr=localhost:8000

         **Tokenizer & parser:** ``agentloop.agentic_tokenizer: "glm47"`` must match ``rollout.sglang.tool_call_parser: glm47``.

         **Single-node parallelism:** Megatron actor uses TP4 / EP8 / DP2 and SGLang rollout uses ADP2 / EP8; keep ``rollout.cuda_graph_max_bs`` (256) at least as large as per-rank ``max_running_requests`` (512 / 2).

         **2-node parallelism:** cross-node rollout requires ``rollout.sglang.backend_type: server``, equal rollout GPUs per node with ``tp×pp`` divisible by per-node count, and ``enable_dp_attention: True`` when ``dp_size > 1`` (DP4 / EP16 rollout; TP4 / EP8 / DP4 actor with intra-node EP). Restrict ``NCCL_IB_HCA`` if RoCE/IB NICs span subnets.

   .. tab-item:: 4. Async Training
      :sync: searchr1-async

      Overlap rollout and training across **2 nodes of 8×A100-80G** (``cluster.num_nodes: 2``): Megatron actor on node 0, SGLang server rollout on node 1, and rule-based rewards computed in the agent loop. Complete Install and Download from the **GLM-4.7-Flash** tab on both nodes, start the Qdrant retriever on node 0, and set up the 2-node Ray cluster (:doc:`Multi-node guide </rst_source/guides/multi_node>`).

      **Launch (2×8 A100 Async)**

      Launch ``examples/agent/searchr1/config/train_glm4.7_flash_2node_async.yaml`` from the head node (enables ``runner.async_mode: True``, ``rollout.sglang.backend_type: server``, and ``agentloop.task_name``):

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_glm4.7_flash_2node_async \
           rollout.model.model_path=./GLM-4.7-Flash \
           'data.train_data_paths=["./Search-R1-Data/searchr1_train_glm.jsonl"]' \
           tools.search.server_addr=localhost:8000

      See :doc:`Async Agentic RL </rst_source/guides/async_agentic>` for tuning ``max_inflight_groups``, ``max_buffered_groups``, ``max_actor_accept_staleness``, and ``rollout.pause_mode`` (retract, in_place, drain). To run with the session server, add ``session_server.enable=true`` (see :doc:`Session Server </rst_source/guides/session_server>`).

      .. dropdown:: Async metrics, resume, and limitations

         **Metrics:** watch ``rollout/buffer/inflight_groups``, ``rollout/buffer/size``, ``rollout/buffer/admission_wait_*`` (inflight limit vs buffer full), ``rollout/data_staleness/mean``, and ``rollout/buffer/dropped_groups``.

         **Resume:** checkpoints save the dataloader position to ``data/data.pt`` and skip prompts already issued before saving (192 on the default Qwen config: 128 in-flight/buffered + 64 prefetched).

         **Limitations:** requires Megatron actor, multi-turn agent loop, SGLang server backend, and disaggregated placement; uses rollout logprobs directly (``algorithm.recompute_logprobs: False``); monitor ``dropped_groups`` when enabling ``reject_equal_reward`` (``max_dropped_groups_per_step`` defaults to 10 × batch size).

Visualization and Results
-------------------------

Inspect reward and step-time curves with ``tensorboard --logdir ./logs``; see :doc:`Training metrics </rst_source/reference/metrics>` for W&B/SwanLab and metric definitions.

.. list-table:: **Search-R1 reference results**
   :header-rows: 1
   :widths: 24 16 14 14 32

   * - Setup
     - Hardware
     - Step time
     - Throughput
     - Key outcome
   * - Qwen2.5-3B (GRPO)
     - 8×H100
     - 55% faster
     - —
     - Matches original reward (vs 133 s/step baseline); PPO/GRPO match on 8×A800 (batch 1024).
   * - GLM-4.7-Flash (Sync)
     - 8×A100-80G
     - ~870 s
     - —
     - Reward 0.28 → 0.54, turns 1.7 → 2.0 over 100 steps (400 updates, 512×4).
   * - GLM-4.7-Flash (Sync)
     - 2×8 A100-80G
     - 278.6 s
     - 1159 tok/s
     - Reward 0.25–0.47, ``ratio_abs`` 0.052–0.062 (steps 2–19, 128×4).
   * - GLM-4.7-Flash (Async)
     - 2×8 A100-80G
     - **139.0 s**
     - **2024 tok/s**
     - **1.75× throughput**, staleness 1.3–1.4, reward 0.26–0.42, ``ratio_abs`` 0.053–0.063.

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png" style="width: 100%;"/><p><em>Qwen2.5-3B-Instruct reward &amp; step time in RLinf</em></p></div>
     <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_ppo_grpo_reward.png" style="width: 100%; max-width: 410px;"/><p><em>Qwen2.5-3B GRPO (orange) vs PPO (blue) on 8×A800</em></p></div>
     <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_reward.png" style="width: 100%;"/><p><em>GLM-4.7-Flash trajectory reward (8×A100, 400 updates)</em></p></div>
     <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_step_time.png" style="width: 100%;"/><p><em>GLM-4.7-Flash sync vs async step time (2×8 A100)</em></p></div>
   </div>

.. dropdown:: Additional training curves and async limit-tuning plots

   .. raw:: html

      <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1_orig_impl_time.png" style="width: 100%;"/><p><em>Original Search-R1 step time</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_length.png" style="width: 100%;"/><p><em>GLM-4.7-Flash total sequence length</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_time.png" style="width: 100%;"/><p><em>GLM-4.7-Flash time per rollout step</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_turn.png" style="width: 100%;"/><p><em>GLM-4.7-Flash turns per trajectory</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_reward_walltime.png" style="width: 100%;"/><p><em>Async reward vs wall-clock time</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_reward_step.png" style="width: 100%;"/><p><em>Async trajectory reward per step</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_ratio_abs.png" style="width: 100%;"/><p><em>Async rollout-training ratio_abs</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_cap_step_time.png" style="width: 100%;"/><p><em>In-flight cap 77 vs 160: step time</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_cap_staleness.png" style="width: 100%;"/><p><em>In-flight cap 77 vs 160: staleness</em></p></div>
        <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_async_buffer_step_time.png" style="width: 100%;"/><p><em>Qwen2.5-1.5B buffer cap 1×B vs 2×B: step time (11 s vs 3.94 s)</em></p></div>
        <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_async_buffer_size.png" style="width: 100%;"/><p><em>Qwen2.5-1.5B buffer cap 1×B vs 2×B: buffer size</em></p></div>
      </div>
