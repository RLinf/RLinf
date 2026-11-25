.. _wideseek-r1-example:

WideSeek-R1
===========

Train and evaluate WideSeek-R1, a lead-agent and parallel-subagent framework for broad information seeking trained with multi-agent reinforcement learning (MARL) on a shared LLM. On WideSearch, WideSeek-R1-4B reaches ``40.0%`` item F1 — matching single-agent DeepSeek-R1-671B while scaling with parallel subagents (:doc:`publication </rst_source/resources/publications/wideseek_r1>`, `project page <https://wideseek-r1.github.io>`__, `arXiv <https://arxiv.org/abs/2602.04634>`__, `code <https://github.com/RLinf/RLinf/tree/main/examples/agent/wideseek_r1>`__).

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen3-4B · Qwen3 dense · WideSeek-R1-4B

   .. grid-item-card:: Algorithm
      :text-align: center

      Hybrid Multi-Agent RL (MARL)

   .. grid-item-card:: Tools
      :text-align: center

      Offline Qdrant · Online Serper / Jina

   .. grid-item-card:: Hardware
      :text-align: center

      8×H100 + 8×H100 judge (or single-node local judge)

| **What you'll do:** configure search tools & LLM judge → launch hybrid MARL training → evaluate on WideSearch or standard QA.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · :doc:`Multi-node guide </rst_source/guides/multi_node>` (optional scaling).

.. _wideseek-r1-setup:
.. _wideseek-r1-eval:

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images (including FA4 builds for Hopper+) are on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__; ``install.sh agentic`` auto-selects FA2 on pre-Hopper GPUs and FA4 on Hopper+.

.. tab-set::

   .. tab-item:: 1. Tools & Judge
      :sync: wideseek-tools

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Search Backend**

      Training and standard QA use **offline Qdrant** retrieval; WideSearch evaluation uses **online Serper + Jina** search.

      **Offline Qdrant:** reuse the prebuilt Wiki-2018 server from :doc:`Search-R1 </rst_source/examples/agentic/searchr1/index>` (install ``qdrant-client==1.16.2``, download ``RLinf/Wiki-2018-Corpus`` and ``intfloat/e5-base-v2``, run Qdrant, then launch ``examples/agent/tools/search_local_server_qdrant/launch_local_server.sh`` on port 8000).

      **Online Serper / Jina:** export API keys from `Serper <https://serper.dev>`__ and `Jina AI <https://jina.ai>`__ before launching:

      .. code-block:: bash

         export SERPER_API_KEY=your_serper_api_key
         export JINA_API_KEY=your_jina_api_key

      **3. LLM Judge**

      Serve `Qwen3-30B-A3B-Instruct-2507 <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>`__ via SGLang (default port ``30000``; adjust ``--dp`` for your GPUs), or skip this step and use RLinf's built-in local judge configs (``train_qwen3_hybrid_local_judge`` / ``eval_qwen3_widesearch_local_judge``):

      .. code-block:: bash

         hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir ./Qwen3-30B-A3B-Instruct-2507
         python3 -m sglang.launch_server --model-path ./Qwen3-30B-A3B-Instruct-2507 \
           --host 0.0.0.0 --log-level info --context-length 32768 --dp 8

      **4. Quick Check**

      Verify the search backend (``--is_online true`` or ``false``) and external judge server (replace ``127.0.0.1`` with your judge host IP from ``hostname -I``):

      .. code-block:: bash

         python rlinf/agents/wideseek_r1/tools.py --is_online false
         python rlinf/agents/wideseek_r1/utils/sglang_client.py --llm-ip 127.0.0.1

      .. dropdown:: Online/offline YAML keys, built-in local judge, and multi-node notes

         **Online YAML** (``examples/agent/wideseek_r1/config/``): set ``tools.online: True``, ``tools.use_jina: True``, ``tools.enable_cache: True``, and ``tools.cache_file: "./webpage_cache.json"``.

         **Offline YAML:** set ``tools.online: False``; if Qdrant runs remotely, set ``tools.search.server_addr: "HOST:8000"`` (default ``localhost:8000``).

         **Built-in local judge:** set ``agentloop.use_local_judge: true`` and configure ``rollout_judge`` (see ``examples/agent/wideseek_r1/config/train_qwen3_hybrid_local_judge.yaml`` and ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch_local_judge.yaml``). No separate judge server is needed.

         **Multi-node scaling:** multi-agent generation scales across multiple 8-GPU nodes; see :doc:`Multi-node guide </rst_source/guides/multi_node>`.

   .. tab-item:: 2. Train
      :sync: wideseek-train

      Ensure offline Qdrant retrieval and the LLM judge are running (or switch to ``train_qwen3_hybrid_local_judge`` on a single 8-GPU host).

      **1. Download**

      Download `Qwen3-4B <https://huggingface.co/Qwen/Qwen3-4B>`__ and `WideSeek-R1-train-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data>`__ (uses ``hybrid_20k.jsonl``; `WideSeek-R1-Corpus <https://huggingface.co/datasets/RLinf/WideSeek-R1-Corpus>`__ is also published for corpus inspection):

      .. code-block:: bash

         hf download Qwen/Qwen3-4B --local-dir ./Qwen3-4B
         hf download --repo-type dataset RLinf/WideSeek-R1-train-data --local-dir ./WideSeek-R1-train-data

      **2. Launch**

      Launch hybrid MARL training with ``examples/agent/wideseek_r1/config/train_qwen3_hybrid.yaml`` (replace ``127.0.0.1`` with your judge IP):

      .. code-block:: bash

         bash examples/agent/wideseek_r1/run_train.sh train_qwen3_hybrid \
           rollout.model.model_path=./Qwen3-4B \
           data.train_data_paths=./WideSeek-R1-train-data/hybrid_20k.jsonl \
           agentloop.llm_ip=127.0.0.1

      .. dropdown:: Single-source datasets and single-agent workflow

         **Mixed vs single-source data:** ``data.is_hybrid: True`` mixes WideSearch-style and QA data (``hybrid_20k.jsonl``). For single-source sets like width_20k or depth_20k, set ``data.is_hybrid: False`` and match ``data.is_markdown`` (True for width_20k, False for depth_20k).

         **Workflow switch:** set ``agentloop.workflow`` to ``mas`` (multi-agent) or ``sa`` (single-agent, comparable to `ASearcher <https://github.com/inclusionAI/ASearcher>`__).

   .. tab-item:: 3. Evaluate
      :sync: wideseek-eval

      .. note::

         `WideSeek-R1-4B <https://huggingface.co/RLinf/WideSeek-R1-4b>`__ stores weights in fp16 while its ``config.json`` declares ``bfloat16``. Shipped RLinf configs already set ``rollout.model.precision: fp16``; when serving outside RLinf, pass ``--dtype float16`` to avoid a ~3-point avg@4 drop on WideSearch (55.9 vs 59.0; see `issue #1662 <https://github.com/RLinf/RLinf/issues/1662>`__).

      **1. Download**

      Download the released checkpoint plus `WideSeek-R1-test-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-test-data>`__ (WideSearch) or `ASearcher-test-data <https://huggingface.co/datasets/inclusionAI/ASearcher-test-data>`__ (standard QA):

      .. code-block:: bash

         hf download RLinf/WideSeek-R1-4b --local-dir ./WideSeek-R1-4b
         hf download --repo-type dataset RLinf/WideSeek-R1-test-data --local-dir ./WideSeek-R1-test-data
         hf download --repo-type dataset inclusionAI/ASearcher-test-data --local-dir ./ASearcher-test-data

      **2. Launch Evaluation**

      Full WideSearch (200 examples, ``data.data_size=-1``) takes ~7 hours on 8 generation + 8 judge GPUs; set a smaller ``data.data_size`` for a quick check.

      .. code-block:: bash

         # WideSearch (online Serper/Jina tools, data.is_markdown=True)
         bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_widesearch \
           rollout.model.model_path=./WideSeek-R1-4b \
           data.val_data_paths=./WideSeek-R1-test-data \
           agentloop.llm_ip=127.0.0.1

         # Standard QA (offline Qdrant tools, data.is_markdown=False)
         bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_qa \
           rollout.model.model_path=./WideSeek-R1-4b \
           data.val_data_paths=./ASearcher-test-data \
           agentloop.llm_ip=127.0.0.1

      **3. Score WideSearch**

      Standard QA writes LLM-judge scores directly to ``metric.json``. For final WideSearch benchmark scores, score the saved ``responses/`` directory with `WideSeek-R1-Eval <https://github.com/RLinf/WideSeek-R1-Eval>`__.

      .. dropdown:: Output files, two-engine evaluation, and workflow switch

         **Outputs** (under ``${runner.output_dir}/${runner.experiment_name}``): ``metric.json`` (aggregate metrics and QA judge scores), ``allresult.json`` (multi-turn interaction logs), and ``responses/`` (final answers).

         **Two-engine evaluation** (``examples/agent/wideseek_r1/config/eval_qwen3_qa_2eng.yaml``): set ``rollout.use_fixed_worker: True`` and ``agentloop.fixed_role: worker`` (or planner) to assign separate models via ``rollout.model.model_path`` and ``rollout_fixed_worker.model.model_path``.

         **Workflow switch:** set ``agentloop.workflow`` to ``mas`` (multi-agent) or ``sa`` (single-agent).

Visualization and Results
-------------------------

Training and evaluation outputs live under ``${runner.output_dir}/${runner.experiment_name}``; run ``tensorboard --logdir ./logs`` or see :doc:`Training metrics </rst_source/reference/metrics>` for logging backends and metric definitions.

.. list-table:: **WideSeek-R1 headline benchmark**
   :header-rows: 1
   :widths: 30 22 20 28

   * - Model
     - Benchmark
     - Metric
     - Notes
   * - WideSeek-R1-4B (MARL)
     - WideSearch
     - **40.0% item F1**
     - Comparable to single-agent DeepSeek-R1-671B; scales with parallel subagents.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wideseek_r1/scaling.png
   :align: center
   :width: 85%

   WideSeek-R1-4B performance scaling on WideSearch with parallel subagents (full ablations in :doc:`WideSeek-R1 publication </rst_source/resources/publications/wideseek_r1>`).
