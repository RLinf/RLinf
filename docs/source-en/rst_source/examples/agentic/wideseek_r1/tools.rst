.. _wideseek-r1-tools:

Environment and Tool Setup
==========================

Prepare the RLinf environment, search backends, and judge model before
WideSeek-R1 training or evaluation. This page also covers multi-node launches
when a single eight-GPU machine is too slow.

Overview
--------

Work through installation, tool backends, the judge, and multi-node setup in
that order.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Installation
      :text-align: center

      Docker image or ``bash requirements/install.sh agentic``

   .. grid-item-card:: Tool Backends
      :text-align: center

      Offline Qdrant retrieval or online Serper/Jina search

   .. grid-item-card:: Judge Model
      :text-align: center

      External SGLang server or RLinf local judge

   .. grid-item-card:: Multi-Node
      :text-align: center

      Scale training and evaluation beyond one eight-GPU machine

.. contents::
   :depth: 2
   :local:

.. _wideseek-r1-install:

Installation
------------

For the base environment, follow the RLinf
:doc:`installation guide <../../../start/installation>`.

We recommend the prebuilt Docker image:

.. code-block:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.4-torch2.11.0-sglang0.5.12.post1-vllm0.23.0-megatron0.17.0-te2.17

If you prefer a local environment, install the agentic stack:

.. code-block:: bash

   bash requirements/install.sh agentic

.. note::
   If you run the prebuilt image on a GPU older than ``sm90`` (such as an
   ``A100``), you may hit an architecture-not-supported error. Uninstall
   ``flash-attn-4`` first:

   .. code-block:: bash

      uv pip uninstall flash-attn-4

   If you installed the stack locally with
   ``bash requirements/install.sh agentic``, this is handled automatically
   and you do not need to uninstall it.

Startup scripts and configuration files are in ``examples/agent/wideseek_r1``.

.. list-table::
   :header-rows: 1

   * - Path
     - Role
   * - ``examples/agent/wideseek_r1/config``
     - YAML configuration files for training and evaluation.
   * - ``examples/agent/tools/search_local_server_qdrant``
     - Search engine implementation used by offline tools.
   * - ``examples/agent/wideseek_r1/run_train.sh`` / ``examples/agent/wideseek_r1/run_eval.sh``
     - Main entry points for training and evaluation.

.. _wideseek-r1-tool-backends:

Tool Backends
-------------

WideSeek-R1 provides two search backends:

- ``online`` mode for live web search and webpage access.
- ``offline`` mode for retrieval against a local Qdrant-based knowledge base.

In the standard workflow, offline tools are used for training and standard QA
evaluation, while online tools are used for WideSearch evaluation. Choose one
backend and start the corresponding service before launching training or
evaluation.

.. _wideseek-r1-online-tools:

Online Mode
~~~~~~~~~~~

Online mode uses `Serper <https://serper.dev>`__ for web search and
`Jina AI <https://jina.ai>`__ for webpage access.

API Keys
^^^^^^^^

Export the required API keys before running training or evaluation:

.. code-block:: bash

   export SERPER_API_KEY=your_serper_api_key
   export JINA_API_KEY=your_jina_api_key

Configuration
^^^^^^^^^^^^^

In the YAML config under ``examples/agent/wideseek_r1/config``, set:

.. code-block:: yaml

   tools:
     online: True
     use_jina: True
     enable_cache: True
     cache_file: "./webpage_cache.json"

.. _wideseek-r1-offline-tools:

Offline Mode
~~~~~~~~~~~~

Offline mode uses a local Qdrant retrieval service together with a local corpus
and webpage store.

Prerequisites
^^^^^^^^^^^^^

After completing the base setup from the
:doc:`installation guide <../../../start/installation>`, install the Qdrant
client:

.. code-block:: bash

   uv pip install qdrant-client==1.16.2

Download the Corpus and Retriever
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prepare the following assets:

- `Wiki-2018-Corpus <https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus>`__
- `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__

The corpus package includes:

- ``wiki_corpus.jsonl`` for retrieval snippets.
- ``wiki_webpages.jsonl`` for webpage content lookup.
- ``qdrant/`` containing the Qdrant collection files.

Launch the Retrieval Service
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Start Qdrant in the corpus directory:

   .. code-block:: bash

      cd /PATH/TO/Wiki-2018-Corpus/qdrant
      ./qdrant

   This process must stay alive. Running it inside ``tmux`` is recommended.

2. Get the host IP address for the Qdrant service:

   .. code-block:: bash

      hostname -I

3. Edit
   `examples/agent/tools/search_local_server_qdrant/launch_local_server.sh`
   and update these variables:

   - ``WIKI2018_DIR``: ``/PATH/TO/Wiki-2018-Corpus``
   - ``retriever_path``: ``/PATH/TO/e5-model``
   - ``qdrant_url``: for example ``http://<host_ip>:6333``
   - ``qdrant_collection_name``: set it to ``wiki_collection_m32_cef512``.
   - ``qdrant_search_param``: set it to ``{"hnsw_ef":256}``.

4. Start the retrieval service:

   .. code-block:: bash

      bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

We recommend running this retrieval service on the same machine as training or
evaluation to avoid unnecessary network latency. If you run it elsewhere,
configure ``tools.search.server_addr`` accordingly. The default address is
``localhost:8000``.

The retrieval service listens on port ``8000`` by default and exposes:

- ``POST /retrieve`` for vector retrieval.
- ``POST /access`` for webpage content lookup.

Because Qdrant retrieval runs on CPU, only the E5 retriever model consumes GPU
memory after the service starts.

Configuration
^^^^^^^^^^^^^

In your YAML config, set:

.. code-block:: yaml

   tools:
     online: False

If the retrieval service is not running on the local machine, also set:

.. code-block:: yaml

   tools:
     search:
       server_addr: "HOST:8000"

.. _wideseek-r1-tool-test:

Test the Tools
~~~~~~~~~~~~~~

You can test the WideSeek-R1 tool worker directly.

Online mode:

.. code-block:: bash

   python rlinf/agents/wideseek_r1/tools.py --is_online true

Offline mode:

.. code-block:: bash

   python rlinf/agents/wideseek_r1/tools.py --is_online false

The online test requires ``SERPER_API_KEY`` and ``JINA_API_KEY``.

The offline test requires the local retrieval service to be reachable at the
configured ``server_addr``.

.. _wideseek-r1-judge:

Judge Model
-----------

Before running either training or evaluation, start the judge model.
WideSeek-R1 uses an LLM judge to provide more reliable feedback than exact-match
scoring alone. You can use an external judge server or the built-in judge.

External Judge Model Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default setup uses
`Qwen3-30B-A3B-Instruct-2507 <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>`__
as the judge model.

Start the judge server with SGLang:

.. code-block:: bash

   python3 -m sglang.launch_server \
      --model-path /PATH/TO/Qwen3-30B-A3B-Instruct-2507 \
      --host 0.0.0.0 \
      --log-level info \
      --context-length 32768 \
      --dp 8

In the main experiments, the judge model was served on 8 H100 GPUs. You can
reduce or increase ``--dp`` based on your available hardware and throughput
requirements.

Then obtain the host IP address, for example:

.. code-block:: bash

   hostname -I

Use that IP address in the YAML configuration through the following fields. The
default port is ``30000``.

.. code-block:: yaml

   agentloop:
     llm_ip: LLM_JUDGE_IP
     llm_port: LLM_JUDGE_PORT

You can test it with:

.. code-block:: bash

   python rlinf/agents/wideseek_r1/utils/sglang_client.py --llm-ip LLM_JUDGE_IP

Using RLinf Built-in Rollout Engine as Judge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can use RLinf's built-in rollout engine as the judge instead
of an external server. This approach runs the judge LLM within the RLinf
framework, which can be more convenient for local development and testing.

To use the built-in rollout engine as judge, set the following configuration in
your YAML file:

.. code-block:: yaml

   agentloop:
     use_local_judge: true  # Enable local judge within RLinf framework

Then configure the ``rollout_judge`` section with your desired model and
settings:

.. code-block:: yaml

   rollout_judge:
     group_name: "RolloutJudgeGroup"
     gpu_memory_utilization: 0.5
     model:
       model_type: qwen3
       model_path: /PATH/TO/YOUR/JUDGE/MODEL  # Replace with actual path
       precision: fp16
     rollout_backend: sglang
     tensor_parallel_size: 1
     pipeline_parallel_size: 1
     max_running_requests: 64

Example configuration files using the built-in judge:

.. list-table::
   :header-rows: 1

   * - Config
     - Purpose
   * - ``examples/agent/wideseek_r1/config/train_qwen3_hybrid_local_judge.yaml``
     - Train with the local judge.
   * - ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch_local_judge.yaml``
     - Evaluate WideSearch with the local judge.

When using the built-in judge, you do not need to start a separate judge server.
The judge model is loaded and managed by RLinf's rollout engine.

.. _wideseek-r1-multinode:

Multi-Node
----------

Multi-agent generation is expensive, so training and evaluation on a single
eight-GPU machine can be slow. WideSeek-R1 supports multi-node training and
evaluation. See :doc:`../../../guides/multi_node`.
