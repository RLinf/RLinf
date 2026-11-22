Local Wiki Retriever
====================

Start a local Wiki-2018 retrieval server before launching any Search-R1 training or evaluation recipe. Both supported backends expose the same HTTP API on port ``8000`` (``POST /retrieve`` and ``POST /access``) using the `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ encoder.

Prerequisites
-------------

Install the base RLinf agentic environment from :doc:`RLinf Installation </rst_source/start/installation>`. In mainland China, run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

Choose a Retrieval Backend
--------------------------

Qdrant with the prebuilt `RLinf/Wiki-2018-Corpus <https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus>`__ is recommended: you skip hours of index building, and Qdrant runs on CPU so only the ``e5-base-v2`` encoder uses GPU memory (about 1 GB). Use FAISS GPU if you want to build and serve the index with ASearcher's original pipeline.

.. tab-set::

   .. tab-item:: Qdrant (Recommended)

      **1. Install the Qdrant client**

      Install ``qdrant-client`` in your RLinf environment:

      .. code-block:: bash

         uv pip install qdrant-client==1.16.2

      **2. Download the prebuilt corpus and retriever**

      Download `RLinf/Wiki-2018-Corpus <https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus>`__ (includes ``wiki_corpus.jsonl``, ``wiki_webpages.jsonl``, and the prebuilt ``qdrant/`` index) and `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__:

      .. code-block:: bash

         hf download --repo-type dataset RLinf/Wiki-2018-Corpus --local-dir ./Wiki-2018-Corpus
         hf download intfloat/e5-base-v2 --local-dir ./e5-base-v2

      **3. Start the Qdrant daemon**

      Run the bundled Qdrant binary inside ``tmux`` or another persistent session:

      .. code-block:: bash

         cd ./Wiki-2018-Corpus/qdrant
         ./qdrant

      **4. Launch the retrieval server**

      Update these variables in ``examples/agent/tools/search_local_server_qdrant/launch_local_server.sh``:

      - ``WIKI2018_DIR``: path to ``./Wiki-2018-Corpus``.
      - ``retriever_path``: path to ``./e5-base-v2``.
      - ``qdrant_url``: Qdrant address, such as ``http://localhost:6333`` (or ``http://<host_ip>:6333``).
      - ``qdrant_collection_name``: ``wiki_collection_m32_cef512``.
      - ``qdrant_search_param``: ``'{"hnsw_ef":256}'``.

      Start the server and wait until it prints its listening address:

      .. code-block:: bash

         bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

      .. dropdown:: Build a custom Qdrant collection from scratch

         To build a Qdrant collection with custom HNSW parameters instead of using the prebuilt ``qdrant/`` directory, download a `Qdrant release binary <https://github.com/qdrant/qdrant/releases>`__, start it in a dedicated folder, update ``WIKI2018_DIR``, ``retriever_path``, and ``qdrant_path`` in ``examples/agent/tools/search_local_server_qdrant/build_index.sh``, and run:

         .. code-block:: bash

            bash examples/agent/tools/search_local_server_qdrant/build_index.sh

         For HNSW tuning options, see the `Qdrant optimization guide <https://qdrant.tech/documentation/guides/optimize/>`__.

   .. tab-item:: FAISS GPU (ASearcher)

      **1. Create the retriever conda environment**

      Set up a dedicated conda environment with GPU FAISS:

      .. code-block:: bash

         conda create -n retriever python=3.10 -y
         conda activate retriever
         conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
         conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
         pip install transformers datasets pyserini uvicorn fastapi

      **2. Download the corpus and retriever**

      Download the ASearcher Wiki-2018 files (about 50–60 GB) and ``intfloat/e5-base-v2``:

      .. code-block:: bash

         conda activate retriever
         python examples/agent/searchr1/download.py --save_path ./wiki2018
         hf download intfloat/e5-base-v2 --local-dir ./e5-base-v2

      **3. Build the FAISS index**

      Set ``WIKI2018_WORK_DIR`` and ``retriever_model`` in ``examples/agent/tools/search_local_server_faiss/build_index.sh``, then build the index:

      .. code-block:: bash

         bash examples/agent/tools/search_local_server_faiss/build_index.sh

      **4. Launch the retrieval server**

      Set ``WIKI2018_WORK_DIR`` and ``retriever_path`` in ``examples/agent/tools/search_local_server_faiss/launch_local_server.sh``, then launch the server and wait until it prints its host IP and port:

      .. code-block:: bash

         bash examples/agent/tools/search_local_server_faiss/launch_local_server.sh

Connect Training to the Retrieval Server
----------------------------------------

Run the retrieval server on the training host when possible to avoid network latency. If it runs on another machine, get its IP with ``hostname -I`` and set ``tools.search.server_addr`` in your Search-R1 YAML config (defaults to ``localhost:8000``):

.. code-block:: yaml

   tools:
     search:
       server_addr: "<server_ip>:8000"

Next Steps
----------

Once the retrieval server is listening on port ``8000``, pick a Search-R1 training recipe:

- :doc:`Qwen2.5-3B Search-R1 <qwen2.5>`: GRPO, PPO, and MA-FSDP recipes on 8×H100.
- :doc:`GLM-4.7-Flash Search-R1 <glm4.7_flash>`: MoE training on 8×A100 or 2×8 A100.
- :doc:`Asynchronous Search-R1 <async>`: overlapped rollout and training on 2×8 A100.

WideSeek-R1 offline training shares the same prebuilt Qdrant Wiki-2018 setup; see :doc:`WideSeek-R1 Environment and Tool Setup <../wideseek_r1/tools>`.

References
----------

- `Search-R1 <https://github.com/PeterGriffinJin/Search-R1>`__ (`retriever docs <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__)
- `Search-R1 & veRL-SGLang Tutorial <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__
- `ASearcher <https://github.com/inclusionAI/ASearcher>`__
