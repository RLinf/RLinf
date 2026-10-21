Reinforcement Learning Training of Search-R1
============================================

Multi-turn RL with tool calls extends the interaction boundary of large language models (LLMs) to the real world.
Reproduce the experiments from
`Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning <https://arxiv.org/abs/2503.09516>`__
in RLinf, training LLMs to answer questions by invoking search tools.

This page covers the environment and the local wiki retrieval server shared by
all models. Choose a model below for its training recipe and results.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 640px; margin: 0 auto;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="qwen2.5.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png"
            style="width: 100%; height: 200px; object-fit: cover; object-position: left center; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="qwen2.5.html" style="text-decoration: underline; color: blue;">
          <b>Qwen2.5-3B Search-R1</b>
         </a><br>
         Search-R1 on 8×H100, 55% faster than the original implementation
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="glm4.7_flash.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_reward.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="glm4.7_flash.html" style="text-decoration: underline; color: blue;">
          <b>GLM-4.7-Flash Search-R1</b>
         </a><br>
         Search-R1 for an MoE model (~3B active) on 8×A100
       </p>
     </div>
   </div>

Environment
-----------

RLinf Environment
~~~~~~~~~~~~~~~~~

RLinf environment setup follows :doc:`RLinf Installation </rst_source/start/installation>`.

Local Wiki Server Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the local retrieval server from the Search-R1 example.
Install faiss via conda; details in
`SearchR1 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__
and installation reference in
`Search-R1 & veRL-SGLang <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__
The environment is also configured via conda.

.. code-block:: bash

   conda create -n retriever python=3.10 -y
   conda activate retriever

   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install transformers datasets pyserini huggingface_hub

   # Install GPU version of faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   pip install uvicorn fastapi

Wiki Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~

We use the local retrieval files provided by Asearcher.
The downloaded files are approximately 50–60 GB in size.

.. code-block:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/agent/searchr1/download.py --save_path $save_path

Download the `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding model from HuggingFace,
and build the index

.. code-block:: bash

   bash examples/agent/tools/search_local_server_faiss/build_index.sh

Write the paths to the previously downloaded wiki files and the index into examples/agent/searchr1/launch_local_server.sh

.. code-block:: bash

   #!/bin/bash

   set -ex

   WIKI2018_WORK_DIR=$save_path

   index_file=$WIKI2018_WORK_DIR/e5.index/e5_Flat.index
   corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
   pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl
   retriever_name=e5
   retriever_path=path/to/intfloat/e5-base-v2

   python3  ./local_retrieval_server.py --index_path $index_file \
                                               --corpus_path $corpus_file \
                                               --pages_path $pages_file \
                                               --topk 3 \
                                               --retriever_name $retriever_name \
                                               --retriever_model $retriever_path \
                                               --faiss_gpu --port 8000

Run `launch_local_server.sh` to start the Local Wiki Server.
Wait until server IP information is printed — indicating successful startup.

(Optional) Using Qdrant as Local Wiki Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also support qdrant as the wiki server as well. If you don't want to use the qdrant, move on to the Training section.

Download the local retrieval wiki corpus files provided by ASearcher using the method mentioned in the previous section.

Download the `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding model from HuggingFace.

Download `qdrant <https://github.com/qdrant/qdrant/releases>`__ binary file and build a qdrant collection with follwing steps. First, Create a new folder and put the qdrant binary into this folder, to facilitate the subsequent storage of qdrant binary and constructed collection files.

In `examples/agent/tools/search_local_server_qdrant/build_index.sh` and `examples/agent/tools/search_local_server_qdrant/launch_local_server.sh`, update the file paths for `WIKI2018_DIR`, `retriever_path`, and `qdrant_path` according to your downloaded wiki corpus, e5-base-v2, and qdrant paths.

Use the following commands to build the qdrant wiki server collection:

.. code-block:: bash

   # Create folder for qdrant
   mkdir -p /path/to/qdrant
   # Copy the binary
   cp qdrant /path/to/qdrant

   # Launch qdrant server
   /path/to/qdrant/qdrant &

   # Build qdrant collection
   bash examples/agent/tools/search_local_server_qdrant/build_index.sh

Run launch_local_server.sh to start the Local Qdrant Wiki Server. Wait until server IP information is printed — indicating successful startup.

.. code-block:: bash

   # Launch qdrant server
   /path/to/qdrant/qdrant &

   # Launch qdrant-based wiki server
   bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

Qdrant uses the HNSW graph index algorithm by default. For details on optimizing the HNSW graph index, please refer to the `Qdrant documentation <https://qdrant.tech/documentation/guides/optimize/>`__.


References
----------

search-r1: https://github.com/PeterGriffinJin/Search-R1

Search-R1 & veRL-SGLang:
https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md

Asearcher: https://github.com/inclusionAI/ASearcher

.. toctree::
   :hidden:
   :maxdepth: 1

   Qwen2.5-3B Search-R1 <qwen2.5>
   GLM-4.7-Flash Search-R1 <glm4.7_flash>
