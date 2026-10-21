Search-R1的强化学习训练
========================================

使用本配方复现 `Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning <https://arxiv.org/abs/2503.09516>`__ 的搜索增强推理实验，通过强化学习训练大语言模型调用搜索工具回答问题。

本页介绍所有模型共用的环境和本地 wiki 检索服务。请在下方选择模型，查看对应的训练配方与结果。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 640px; margin: 0 auto;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="qwen2.5.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png"
            style="width: 100%; height: 200px; object-fit: cover; object-position: left center; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="qwen2.5.html" style="text-decoration: underline; color: blue;">
          <b>Qwen2.5-3B Search-R1</b>
         </a><br>
         在 8×H100 上训练 Search-R1，相比原版实现加速 55%
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="glm4.7_flash.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_reward.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="glm4.7_flash.html" style="text-decoration: underline; color: blue;">
          <b>GLM-4.7-Flash Search-R1</b>
         </a><br>
         在 8×A100 上训练 MoE 模型（激活参数约 3B）
       </p>
     </div>
   </div>

环境
----------------------------------------

RLinf环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 环境配置参照 :doc:`RLinf Installation </rst_source/start/installation>`。

Local Wiki Server运行环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们使用search-R1示例中的local retrieve
server，通过conda安装faiss，详细文档见\ `SearchR1 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__\ ，安装过程参考\ `Search-R1 &
veRL-SGLang <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__\ ，同样使用conda来配置环境

.. code-block:: bash

   conda create -n retriever python=3.10 -y
   conda activate retriever

   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install transformers datasets pyserini huggingface_hub

   #  安装 GPU 版 faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   pip install uvicorn fastapi

Wiki配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们使用Asearcher提供的本地检索文件，下载文件大约 50~60GB

.. code-block:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/agent/searchr1/download.py --save_path $save_path

从huggingface上下载\ `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding模型，并生成index

.. code-block:: bash

   bash examples/agent/tools/search_local_server_faiss/build_index.sh

将之前下载好的wiki文件路径和index路径等写入examples/agent/searchr1/launch_local_server.sh

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

运行launch_local_server.sh启动Local Wiki Server，等待直至输出server ip等信息，代表server启动完成

(Optional) 使用Qdrant作为Wiki Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们也支持使用 qdrant 作为 wiki 服务器。如果你不打算使用 qdrant，可以直接跳到 训练 部分。

使用上一部分中提到的方式准备 Asearcher 提供的本地 wiki corpus 检索文件。

从 huggingface 上下载\ `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding 模型。

下载 `qdrant <https://github.com/qdrant/qdrant/releases>`__ 并按照以下步骤构建 qdrant collection。首先，创建一个文件夹并把下载好的 qdrant 二进制文件放入该文件夹中，方便后续存储 qdrant 程序及其构建的 collection 文件。

在 `examples/agent/tools/search_local_server_qdrant/build_index.sh` 和 `examples/agent/tools/search_local_server_qdrant/launch_local_server.sh` 中，根据之前下载的 wiki corpus, e5-base-v2 和 qdrant 路径更新 `WIKI2018_DIR`、 `retriever_path` 和 `qdrant_path` 的文件路径。

使用以下指令构建 qdrant wiki 服务器的 collection：

.. code-block:: bash

   # 创建 qdrant 存放的文件夹
   mkdir -p /path/to/qdrant
   # 拷贝二进制执行文件
   cp qdrant /path/to/qdrant

   # 启动 qdrant server
   /path/to/qdrant/qdrant &

   # 构建 qdrant collection
   bash examples/agent/tools/search_local_server_qdrant/build_index.sh

运行 launch_local_server.sh 启动 Local Qdrant Wiki Server ，等待直至输出 server ip 等信息，代表 server 启动完成

.. code-block:: bash

   # 启动 qdrant server
   /path/to/qdrant/qdrant &

   # 启动基于 qdrant 的 wiki server
   bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

Qdrant 默认使用 HNSW 图索引算法。关于 HNSW 图索引的优化,请参考 `Qdrant 文档 <https://qdrant.tech/documentation/guides/optimize/>`__。

References
----------

search-r1: https://github.com/PeterGriffinJin/Search-R1

Search-R1 &
veRL-SGLang:
https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md

Asearcher: https://github.com/inclusionAI/ASearcher

.. toctree::
   :hidden:
   :maxdepth: 1

   Qwen2.5-3B Search-R1 <qwen2.5>
   GLM-4.7-Flash Search-R1 <glm4.7_flash>
