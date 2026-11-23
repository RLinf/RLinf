本地 Wiki 检索服务
==================

启动任意 Search-R1 训练或评测配方前，先部署本地 Wiki-2018 检索服务。Qdrant 与 FAISS 两种后端都使用 `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ 编码，并在 ``8000`` 端口暴露相同的 HTTP 接口（``POST /retrieve`` 与 ``POST /access``）。

前置条件
--------

先按 :doc:`RLinf 安装指南 </rst_source/start/installation>` 完成 agentic 基础环境安装。中国大陆用户在运行 ``hf download`` 前先执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

选择检索后端
------------

推荐使用预构建的 Qdrant 方案（`RLinf/Wiki-2018-Corpus <https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus>`__）：下载后即可使用，省去数小时建索引的时间，且 Qdrant 检索运行在 CPU 上，仅 ``e5-base-v2`` 编码器占用约 1 GB 显存。如需沿用 ASearcher 原版索引流程，可切换到 FAISS GPU 标签页。

.. tab-set::

   .. tab-item:: Qdrant（推荐）

      **1. 安装 Qdrant 客户端**

      在 RLinf 环境中安装 ``qdrant-client``：

      .. code-block:: bash

         uv pip install qdrant-client==1.16.2

      **2. 下载预构建语料与检索模型**

      下载 `RLinf/Wiki-2018-Corpus <https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus>`__\ （包含 ``wiki_corpus.jsonl``、``wiki_webpages.jsonl`` 及预构建的 ``qdrant/`` 索引）和 `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__：

      .. code-block:: bash

         hf download --repo-type dataset RLinf/Wiki-2018-Corpus --local-dir ./Wiki-2018-Corpus
         hf download intfloat/e5-base-v2 --local-dir ./e5-base-v2

      **3. 启动 Qdrant 服务**

      在 ``tmux`` 等常驻会话中启动自带的 Qdrant 二进制程序：

      .. code-block:: bash

         cd ./Wiki-2018-Corpus/qdrant
         ./qdrant

      **4. 启动检索服务**

      修改 ``examples/agent/tools/search_local_server_qdrant/launch_local_server.sh`` 中的以下变量：

      - ``WIKI2018_DIR``：``./Wiki-2018-Corpus`` 的路径。
      - ``retriever_path``：``./e5-base-v2`` 的路径。
      - ``qdrant_url``：Qdrant 服务地址，例如 ``http://localhost:6333``\ （或 ``http://<host_ip>:6333``）。
      - ``qdrant_collection_name``：``wiki_collection_m32_cef512``。
      - ``qdrant_search_param``：``'{"hnsw_ef":256}'``。

      运行启动脚本，等待终端输出监听地址：

      .. code-block:: bash

         bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

      .. dropdown:: 从零构建自定义 Qdrant 索引

         如需用自定义 HNSW 参数重建索引，先从 `Qdrant Releases <https://github.com/qdrant/qdrant/releases>`__ 下载二进制文件并在独立目录中启动，更新 ``examples/agent/tools/search_local_server_qdrant/build_index.sh`` 中的 ``WIKI2018_DIR``、``retriever_path`` 与 ``qdrant_path``，再运行：

         .. code-block:: bash

            bash examples/agent/tools/search_local_server_qdrant/build_index.sh

         HNSW 参数调优见 `Qdrant 优化文档 <https://qdrant.tech/documentation/guides/optimize/>`__。

   .. tab-item:: FAISS GPU（ASearcher）

      **1. 创建 retriever conda 环境**

      新建独立的 conda 环境并安装 GPU 版 FAISS：

      .. code-block:: bash

         conda create -n retriever python=3.10 -y
         conda activate retriever
         conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
         conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
         pip install transformers datasets pyserini uvicorn fastapi

      **2. 下载语料与检索模型**

      下载 ASearcher Wiki-2018 语料（约 50–60 GB）与 ``intfloat/e5-base-v2``：

      .. code-block:: bash

         conda activate retriever
         python examples/agent/searchr1/download.py --save_path ./wiki2018
         hf download intfloat/e5-base-v2 --local-dir ./e5-base-v2

      **3. 构建 FAISS 索引**

      在 ``examples/agent/tools/search_local_server_faiss/build_index.sh`` 中填好 ``WIKI2018_WORK_DIR`` 与 ``retriever_model``，运行建索引脚本：

      .. code-block:: bash

         bash examples/agent/tools/search_local_server_faiss/build_index.sh

      **4. 启动检索服务**

      在 ``examples/agent/tools/search_local_server_faiss/launch_local_server.sh`` 中填好 ``WIKI2018_WORK_DIR`` 与 ``retriever_path``，启动服务并等待终端打印主机 IP 与端口：

      .. code-block:: bash

         bash examples/agent/tools/search_local_server_faiss/launch_local_server.sh

将训练连接到检索服务
--------------------

尽量将检索服务与训练部署在同一台机器上以减少网络延迟。若部署在独立机器上，用 ``hostname -I`` 获取 IP，并在 Search-R1 YAML 配置中设置 ``tools.search.server_addr``\ （默认 ``localhost:8000``）：

.. code-block:: yaml

   tools:
     search:
       server_addr: "<server_ip>:8000"

下一步
------

检索服务在 ``8000`` 端口就绪后，选择对应的 Search-R1 训练配方：

- :doc:`Qwen2.5-3B Search-R1 <qwen2.5>`：8×H100 上的 GRPO、PPO 与 MA-FSDP 配方。
- :doc:`GLM-4.7-Flash Search-R1 <glm4.7_flash>`：8×A100 或 2×8 A100 上的 MoE 模型训练。
- :doc:`Search-R1 异步训练 <async>`：2×8 A100 上 rollout 与训练重叠运行。

WideSeek-R1 离线训练共用同一套预构建 Qdrant Wiki-2018 服务，见 :doc:`WideSeek-R1 环境与工具配置 <../wideseek_r1/tools>`。

参考资料
--------

- `Search-R1 <https://github.com/PeterGriffinJin/Search-R1>`__\ （`retriever 文档 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__）
- `Search-R1 & veRL-SGLang 教程 <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__
- `ASearcher <https://github.com/inclusionAI/ASearcher>`__
