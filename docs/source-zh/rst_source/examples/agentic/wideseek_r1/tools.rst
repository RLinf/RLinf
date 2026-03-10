.. _wideseek-r1-tools:

工具配置
========

WideSeek-R1 提供两种搜索后端：

- ``online`` 模式，用于实时网页搜索和网页访问。
- ``offline`` 模式，用于基于本地 Qdrant 知识库的检索。

在标准工作流中，离线工具通常用于训练和标准 QA 评测，而在线工具用于 WideSearch 评测。

.. contents::
   :depth: 2
   :local:

.. _wideseek-r1-online-tools:

在线模式
--------

在线模式使用 `Serper <https://serper.dev>`__ 进行网页搜索，并使用 `Jina AI <https://jina.ai>`__ 进行网页访问。

API 密钥
~~~~~~~~

在运行训练或评测之前，请先导出所需的 API 密钥：

.. code-block:: bash

   export SERPER_API_KEY=your_serper_api_key
   export JINA_API_KEY=your_jina_api_key

配置
~~~~

在 ``examples/wideseek_r1/config`` 下的 YAML 配置中设置：

.. code-block:: yaml

   tools:
     online: True
     use_jina: True
     enable_cache: True
     cache_file: "./webpage_cache.json"

.. _wideseek-r1-offline-tools:

离线模式
--------

离线模式使用本地 Qdrant 检索服务，并配合本地语料库与网页存储。

前置条件
~~~~~~~~

完成 :doc:`安装指南 <../../../start/installation>` 中的基础环境配置后，安装 Qdrant 客户端：

.. code-block:: bash

   uv pip install qdrant-client==1.16.2

下载语料库与检索器
~~~~~~~~~~~~~~~~~~

准备以下资源：

- `Wiki-2018-Corpus <https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus>`__
- `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__

语料包包含：

- ``wiki_corpus.jsonl``，用于检索片段。
- ``wiki_webpages.jsonl``，用于网页内容查找。
- ``qdrant/``，其中包含 Qdrant collection 文件。

启动检索服务
~~~~~~~~~~~~

1. 在语料目录中启动 Qdrant：

   .. code-block:: bash

      cd /PATH/TO/Wiki-2018-Corpus/qdrant
      ./qdrant

   该进程必须持续运行。建议在 ``tmux`` 中启动。

2. 获取 Qdrant 服务所在主机的 IP 地址：

   .. code-block:: bash

      hostname -I

3. 编辑 `examples/wideseek_r1/search_engine/launch_qdrant.sh` 并更新以下变量：

   - ``pages_file``：``/PATH/TO/Wiki-2018-Corpus/wiki_webpages.jsonl``
   - ``retriever_path``：``/PATH/TO/e5-model``
   - ``qdrant_url``：例如 ``http://<host_ip>:6333``

4. 启动检索服务：

   .. code-block:: bash

      bash examples/wideseek_r1/search_engine/launch_qdrant.sh

我们建议将该检索服务部署在与训练或评测相同的机器上，以避免不必要的网络延迟。如果部署在其他机器上，请相应配置 ``tools.search.server_addr``。默认地址为 ``localhost:8000``。

检索服务默认监听 ``8000`` 端口，并暴露以下接口：

- ``POST /retrieve`` 用于向量检索。
- ``POST /access`` 用于网页内容查找。

由于 Qdrant 检索运行在 CPU 上，服务启动后只有 E5 检索模型会占用 GPU 显存。

配置
~~~~

在 YAML 配置中设置：

.. code-block:: yaml

   tools:
     online: False

如果检索服务不运行在本机上，还需要设置：

.. code-block:: yaml

   tools:
     search:
       server_addr: "HOST:8000"

.. _wideseek-r1-tool-test:

测试工具
--------

你可以直接测试 WideSeek-R1 的工具 worker。

在线模式：

.. code-block:: bash

   python rlinf/agents/wideseek_r1/tools.py --is_online true

离线模式：

.. code-block:: bash

   python rlinf/agents/wideseek_r1/tools.py --is_online false

在线测试需要 ``SERPER_API_KEY`` 和 ``JINA_API_KEY``。

离线测试要求本地检索服务能够通过已配置的 ``server_addr`` 访问。
