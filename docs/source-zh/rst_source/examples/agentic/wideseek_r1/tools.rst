.. _wideseek-r1-tools:

环境与工具配置
==============

在 WideSeek-R1 训练或评测之前，先准备 RLinf 环境、搜索后端和评判模型。单机 8 卡过慢时，本页也说明如何做多节点启动。


概述
----

按安装、工具后端、评判模型、多节点的顺序完成本页。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 安装
      :text-align: center

      Docker 镜像或 ``bash requirements/install.sh agentic``

   .. grid-item-card:: 工具后端
      :text-align: center

      离线 Qdrant 检索或在线 Serper/Jina 搜索

   .. grid-item-card:: 评判模型
      :text-align: center

      外部 SGLang 服务或 RLinf 本地评判器

   .. grid-item-card:: 多节点
      :text-align: center

      将训练和评测扩展到单机 8 卡之外

.. contents::
   :depth: 2
   :local:

.. _wideseek-r1-install:

安装
----

基础环境请参考 RLinf 的 :doc:`安装指南 <../../../start/installation>`。

我们推荐使用预构建的 Docker 镜像：

.. code-block:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.4-torch2.11.0-sglang0.5.12.post1-vllm0.23.0-megatron0.17.0-te2.17

如果你更倾向于本地环境，请安装 agentic 依赖栈：

.. code-block:: bash

   bash requirements/install.sh agentic

.. note::
   若您在早于 ``sm90`` 的 GPU（如 ``A100``）上运行预构建镜像，则可能会出现架构不支持相关报错，此时需要先卸载 ``flash-attn-4``：

   .. code-block:: bash

      uv pip uninstall flash-attn-4

   若您是在本地环境用 ``bash requirements/install.sh agentic`` 安装的依赖，则上述问题会自动解决，无需手动卸载。

启动脚本和配置文件位于 ``examples/agent/wideseek_r1``。

.. list-table::
   :header-rows: 1

   * - 路径
     - 作用
   * - ``examples/agent/wideseek_r1/config``
     - 用于训练和评测的 YAML 配置文件。
   * - ``examples/agent/tools/search_local_server_qdrant``
     - 离线工具使用的搜索引擎实现。
   * - ``examples/agent/wideseek_r1/run_train.sh`` / ``examples/agent/wideseek_r1/run_eval.sh``
     - 训练和评测的主要入口脚本。

.. _wideseek-r1-tool-backends:

工具后端
--------

WideSeek-R1 提供两种搜索后端：

- ``online`` 模式，用于实时网页搜索和网页访问。
- ``offline`` 模式，用于基于本地 Qdrant 知识库的检索。

在标准工作流中，离线工具用于训练和标准 QA 评测，在线工具用于 WideSearch 评测。启动训练或评测前，请先选择一种后端启动相应的服务。

.. _wideseek-r1-online-tools:

在线模式
~~~~~~~~

在线模式使用 `Serper <https://serper.dev>`__ 进行网页搜索，并使用 `Jina AI <https://jina.ai>`__ 进行网页访问。

API 密钥
^^^^^^^^

在运行训练或评测之前，请先导出所需的 API 密钥：

.. code-block:: bash

   export SERPER_API_KEY=your_serper_api_key
   export JINA_API_KEY=your_jina_api_key

配置
^^^^

在 ``examples/agent/wideseek_r1/config`` 下的 YAML 配置中设置：

.. code-block:: yaml

   tools:
     online: True
     use_jina: True
     enable_cache: True
     cache_file: "./webpage_cache.json"

.. _wideseek-r1-offline-tools:

离线模式
~~~~~~~~

离线模式使用本地 Qdrant 检索服务，并配合本地语料库与网页存储。

前置条件
^^^^^^^^

完成 :doc:`安装指南 <../../../start/installation>` 中的基础环境配置后，安装 Qdrant 客户端：

.. code-block:: bash

   uv pip install qdrant-client==1.16.2

下载语料库与检索器
^^^^^^^^^^^^^^^^^^

准备以下资源：

- `Wiki-2018-Corpus <https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus>`__
- `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__

语料包包含：

- ``wiki_corpus.jsonl``，用于检索片段。
- ``wiki_webpages.jsonl``，用于网页内容查找。
- ``qdrant/``，其中包含 Qdrant collection 文件。

启动检索服务
^^^^^^^^^^^^

1. 在语料目录中启动 Qdrant：

   .. code-block:: bash

      cd /PATH/TO/Wiki-2018-Corpus/qdrant
      ./qdrant

   该进程必须持续运行。建议在 ``tmux`` 中启动。

2. 获取 Qdrant 服务所在主机的 IP 地址：

   .. code-block:: bash

      hostname -I

3. 编辑 `examples/agent/tools/search_local_server_qdrant/launch_local_server.sh` 并更新以下变量：

   - ``WIKI2018_DIR``： ``/PATH/TO/Wiki-2018-Corpus``
   - ``retriever_path``： ``/PATH/TO/e5-model``
   - ``qdrant_url``：例如 ``http://<host_ip>:6333``
   - ``qdrant_collection_name``：设置为 ``wiki_collection_m32_cef512``.
   - ``qdrant_search_param``：设置为 ``{"hnsw_ef":256}``.

4. 启动检索服务：

   .. code-block:: bash

      bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

我们建议将该检索服务部署在与训练或评测相同的机器上，以避免不必要的网络延迟。如果部署在其他机器上，请相应配置 ``tools.search.server_addr``。默认地址为 ``localhost:8000``。

检索服务默认监听 ``8000`` 端口，并暴露以下接口：

- ``POST /retrieve`` 用于向量检索。
- ``POST /access`` 用于网页内容查找。

由于 Qdrant 检索运行在 CPU 上，服务启动后只有 E5 检索模型会占用 GPU 显存。

配置
^^^^

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
~~~~~~~~

你可以直接测试 WideSeek-R1 的工具 worker。

在线模式：

.. code-block:: bash

   python rlinf/agents/wideseek_r1/tools.py --is_online true

离线模式：

.. code-block:: bash

   python rlinf/agents/wideseek_r1/tools.py --is_online false

在线测试需要 ``SERPER_API_KEY`` 和 ``JINA_API_KEY``。

离线测试要求本地检索服务能够通过已配置的 ``server_addr`` 访问。

.. _wideseek-r1-judge:

评判模型
--------

在运行训练或评测之前，请先启动评判模型。WideSeek-R1 使用 LLM 评判器，相比仅依赖精确匹配打分，能够提供更可靠的反馈。我们提供外部评判模型服务和内置评判模型两种选择。

外部评判模型服务
~~~~~~~~~~~~~~~~

默认配置使用 `Qwen3-30B-A3B-Instruct-2507 <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>`__ 作为评判模型。

使用 SGLang 启动评判服务：

.. code-block:: bash

   python3 -m sglang.launch_server \
      --model-path /PATH/TO/Qwen3-30B-A3B-Instruct-2507 \
      --host 0.0.0.0 \
      --log-level info \
      --context-length 32768 \
      --dp 8

在主实验中，评判模型部署在 8 张 H100 GPU 上。你可以根据可用硬件和吞吐需求减少或增加 ``--dp`` 的值。

然后获取主机 IP 地址，例如：

.. code-block:: bash

   hostname -I

在 YAML 配置中通过以下字段使用该 IP 地址。默认端口为 ``30000``。

.. code-block:: yaml

   agentloop:
     llm_ip: LLM_JUDGE_IP
     llm_port: LLM_JUDGE_PORT

你可以通过以下命令测试：

.. code-block:: bash

   python rlinf/agents/wideseek_r1/utils/sglang_client.py --llm-ip LLM_JUDGE_IP

使用 RLinf 内置的 Rollout Engine 作为评判器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你也可以使用 RLinf 内置的 rollout engine 作为评判器，而不是使用外部服务器。这种方式在 RLinf 框架内部运行评判 LLM，对于本地开发和测试更加方便。

要使用内置的 rollout engine 作为评判器，请在 YAML 配置文件中设置：

.. code-block:: yaml

   agentloop:
     use_local_judge: true  # 在 RLinf 框架内启用本地评判器

然后配置 ``rollout_judge`` 部分，设置你所需的模型和参数：

.. code-block:: yaml

   rollout_judge:
     group_name: "RolloutJudgeGroup"
     gpu_memory_utilization: 0.5
     model:
       model_type: qwen3
       model_path: /PATH/TO/YOUR/JUDGE/MODEL  # 替换为实际路径
       precision: fp16
     rollout_backend: sglang
     tensor_parallel_size: 1
     pipeline_parallel_size: 1
     max_running_requests: 64

使用内置评判器的示例配置文件：

.. list-table::
   :header-rows: 1

   * - 配置
     - 用途
   * - ``examples/agent/wideseek_r1/config/train_qwen3_hybrid_local_judge.yaml``
     - 使用本地评判器训练。
   * - ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch_local_judge.yaml``
     - 使用本地评判器评测 WideSearch。

使用内置评判器时，你不需要启动单独的评判服务器。评判模型将由 RLinf 的 rollout engine 加载和管理。

.. _wideseek-r1-multinode:

多节点
------

多智能体生成的时间开销较大，使用单机 8 卡进行训练和评测会显著降低实验效率，因此 WideSeek-R1 支持多节点训练与评测。详细内容请参阅 :doc:`../../../guides/multi_node`。
