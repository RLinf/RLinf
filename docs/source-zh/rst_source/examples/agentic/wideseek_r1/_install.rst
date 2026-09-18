.. This file is a reusable include, not a standalone page.
   Include it from the training and evaluation recipes.

基础环境请参考 RLinf 的 :doc:`安装指南 <../../../start/installation>`。

我们推荐使用预构建的 Docker 镜像：

.. code-block:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.4-torch2.11.0-sglang0.5.12.post1-vllm0.23.0-megatron0.17.0-te2.17

如果你更倾向于本地环境，请安装 agentic 依赖栈：

.. code-block:: bash

   bash requirements/install.sh agentic

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
