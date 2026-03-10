.. _wideseek-r1-example:

WideSeek-R1
===========

WideSeek-R1 是一个面向广域信息检索任务的主智能体与子智能体框架，通过多智能体强化学习（MARL）进行训练。它通过共享 LLM、隔离的智能体上下文以及专用工具，实现了可扩展的编排与并行执行。

在 WideSearch 基准上，WideSeek-R1-4B 的 item F1 分数达到 ``40.0%``。这一结果可与单智能体 DeepSeek-R1-671B 相当，并且随着并行子智能体数量的增加仍在持续提升。

有关完整方法和实验结果，请参见 :doc:`WideSeek-R1 论文页面 <../../../publications/wideseek_r1>`、`项目主页 <https://wideseek-r1.github.io>`__、`arXiv 论文 <https://arxiv.org/abs/2602.04634>`__，以及 `RLinf 中的示例代码 <https://github.com/RLinf/RLinf/tree/main/examples/wideseek_r1>`__。

.. contents::
   :depth: 2
   :local:

安装
----

基础环境请参考 RLinf 的 :doc:`安装指南 <../../../start/installation>`。

我们推荐使用预构建的 Docker 镜像：

.. code-block:: bash

   docker pull rlinf/rlinf:math-rlinf0.1-torch2.6.0-sglang0.4.6.post5-vllm0.8.5-megatron0.13.0-te2.1

如果你更倾向于本地环境，请安装 agentic 依赖栈：

.. code-block:: bash

   bash requirements/install.sh agentic

启动脚本和配置文件位于 `examples/wideseek_r1` 目录下。

- ``examples/wideseek_r1/config`` 包含用于训练和评估的 YAML 配置文件。
- ``examples/wideseek_r1/search_engine`` 提供离线工具使用的搜索引擎实现。
- ``examples/wideseek_r1/run_train.sh`` 和 ``examples/wideseek_r1/run_eval.sh`` 分别是训练和评估的主要入口脚本。


工具后端
--------

WideSeek-R1 支持两种工具后端：

- :ref:`wideseek-r1-offline-tools`，用于训练和标准 QA 评测。
- :ref:`wideseek-r1-online-tools`，用于 WideSearch 评测。

完整配置流程请参见 :doc:`工具配置 <tools>`。

快速开始
--------

在运行训练或评测之前，请先启动评判模型服务。WideSeek-R1 使用 LLM 评判器，相比仅依赖精确匹配打分，能够提供更可靠的反馈。

评判模型
~~~~~~~~

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

你可以通过一下命令测试:

.. code-block:: bash

   python rlinf/agents/wideseek_r1/utils/sglang_client.py --llm-ip LLM_JUDGE_IP
   
多节点
~~~~~~~~~~~~

由于多智能体生成的时间开销较大，使用单机 8 卡进行训练和评估会显著降低实验效率，因此 WideSeek-R1 支持多节点训练与评估。详细内容请参阅 :doc:`../../../start/distribute`.

后续步骤
~~~~~~~~

- 工具配置请参见 :doc:`tools`。
- 完整训练流程请参见 :doc:`train`。
- 完整评测流程请参见 :doc:`eval`。

.. toctree::
   :hidden:
   :maxdepth: 2

   tools
   train
   eval
