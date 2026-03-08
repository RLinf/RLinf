.. _wideseek-r1-example:

WideSeek-R1
===========

WideSeek-R1 is a lead-agent and subagent framework trained with multi-agent
reinforcement learning (MARL) for broad information-seeking tasks. It combines
scalable orchestration and parallel execution through a shared LLM, isolated
agent contexts, and specialized tools.

On the WideSearch benchmark, WideSeek-R1-4B reaches an item F1 score of
``40.0%``. This is comparable to single-agent DeepSeek-R1-671B while continuing
to improve as the number of parallel subagents increases.

For the full method and results, see the
:doc:`WideSeek-R1 publication <../../../publications/wideseek_r1>`, the
`project page <https://wideseek-r1.github.io>`__, the
`paper on arXiv <https://arxiv.org/abs/2602.04634>`__, and the
`example code in RLinf <https://github.com/RLinf/RLinf/tree/main/examples/wideseek_r1>`__.

.. contents::
   :depth: 2
   :local:

Installation
------------

For the base environment, follow the RLinf
:doc:`installation guide <../../../start/installation>`.

We recommend the prebuilt Docker image:

.. code-block:: bash

   docker pull rlinf/rlinf:math-rlinf0.1-torch2.6.0-sglang0.4.6.post5-vllm0.8.5-megatron0.13.0-te2.1

If you prefer a local environment, install the agentic stack:

.. code-block:: bash

   bash requirements/install.sh agentic

Tool Backends
-------------

WideSeek-R1 supports two tool backends:

- :ref:`wideseek-r1-offline-tools` for training and standard QA evaluation.
- :ref:`wideseek-r1-online-tools` for WideSearch evaluation.

See :doc:`Tool Setup <tools>` for the full configuration workflow.

Quick Start
-----------

Before running either training or evaluation, start the judge model server.
WideSeek-R1 uses an LLM judge to provide more reliable feedback than exact-match
scoring alone.

Judge Model
~~~~~~~~~~~

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

Use that IP address in the YAML configuration through the following fields. The default port is ``30000``.

.. code-block:: yaml

   agentloop:
     llm_ip: LLM_JUDGE_IP
     llm_port: LLM_JUDGE_PORT

you can test it by:

.. code-block:: bash

   python rlinf/agents/wideseek_r1/utils/sglang_client.py --llm-ip LLM_JUDGE_IP

Multi-node 
~~~~~~~~~~~~

Since multi-agent generation incurs substantial time overhead, training and evaluation on a single machine with eight GPUs can significantly slow down experiments; therefore, 
WideSeek-R1 supports multi-node training and evaluation. Please refer to the documentation :doc:`../../../start/distribute`.

Next Steps
~~~~~~~~~~

- For tool configuration, see :doc:`tools`.
- For the full training procedure, see :doc:`train`.
- For the full evaluation procedure, see :doc:`eval`.

.. toctree::
   :hidden:
   :maxdepth: 2

   tools
   train
   eval
