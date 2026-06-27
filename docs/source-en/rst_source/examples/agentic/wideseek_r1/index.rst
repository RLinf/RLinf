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
:doc:`WideSeek-R1 publication <../../../resources/publications/wideseek_r1>`, the
`project page <https://wideseek-r1.github.io>`__, the
`paper on arXiv <https://arxiv.org/abs/2602.04634>`__, and the
`example code in RLinf <https://github.com/RLinf/RLinf/tree/main/examples/agent/wideseek_r1>`__.

Overview
--------

Use this guide as the entry point for WideSeek-R1 setup, training, and evaluation.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen3-4B and Qwen3-series dense models

   .. grid-item-card:: Algorithm
      :text-align: center

      Multi-agent RL for broad information seeking

   .. grid-item-card:: Tools
      :text-align: center

      Online web search or offline Qdrant retrieval

   .. grid-item-card:: Hardware
      :text-align: center

      Single-node quick start or multi-node scaling

.. contents::
   :depth: 2
   :local:

Installation
------------

For the base environment, follow the RLinf
:doc:`installation guide <../../../start/installation>`.

We recommend the prebuilt Docker image:

.. code-block:: bash

   docker pull rlinf/rlinf:math-rlinf0.2-torch2.6.0-sglang0.4.6.post5-vllm0.8.5-megatron0.13.0-te2.1

If you prefer a local environment, install the agentic stack:

.. code-block:: bash

   bash requirements/install.sh agentic

Our startup scripts and configuration files are located in ``examples/agent/wideseek_r1``.

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

Tool Backends
-------------

WideSeek-R1 supports two tool backends:

.. list-table::
   :header-rows: 1

   * - Backend
     - Use it for
   * - :ref:`Offline tools <wideseek-r1-offline-tools>`
     - Training and standard QA evaluation.
   * - :ref:`Online tools <wideseek-r1-online-tools>`
     - WideSearch evaluation.

See :doc:`Tool Setup <tools>` for the full configuration workflow.

Run It
------

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

Using RLinf Built-in Rollout Engine as Judge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can use RLinf's built-in rollout engine as the judge instead of an external server. This approach runs the judge LLM within the RLinf framework, which can be more convenient for local development and testing.

To use the built-in rollout engine as judge, set the following configuration in your YAML file:

.. code-block:: yaml

   agentloop:
     use_local_judge: true  # Enable local judge within RLinf framework

Then configure the rollout_judge section with your desired model and settings:

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

Example configuration files using the built-in judge can be found in:

.. list-table::
   :header-rows: 1

   * - Config
     - Purpose
   * - ``examples/agent/wideseek_r1/config/train_qwen3_hybrid_local_judge.yaml``
     - Train with the local judge.
   * - ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch_local_judge.yaml``
     - Evaluate WideSearch with the local judge.

When using the built-in judge, you don't need to start a separate judge server. The judge model will be loaded and managed by RLinf's rollout engine.

Multi-node
~~~~~~~~~~~~

Since multi-agent generation incurs substantial time overhead, training and evaluation on a single machine with eight GPUs can significantly slow down experiments; therefore,
WideSeek-R1 supports multi-node training and evaluation. Please refer to :doc:`../../../guides/multi_node`.

Next Steps
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Page
     - Next step
   * - :doc:`Tool Setup <tools>`
     - Configure offline and online tool backends.
   * - :doc:`Training <train>`
     - Run the full training procedure.
   * - :doc:`Evaluation <eval>`
     - Run the full evaluation procedure.

.. toctree::
   :hidden:
   :maxdepth: 2

   tools
   train
   eval
