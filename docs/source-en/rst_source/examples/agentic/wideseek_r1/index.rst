.. _wideseek-r1-example:

WideSeek-R1
===========

Overview
--------

WideSeek-R1 is a lead-agent and subagent framework trained with multi-agent
reinforcement learning (MARL) for broad information-seeking tasks. It combines
scalable orchestration with parallel execution by using a shared LLM, isolated
contexts, and specialized tools. On the WideSearch benchmark, WideSeek-R1-4B
achieves an item F1 score of 40.0%, which is comparable to single-agent
DeepSeek-R1-671B, while continuing to improve as the number of parallel
subagents increases.

For the full method, experiments, and results, see the
:doc:`WideSeek-R1 publication <../../../publications/wideseek_r1>`.

Installation
------------

For general environment setup, see the RLinf
:doc:`installation guide <../../../start/installation>`.

We recommend using the prebuilt Docker image:

.. code-block:: bash

   docker pull rlinf/rlinf:math-rlinf0.1-torch2.6.0-sglang0.4.6.post5-vllm0.8.5-megatron0.13.0-te2.1

If you prefer a local environment, install the agentic stack:

.. code-block:: bash

   bash requirements/install.sh agentic

Tool Setup
----------

WideSeek-R1 supports two tool backends:

- :ref:`wideseek-r1-offline-tools` for standard training and QA-style evaluation.
- :ref:`wideseek-r1-online-tools` for WideSearch evaluation.

For complete setup instructions, see :doc:`Tool Setup <tools>`.

Run the Script
--------------

Judge Model
~~~~~~~~~~~

Our default setup uses ``Qwen3-30B-A3B-Instruct-2507`` as the LLM judge.
Download the model from
`Hugging Face <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>`__,
then update the model path in ``examples/wideseek_r1/judge_llm.sh`` if needed.

Start the judge server with:

.. code-block:: bash

   bash examples/wideseek_r1/judge_llm.sh

Training
~~~~~~~~

Before training, review the configuration file
``examples/wideseek_r1/config/train_mas_qwen3-4b_hybrid.yaml`` and update the
model, data, output, and tool-related paths for your environment.

Then launch the main experiment:

.. code-block:: bash

   bash examples/wideseek_r1/run_train.sh train_mas_qwen3-4b_hybrid

.. toctree::
   :hidden:
   :maxdepth: 2

   tools
