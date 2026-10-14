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

This page introduces the example and routes you to environment and tool setup,
training, and evaluation. Start with
:doc:`Environment and Tool Setup <tools>`, then open the page that matches the
step you are on.

Overview
--------

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

Choose a Page
-------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Environment and Tool Setup
      :link: tools
      :link-type: doc

      Install the stack, configure search backends, start a judge, and scale
      across nodes.

   .. grid-item-card:: Training
      :link: train
      :link-type: doc

      Prepare the model and data, then launch hybrid multi-agent RL.

   .. grid-item-card:: Evaluation
      :link: eval
      :link-type: doc

      Evaluate on WideSearch or standard QA with the matching tool backend.

.. toctree::
   :hidden:
   :maxdepth: 2

   Environment and Tool Setup <tools>
   Training <train>
   Evaluation <eval>
