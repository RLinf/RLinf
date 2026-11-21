WideSeek-R1 Evaluation
======================

This page describes how to evaluate WideSeek-R1 in RLinf.

The provided scripts support two evaluation settings:

- WideSearch benchmark evaluation.
- Standard QA evaluation.

The reference configs use Qwen3-series dense models.

Overview
--------

Use this page to evaluate a released WideSeek-R1 checkpoint or your own Qwen3
dense model.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      WideSeek-R1-4B or Qwen3-series dense models

   .. grid-item-card:: Benchmarks
      :text-align: center

      WideSearch and standard QA

   .. grid-item-card:: Tools
      :text-align: center

      Online search for WideSearch; offline retrieval for QA

   .. grid-item-card:: Outputs
      :text-align: center

      ``metric.json`` and generated responses

.. contents::
   :depth: 2
   :local:

Prerequisites
-------------

Before evaluation, complete
:doc:`Environment and Tool Setup <tools>`:

- The RLinf environment is installed.
- The judge model is running.
- The appropriate tool backend is configured.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run
``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt
images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.
Complete :doc:`Environment and Tool Setup <tools>` so that the judge model and
matching tool backend (online search for WideSearch, offline retrieval for QA)
are configured before launching evaluation.

.. note::

   The released `WideSeek-R1-4B <https://huggingface.co/RLinf/WideSeek-R1-4b>`__
   checkpoint stores its weights in fp16 while its ``config.json`` declares
   ``bfloat16``. Serving engines that follow the config dtype by default
   (vLLM/SGLang/HF Transformers with ``dtype="auto"``) therefore upcast the
   weights, which costs about 3 points avg@4 on WideSearch (55.9 vs 59.0).
   When serving this checkpoint outside RLinf, pass an explicit
   ``--dtype float16``. Inside RLinf no extra step is needed: the shipped eval
   and train configs set ``rollout.model.precision: fp16``
   (see `issue #1662 <https://github.com/RLinf/RLinf/issues/1662>`__).

.. tab-set::

   .. tab-item:: WideSearch Benchmark

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the released checkpoint (or use your own Qwen3 dense model) and the formatted `WideSeek-R1-test-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-test-data>`__:

      .. code-block:: bash

         hf download RLinf/WideSeek-R1-4b --local-dir ./WideSeek-R1-4b
         hf download --repo-type dataset RLinf/WideSeek-R1-test-data --local-dir ./WideSeek-R1-test-data

      **3. Launch**

      Update ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch.yaml`` (keep ``data.is_markdown: True``; ``data_size: -1`` evaluates the full 200-example set, which takes about **7 hours** with 8 GPUs for generation and 8 GPUs for the judge; set ``agentloop.llm_ip`` to your judge host):

      .. code-block:: yaml

         rollout:
           model:
             model_type: qwen3
             model_path: /PATH/TO/MODEL

         data:
           is_markdown: True
           val_data_paths: /PATH/TO/EVAL/WIDESEARCH/DATASET
           data_size: -1

      Run evaluation (for multi-node launches, see :ref:`wideseek-r1-multinode`):

      .. code-block:: bash

         bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_widesearch

   .. tab-item:: Standard QA

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the released checkpoint and `ASearcher-test-data <https://huggingface.co/datasets/inclusionAI/ASearcher-test-data>`__ (single-hop and multi-hop QA):

      .. code-block:: bash

         hf download RLinf/WideSeek-R1-4b --local-dir ./WideSeek-R1-4b
         hf download --repo-type dataset inclusionAI/ASearcher-test-data --local-dir ./ASearcher-test-data

      **3. Launch**

      Update ``examples/agent/wideseek_r1/config/eval_qwen3_qa.yaml`` (keep ``data.is_markdown: False``; start with a smaller ``data_size`` for a quick sanity check; set ``agentloop.llm_ip`` to your judge host):

      .. code-block:: yaml

         rollout:
           model:
             model_type: qwen3
             model_path: /PATH/TO/MODEL

         data:
           is_markdown: False
           val_data_paths: /PATH/TO/EVAL/QA/DATASET
           data_size: -1

      Run evaluation (for multi-node launches, see :ref:`wideseek-r1-multinode`):

      .. code-block:: bash

         bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_qa

Output Files
------------

Evaluation outputs are written to:

.. code-block:: text

   ${runner.output_dir}/${runner.experiment_name}

Important files include:

- ``metric.json``: aggregate metrics such as output length and tool usage.
- ``allresult.json``: full multi-turn interaction logs.
- ``responses/``: final model answers for each example.

For standard QA evaluation, ``metric.json`` also includes the final LLM-judge
results.

For WideSearch evaluation, RLinf stores the generated responses so they can be
scored with the official WideSearch evaluation pipeline.

Additional WideSearch Scoring
-----------------------------

For final WideSearch benchmark scoring, use the dedicated evaluation repository:

- `WideSeek-R1-Eval <https://github.com/RLinf/WideSeek-R1-Eval>`__

Refer to the repository README for the complete procedure.

Two-Engine Evaluation
---------------------

WideSeek-R1 also supports evaluation with two separate model instances in the
multi-agent setting, so the planner and worker roles can use different models.

Use
`examples/agent/wideseek_r1/config/eval_qwen3_qa_2eng.yaml`.
The relevant fields are:

.. code-block:: yaml

   agentloop:
     fixed_role: worker  # planner or worker

   rollout:
     use_fixed_worker: True

``use_fixed_worker`` enables the second model instance. ``fixed_role`` selects
which role uses that second model.

You can then set different model paths under ``rollout.model.model_path`` and
``rollout_fixed_worker.model.model_path``.

Notes
-----

As in training, ``agentloop.workflow`` controls whether evaluation uses
single-agent or multi-agent execution:

- ``mas``: multi-agent evaluation.
- ``sa``: single-agent evaluation.

The single-agent mode is designed to be comparable to
`ASearcher <https://github.com/inclusionAI/ASearcher>`__.
