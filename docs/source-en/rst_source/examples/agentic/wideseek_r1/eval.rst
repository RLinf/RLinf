Evaluation
==========

This page describes how to evaluate WideSeek-R1 in RLinf.

The provided scripts support two evaluation settings:

- WideSearch benchmark evaluation.
- Standard QA evaluation.

The reference configs use Qwen3-series dense models.

.. contents::
   :depth: 2
   :local:

Prerequisites
-------------

Before evaluation, make sure the following components are ready:

- The RLinf environment is installed. See :doc:`../../../start/installation`.
- The judge model server is running. See :doc:`index`.
- The appropriate tool backend is configured. See :doc:`tools`.

Download the Model
------------------

The released checkpoint is available at:

- `WideSeek-R1-4B <https://huggingface.co/RLinf/WideSeek-R1-4b>`__

You may also evaluate your own Qwen3-series dense model.

After downloading the model, set the local path in the evaluation config:

.. code-block:: yaml

   rollout:
     model:
       model_type: qwen3
       model_path: /PATH/TO/MODEL

Evaluation Datasets
-------------------

WideSeek-R1 currently supports two dataset types for evaluation.

WideSearch Benchmark
~~~~~~~~~~~~~~~~~~~~

Use the formatted WideSearch evaluation set from Hugging Face:

- `WideSeek-R1-test-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-test-data>`__

Compared with the original raw benchmark, this version is converted into the
format expected by RLinf and includes several data fixes.

Update
`examples/agent/wideseek_r1/config/eval_qwen3_widesearch.yaml`
as follows:

.. code-block:: yaml

   data:
     is_markdown: True
     val_data_paths: /PATH/TO/EVAL/WIDESEARCH/DATASET
     data_size: -1

Key fields:

- ``is_markdown`` should remain ``True`` for the WideSearch dataset.
- ``val_data_paths`` points to the evaluation dataset.
- ``data_size: -1`` means to evaluate on the full dataset.

For a quick sanity check, start with a smaller ``data_size``.

In the reference setup, full evaluation on 200 WideSearch examples took about
7 hours with 8 GPUs for generation and 8 GPUs for the judge model.

Standard QA Evaluation
~~~~~~~~~~~~~~~~~~~~~~

For standard QA evaluation, use the dataset released by ASearcher:

- `ASearcher-test-data <https://huggingface.co/datasets/inclusionAI/ASearcher-test-data>`__

This dataset includes both single-hop tasks, such as Natural Questions, and
multi-hop tasks, such as HotpotQA.

Update
`examples/agent/wideseek_r1/config/eval_qwen3_qa.yaml`
as follows:

.. code-block:: yaml

   data:
     is_markdown: False
     val_data_paths: /PATH/TO/EVAL/QA/DATASET
     data_size: -1

Here ``is_markdown`` must be ``False``.

Run Evaluation
--------------

Before launching evaluation, verify all of the following:

- ``rollout.model.model_path`` points to the model you want to evaluate.
- ``data.val_data_paths`` points to the correct dataset.
- ``agentloop.llm_ip`` is set correctly.
- The required tools are configured. See :doc:`tools`.

Then run one of the following commands:

.. code-block:: bash

   bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_widesearch
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
