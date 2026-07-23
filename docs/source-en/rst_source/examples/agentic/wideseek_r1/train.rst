WideSeek-R1 Training
====================

This page describes how to reproduce WideSeek-R1 training in RLinf.

The reference configuration uses ``Qwen3-4B``, but the current pipeline is also
compatible with other dense models in the Qwen3 family.

Overview
--------

Use this page to prepare the model, data, and config for WideSeek-R1 training.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen3-4B

   .. grid-item-card:: Algorithm
      :text-align: center

      Hybrid multi-agent RL

   .. grid-item-card:: Data
      :text-align: center

      ``hybrid_20k.jsonl`` from WideSeek-R1-train-data

   .. grid-item-card:: Tools
      :text-align: center

      Offline retrieval and judge model server

.. contents::
   :depth: 2
   :local:

Prerequisites
-------------

Before launching training, make sure the following components are ready:

- The RLinf environment is installed. See :doc:`../../../start/installation`.
- The judge model server is running. See :doc:`index`.
- The offline retrieval tools are configured. See :doc:`tools`.

Download the Base Model
-----------------------

The main experiments use
`Qwen3-4B <https://huggingface.co/Qwen/Qwen3-4B>`__.

After downloading the model, update
`examples/agent/wideseek_r1/config/train_qwen3_hybrid.yaml`
with the local model path:

.. code-block:: yaml

   rollout:
     model:
       model_type: qwen3
       model_path: /PATH/TO/MODEL

Download the Training Data
--------------------------

WideSeek-R1 training uses a 20k hybrid dataset that combines broad
information-seeking data with standard QA data. The dataset is available on
Hugging Face:

- `WideSeek-R1-train-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data>`__

The main experiments use ``hybrid_20k.jsonl`` from
``WideSeek-R1-train-data``.

A separate `WideSeek-R1-Corpus <https://huggingface.co/datasets/RLinf/WideSeek-R1-Corpus>`__
artifact is also available for users who want to inspect or reuse the public
corpus resources. It is not the ``hybrid_20k.jsonl`` training file used by the
main experiments.

After downloading the data, update
`examples/agent/wideseek_r1/config/train_qwen3_hybrid.yaml`
with the dataset path:

.. code-block:: yaml

   data:
     train_data_paths: /PATH/TO/TRAIN/DATASET/hybrid_20k.jsonl
     is_hybrid: True

About ``is_hybrid`` and ``is_markdown``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``is_hybrid`` indicates whether the training set mixes WideSearch-style data and
standard QA data.

- Set ``is_hybrid: True`` for the provided hybrid training set.
- If you train on a single-source dataset such as ``width_20k`` or ``depth_20k``,
  set ``is_hybrid: False``.
- When ``is_hybrid`` is ``False``, make sure ``data.is_markdown`` matches the
  dataset format you use (True for ``width_20k``, False for ``depth_20k``).

Run It
------

Before starting training, verify all of the following:

- ``rollout.model.model_path`` points to the downloaded base model.
- ``data.train_data_paths`` points to the training dataset.
- ``agentloop.llm_ip`` is set correctly.
- Offline tools are configured and reachable. See :doc:`tools`.

Then run:

.. code-block:: bash

   bash examples/agent/wideseek_r1/run_train.sh train_qwen3_hybrid

Visualization and Results
-------------------------

Training outputs are written to:

.. code-block:: text

   ${runner.output_dir}/${runner.experiment_name}

You can inspect the TensorBoard files in that directory to monitor training
metrics.

Notes
-----

WideSeek-R1 supports both single-agent and multi-agent execution. Switch between
them with ``agentloop.workflow`` in the YAML config:

- ``mas``: multi-agent training.
- ``sa``: single-agent training.

The single-agent mode is designed to be comparable to
`ASearcher <https://github.com/inclusionAI/ASearcher>`__.
