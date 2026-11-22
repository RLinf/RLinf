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

   .. grid-item-card:: Hardware
      :text-align: center

      Reference run: 8×H100 training + 8×H100 external judge (16 H100 total)

.. contents::
   :depth: 2
   :local:

Prerequisites
-------------

Before launching training, complete
:doc:`Environment and Tool Setup <tools>`:

- **Hardware**: the reference setup uses **8×H100** for RLinf training plus **8×H100** for the external ``Qwen3-30B-A3B-Instruct-2507`` judge server (8+8 H100). On smaller clusters, reduce ``--dp`` on the judge server or switch to the single-node local judge config ``train_qwen3_hybrid_local_judge.yaml``.
- The RLinf environment is installed.
- The judge model is running.
- The offline retrieval tools are configured.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run
``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt
images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.
Complete :doc:`Environment and Tool Setup <tools>` so that the judge model and
offline retrieval tools are running before launching training.

.. tab-set::

   .. tab-item:: Qwen3-4B (Hybrid Multi-Agent RL)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download `Qwen3-4B <https://huggingface.co/Qwen/Qwen3-4B>`__ and `WideSeek-R1-train-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data>`__ (the main experiments use ``hybrid_20k.jsonl``; a separate `WideSeek-R1-Corpus <https://huggingface.co/datasets/RLinf/WideSeek-R1-Corpus>`__ artifact is also available for corpus inspection):

      .. code-block:: bash

         hf download Qwen/Qwen3-4B --local-dir ./Qwen3-4B
         hf download --repo-type dataset RLinf/WideSeek-R1-train-data --local-dir ./WideSeek-R1-train-data

      **3. Launch**

      Update ``rollout.model.model_path``, ``data.train_data_paths``, and ``agentloop.llm_ip`` in ``examples/agent/wideseek_r1/config/train_qwen3_hybrid.yaml``:

      .. code-block:: yaml

         rollout:
           model:
             model_type: qwen3
             model_path: /PATH/TO/MODEL

         data:
           train_data_paths: /PATH/TO/TRAIN/DATASET/hybrid_20k.jsonl
           is_hybrid: True

      Start training (for multi-node launches, see :ref:`wideseek-r1-multinode`):

      .. code-block:: bash

         bash examples/agent/wideseek_r1/run_train.sh train_qwen3_hybrid

Hybrid and Single-Source Datasets
---------------------------------

``is_hybrid`` indicates whether the training set mixes WideSearch-style data and
standard QA data.

- Set ``is_hybrid: True`` for the provided hybrid training set.
- If you train on a single-source dataset such as ``width_20k`` or ``depth_20k``,
  set ``is_hybrid: False``.
- When ``is_hybrid`` is ``False``, make sure ``data.is_markdown`` matches the
  dataset format you use (True for ``width_20k``, False for ``depth_20k``).

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
