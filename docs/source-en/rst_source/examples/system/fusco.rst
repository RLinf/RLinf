FUSCO Communication
===================

FUSCO is a high-performance distributed All-to-All communication library designed specifically for MoE (Mixture of Experts) training and inference scenarios.
By fusing data transformation and communication, FUSCO significantly improves communication efficiency for large-scale MoE models. Use it to accelerate MoE communication in RLinf.

Overview
--------

Use this page to enable FUSCO acceleration for Megatron-backed MoE training in RLinf.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Component
      :text-align: center

      MoE All-to-All token dispatcher

   .. grid-item-card:: Backend
      :text-align: center

      Megatron-LM actor training

   .. grid-item-card:: Config
      :text-align: center

      ``moe_token_dispatcher_type: alltoall``

   .. grid-item-card:: Test
      :text-align: center

      Reasoning e2e FUSCO verification

Setup and Run
-------------

RLinf integrates FUSCO through patching, supporting MoE training with Megatron-LM as the backend. In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Megatron MoE

      **1. Install**

      Install RLinf and FUSCO (see the `FUSCO repository <https://github.com/infinigence/FUSCO>`__), then download ``libfusco.so``:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

         git clone https://github.com/infinigence/FUSCO.git
         cd FUSCO && python setup.py install && cd ..
         mkdir -p lib
         curl -L -o lib/libfusco.so https://github.com/infinigence/FUSCO/releases/download/v0.1/libfusco.so

      **2. Configure**

      When ``moe_token_dispatcher_type`` is ``alltoall``, ``expert_model_parallel_size > 1``, ``expert_tensor_parallel_size == 1``, and ``variable_seq_lengths`` is ``false``, RLinf automatically replaces Megatron's MoE all-to-all token dispatcher with FUSCO:

      .. code-block:: yaml

         actor:
           model:
             moe_token_dispatcher_type: alltoall
             expert_model_parallel_size: 2
             expert_tensor_parallel_size: 1
             variable_seq_lengths: false

      **3. Launch**

      Verify FUSCO with the reasoning MoE end-to-end test:

      .. code-block:: bash

         FUSCO_SO_PATH=$PWD/lib/libfusco.so \
         REPO_PATH=$PWD \
         bash tests/e2e_tests/reasoning/run.sh \
           qwen3-moe-2.5b-collocated-mg-sgl-ep-fusco-test

.. note::

   **References:** `FUSCO repository <https://github.com/infinigence/FUSCO>`__ · `FUSCO paper <https://arxiv.org/pdf/2512.22036>`__.
