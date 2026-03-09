FUSCO High-Performance MoE Communication Library
====================================

FUSCO is a high-performance distributed All-to-All communication library designed specifically for MoE (Mixture of Experts) training and inference scenarios.
By fusing data transformation and communication, FUSCO significantly improves communication efficiency for large-scale MoE models. This document introduces how to use FUSCO for acceleration within the RLinf framework.

Installation
------------

Refer to the installation guide provided in the official FUSCO repository (https://github.com/infinigence/FUSCO.git).

.. code-block:: bash

   # clone and install
   git clone https://github.com/infinigence/FUSCO.git
   cd FUSCO/
   python setup.py install

   # download the shared library
   mkdir -p lib
   curl -L -o lib/libfusco.so https://ghfast.top/https://github.com/infinigence/FUSCO/releases/download/v0.1/libfusco.so


Quick Start
-----------

RLinf currently integrates FUSCO through patching, supporting MoE training with Megatron-LM as the backend. When the training configuration meets the requirements, the system will automatically replace the MoEAlltoAllTokenDispatcher class in Megatron and use FUSCO's implementation for acceleration.
The configuration example for enabling FUSCO is as follows:

.. code-block:: yaml

  actor:
    model:
      moe_token_dispatcher_type: alltoall
      expert_model_parallel_size: 2
      expert_tensor_parallel_size: 1
      variable_seq_lengths: false

Configuration Description:

- ``moe_token_dispatcher_type``: Set to ``alltoall``
- ``expert_model_parallel_size``: Set to greater than 1
- ``expert_tensor_parallel_size``: Set to equal to 1
- ``variable_seq_lengths``: Set to ``false``

After meeting the above conditions and installing FUSCO correctly, RLinf will automatically enable FUSCO.

You can test with the following command:

.. code-block:: bash 

  FUSCO_SO_PATH=/path/to/libfusco.so \
  REPO_PATH=/path/to/RLinf/ \
  bash tests/e2e_tests/reasoning/run.sh \
  qwen3-moe-2.5b-collocated-mg-sgl-ep-fusco-test


References
----------
- **FUSCO Repository**: `High-performance distributed data shuffling (all-to-all) library for MoE training and inference <https://github.com/infinigence/FUSCO>`_.
- **FUSCO Paper**: `FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion <https://arxiv.org/pdf/2512.22036>`_.
