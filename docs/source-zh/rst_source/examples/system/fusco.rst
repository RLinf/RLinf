FUSCO高性能MoE通信库
====================================

FUSCO是一个高性能的分布式 All-to-All 通信库，专为 MoE（Mixture of Experts）训练和推理场景设计。
通过融合数据变换与通信过程，FUSCO 显著提升了大规模 MoE 模型的通信效率。本文档介绍了如何在 RLinf 框架下使用 FUSCO 进行加速。

安装
----

参考 FUSCO 官方仓库(https://github.com/infinigence/FUSCO.git) 给出的安装指南。

.. code-block:: bash

   # clone and install
   git clone https://github.com/infinigence/FUSCO.git
   cd FUSCO/
   python setup.py install

   # download the shared library
   mkdir -p lib
   curl -L -o lib/libfusco.so https://ghfast.top/https://github.com/infinigence/FUSCO/releases/download/v0.1/libfusco.so


快速开始
--------

RLinf 目前通过 Patch 方式集成 FUSCO，支持以 Megatron-LM 为后端的 MoE 训练。当训练配置满足条件时，系统会自动替换 Megatron 中的 MoEAlltoAllTokenDispatcher 类，并使用 FUSCO 的实现进行加速。
启用 FUSCO 的配置示例如下：

.. code-block:: yaml

  actor:
    model:
      moe_token_dispatcher_type: alltoall
      expert_model_parallel_size: 2
      expert_tensor_parallel_size: 1
      variable_seq_lengths: false

配置说明：

- ``moe_token_dispatcher_type``: 设置为 ``alltoall``
- ``expert_model_parallel_size``: 设置为大于1
- ``expert_tensor_parallel_size``: 设置为等于1
- ``variable_seq_lengths``: 设置为 ``false``

满足以上条件且正确安装 FUSCO 后，RLinf 会自动启用 FUSCO。

可以通过以下命令进行测试：

.. code-block:: bash 

  FUSCO_SO_PATH=/path/to/libfusco.so \
  REPO_PATH=/path/to/RLinf/ \
  bash tests/e2e_tests/reasoning/run.sh \
  qwen3-moe-2.5b-collocated-mg-sgl-ep-fusco-test


参考
--------
- **FUSCO 仓库**：`High-performance distributed data shuffling (all-to-all) library for MoE training and inference <https://github.com/infinigence/FUSCO>`_。
- **FUSCO 论文**：`FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion <https://arxiv.org/pdf/2512.22036>`_。