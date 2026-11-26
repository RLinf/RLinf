FUSCO 集合通信加速
==================

使用 FUSCO 为 RLinf 中的 MoE 训练和推理加速分布式 All-to-All 通信。FUSCO 融合数据变换与通信过程，降低大规模 MoE 模型的通信开销。

概览
----------------------------------------

使用本页为 RLinf 中基于 Megatron 的 MoE 训练启用 FUSCO 加速。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 组件
      :text-align: center

      MoE All-to-All token dispatcher

   .. grid-item-card:: 后端
      :text-align: center

      Megatron-LM actor training

   .. grid-item-card:: 配置
      :text-align: center

      ``moe_token_dispatcher_type: alltoall``

   .. grid-item-card:: 测试
      :text-align: center

      Reasoning 端到端 FUSCO 快速验证

安装与运行
----------

RLinf 通过 Patch 方式集成 FUSCO，支持以 Megatron-LM 为后端的 MoE 训练。中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: Megatron MoE

      **1. 安装环境**

      安装 RLinf 与 FUSCO（参考 `FUSCO 官方仓库 <https://github.com/infinigence/FUSCO>`__），并下载 ``libfusco.so``：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

         git clone https://github.com/infinigence/FUSCO.git
         cd FUSCO && python setup.py install && cd ..
         mkdir -p lib
         curl -L -o lib/libfusco.so https://github.com/infinigence/FUSCO/releases/download/v0.1/libfusco.so

      **2. 配置参数**

      配置满足 ``moe_token_dispatcher_type: alltoall``、``expert_model_parallel_size > 1``、``expert_tensor_parallel_size == 1`` 且 ``variable_seq_lengths: false`` 后，RLinf 会自动将 Megatron 的 MoE all-to-all token dispatcher 替换为 FUSCO 加速实现：

      .. code-block:: yaml

         actor:
           model:
             moe_token_dispatcher_type: alltoall
             expert_model_parallel_size: 2
             expert_tensor_parallel_size: 1
             variable_seq_lengths: false

      **3. 启动测试**

      使用 reasoning MoE 端到端测试验证 FUSCO：

      .. code-block:: bash

         FUSCO_SO_PATH=$PWD/lib/libfusco.so \
         REPO_PATH=$PWD \
         bash tests/e2e_tests/reasoning/run.sh \
           qwen3-moe-2.5b-collocated-mg-sgl-ep-fusco-test

.. note::

   **参考：** `FUSCO 仓库 <https://github.com/infinigence/FUSCO>`__ · `FUSCO 论文 <https://arxiv.org/pdf/2512.22036>`__。
