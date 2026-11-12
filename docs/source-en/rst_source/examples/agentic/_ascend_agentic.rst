Install the FSDP2/SGLang environment on an Ascend 910B host with CANN 8.5.0 and NNAL already configured:

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   bash requirements/install.sh --platform ascend --no-root agentic --venv .venv-ascend
   source .venv-ascend/bin/activate

Use the CANN installation path on your host if it differs. The installer uses Python 3.11, PyTorch 2.8.0, torch-npu 2.8.0.post2, SGLang 0.5.12.post1, and Transformers 5.6.1. Set ``ASCEND_CHIP=a3`` for A3 kernel packages; the default is ``910b``.

Enable multiple worker processes per NPU and preserve device visibility when Ray starts:

.. code-block:: bash

   export HCCL_INTRA_ROCE_ENABLE=1
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

The Ascend recipes use FSDP2, packed ``ascend_fusion`` attention, and the SGLang engine backend. Set model and data paths before launching. The SGLang server backend and A3 hardware have not been validated for these recipes.
