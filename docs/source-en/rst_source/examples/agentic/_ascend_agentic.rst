Install the FSDP2/SGLang environment on an Ascend 910B host with CANN 8.5.0 and NNAL already configured:

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   bash requirements/install.sh --platform ascend --no-root agentic --venv .venv-ascend
   source .venv-ascend/bin/activate

Use the CANN installation path on your host if it differs. The installer uses Python 3.11, PyTorch 2.8.0, torch-npu 2.8.0.post2, SGLang 0.5.12.post1, and Transformers 5.6.1. Set ``ASCEND_CHIP=a3`` for A3 kernel packages; the default is ``910b``.

For manual installation, follow the `SGLang Ascend guide for this version <https://github.com/sgl-project/sglang/blob/v0.5.12.post1/docs/platforms/ascend/ascend_npu.md>`_ and the `TorchNPU compatibility table <https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md>`_. Keep Triton Ascend at ``3.2.0`` and use the ``2026.05.01`` SGLang NPU kernel release for CANN 8.5.0, including its DeepEP and memory saver wheels. The ``2026.05.01.post3`` kernel imports ``triton.language.extra.cann``, which Triton Ascend 3.2.0 does not provide. These releases share the kernel package version ``2026.5.1``, so record the release URL as well as the installed version. These Python environment instructions do not require changing CANN or the driver.

Enable multiple worker processes per NPU and preserve device visibility when Ray starts:

.. code-block:: bash

   export HCCL_INTRA_ROCE_ENABLE=1
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

The Ascend recipes use FSDP2, packed ``ascend_fusion`` attention, and the SGLang engine backend. Set model and data paths before launching. The SGLang server backend and A3 hardware have not been validated for these recipes.
