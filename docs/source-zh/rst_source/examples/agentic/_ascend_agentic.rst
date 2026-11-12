在已配置 CANN 8.5.0 和 NNAL 的 Ascend 910B 机器上安装 FSDP2/SGLang 环境：

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   bash requirements/install.sh --platform ascend --no-root agentic --venv .venv-ascend
   source .venv-ascend/bin/activate

若机器上的 CANN 安装位置不同，请替换上述路径。安装脚本使用 Python 3.11、PyTorch 2.8.0、torch-npu 2.8.0.post2、SGLang 0.5.12.post1 和 Transformers 5.6.1。A3 机器需设置 ``ASCEND_CHIP=a3`` 以选择对应的 kernel 包，默认值为 ``910b``。

启动 Ray 前设置以下环境变量，以支持单张 NPU 上的多个 worker 进程并保留设备可见性设置：

.. code-block:: bash

   export HCCL_INTRA_ROCE_ENABLE=1
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

Ascend 配方使用 FSDP2、处理 packed inputs 的 ``ascend_fusion`` attention 和 SGLang engine backend。启动前请设置模型和数据路径。这些配方尚未验证 SGLang server backend 和 A3 硬件。
