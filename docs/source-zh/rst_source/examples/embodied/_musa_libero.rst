MUSA 使用摩尔线程 training-suite 镜像中相互匹配的 PyTorch 与 ``torch_musa``。通过 ``mthreads`` runtime 启动容器，才能在容器内访问驱动库与设备：

.. code-block:: bash

   export MUSA_IMAGE=registry.mthreads.com/mcctest/ai/training-suite:v2.1.5-musa4.3.7
   docker run -it --rm --runtime=mthreads \
      --ipc=host --shm-size=100g \
      -e MTHREADS_VISIBLE_DEVICES=all \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      "$MUSA_IMAGE" bash

安装模型环境前，先确认设备可用：

.. code-block:: bash

   mthreads-gmi
   python -c "import torch, torch_musa; print(torch.musa.device_count())"

.. warning::

   若设备数量为零，请先解决设备访问问题。保留镜像中配套的 PyTorch 与 ``torch_musa``，安装通用 torch wheel 可能覆盖 MUSA 版本。具身示例使用 Hugging Face rollout 后端，因为 MUSA 安装会跳过仅支持 CUDA 的 vLLM 与 SGLang kernel。

如需从当前代码构建 RLinf 镜像，在宿主机使用 BuildKit：

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=musa \
      --build-arg MUSA_VER=v2.1.5-musa4.3.7 \
      --build-arg BUILD_TARGET=embodied-maniskill_libero \
      -t rlinf:embodied-maniskill_libero-musa .

MUSA 驱动库由 runtime 注入，因此设备感知容器启动前，镜像构建步骤不能导入 torch。BuildKit 还能避免解析无关的 CUDA 与 ROCm 基础镜像阶段。
