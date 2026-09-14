ROCm LIBERO 容器需要访问 AMD 内核与渲染设备。在宿主机的仓库根目录启动容器：

.. code-block:: bash

   docker run -it --rm \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --ipc=host --shm-size 20g --network host \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4 bash

ROCm 7.2.3 对应的镜像 tag 为 ``agentic-rlinf0.3-libero-rocm7.2.3``。中国大陆用户可使用 ``docker.1ms.run/rlinf/rlinf`` 下的同名 tag。

如需从当前代码构建包含这些模型的共用镜像，在宿主机运行以下命令，再将上面的容器镜像替换为 ``rlinf:embodied-maniskill_libero-rocm6.4``：

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=amd \
      --build-arg ROCM_VER=6.4 \
      --build-arg 'ROCM_ARCHS=gfx90a;gfx942' \
      --build-arg BUILD_TARGET=embodied-maniskill_libero \
      -t rlinf:embodied-maniskill_libero-rocm6.4 .

.. warning::

   ``ROCM_ARCHS`` 必须与目标 GPU 匹配。Docker 构建期间可能无法访问设备，``flash-attn`` 等扩展需要显式指定架构。Dockerfile 会将这些值传给 ROCm 构建工具。
