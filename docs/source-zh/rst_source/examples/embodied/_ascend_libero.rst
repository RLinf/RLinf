昇腾 LIBERO 镜像需要访问宿主机的 NPU 驱动。在宿主机的仓库根目录运行以下命令，将驱动目录挂载到容器中：

.. code-block:: bash

   docker run -it --rm \
      --privileged \
      --ipc=host \
      --shm-size 20g \
      --network host \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /var/log/npu:/usr/slog \
      -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
      -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
      -v "$PWD":/workspace/RLinf \
      -w /workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-libero-cann9.0 bash

中国大陆用户可使用 ``docker.1ms.run/rlinf/rlinf`` 下的同名镜像 tag。若要指定可用的 NPU，可将 ``--privileged`` 替换为以下设备参数，并为每张需要使用的 NPU 添加一项 ``/dev/davinciN``：

.. code-block:: text

   --device=/dev/davinci_manager
   --device=/dev/devmm_svm
   --device=/dev/hisi_hdc
   --device=/dev/davinci0

如需从当前代码构建镜像，在宿主机执行以下命令，再将上面启动命令中的镜像替换为 ``rlinf-libero-cann9``：

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=ascend \
      --build-arg CANN_VER=9.0.0-910b \
      --build-arg UBUNTU_VER=22.04 \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-cann9 .

``CANN_VER`` 包含昇腾基础镜像 tag 中的硬件后缀。也可以通过 Dockerfile 的 ``ASCEND_BASE_IMAGE`` 参数指定完整的基础镜像地址。
