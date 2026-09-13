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
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-cann9.1.1 bash

该镜像基于面向 910B 的 CANN 9.1.1，包含全部 ManiSkill 与 LIBERO 模型环境。上面的 tag 同时包含 arm64 与 amd64 两个版本，Docker 会按宿主机架构自动选择；如需指定单一架构，可使用带 ``-arm64`` 或 ``-amd64`` 后缀的 tag。此前仅含 LIBERO 的镜像仍可使用，tag 为 ``agentic-rlinf0.3-libero-cann9.0``。中国大陆用户可使用 ``infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf`` 下的同名镜像 tag。若要指定可用的 NPU，可将 ``--privileged`` 替换为以下设备参数，并为每张需要使用的 NPU 添加一项 ``/dev/davinciN``：

.. code-block:: text

   --device=/dev/davinci_manager
   --device=/dev/devmm_svm
   --device=/dev/hisi_hdc
   --device=/dev/davinci0

如需从当前代码构建镜像，在宿主机执行以下命令，再将上面启动命令中的镜像替换为 ``rlinf-maniskill_libero-cann9``：

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=ascend \
      --build-arg CANN_VER=9.1.1-910b \
      --build-arg UBUNTU_VER=22.04 \
      --build-arg BUILD_TARGET=embodied-maniskill_libero \
      -t rlinf-maniskill_libero-cann9 .

``CANN_VER`` 包含昇腾基础镜像 tag 中的硬件后缀。也可以通过 Dockerfile 的 ``ASCEND_BASE_IMAGE`` 参数指定完整的基础镜像地址。

昇腾 950 需要 CANN 9.1；上面的镜像面向 Atlas 800T A2（910B），使用 CANN 9.0。``requirements/install.sh`` 会通过 ``npu-smi`` 读取芯片型号并安装对应版本：昇腾 950 使用 torch 2.10 与 ``torch-npu`` 2.10.0.post4，910B 使用 torch 2.6。在没有 NPU 驱动的机器上（例如构建镜像时），用 ``ASCEND_CHIP`` 指定芯片：

.. code-block:: bash

   ASCEND_CHIP=Ascend950PR bash requirements/install.sh --platform ascend embodied --model openpi --env libero

用 ``--torch <version>`` 可以覆盖默认版本。
