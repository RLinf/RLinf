AMD ROCm 平台上的 LIBERO 强化学习
=================================

本文介绍在 AMD ROCm 平台上运行 RLinf LIBERO 强化学习示例所需的配置。
本文重点说明依赖安装和运行时环境变量。LIBERO 任务说明、PPO/GRPO
算法、模型下载、配置文件、指标和结果均与平台无关；这些内容请参考
:doc:`基于LIBERO评测平台的强化学习训练 <libero>`。

依赖安装
--------

与 NVIDIA 流程相比，AMD 平台的关键区别是依赖需要使用 ROCm 后端安装，
并且 LIBERO 建议使用 OSMesa 进行 CPU 渲染。

方式 1：Docker 镜像
~~~~~~~~~~~~~~~~~~~

优先使用 ROCm LIBERO 镜像：

.. code-block:: bash

   docker run -it --rm \
      --device=/dev/kfd \
      --device=/dev/dri \
      --group-add video \
      --ipc=host \
      --shm-size 20g \
      --network host \
      --name rlinf-amd-libero \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4
      # 对于中国大陆用户，可以使用以下方式加速下载：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4
      # rocm7.2.3: rlinf/rlinf:agentic-rlinf0.2-libero-rocm7.2.3

进入容器后，切换到 OpenVLA-OFT 环境：

.. code-block:: bash

   source switch_env openvla-oft

如果需要自行构建镜像，请显式指定 AMD 平台和 ROCm 版本：

.. code-block:: bash

   docker build \
      --build-arg PLATFORM=amd \
      --build-arg ROCM_VER=6.4 \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-rocm6.4 .

当 Docker 构建环境无法看到 AMD 设备时，``flash-attn`` 等 ROCm 扩展无法自动检测
GPU 架构。此时需要显式传入目标架构列表：

.. code-block:: bash

   docker build \
      --build-arg PLATFORM=amd \
      --build-arg ROCM_VER=6.4 \
      --build-arg 'ROCM_ARCHS=gfx90a;gfx942' \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-rocm6.4 .

请根据目标 AMD GPU 选择对应的 ``gfx`` 值。RLinf 会将 ``ROCM_ARCHS`` 传递给
``GPU_ARCHS`` 等 ROCm 构建变量，避免源码构建在 Docker 中回退到
``--offload-arch=native``。

方式 2：本地安装
~~~~~~~~~~~~~~~~

使用 ``install.sh`` 安装依赖，并传入 ``--platform amd``：

.. code-block:: bash

   bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env libero
   source .venv/bin/activate

如果 ROCm 安装在标准路径下，通常可以省略 ``--rocm``，安装脚本会自动检测。
国内用户可以添加 ``--use-mirror`` 加速下载：

.. code-block:: bash

   bash requirements/install.sh --use-mirror --platform amd --rocm 6.4 embodied --model openvla-oft --env libero

LIBERO CPU 渲染
---------------

AMD 平台运行 LIBERO 时建议使用 CPU 渲染。启动训练前设置以下环境变量：

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

``examples/embodiment/run_embodiment.sh`` 会保留这些环境变量。如果未设置，
脚本仍使用其他示例默认的 EGL 渲染方式。

在 AMD 上启动 LIBERO
--------------------

依赖和模型路径准备完成后，使用 :doc:`LIBERO 主文档 <libero>` 中相同的配置，
但保持 OSMesa 渲染：

.. code-block:: bash

   MUJOCO_GL=osmesa \
   PYOPENGL_PLATFORM=osmesa \
   ROBOT_PLATFORM=LIBERO \
   bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

如果运行 PPO，请使用原 LIBERO 文档中的 PPO 配置：

.. code-block:: bash

   MUJOCO_GL=osmesa \
   PYOPENGL_PLATFORM=osmesa \
   ROBOT_PLATFORM=LIBERO \
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openvlaoft

保持不变的部分
--------------

- 使用 :doc:`基于LIBERO评测平台的强化学习训练 <libero>` 中相同的 LIBERO 配置。
- 使用相同的模型下载和 ``model_path`` 配置流程。
- 使用相同的 PPO/GRPO 算法设置和 placement 概念。
- 监控相同的训练、rollout 和环境指标。
