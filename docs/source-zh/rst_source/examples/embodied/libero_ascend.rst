Ascend CANN 平台上的 LIBERO 强化学习
====================================

本文介绍在 Ascend CANN 平台上运行 RLinf LIBERO 强化学习示例所需的配置。
本文重点说明依赖安装以及容器访问宿主机 Ascend 驱动的运行方式。LIBERO
任务说明、PPO/GRPO 算法、模型下载、配置文件、指标和结果均与平台无关；
这些内容请参考 :doc:`基于LIBERO评测平台的强化学习训练 <libero>`。

依赖安装
--------

与 NVIDIA 流程相比，Ascend 平台的关键区别是依赖需要使用 Ascend 后端安装。
``install.sh`` 会安装 CPU PyTorch wheel，然后安装与 PyTorch 版本匹配的
``torch-npu`` 包。

方式 1：Docker 镜像
~~~~~~~~~~~~~~~~~~~

使用 Ascend LIBERO 镜像，或从 RLinf Dockerfile 自行构建镜像。容器需要以
privileged 模式运行，并挂载宿主机 Ascend 驱动目录：

.. code-block:: bash

   docker run -it --rm \
      --privileged \
      --ipc=host \
      --shm-size 20g \
      --network host \
      --name rlinf-ascend-libero \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-libero-cann9.0
      # 为提升国内下载速度，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-libero-cann9.0

进入容器后，切换到 OpenVLA-OFT 环境：

.. code-block:: bash

   source switch_env openvla-oft

如果需要自行构建镜像，请显式指定 Ascend 平台和 CANN 镜像版本。
``CANN_VER`` 包含基础镜像使用的硬件标签：

.. code-block:: bash

   docker build \
      --build-arg PLATFORM=ascend \
      --build-arg CANN_VER=9.0.0-910b \
      --build-arg UBUNTU_VER=22.04 \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-cann9 .

Dockerfile 使用以下 CANN 基础镜像：

.. code-block:: text

   swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VER}-ubuntu${UBUNTU_VER}-py3.11

方式 2：本地安装
~~~~~~~~~~~~~~~~

使用 ``install.sh`` 安装依赖，并传入 ``--platform ascend``：

.. code-block:: bash

   bash requirements/install.sh --platform ascend embodied --model openvla-oft --env libero
   source .venv/bin/activate

国内用户可以添加 ``--use-mirror`` 加速下载：

.. code-block:: bash

   bash requirements/install.sh --use-mirror --platform ascend embodied --model openvla-oft --env libero

LIBERO CPU 渲染
---------------

Ascend 平台运行 LIBERO 时建议使用 CPU 渲染。启动训练前设置以下环境变量：

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

``examples/embodiment/run_embodiment.sh`` 会保留这些环境变量。如果未设置，
脚本仍使用其他示例默认的 EGL 渲染方式。

在 Ascend 上启动 LIBERO
-----------------------

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
