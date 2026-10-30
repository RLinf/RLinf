在 Ascend CANN 上运行 RLinf
============================

在 Ascend CANN 上运行 RLinf 示例。本文说明 CANN 相关的依赖安装和容器访问宿主机
Ascend 驱动的运行方式。任务说明、算法、模型下载、配置文件、指标和结果仍以各示例
文档为准。

安装
----

与 NVIDIA 流程相比，Ascend 芯片的关键区别是依赖需要使用 Ascend 后端安装。
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
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /var/log/npu:/usr/slog \
      -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
      -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-libero-cann9.0
      # 为提升国内下载速度，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-libero-cann9.0

如果不想使用privileged，则需要额外添加设备，并手动添加NPU：

.. code-block:: bash

      # 上述指令中添加下面字段
      --device=/dev/davinci_manager \
      --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      --device=/dev/davinci0 # 第一个npu为例

进入容器后，切换到 OpenVLA-OFT 环境：

.. code-block:: bash

   source switch_env openvla-oft

如果需要自行构建镜像，请显式指定 Ascend 芯片和 CANN 镜像版本。
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

Agentic RL 示例使用 Agentic 依赖栈：

.. code-block:: bash

   bash requirements/install.sh --platform ascend agentic
   source .venv/bin/activate

LIBERO CPU 渲染
---------------

Ascend 芯片运行 LIBERO 时建议使用 CPU 渲染。启动训练前设置以下环境变量：

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

``examples/embodiment/run_embodiment.sh`` 会保留这些环境变量。如果未设置，
脚本仍使用其他示例默认的 EGL 渲染方式。

在 Ascend 上启动 LIBERO
-----------------------

依赖和模型路径准备完成后，使用 :doc:`LIBERO 主文档 <../examples/embodied/libero>` 中相同的配置，
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

Ascend 上的 GR00T N1.5
----------------------

GR00T N1.5 同样可以在 Ascend 上运行。使用 ``gr00t`` 模型和涵盖 LIBERO 任务的
``maniskill_libero`` 环境进行安装：

.. code-block:: bash

   bash requirements/install.sh --platform ascend embodied --model gr00t --env maniskill_libero
   source .venv/bin/activate

在 Ascend 上，``install.sh`` 会从源码编译 ``decord``\ （aarch64 没有官方 wheel），
并固定一个为 GR00T 验证过的 TensorFlow 版本。flash-attention 会被跳过，GR00T 在
加载时自动切换到 NPU 算子，因此无需修改配置。

使用 OSMesa 渲染启动 GR00T 的 LIBERO 训练，配置见
:doc:`GR00T 示例 <../examples/embodied/gr00t>`\ ：

.. code-block:: bash

   MUJOCO_GL=osmesa \
   PYOPENGL_PLATFORM=osmesa \
   ROBOT_PLATFORM=LIBERO \
   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t

在 Ascend 上运行 Agentic RL
----------------------------

以下配方使用 FSDP2 和 SGLang 在 Ascend 上运行 Agentic RL 任务，模型和数据集与对应的 NVIDIA 配方一致。

SearchR1
~~~~~~~~

SearchR1 需要本地 Qdrant wiki 检索 server。按照 :doc:`SearchR1 文档 <../examples/agentic/searchr1>` 下载 SearchR1 数据和 ``e5-base-v2`` retriever，在以下两个脚本中设置路径，并在不同终端启动索引构建和 server：

.. code-block:: bash

   bash examples/agent/tools/search_local_server_qdrant/build_index_ascend.sh
   bash examples/agent/tools/search_local_server_qdrant/launch_local_server_ascend.sh

然后启动 Ascend 配方：

.. code-block:: bash

   bash examples/agent/searchr1/run_train_ascend.sh train_qwen2.5_ascend

AgentLightning calc_x
~~~~~~~~~~~~~~~~~~~~~

按照 :doc:`calc_x 文档 <../examples/agentic/agentlightning_calc_x>` 下载 ``calc_x`` 数据集，在 Ascend 配置中设置模型和数据集路径，然后运行：

.. code-block:: bash

   bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn-fsdp_ascend

使用 LLM judge 的代码强化学习
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

离线 LLM judge 配方需要 Code-FIM 数据集、Qwen2.5-Coder 模型和 judge endpoint 的凭据。启动 Ascend 配置前导出 endpoint 信息：

.. code-block:: bash

   export LLMASJUDGE_API_URL=https://your-judge-endpoint/v1/chat/completions
   export LLMASJUDGE_API_KEY=your-api-key
   export LLMASJUDGE_MODEL=your-judge-model
   bash examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh qwen2.5-1.5b-grpo-llm_judge-fsdp_ascend

LIBERO 中保持不变的设置
------------------------

- 使用 :doc:`基于 LIBERO 的强化学习训练 <../examples/embodied/libero>` 中相同的 LIBERO 配置。
- 使用相同的模型下载和 ``model_path`` 配置流程。
- 使用相同的 PPO/GRPO 算法设置和 placement 概念。
- 监控相同的训练、rollout 和环境指标。
