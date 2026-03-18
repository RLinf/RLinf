Lingbot-VLA模型强化学习
=========================

本文档介绍如何将 Lingbot-VLA 作为原生插件接入 RLinf 框架，并在 RoboTwin 2.0 仿真环境中进行端到端的策略评估与强化学习微调。与传统的 WebSocket 通信模式不同，原生接入模式将 Lingbot-VLA 彻底融入 RLinf 的 Python 内存空间中，以实现最高效的交互与训练。

主要目标是让模型具备以下能力：

* **视觉理解**：处理来自机器人相机（如头部、腕部）的多视角 RGB 图像。
* **语言理解**：理解并泛化自然语言任务描述。
* **动作生成**：通过大模型底座（基于 Qwen2.5-VL）直接自回归生成高维连续动作块（Action Chunks）。
* **原生交互**：在 RLinf 框架内直接与 RoboTwin 仿真环境进行零延迟的 Tensor 级交互。

环境
----

**RoboTwin 环境**

* **Environment**：基于 Sapien 的 RoboTwin 2.0 物理仿真基准。
* **Task**：指挥 ALOHA 等双臂/单臂机器人完成复杂家居与操作技能（如 ``click_bell``, ``open_microwave``, ``stack_blocks_three`` 等）。
* **Observation**：多相机视角采集的 RGB 图像。
* **Action Space**：14 维连续动作（以双臂 ALOHA 为例），包含双臂的绝对位姿（x, y, z, roll, pitch, yaw）及夹爪开合度。

任务描述格式
------------

Lingbot-VLA 直接使用环境提供的自然语言任务描述作为视觉语言大模型（VLM）的文本 Prompt 输入。

数据结构
--------

* **Images**：主视角（Head）与左右腕部（Wrist）视角的 RGB 图像。
* **Task Descriptions**：自然语言指令（如 "click the bell"）。
* **Actions**：长度为 50（可配置）的动作块（Action Chunks），采用基于历史观测的开环/闭环执行策略。

算法
----

**核心算法组件**

* **GRPO (Group Relative Policy Optimization)**
    * 基于组内相对奖励的优势估计。
    * 带比例限制的策略裁剪。
    * KL 散度正则化。

* **Lingbot-VLA (基于 Qwen2.5-VL)**
    * 通过视觉语言大模型底座自回归生成动作块。
    * 基于 Flow-SDE (随机微分方程) 的动作去噪与生成。
    * 支持配置 ``noise_method``（如 ``flow_sde``）、``noise_level`` 和 ``num_steps`` 等去噪参数。

依赖安装
--------

为了实现高版本 Torch (2.8.0) 与 RLinf (Python 3.10) 的完美兼容，我们已将复杂的依赖隔离逻辑封装至安装脚本中。请按以下步骤构建混合环境。

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

首先克隆 RLinf 仓库并进入主目录：

.. code-block:: bash

    git clone https://github.com/RLinf/RLinf.git
    cd RLinf
    export RLINF_PATH=$(pwd)

2. 安装依赖
~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行基于 RoboTwin 的具身训练：

.. code-block:: bash

    docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:embodied-rlinf0.1-robotwin

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code-block:: bash

    source switch_env lingbotvla

**选项 2：自定义环境**

在本地环境中一键安装 Lingbot-VLA 原生环境与 RoboTwin 基础依赖（脚本将自动拉取 Lingbot-VLA 源码至 `.venv/lingbot-vla` 目录，并处理所有高危依赖冲突）：

.. code-block:: bash

    bash requirements/install.sh embodied --model lingbotvla --env robotwin --use-mirror
    source .venv/bin/activate

RoboTwin 仓库克隆与资产下载
---------------------------

RoboTwin Assets 是 RoboTwin 环境运行所需的资源文件，需要从 HuggingFace 下载。

.. code-block:: bash

   # 1. 克隆 RoboTwin 仓库
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support

   # 2. 下载并解压 Assets 文件
   bash script/_download_assets.sh

模型下载
--------

开始训练前，请从 HuggingFace 下载 Lingbot-VLA 基础权重和 Qwen 底座模型：

.. code-block:: bash

    # 方法 1：使用 git clone
    git lfs install
    git clone https://huggingface.co/robbyant/lingbot-vla-4b
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

    # 方法 2：使用 huggingface-hub
    pip install huggingface-hub
    huggingface-cli download robbyant/lingbot-vla-4b --local-dir lingbot-vla-4b
    huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B-Instruct


然后在配置中将 ``rollout.model.model_path`` 和 ``actor.model.model_path`` 设为本地模型路径（如 ``/path/to/model/lingbot-vla-4b`` 或 ``./lingbot-vla-4b``），并**务必**将对应的 ``tokenizer_path`` 设为下载的 Tokenizer 路径（如 ``/path/to/model/Qwen2.5-VL-3B-Instruct``），否则 Rollout 节点在解析文本指令时会报错。

快速开始
--------

配置文件
~~~~~~~~

RLinf 支持对 Lingbot-VLA 进行全参监督微调（SFT）与强化学习对齐（GRPO）。相关配置文件如下：

* **SFT (行为克隆)**:
  ``examples/sft/config/robotwin_sft_lingbotvla.yaml``
* **GRPO (强化学习)**:
  ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``

关键配置片段 (SFT)
^^^^^^^^^^^^^^^^^^

SFT 阶段的核心在于指定离线数据集格式（LeRobot Parquet 格式）、FSDP 训练后端以及批次大小。

.. code-block:: yaml

    runner:
      task_type: sft
      max_epochs: 30000

    data:
      # 指向转换好的 LeRobot 格式离线数据集目录
      train_data_paths: "/path/to/lerobot_data"

    actor:
      training_backend: "fsdp"
      micro_batch_size: 1
      global_batch_size: 8
      model:
        model_type: "lingbotvla"
        model_path: "path/to/lingbot_model"
        tokenizer_path: "/path/to/model/Qwen2.5-VL-3B-Instruct"
        precision: bf16
        num_action_chunks: 50
        action_dim: 14

关键配置片段 (GRPO)
^^^^^^^^^^^^^^^^^^^

GRPO 顶层文件通过 Hydra 动态组装了环境与模型，并直接在 ``actor.model`` 下覆写了强化学习所需的核心 SDE 采样参数。

**注意**：由于 Lingbot-VLA 使用的是 ``robotwin_50.json`` 中统一的全局归一化键值（如 ``action.arm.position``），因此在不同任务间切换时，**无需再配置或覆写** ``unnorm_key``，实现了真正的多任务平滑迁移。

.. code-block:: yaml

    rollout:
      model:
        model_type: "lingbotvla"
        

    actor:
      model:
        model_path: "/path/to/lingbot_sft_model"
        tokenizer_path: "/path/to/model/Qwen2.5-VL-3B-Instruct"
        model_type: "lingbotvla"
        lingbotvla:
            config_path: "/path/to/lingbot-vla-4b"
        action_dim: 14
        num_action_chunks: 50
        num_steps: 10              
        noise_method: "flow_sde"   
        noise_level: 0.5           
        action_env_dim: 14         


启动命令
~~~~~~~~

要使用选定的配置开始训练，请运行相应的启动脚本。

**注意**：由于默认任务使用的是双臂机器人，在执行任何启动脚本前，请务必在终端中声明机器人平台为 ALOHA，否则环境将无法正确加载动作空间：

.. code-block:: bash

    export ROBOT_PLATFORM="ALOHA"
    # 设置 ROBOTWIN_PATH 环境变量
    export ROBOTWIN_PATH=/path/to/RoboTwin
    # 设置 install.sh 自动生成的 lingbot-vla 目录
    export LINGBOT_VLA_PATH=$(python -c "import lingbotvla; import os; print(os.path.dirname(lingbotvla.__path__[0]))")


**1. 启动 SFT 训练**

使用转换好的离线数据进行监督微调：

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvla

**2. 启动 GRPO 训练**

例如，要在 RoboTwin Click Bell 任务上使用 GRPO 算法对 SFT 后的模型进行强化学习微调：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla

评估
----

Lingbot-VLA 在 RoboTwin 环境中提供了针对各项任务的端到端评估脚本（以按铃任务为例）：

.. code-block:: bash

    export ROBOT_PLATFORM="ALOHA"
    bash examples/embodiment/eval_embodiment.sh robotwin_click_bell_grpo_lingbotvla_eval

如需了解 RLinf 统一的 VLA 评估流程，请参考 :doc:`VLA 评估文档 <../../start/vla-eval>`。

可视化与结果
------------

**TensorBoard 日志**

.. code-block:: bash

    tensorboard --logdir ../results --port 6006

**关键指标**

* **训练**: ``train/actor/policy_loss``, ``train/actor/entropy_loss``, ``train/actor/approx_kl``
* **环境**: ``env/success_once`` (回合成功率), ``env/episode_len``, ``env/reward``
