Lingbot-VLA 模型原生接入与评估
==============================

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

    bash requirements/install.sh embodied --model lingbotvla --env robotwin --use-mirror --no-root
    source .venv/bin/activate

3. RoboTwin 环境配置
~~~~~~~~~~~~~~~~~~~~

由于 RLinf 内置环境不包含完整的 RoboTwin 源码，需要手动拉取 RoboTwin 的 ``RLinf_support`` 分支。

.. code-block:: bash

    cd ${RLINF_PATH}
    git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
    cd RoboTwin
    export ROBOTWIN_PATH=$(pwd)
    export HF_ENDPOINT=https://hf-mirror.com
    bash script/_download_assets.sh

模型下载
--------

开始训练前，请从 HuggingFace 下载 Lingbot-VLA 基础权重和 Qwen 底座模型：

.. code-block:: bash

    # 进入 install.sh 自动生成的 lingbot-vla 目录
    export LINGBOTVLA_PATH="${RLINF_PATH}/.venv/lingbot-vla"
    cd ${LINGBOTVLA_PATH}

    # 方法 1：使用 git clone
    git lfs install
    git clone https://huggingface.co/robbyant/lingbot-vla-4b
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

    # 方法 2：使用 huggingface-hub
    pip install huggingface-hub
    huggingface-cli download robbyant/lingbot-vla-4b --local-dir lingbot-vla-4b
    huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B-Instruct
    
    # 消除下载可能产生的嵌套文件夹陷阱
    cd lingbot-vla-4b
    mv lingbot-vla-4b/* . 
    rmdir lingbot-vla-4b
    cd ..

然后在配置中将 ``rollout.model.model_path`` 和 ``actor.model.model_path`` 设为本地路径（如 ``/path/to/model/lingbot-vla-4b`` 或 ``./lingbot-vla-4b``）。

快速开始
--------

配置文件
~~~~~~~~

* Lingbot-VLA + GRPO + RoboTwin:
  ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``

关键配置片段
^^^^^^^^^^^^

该顶层文件通过 Hydra 动态组装了环境与模型，并直接在 `actor.model` 下覆写了 GRPO 强化学习所需的核心 SDE 采样参数。

**注意**：由于 Lingbot-VLA 使用的是 `robotwin_50.json` 中统一的全局归一化键值（如 `action.arm.position`），因此在不同任务间切换时，**无需再配置或覆写** ``unnorm_key``，实现了真正的多任务平滑迁移。

.. code-block:: yaml

    rollout:
      model:
        model_type: "lingbotvla"
        model_path: "/path/to/model/lingbotvla"

    actor:
      model:
        model_path: "/path/to/model/lingbotvla"
        model_type: "lingbotvla"
        action_dim: 14
        num_action_chunks: 50
        num_steps: 10              
        noise_method: "flow_sde"   
        noise_level: 0.5           
        action_env_dim: 14         


启动命令
~~~~~~~~

要使用选定的配置开始训练，请运行以下命令：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，要在 RoboTwin Click Bell 任务上使用 GRPO 算法训练 Lingbot-VLA 模型，请运行：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla

评估
----

Lingbot-VLA 在 RoboTwin 环境中提供了针对各项任务的端到端评估脚本（以按铃任务为例）：

.. code-block:: bash

    bash examples/embodiment/eval_embodiment.sh robotwin_click_bell_eval_lingbotvla

RLinf 统一的 VLA 评估流程详见 `VLA 评估文档 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/vla-eval.html>`_。

可视化与结果
------------

**TensorBoard 日志**

.. code-block:: bash

    tensorboard --logdir ../results --port 6006

**关键指标**

* **训练**: ``train/actor/policy_loss``, ``train/actor/entropy_loss``, ``train/actor/approx_kl``
* **环境**: ``env/success_once`` (回合成功率), ``env/episode_len``, ``env/reward``
