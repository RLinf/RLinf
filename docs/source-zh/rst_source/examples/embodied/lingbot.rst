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

1. 克隆 RLinf 与一键安装环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

首先克隆 RLinf 仓库，并使用 ``install.sh`` 一键配置 Lingbot-VLA 专属的底层环境（脚本将自动拉取 Lingbot 源码至 ``.venv/lingbot-vla`` 目录，并处理所有高危依赖冲突）：

.. code-block:: bash

    export WORK_DIR="/path/to/your/workspace"
    mkdir -p ${WORK_DIR} && cd ${WORK_DIR}

    # 克隆 RLinf 仓库
    git clone https://github.com/RLinf/RLinf.git
    cd RLinf
    export RLINF_PATH=$(pwd)

    # 一键安装 Lingbot-VLA 原生环境与 RoboTwin 基础依赖
    bash requirements/install.sh embodied --model lingbot-vla --env robotwin --use-mirror --no-root
    source .venv/bin/activate

2. RoboTwin 环境配置
~~~~~~~~~~~~~~~~~~~~

由于 RLinf 内置环境不包含完整的 RoboTwin 源码，需要手动拉取 RoboTwin 的 ``RLinf_support`` 分支（该分支已包含与 Lingbot-VLA 兼容的所有必要补丁）。

.. code-block:: bash

    cd ${RLINF_PATH}
    git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
    cd RoboTwin
    export ROBOTWIN_PATH=$(pwd)
    export HF_ENDPOINT=https://hf-mirror.com
    bash script/_download_assets.sh

模型下载
--------

开始评估或训练前，请前往自动生成的 Lingbot-VLA 源码目录中，从 HuggingFace 下载基础权重和 Qwen 底座模型：

.. code-block:: bash

    # 进入 install.sh 自动生成的 lingbot 目录
    export LINGBOT_PATH="${RLINF_PATH}/.venv/lingbot-vla"
    cd ${LINGBOT_PATH}

    # 1. 下载 Lingbot 4B 基础模型权重
    python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-4b --local_dir lingbot-vla-4b

    # 【重要路径修复】消除下载脚本产生的嵌套文件夹陷阱
    cd lingbot-vla-4b
    mv lingbot-vla-4b/* .
    rmdir lingbot-vla-4b
    cd ..

    # 2. 下载 Qwen 底座权重
    huggingface-cli download --repo-type model Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B-Instruct

快速开始
--------

配置文件
~~~~~~~~

为 Lingbot 准备 RLinf 的评估配置文件： ``examples/embodiment/config/robotwin_click_bell_eval_lingbot.yaml``

**关键配置片段：**

.. code-block:: yaml

    rollout:
      group_name: "RolloutGroup"
      backend: "huggingface"
      enable_offload: False
      pipeline_stage_num: 1
      model:
        model_type: "lingbot"
        model_path: /path/to/RLinf/.venv/lingbot-vla/lingbot-vla-4b  # 请替换为实际的权重绝对路径
        tokenizer_path: /path/to/RLinf/.venv/lingbot-vla/Qwen2.5-VL-3B-Instruct
        action_dim: 14
        num_action_chunks: 50

    env:
      eval:
        total_num_envs: 16
        max_episode_steps: 300
        is_eval: True
        video_cfg:
          save_video: True
          video_base_dir: /path/to/RLinf/RoboTwin/eval_result/click_bell/policy.lingbot_wrapper/demo_clean

评估
----

模型配置完成后，使用 RLinf 官方评估脚本拉起 Ray 集群进行分布式并行评估。
**注意：** 为防止单节点渲染导致 CPU/GPU 内存溢出（OOM）及超时崩溃，建议将高精度物理仿真（RoboTwin）的并发评估环境数量 (``total_num_envs``) 限制为 4。

.. code-block:: bash

    cd ${RLINF_PATH}
    source .venv/bin/activate

    # 1. 启动纯净的 Ray 集群 (绑定本地 IP 并自定义端口，防止端口冲突引发 Dashboard 崩溃)
    ray stop --force
    export RAY_TMPDIR=/tmp/ray_private_tmp
    ray start --head --node-ip-address=127.0.0.1 --dashboard-host=127.0.0.1 --dashboard-port=8277 --include-dashboard=True

    # 2. 声明业务与离线防断流环境变量
    unset RAY_ADDRESS
    export ROBOT_PLATFORM=ALOHA
    export ROBOTWIN_PATH=${ROBOTWIN_PATH}
    export LINGBOT_PATH="${RLINF_PATH}/.venv/lingbot-vla"
    export LINGBOT_VLA_PATH="${LINGBOT_PATH}"   # 指向模型训练/权重根目录（根据实际情况调整）
    export PYTHONPATH=${RLINF_PATH}:${LINGBOT_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

    # 【可选】对于无网离线计算节点，强制阻止 HuggingFace 联网寻找文件
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1

    # 【强制】劫持 imageio 使用系统级完整版 ffmpeg（解决原生无 libopenh264 编码器报错问题）
    export IMAGEIO_FFMPEG_EXE=$(which ffmpeg)

    # 3. 修复执行脚本中未定义 ROBOTWIN_PATH 的硬编码问题
    sed -i 's|export ROBOTWIN_PATH="/path/to/RoboTwin"|export ROBOTWIN_PATH=${ROBOTWIN_PATH}|g' examples/embodiment/eval_embodiment.sh

    # 4. 执行评估指令 (请将 ${LINGBOT_VLA_PATH} 替换为实际路径或保持如上设置)
    bash examples/embodiment/eval_embodiment.sh robotwin_click_bell_eval_lingbot ALOHA \
        ++rollout.model.model_path="${LINGBOT_VLA_PATH}/output_mixed_5tasks_aloha/checkpoints/global_step_46400/hf_ckpt" \
        ++actor.model.model_path="${LINGBOT_VLA_PATH}/output_mixed_5tasks_aloha/checkpoints/global_step_46400/hf_ckpt" \
        ++rollout.model.tokenizer_path="${LINGBOT_VLA_PATH}/Qwen2.5-VL-3B-Instruct" \
        ++actor.model.tokenizer_path="${LINGBOT_VLA_PATH}/Qwen2.5-VL-3B-Instruct"

可视化与结果
------------

测试完成后，相关的成功率统计、动作日志及渲染视频将统一保存在配置文件中 ``video_base_dir`` 所指定的路径下（例如 ``RoboTwin/eval_result/click_bell/...``）。

* **视频记录**：可通过生成的 ``.mp4`` 录像确认 Lingbot-VLA 模型的空间定位、夹爪动作与轨迹顺滑度是否达标。
* **成功率**：输出的日志文件中将记录测试种子的整体任务成功率。