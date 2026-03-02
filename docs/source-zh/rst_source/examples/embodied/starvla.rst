StarVLA 模型强化学习训练
=========================

本文档介绍如何使用 **RLinf** 框架对 **StarVLA** 模型进行强化学习微调。StarVLA 是一个开源的 Vision-Language-Action (VLA) 工具箱，支持将 VLM backbone 与 action head 以模块化方式组合，从而训练与部署多类 VLA 模型。本示例以 **LIBERO Spatial** 基准为例，使用 **QwenOFT** 模型。

训练目标概述
------------

本次训练的主要目标包括：

1. 视觉理解：处理来自相机的 RGB 图像。
2. 语言理解：理解自然语言任务描述。
3. 动作生成：读取 VLM backbone 的 hidden states，通过 MLP action head **一次性并行回归** 一段连续动作序列（action chunk）。
4. 强化学习：结合环境反馈，使用 **GRPO** 算法优化策略。

环境与接口约定
--------------

LIBERO 环境
^^^^^^^^^^^

* **Environment**：基于 robosuite (MuJoCo) 的 LIBERO 仿真基准。
* **Task**：控制 7 自由度机械臂完成家务操作技能。
* **Observation**：多视角 RGB 图像 +（可选）proprio/state。
* **Action Space**：连续动作，常见为 7 维（末端位姿增量 6D + gripper 1D）。
* **Robot Platform**：RLinf 通过环境变量 ``ROBOT_PLATFORM`` 区分平台并选择合适的动作维度与（反）归一化逻辑。本文默认 ``ROBOT_PLATFORM=libero``。

任务描述格式
^^^^^^^^^^^^

StarVLA 直接使用环境提供的自然语言任务描述作为语言模型输入。

环境观测数据结构（``env_obs``）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 RLinf 的 StarVLA wrapper 中，环境侧观测 ``env_obs`` 建议组织为一个 dict，并遵循 batch-first（第 0 维为 batch）的约定。令 ``B`` 为并行环境数（batch size）。

必选字段：

* ``main_images``：主视角 RGB，``torch.uint8``，形状 ``[B, H, W, 3]``（常用 ``H=W=224``）。
* ``states``：本体状态，``torch.float32``，形状 ``[B, D_state]``。
* ``task_descriptions``：自然语言任务描述，``list[str]``，长度为 ``B``。

可选字段：

* ``wrist_images``：腕部视角 RGB，``torch.uint8``，形状 ``[B, H, W, 3]``。
* ``extra_view_images``：其他视角 RGB，``torch.uint8``，推荐形状 ``[B, V, H, W, 3]``（``V`` 为额外视角数）。若仅提供单个额外视角，也允许 ``[B, H, W, 3]``，并在 wrapper 内等价视为 ``V=1``。

在 LIBERO 的默认实现中，``states`` 的常见定义为：

* 末端位置 ``(x, y, z)`` （3 维）
* 末端姿态轴角 ``(rx, ry, rz)`` （3 维）
* 夹爪状态（原始为 2 维）

因此常见 ``D_state = 3 + 3 + 2 = 8``。若 checkpoint 期望 7 维状态，wrapper 会将 2 维夹爪状态做压缩（取均值）并适配为：

``[x, y, z, rx, ry, rz, g_mean]``，其中 ``g_mean = 0.5 * (g0 + g1)``。

动作块（action chunk）接口
^^^^^^^^^^^^^^^^^^^^^^^^^^

StarVLA 推理接口输出动作块：

* ``actions``：``torch.float32``，形状 ``[B, T, D_action]``。
* ``T = actor.model.num_action_chunks``：动作块长度 / planning horizon（可配置）。
* ``D_action = actor.model.action_dim``：动作维度（LIBERO 常用 7）。

Rollout 的执行策略通常采用 receding-horizon：策略每次 forward 产生一段长度 ``T`` 的动作序列，环境按设定执行其中前 ``N`` 步后重新规划下一段动作（``1 <= N <= T``）。若采用默认“整块执行再重规划”，则 ``N = T``。

算法说明
--------

* **StarVLA（QwenOFT）**：VLM backbone 提供多模态理解能力；MLP action head 以并行解码方式回归连续动作序列（非扩散式采样）。
* **GRPO**：基于环境反馈进行策略优化，兼容 RLinf 的具身训练流水线与 LIBERO 环境。

依赖安装
--------

1. 克隆 RLinf 仓库
^^^^^^^^^^^^^^^^^^

若你在开发/测试 PR，可克隆你的 fork 并切到对应分支；若已合入主仓库，直接克隆上游即可。

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
^^^^^^^^^^^^

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env starvla

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model starvla --env maniskill_libero
   source .venv/bin/activate

模型下载
---------

训练开始前，请从 HuggingFace 下载所需的 StarVLA QwenOFT checkpoint 与 base VLM：

* ``StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1``
* ``Qwen/Qwen2.5-VL-3B-Instruct``

.. code-block:: bash

   # 方式1：使用 git clone
   git lfs install
   git clone https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1
   git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

   # 方式2：使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install -U huggingface-hub
   hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
   hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

.. note::

   下载完成后，请修改 ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` 中的 ``framework.qwenvl.base_vlm``，
   使其指向 ``Qwen2.5-VL-3B-Instruct`` 的本地路径。


快速开始
--------

配置文件
^^^^^^^^

StarVLA + GRPO + LIBERO（10 tasks）示例配置：

* ``examples/embodiment/config/libero_10_grpo_starvla.yaml``

关键配置片段
^^^^^^^^^^^^

.. code-block:: yaml

   rollout:
     model:
       model_path: "/path/to/model/Qwen2.5-VL-OFT-LIBERO-4in1"

   actor:
     model:
       model_path: "/path/to/model/Qwen2.5-VL-OFT-LIBERO-4in1"
       action_dim: 7
       num_action_chunks: 8
       starvla:
         framework_name: "QwenOFT"
         expected_action_dim: ${actor.model.action_dim}
         expected_num_action_chunks: ${actor.model.num_action_chunks}

启动命令
^^^^^^^^

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_grpo_starvla

评估
----

评估建议采用 RLinf 统一的 VLA 评估流程（参见 RLinf 文档中的 Embodied VLA Evaluation 教程）。
