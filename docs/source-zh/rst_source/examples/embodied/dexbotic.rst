Dexbotic模型强化学习训练
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何使用 RLinf 框架对 **Dexbotic** VLA 模型进行强化学习微调。
Dexbotic (`<https://github.com/dexmal/dexbotic>`__) 是 Dexmal 推出的开源 Vision-Language-Action 工具箱，
实现了多种具身模型的统一接口。本示例以 LIBERO Spatial 基准为例，使用 Dexbotic π\ :sub:`0`\ 模型。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像
2. **语言理解**：理解自然语言任务描述
3. **动作生成**：通过基于流的扩散去噪产生精确的机器人动作
4. **强化学习**：结合环境反馈，使用 PPO 优化策略

环境
-----------

**LIBERO 环境**

- **Environment**：基于 *robosuite* （MuJoCo）的 LIBERO 仿真基准
- **Task**：指挥 7 自由度机械臂完成家居操作技能（抓取放置、叠放、空间重排等）
- **Observation**：工作区周围离屏相机采集的 RGB 图像（常见分辨率 128×128 或 224×224）
- **Action Space**：7 维连续动作
  - 末端执行器三维位置控制（x, y, z）
  - 三维旋转控制（roll, pitch, yaw）
  - 夹爪控制（开/合）

**任务描述格式**

Dexbotic 直接使用环境提供的自然语言任务描述作为语言模型输入。

**数据结构**

- **Images**：主视角与腕部视角 RGB 张量，形状为 ``[batch_size, 224, 224, 3]``
- **States**：末端执行器位姿（位置 + 朝向）及夹爪状态
- **Task Descriptions**：自然语言指令
- **Actions**：长度为 50 的动作块（可配置）；每 N 步重新规划动作

算法
---------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE 进行优势估计
   - 基于比率的策略裁剪
   - 价值函数裁剪
   - 熵正则化

2. **Dexbotic（基于 π\ :sub:`0.5`\ 的 VLA）**

   - 基于 flow-matching / flow-SDE 的动作生成
   - 用于动作块的扩散去噪
   - 带 Value Head 的 Critic 功能
   - 可配置 ``noise_method``（如 ``flow_sde``）、``noise_level`` 与 ``num_steps``

依赖安装
-----------------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像进行 LIBERO 具身训练：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env dexbotic

**选项 2：自定义环境**

在本地环境中安装依赖：

.. code:: bash

   bash requirements/install.sh embodied --model dexbotic --env maniskill_libero
   source .venv/bin/activate

模型下载
--------------

开始训练前，请从 HuggingFace 下载 Dexbotic SFT 模型：

.. code:: bash

   # 方法 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/Dexmal/libero-db-pi0

   # 方法 2：使用 huggingface-hub
   pip install huggingface-hub
   huggingface-cli download Dexmal/libero-db-pi0 --local-dir libero-db-pi0

然后在配置中将 ``rollout.model.model_path`` 和 ``actor.model.model_path``
设为本地路径（如 ``/path/to/model/Dexbotic-Pi05-SFT`` 或 ``./libero-db-pi0``）。

快速开始
-----------

**配置文件**

- **Dexbotic + PPO + LIBERO Spatial**：
  ``examples/embodiment/config/libero_spatial_ppo_dexbotic_pi0.yaml``

**关键配置片段**

.. code:: yaml

   rollout:
     model:
       model_path: "/path/to/model/Dexbotic-Pi05-SFT"
   actor:
     model:
       model_path: "/path/to/model/Dexbotic-Pi05-SFT"
       num_action_chunks: 5
       num_steps: 4
       action_dim: 7
       add_value_head: True
       dexbotic:
         num_images_in_input: 2
         noise_level: 0.5
         noise_method: "flow_sde"
         train_expert_only: True
         detach_critic_input: True

**启动命令**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_dexbotic_pi0

评估
----------

Dexbotic 在 ``toolkits/eval_scripts_dexbotic/`` 中提供了针对 LIBERO 的评估脚本：

.. code-block:: bash

   python toolkits/eval_scripts_dexbotic/libero_eval.py \
      --config libero_spatial_dexbotic \
      --ckpt_path /path/to/checkpoint \
      --action_chunk_size 50 \
      --num_diffusion_steps 10

亦可使用 RLinf 统一的 VLA 评估流程，详见
:doc:`VLA 评估文档 <../../start/vla-eval>`。

可视化与结果
-------------------------

**TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**关键指标**

- **训练**：``train/actor/policy_loss``、``train/critic/value_loss``、
  ``train/actor/approx_kl``
- **环境**：``env/success_once``（回合成功率）、``env/episode_len``
