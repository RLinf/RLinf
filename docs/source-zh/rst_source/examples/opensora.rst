基于 OpenSora 世界模型的强化学习
==================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档提供在 **RLinf** 框架中启动与管理 Vision-Language-Action Models (VLAs) 训练任务的完整指南，
使用 **Action-conditioned OpenSora 世界模型** （下文简称 OpenSora）作为环境后端。

核心目标是在无需真实机器人或传统物理仿真器的情况下，通过视觉生成模型模拟环境随动作的动态变化，
为策略优化提供闭环训练。

与在 LIBERO 环境中微调 VLA 的流程类似，本指南重点介绍如何在基于 OpenSora 的仿真环境中运行强化学习训练任务，
并展示该框架下模型具备的关键能力。

OpenSora 主要希望赋予模型以下能力：

1. **视觉理解**：OpenSora 借助当前观测图像与给定动作序列生成未来视频帧，为策略提供连续视觉反馈，使模型能够处理来自真实机器人相机的 RGB 图像。
2. **语言理解**：理解自然语言任务描述。
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **策略提升**：借助 OpenSora 生成的“想象”轨迹，使用 PPO 等强化学习方法优化 VLA 策略。

与 LIBERO 环境下微调 VLA 的流程类似，本文档重点介绍如何在基于 OpenSora 的仿真环境中运行 RL 训练任务。

环境
-----------------------

作为世界模型，OpenSora 理论上可以拟合任意环境的任意任务并保持接口一致。以 **LIBERO 环境** 为例，环境接口与定义如下：

**OpenSora 模拟 LIBERO 环境**

- **Environment**：视觉生成模型
- **Task**：指挥一台 7 自由度机械臂完成多种家居操作技能（抓取放置、叠放、开抽屉、空间重排等）
- **Observation**：视觉生成模型返回的图像
- **Action Space**：7 维连续动作
  - 末端执行器三维位置控制（x, y, z）
  - 三维旋转控制（roll, pitch, yaw）
  - 夹爪控制（开 / 合）

**OpenSora 模拟 LIBERO 环境重置**

不同于传统仿真器可通过 reset() 直接重置，OpenSora 需要接收初始帧与任务描述进行初始化与重置。
因此需提前下载初始化数据集并在配置中指定路径。

**数据结构**

- **Images**：RGB 张量 ``[batch_size, 256, 256, 3]``
- **Task Descriptions**：自然语言指令
- **Actions**：归一化连续值，转换为离散 tokens
- **Rewards**：由世界模型中的奖励判定器给出，范围为 0 到 1

算法
-----------------------------------------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE（Generalized Advantage Estimation）进行优势估计
   - 基于比率的策略裁剪
   - 价值函数裁剪
   - 熵正则化

2. **GRPO（Group Relative Policy Optimization）**

   - 对于每个状态 / 提示，策略生成 *G* 个独立动作
   - 以组内平均奖励为基线，计算每个动作的相对优势

3. **Vision-Language-Action 模型**

   - OpenVLA 架构，多模态融合
   - 动作 token 化与反 token 化
   - 带 Value Head 的 Critic 功能

依赖安装
-----------------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-opensora
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-opensora

**选项 2：自定义环境**

直接在本地环境中安装依赖：

.. code:: bash

   # 为提高国内依赖安装速度，可在 install.sh 中添加 --use-mirror
   bash requirements/install.sh embodied --model openvla-oft --env opensora
   source .venv/bin/activate

VLA 模型下载
------------------

在开始训练之前，需要下载相应预训练模型：

.. code:: bash

   # 使用下面任一方法下载模型
   # 方法 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   # 方法 2：使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

下载完成后，请确保在配置 yaml 文件中正确指定模型路径与 unnorm_key。

.. code:: yaml

   rollout:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   actor:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
         unnorm_key: libero_90_no_noops_trajall # 对于 RLinf-OpenVLAOFT-LIBERO-130-Base-Lora 模型，使用 libero_130_no_noops_trajall

WM (World Model) 模型下载
--------------------------------

除 VLA 模型之外，还需下载 OpenSora 权重与用于仿真初始化的数据集。
当前 RLinf 仅提供 libero-spatial 与 libero-object 的权重与数据，下载方法如下：

.. code:: bash

   # 下载权重与初始化数据
   # 方法 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenSora-LIBERO-Spatial

   # 方法 2：使用 huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenSora-LIBERO-Spatial --local-dir RLinf-OpenSora-LIBERO-Spatial

RLinf-OpenSora-LIBERO-Spatial 的目录结构如下：

.. code-block:: text

    RLinf-OpenSora-LIBERO-Spatial/
        ├── dataset_statistics.json             # 数据集归一化统计信息
        ├── dataset/                            # 仿真初始化数据集
        │   ├── traj0.npy
        │   ├── traj1.npy
        │   ├── ...
        │   └── trajN.npy
        ├── model-00001.safetensors              # 世界模型权重文件
        ├── model.safetensors.index.json
        ├── config.json
        ├── resnet_rm.pth                        # 奖励模型权重文件
        └── vae/                                 # VAE 模型权重文件

下载完成后，请确保在配置 yaml 文件中正确指定模型路径。

.. code:: yaml

    env:
        train:
            opensora_wm_hf_ckpt_path: /Pathto/model/RLinf-OpenSora-LIBERO-Spatial/

运行脚本
-------------------

请确保在运行下面命令前已激活正确的 Python 虚拟环境（venv）。
如果使用官方 Docker 镜像，请通过 `source switch_env openvla-oft` 切换到 `openvla-oft` 环境。

**1. 关键参数配置**

以 OpenVLA-OFT 模型为例，在 ``actor.model`` 中需要配置以下关键参数：

.. code-block:: yaml

   actor:
     model:
       model_path: "/path/to/model/Openvla-oft-SFT-libero-spatial-traj1/"    # SFT 模型路径
       model_type: "openvla_oft"                                             # 模型类型设置为 openvla_oft
       use_proprio: False                                                    # 是否使用本体感觉信息
       num_images_in_input: 1                                                # 输入图像数量
       num_action_chunks: 8                                                  # 动作块数量
       unnorm_key: "libero_spatial_no_noops"                                 # 动作归一化键（与 SFT 一致）。RLinf-OpenVLAOFT-LIBERO-130-Base-Lora 使用 libero_130_no_noops_trajall；RLinf-OpenVLAOFT-LIBERO-90-Base-Lora 使用 libero_90_no_noops_trajall。

需要注意的是，world model 不提供本体信息、不生成腕部视角且 chunk 固定，
因此 ``use_proprio`` 默认 False，``num_images_in_input`` 默认 1，``num_action_chunks`` 默认 8。

**2. 环境配置**

在环境配置文件中设置以下关键参数：

.. code-block:: yaml

   # 在 CHOSEN_CONFIG 中覆写

   # 推荐训练使用 opensora_libero_spatial，评估使用 libero_spatial
   env/train: opensora_libero_spatial
   env/eval: libero_spatial
   env:
      train:
         opensora_wm_hf_ckpt_path: /Pathto/model/RLinf-OpenSora-LIBERO-Spatial/

   # 在 env/train/opensora_libero_spatial.yaml 中：

   env_type: opensora_wm
   wm_env_type: libero
   # world model 初始化的初始图像路径
   initial_image_path: ${env.train.opensora_wm_hf_ckpt_path}/dataset_for_rlinf_world_model_init/base_policy_rollout_buffer
   # 不建议修改 world_model_cfg 中的参数
   world_model_cfg:
      # world model 中用于归一化的统计信息路径
      stats_path: /Pathto/model/RLinf-OpenSora-LIBERO-Spatial/best_wm_ckpt/base_policy/dataset_statistics.json
      chunk: 8                     # 与训练和 VLA 推理长度对齐，默认 8
      condition_frame_length: 4    # 与训练对齐的上下文记忆长度，默认 4
      model:
      # 预训练权重
         from_pretrained: /Pathto/model/RLinf-OpenSora-LIBERO-Spatial/best_wm_ckpt/base_policy/model

**3. 配置文件**

目前支持 **OpenVLA-OFT** 模型与 **GRPO** 算法，对应配置文件：

- **OpenVLA-OFT + GRPO**：``examples/embodiment/config/opensora_libero_spatial_grpo_openvlaoft.yaml``

**4. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 OpenSora 环境中使用 GRPO 训练 OpenVLA-OFT 模型：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh opensora_libero_spatial_grpo_openvlaoft

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- **训练指标**：

  - ``train/actor/approx_kl``：近似 KL，用于监控策略更新幅度
  - ``train/actor/clip_fraction``：触发 PPO 裁剪的样本比例
  - ``train/actor/clipped_ratio``：裁剪后概率比的均值，用于衡量策略更新受裁剪影响程度
  - ``train/actor/grad_norm``：梯度范数
  - ``train/actor/lr``：学习率
  - ``train/actor/policy_loss``：PPO/GRPO 的策略损失
  - ``train/critic/value_loss``：价值函数损失
  - ``train/critic/value_clip_ratio``：PPO-style value function clipping 中触发裁剪的比例
  - ``train/critic/explained_variance``：衡量价值函数拟合程度，越接近 1 越好
  - ``train/entropy_loss``：策略熵
  - ``train/loss``：总训练损失（actor_loss + critic_loss + entropy_loss regularization）

- **Rollout 指标**：

  - ``rollout/advantages_max``：优势函数最大值
  - ``rollout/advantages_mean``：优势函数均值
  - ``rollout/advantages_min``：优势函数最小值
  - ``rollout/rewards``：一个 chunk 的奖励（参考 libero_env.py 的 L414）

- **环境指标**：

  - ``env/episode_len``：回合实际经历的环境步数（单位：step）
  - ``env/return``：回合总回报。在 LIBERO 的稀疏奖励设置中该指标不具参考意义，因为回合中几乎始终为 0，仅在成功终止时为 1。
  - ``env/reward``：step-level 奖励（任务未完成时为 0，仅成功终止时为 1）。
    日志数值按回合步数归一化，难以直接反映真实任务表现。
  - ``env/success_once``：推荐用于监控训练效果，直接反映未归一化的任务成功率。

**3. 视频生成**

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

**4. 训练日志工具集成**

.. code-block:: yaml

   runner:
      task_type: embodied
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "libero_10_grpo_openvlaoft"
         logger_backends: ["tensorboard"] # wandb, swanlab

LIBERO 部分结果
~~~~~~~~~~~~~~~~~~~~~~

目前仅测试使用 OpenSora 模拟 libero-spatial 与 libero-object 环境并训练 VLA 模型，更多环境仍在测试中。

对于每个 LIBERO 套件，我们评估所有 task_id 与 trial_id 的组合。Object 与 Spatial 套件共评估 500 个环境（10 个任务 × 50 个试次）。

我们根据模型的训练配置设置评估超参：
对于 SFT 训练（LoRA-base）模型，设置 `do_sample = False`。
对于 RL 训练模型，设置 `do_sample = True`、`temperature = 1.6`，并启用 `rollout_epoch=2` 以获得最佳性能。

.. note::

    选择 OpenSora 作为世界模型模拟器的动机来源于 `WMPO <https://arxiv.org/abs/2511.09515>`_。
    在实际世界模型训练中，我们参考了 `WMPO <https://arxiv.org/abs/2511.09515>`_ 与 `OpenSora <https://github.com/RLinf/opensora>`_。

.. list-table:: **使用 OpenSora 模拟器的 LIBERO 任务组评测结果**
    :header-rows: 1
    :widths: 50 25 25

    * - 模型
      - Object
      - Spatial
    * - |huggingface| `OpenVLA-OFT (LoRA-base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_
      - 50.20%
      - 51.61%
    * - OpenVLA-OFT（OpenSora 作为世界模型的 RLinf-GRPO）
      - 75.5%
      - 64.5%
    * - **效果提升**
      - **+25.3%**
      - **+12.9%**
