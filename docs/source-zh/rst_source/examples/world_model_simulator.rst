基于世界模型模拟器的强化学习训练
===============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档描述如何在 **RLinf** 框架内启动与管理 **EVAC（Enerverse-AC）世界模型模拟器**，
并将其作为环境后端，用于训练与评估 **Vision-Language-Action Models (VLAs)**。
目标是在无需真实机器人或传统物理仿真器的情况下，通过视觉生成模型模拟环境随动作的动态变化，
为策略优化提供一个稳定、可控的训练闭环。

使用方式与在 LIBERO 环境中微调 VLA 类似，本指南侧重介绍如何在基于 EVAC 的模拟环境中
运行强化学习训练任务，并阐述模型在该框架中具备的关键能力。

EVAC 主要希望赋予模型以下能力：

1. **视觉理解**：EVAC 借助当前观测图像与给定的动作序列生成未来视频帧，为策略提供连续的视觉反馈，使模型能够处理来自真实机器人相机的 RGB 图像。 
2. **语言理解**：理解自然语言的任务描述。  
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。 
4. **策略提升**：借助 EVAC 生成的“想象”轨迹，利用 PPO 等强化学习方法对 VLA 策略进行优化。

与 LIBERO 环境下微调 VLA 的流程类似，本文档重点介绍如何在基于 EVAC 的模拟环境中运行 RL 训练任务。

环境
-----------------------

EVAC 作为一个世界模型，理论上可以拟合任意环境的任意任务，并保持接口一致。以 **LIBERO 环境** 为例子，环境各种接口与定义如下：

**EVAC 模拟 LIBERO 环境**

- **Environment**：基于 *robosuite* （MuJoCo）的 LIBERO 仿真基准  
- **Task**：指挥一台 7 自由度机械臂完成多种家居操作技能（抓取放置、叠放、开抽屉、空间重排等）  
- **Observation**：工作区周围离屏相机采集的 RGB 图像（常见分辨率 128×128 或 224×224）  
- **Action Space**：7 维连续动作  
  - 末端执行器三维位置控制（x, y, z）  
  - 三维旋转控制（roll, pitch, yaw）  
  - 夹爪控制（开/合）

**EVAC 模拟 LIBERO 环境重置**

不同于真实仿真器可以直接通过 reset() 进行环境重置，EVAC 需要接收初始帧、任务描述、初始时刻的末端位姿进行初始化并重置。

**任务描述格式**

.. code-block:: text

   In: What action should the robot take to [task_description]?
   Out: 

**数据结构**

- **Images**：RGB 张量 ``[batch_size, 3, 224, 224]``  
- **Task Descriptions**：自然语言指令  
- **Actions**：归一化的连续值，转换为离散 tokens  
- **Rewards**：由世界模型中的奖励判定器给出，为 0-1 奖励

算法
-----------------------------------------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE（Generalized Advantage Estimation）进行优势估计  
   - 基于比率的策略裁剪  
   - 价值函数裁剪  
   - 熵正则化

2. **GRPO（Group Relative Policy Optimization）**

   - 对于每个状态/提示，策略生成 *G* 个独立动作  
   - 以组内平均奖励为基线，计算每个动作的相对优势

3. **Vision-Language-Action 模型**

   - OpenVLA 架构，多模态融合  
   - 动作 token 化与反 token 化  
   - 带 Value Head 的 Critic 功能

模型下载
--------------

在开始训练之前，你需要下载相应的预训练模型：

.. code:: bash

   # 使用下面任一方法下载模型
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   # 方法 2: 使用 huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

除此之外，你还需要下载 EVAC 的相应权重 (此处暂时只提供 libero-spatial 仿真的权重和数据)

.. code:: bash

   # 分别下载 evac 中的 open_clip 模型权重和其他权重与数据
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
   git clone https://huggingface.co/datasets/jzndd/evac_for_rlinf

   # 方法 2: 使用 huggingface-hub
   pip install huggingface-hub
   hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K
   hf download datasets/jzndd/evac_for_rlinf


下载完成后，请确保在配置yaml文件中正确指定模型路径。

运行脚本
-------------------

**1. 关键参数配置**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 4-7

你可以灵活配置 env、rollout、actor 三个组件使用的 GPU 数量。
但注意，由于这里使用了世界模型作为环境，所以如果将环境和策略放在同一张卡上，往往会超出显存。  

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

你还可以重新配置 Placement，实现 **完全分离**：env、rollout、actor 各用各的 GPU、互不干扰，  
这样就不需要 offload 功能。

**2. 配置文件**

   支持 **OpenVLA-OFT** 模型，算法为 **PPO** 与 **GRPO**。  
   对应配置文件：

   - **OpenVLA-OFT + GRPO**：``examples/embodiment/config/libero_spatial_evac_grpo_openvlaoft.yaml``

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，使用 evac 模拟 libero-spatial 环境中并使用 GRPO 训练 OpenVLA-OFT 模型：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_evac_grpo_openvlaoft

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- **训练指标**：

  - ``actor/loss``：PPO 策略损失  
  - ``actor/value_loss``：价值函数损失  
  - ``actor/entropy``：策略熵  
  - ``actor/grad_norm``：梯度范数  
  - ``actor/lr``：学习率  

- **Rollout 指标**：

  - ``rollout/reward_mean``：平均回合奖励  
  - ``rollout/reward_std``：奖励标准差  
  - ``rollout/episode_length``：平均回合长度  
  - ``rollout/success_rate``：任务完成率  

- **环境指标**：

  - ``env/success_rate``：各环境的成功率  
  - ``env/step_reward``：逐步奖励  
  - ``env/termination_rate``：回合终止率  

**3. 视频生成**

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ./logs/video/train

**4. WandB 集成**

.. code-block:: yaml

   trainer:
     logger:
       wandb:
         enable: True
         project_name: "RLinf"
         experiment_name: "openvla-libero"

备注
~~~~~~~~~~~~~~~~~~~

目前，我们仅在 libero-spatial 环境下对 evac 的性能进行了测试，更多环境还在测试中。

在 evac 世界模型的训练中，我们参考了  
`Enerverse-AC <https://github.com/AgibotTech/EnerVerse-ACL>` 和 ，感谢作者开源代码。为了将其适配为世界模型仿真器，我们还额外添加了奖励分类器以提供稀疏奖励，添加动作转换模型以做相对动作到绝对末端位姿的映射
