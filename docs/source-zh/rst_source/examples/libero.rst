基于LIBERO模拟器的强化学习训练
===============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档给出在 RLinf 框架内启动与管理 **Vision-Language-Action Models (VLAs)** 训练任务的完整指南，
在 LIBERO 环境中微调 VLA 模型以完成机器人操作。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。  
2. **语言理解**：理解自然语言的任务描述。  
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。  
4. **强化学习**：结合环境反馈，使用 PPO 优化策略。

环境
-----------------------

**LIBERO 环境**

- **Environment**：基于 *robosuite* （MuJoCo）的 LIBERO 仿真基准  
- **Task**：指挥一台 7 自由度机械臂完成多种家居操作技能（抓取放置、叠放、开抽屉、空间重排等）  
- **Observation**：工作区周围离屏相机采集的 RGB 图像（常见分辨率 128×128 或 224×224）  
- **Action Space**：7 维连续动作  
  - 末端执行器三维位置控制（x, y, z）  
  - 三维旋转控制（roll, pitch, yaw）  
  - 夹爪控制（开/合）

**任务描述格式**

.. code-block:: text

   In: What action should the robot take to [task_description]?
   Out: 

**数据结构**

- **Images**：RGB 张量 ``[batch_size, 3, 224, 224]``  
- **Task Descriptions**：自然语言指令  
- **Actions**：归一化的连续值，转换为离散 tokens  
- **Rewards**：基于任务完成度的逐步奖励

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

下载完成后，请确保在配置yaml文件中正确指定模型路径。

运行脚本
-------------------

**1. 关键参数配置**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-7
         rollout: 8-15
         actor: 0-15

   rollout:
      pipeline_stage_num: 2

你可以灵活配置 env、rollout、actor 三个组件使用的 GPU 数量。  
使用上述配置，可以让 env 与 rollout 之间流水线重叠，并与 actor 共享。  
此外，在配置中设置 `pipeline_stage_num = 2`，可实现 **rollout 与 actor** 之间的流水线重叠，从而提升 rollout 效率。

.. code-block:: yaml
   
   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

你也可以重新配置 Placement，实现 **完全共享**：env、rollout、actor 三个组件共享全部 GPU。

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

   - **OpenVLA-OFT + PPO**：``examples/embodiment/config/libero_10_ppo_openvlaoft.yaml``  
   - **OpenVLA-OFT + GRPO**：``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 LIBERO 环境中使用 PPO 训练 OpenVLA 模型：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openvlaoft

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

LIBERO 结果
~~~~~~~~~~~~~~~~~~~

为了展示 RLinf 在大规模多任务强化学习方面的能力，我们在 LIBERO 的全部130个任务上训练了一个统一模型，并评估了其在 LIBERO 五个任务套件中的表现：LIBERO-Spatial、LIBERO-Goal、LIBERO-Object、LIBERO-Long和LIBERO-90。 

.. note:: 

   该统一基础模型由我们自行微调得来。如需更多详情，请参阅论文 https://arxiv.org/abs/2510.06710。

.. list-table:: **Evaluation results of the unified model on the five LIBERO task groups**
   :header-rows: 1

   * - 模型
     - Spatial
     - Goal
     - Object
     - Long
     - 90
     - Average
   * - `OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_
     - 72.18%
     - 64.06%
     - 71.48%
     - 48.44%
     - 70.97%
     - 65.43
   * - `OpenVLA-OFT (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130>`_
     - **99.40%**
     - **98.79%**
     - **99.80%**
     - **93.95%**
     - **98.59%**
     - **98.11%**
   * - 提升效果
     - +27.22%
     - +34.73%
     - +28.32%
     - +45.51%
     - +27.62%
     - +32.68%

在 Libero 实验中，我们参考了  
`SimpleVLA <https://github.com/PRIME-RL/SimpleVLA-RL>`_，仅做了少量改动。  
感谢作者开源代码。
