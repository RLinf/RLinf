基于ManiSkill模拟器的强化学习训练
==================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档给出在 RLinf 框架内启动与管理 **Vision-Language-Action Models (VLAs)** 训练任务的完整指南，
在ManiSkill3环境中微调VLA模型以完成机器人操作。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。  
2. **语言理解**：理解自然语言的任务描述。  
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。  
4. **强化学习**：结合环境反馈，使用 PPO 优化策略。

环境
-----------------------

**ManiSkill3 环境**

- **Environment**：ManiSkill3 仿真平台  
- **Task**：控制机械臂抓取多种物体  
- **Observation**：第三人称相机的 RGB 图像（224×224）  
- **Action Space**：7 维连续动作  
  - 三维位置控制（x, y, z）  
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

Model Download
--------------

在开始训练之前，你需要下载相应的预训练模型和资产：

.. code:: bash

   # 使用下面任一方法下载模型
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup

   # 方法 2: 使用 huggingface-hub
   pip install huggingface-hub
   hf download gen-robot/openvla-7b-rlvla-warmup

下载完成后，请确保在配置yaml文件中正确指定模型路径。

此外，如果 `Pathto/rlinf/envs/maniskill` 中没有 `assets/` 目录，你还需要添加资产。下载说明可在 `huggingface <https://huggingface.co/datasets/RLinf/maniskill_assets>`_ 中找到。

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

你可以灵活配置 env、rollout、actor 三个组件使用的 GPU等加速器 数量。  
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

支持两种模型：**OpenVLA** 与 **OpenVLA-OFT**；两种算法：**PPO** 与 **GRPO**。  
对应配置文件：

- **OpenVLA + PPO**：``examples/embodiment/config/maniskill_ppo_openvla.yaml``  
- **OpenVLA-OFT + PPO**：``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``  
- **OpenVLA + GRPO**：``examples/embodiment/config/maniskill_grpo_openvla.yaml``  
- **OpenVLA-OFT + GRPO**：``examples/embodiment/config/maniskill_grpo_openvlaoft.yaml``

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 ManiSkill3 环境中使用 PPO 训练 OpenVLA 模型：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla

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
         experiment_name: "openvla-maniskill"

ManiSkill3 结果
~~~~~~~~~~~~~~~~~~~

以下以 ManiSkill3 环境下的 PPO 训练为例：  
在单机 8×H100 的设置下，OpenVLA（左）与 OpenVLA-OFT（右）在 plate-25-main 任务上，成功率达到 90% 以上。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvla.png" style="width: 100%;"/>
       <p><em>OpenVLA</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvlaoft.png" style="width: 100%;"/>
       <p><em>OpenVLA-OFT</em></p>
     </div>
   </div>

我们在训练场景和 OOD（分布外）场景进行了评估。其中 OOD 包括 Vision、Semantic、Execution。  
每类任务最优模型以粗体标注。

.. note::
   为公平对比，这里采用与 `rl4vla` (`论文链接 <https://arxiv.org/abs/2505.19789>`_) 相同的 OOD 测试集。

.. list-table:: **ManiSkill3 上 OpenVLA 与 OpenVLA-OFT 的模型结果**
   :header-rows: 1
   :widths: 40 15 15 15 15 15

   * - 模型
     - 训练场景
     - Vision
     - Semantic
     - Execution
     - 平均值
   * - OpenVLA(Base)
     - 53.91%
     - 38.75%
     - 35.75%
     - 42.11%
     - 39.10%
   * - |huggingface| `rl4vla <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_
     - 93.75%
     - 80.47%
     - 75.00%
     - 81.77%
     - 79.15%
   * - |huggingface| `PPO-OpenVLA <https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood>`_
     - 96.09%
     - 82.03%
     - **78.35%**
     - **85.42%**
     - **81.93%**
   * - |huggingface| `GRPO-OpenVLA <https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood>`_
     - 84.38%
     - 74.69%
     - 72.99%
     - 77.86%
     - 75.15%
   * - OpenVLA-OFT(Base)
     - 28.13%
     - 27.73%
     - 12.95%
     - 11.72%
     - 18.29%
   * - |huggingface| `PPO-OpenVLA-OFT <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood>`_
     - **97.66%**
     - **92.11%**
     - 64.84%
     - 73.57%
     - 77.05%
   * - |huggingface| `GRPO-OpenVLA-OFT <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood>`_
     - 94.14%
     - 84.69%
     - 45.54%
     - 44.66%
     - 60.64%
   

.. note::
   `rl4vla` 指在 **小 batch** 条件下，使用 PPO + OpenVLA 的设置，仅应与我们在类似条件下的 PPO+OpenVLA 对比。  
   而我们的 PPO+OpenVLA 受益于 RLinf 的大规模基础设施，能够使用 **更大的 batch** 进行训练，我们观察到这能显著提升性能。

下面的动图展示了在 RLinf 框架中，使用 PPO 在 ManiSkill3 多任务基准上训练 OpenVLA 模型的效果。

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src=https://github.com/RLinf/misc/raw/main/pic/embody.mp4 type="video/mp4">
     Your browser does not support the video tag.
   </video>
