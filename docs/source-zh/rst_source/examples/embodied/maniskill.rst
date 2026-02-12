基于ManiSkill评测平台的强化学习训练
======================================

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

- **Images**：RGB 张量 ``[batch_size, 224, 224, 3]``  
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

依赖安装
---------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

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

对于不同模型上的实验，请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   # 切换到 OpenVLA 环境
   source switch_env openvla
   # 切换到 OpenVLA-OFT 环境
   source switch_env openvla-oft

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   # 将 --model 参数改为 openvla-oft 可安装 OpenVLA-OFT 环境
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

资源下载
----------------

下载 ManiSkill 资源文件：

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

模型下载
--------------

在开始训练之前，你需要下载相应的预训练模型和资产：

.. code:: bash

   # 使用下面任一方法下载模型
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download gen-robot/openvla-7b-rlvla-warmup --local-dir openvla-7b-rlvla-warmup

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
此外，在配置中设置 `pipeline_stage_num = 2`，可实现 **rollout 与 env** 之间的流水线重叠，从而提升 rollout 效率。

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

  - ``train/actor/approx_kl``: 近似 KL，用于监控策略更新幅度
  - ``train/actor/clip_fraction``: 触发 PPO 的 clip 样本的比例
  - ``train/actor/clipped_ratio``: 被裁剪后的概率比均值，用来衡量策略更新受到 clip 的影响程度
  - ``train/actor/grad_norm``: 梯度范数
  - ``train/actor/lr``: 学习率
  - ``train/actor/policy_loss``: PPO/GRPO的策略损失
  - ``train/critic/value_loss``: 价值函数的损失
  - ``train/critic/value_clip_ratio``: PPO-style value function clipping 中触发 clip 的比例
  - ``train/critic/explained_variance``: 衡量价值函数拟合程度，越接近 1 越好
  - ``train/entropy_loss``: 策略熵
  - ``train/loss``: 策略损失 + 价值损失 + 熵正则的总和  (actor_loss + critic_loss + entropy_loss regularization)

- **Rollout 指标**：

  - ``rollout/advantages_max``: 优势函数的最大值
  - ``rollout/advantages_mean``: 优势函数的均值
  - ``rollout/advantages_min``: 优势函数的最小值
  - ``rollout/rewards``: 一个chunk的奖励

- **环境指标**：

  - ``env/episode_len``：该回合实际经历的环境步数（单位：step）
  - ``env/return``：回合总回报。在 LIBERO 的稀疏奖励设置中，该指标并不具有参考价值，因为奖励在回合中几乎始终为 0，只有在成功结束时才会给出 1
  - ``env/reward``：环境的 step-level 奖励
  - ``env/success_once``：建议使用该指标来监控训练效果，它直接表示未归一化的任务成功率，更能反映策略的真实性能


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
         experiment_name: "maniskill_ppo_openvla"
         logger_backends: ["tensorboard"] # wandb, swanlab

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
   为确保公平对比，我们沿用了 `rl4vla` (`论文链接 <https://arxiv.org/abs/2505.19789>`_) 中使用的同一套 OOD 测试集。
   对于 OpenVLA 模型，我们直接采用了 HuggingFace 上提供的预训练权重 OpenVLA (Base) (aka openvla-7b-rlvla-warmup) <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_.
   对于 OpenVLA-OFT 模型，我们利用采集自 “PutOnPlateInScene25Main-v3” 任务的运动规划数据，自行进行了 LoRA 微调。由此得到的 LoRA 权重也已上传至 HuggingFace OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_。

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
   * - |huggingface| `rl4vla <https://huggingface.co/gen-robot/openvla-7b-rlvla-rl>`_
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
