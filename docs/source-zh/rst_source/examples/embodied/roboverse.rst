基于RoboVerse评测平台的强化学习训练
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档给出在 RLinf 框架内启动与管理 **Vision-Language-Action Models (VLAs)** 训练任务的完整指南，
在RoboVerse环境中微调VLA模型以完成机器人操作。

主要目标是训练能够执行以下任务的视觉-语言-动作模型:

1. **视觉理解**: 处理来自多个摄像头视角的RGB图像。
2. **语言理解**: 解释自然语言任务指令。
3. **操作技能**: 执行复杂的厨房任务，如拾取-放置、开关门和电器控制。

环境
----

**RoboVerse环境**

- **环境**: RoboVerse 仿真平台 
- **观测**: 多视角RGB图像（机器人视角+腕部相机） + 本体感知状态
- **动作空间**：7 维连续动作  
  - 末端执行器三维位置控制（x, y, z）  
  - 三维旋转控制  
  - 夹爪控制（开/合）

**观测结构**

- **主相机图像** (``main_images``): 机器人前方主视角 (224×224 RGB)
- **腕部相机图像** (``wrist_images``): 末端执行器视角相机 (224×224 RGB)
- **本体感知状态** (``states``): 8维向量，包含:
  - ``[0:3]`` 末端执行器位置 (x，y，z)
  - ``[3:6]`` 末端执行器姿态（轴角/旋转向量）
  - ``[6:8]`` 夹爪关节位置

**数据结构**

- **图像**: 主相机RGB张量 ``[batch_size, 3, 224, 224]`` 和腕部相机 ``[batch_size, 3, 224, 224]``。
- **状态**: 本体感知状态张量 ``[batch_size, 8]``。 
- **任务描述**: 自然语言指令
- **动作**: 7维连续动作
- **奖励**: 基于任务完成的稀疏奖励

算法
----

**核心算法组件**

1. **PPO (近端策略优化)**

   - 使用GAE(广义优势估计)进行优势估计

   - 带比率限制的策略裁剪

   - 价值函数裁剪

   - 熵正则化

依赖安装
--------

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
      rlinf/rlinf:agentic-rlinf0.2-roboverse
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-roboverse

**选项 2：自建环境**

通过运行以下命令在您的环境中直接安装依赖：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openpi --env roboverse
   source .venv/bin/activate

资源下载
----------------

下载 RoboVerse 资源文件：

.. code:: bash

   cd <path_to_RLinf>
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset manity/roboverse_data --local-dir .

我们提供了默认任务的资源文件，之后您需要扩展任务时可以按照RoboVerese官方文档中的任务注册指南准备新的资源文件。

模型下载
--------------

.. code-block:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT

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

.. code-block:: yaml

   camera_name: camera0
   camera_heights: 224
   camera_widths: 224
   camera_pos: [0.55, 0.0, 1.38]
   camera_look_at: [-0.28, 0.0, 0.76]
   camera_focal_length: 21

   enable_wrist_camera: true
   wrist_camera_name: robot0_eye_in_hand
   wrist_camera_mount_to: franka
   wrist_camera_mount_link: panda_hand
   wrist_camera_heights: 224
   wrist_camera_widths: 224
   wrist_camera_mount_pos: [0.11, 0.0, -0.06]
   wrist_camera_mount_quat: [-0.1, -0.7, -0.7, -0.1]
   wrist_camera_focal_length: 22

   use_ordered_reset_state_ids: True

   headless: True
   simulator_backend: mujoco
   task_name: libero_90.kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet
   robot_name: franka
   action_dim: 7
   state_dim: 8
   action_clip: 1.0
   ee_pos_delta_scale: 0.02
   ee_rot_delta_scale: 0.05

你可以在配置文件中调整相机参数、启用/禁用腕部相机、修改环境设置等，以适应不同的训练需求和实验设置。

**2. 配置文件**

RoboVerse 当前可直接参考的配置文件如下：

- **OpenPi + PPO**：``examples/embodiment/config/roboverse_ppo_openpi_pi05.yaml``  

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 RoboVerse 环境中使用 PPO 训练 OpenPi 模型：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh roboverse_ppo_openpi_pi05

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
  - ``train/actor/policy_loss``: PPO的策略损失
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
         experiment_name: "roboverse_ppo_openpi_pi05"
         logger_backends: ["tensorboard"] # wandb, swanlab
