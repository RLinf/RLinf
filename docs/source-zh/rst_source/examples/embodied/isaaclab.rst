基于IsaacLab的强化学习训练
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例提供了在 `IsaacLab <https://developer.nvidia.com/isaac/lab>`_ 环境中使用 **RLinf** 框架的完整指南，
通过强化学习对 gr00t 算法进行微调。内容覆盖从环境搭建、核心算法设计到训练配置、评估与可视化的全过程，
并提供可复现的命令与配置片段。

本示例的主要目标是训练一个具备机器人操作能力的模型：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。
2. **语言理解**：理解自然语言形式的任务描述。
3. **动作生成**：输出精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**：通过环境反馈，使用 PPO 优化策略。

环境
----

**IsaacLab 环境**

IsaacLab 是一个高度可定制的仿真平台，允许用户创建自定义环境与任务。
本示例使用 RLinf 自定义环境 `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0` 进行强化学习训练。
如需使用该自定义环境，请按照 **依赖安装** 章节完成环境配置；该环境已默认集成在 RLinf 源码对应的 IsaacLab 库中。

- **环境**：IsaacLab 仿真平台
- **任务**：控制 Franka 机械臂按蓝、红、绿顺序（自下而上）堆叠方块
- **观测**：第三人称相机与机械臂腕部相机的 RGB 图像
- **动作空间**：7 维连续动作

  - 3D 位置控制（x, y, z）
  - 3D 旋转控制（roll, pitch, yaw）
  - 夹爪控制（开/合）

**任务描述**

.. code-block:: text

   Stack the red block on the blue block, then stack the green block on the red block.

**数据结构**

- **图像**：来自主视角与腕部视角的 RGB 张量 ``[batch_size, H, W, 3]`` （``H`` 与 ``W`` 由环境配置中的相机分辨率决定，例如 ``examples/embodiment/config/env/isaaclab_stack_cube.yaml`` 中的 ``256x256``）
- **任务描述**：自然语言指令
- **状态**：末端执行器的位置、姿态与夹爪状态
- **奖励**：0-1 的稀疏成功/失败奖励

**添加自定义任务**

如需添加自定义任务，通常需要以下三步：

1. **自定义 IsaacLab 环境**：可参考 `IsaacLab-Examples <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html>`__ 中的可用环境；自定义环境项目可参考 `IsaacLab-Quickstart <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/own-project/index.html>`__。
2. **在 RLinf 中配置训练环境**：参考 ``rlinf/envs/isaaclab/tasks/stack_cube.py``，将自定义脚本放到 ``rlinf/envs/isaaclab/tasks``，并在 ``rlinf/envs/isaaclab/__init__.py`` 中添加相关代码。
3. **配置任务 ID**：参考 ``examples/embodiment/config/env/isaaclab_stack_cube.yaml``，修改 ``init_params.id`` 为自定义 IsaacLab 任务 ID，并确保 ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml`` 文件开头的 ``defaults`` 引用了正确的环境配置。

算法
----

**核心算法组件**

1. **PPO（Proximal Policy Optimization，默认）**

   - 使用 GAE（Generalized Advantage Estimation）进行优势估计
   - 策略裁剪（ratio limits）
   - 价值函数裁剪
   - 熵正则化

2. **GRPO（Group Relative Policy Optimization，未测试）**

   - 对每个状态/提示，策略生成 *G* 个独立动作
   - 通过减去组内平均奖励来计算每个动作的优势

依赖安装
--------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 国内用户可使用以下地址提高下载速度：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-isaaclab
      # 国内用户如需镜像加速，可使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-isaaclab

**选项 2：自定义环境**

直接在本地环境中安装依赖：

.. code:: bash

   # 国内用户可在 install.sh 中添加 --use-mirror 以加速依赖下载

   bash requirements/install.sh embodied --model gr00t --env isaaclab
   source .venv/bin/activate

Isaac Sim 下载
--------------

使用 IsaacLab 前需要先下载并配置 Isaac Sim：

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip

下载完成后，通过以下方式设置环境变量：

.. code-block:: bash

   source ./setup_conda_env.sh

.. warning::

   每次打开新终端并使用 Isaac Sim 时都需要执行该步骤。

模型下载
--------

.. code-block:: bash

   cd /path/to/save/model
   # 下载 IsaacLab stack_cube few-shot SFT 模型
   # 方法 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Stack-cube

   # 方法 2：使用 huggingface-hub
   # 国内用户可设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir RLinf-Gr00t-SFT-Stack-cube

运行脚本
--------

本示例默认配置文件为 ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml``。
你可以修改该配置文件以调整训练设置（例如 GPU 分配、训练超参数与日志记录选项）。

**1. 关键集群配置**

你可以灵活配置 env、rollout 与 actor 组件使用的 GPU 数量。
此外，通过在配置中设置 ``pipeline_stage_num = 2``，可以实现 rollout 与 env 之间的流水线重叠，提高 rollout 效率。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

也可以重新配置布局为完全共享（env、rollout、actor 全部共享所有 GPU）：

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

也可以重新配置为完全分离（各组件使用独立 GPU，互不干扰，从而减少/避免 offload 需求）：

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

**2. 配置模型路径**

请在配置文件中更新 ``model_path``，将其指向模型下载目录。

**3. 启动命令**

在 IsaacLab 环境中使用 PPO 训练 gr00t：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

在 IsaacLab 环境中评估 gr00t：

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

可视化与结果
------------

**1. TensorBoard 日志**

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

-  **训练指标**

   -  ``train/actor/approx_kl``：近似 KL 散度
   -  ``train/actor/clip_fraction``：裁剪比例
   -  ``train/actor/clipped_ratio``：裁剪后的比率
   -  ``train/actor/dual_cliped_ratio``：双裁剪比率
   -  ``train/actor/entropy_loss``：熵损失
   -  ``train/actor/grad_norm``：梯度范数
   -  ``train/actor/lr``：学习率
   -  ``train/actor/policy_loss``：策略损失
   -  ``train/actor/total_loss``：总损失
   -  ``train/critic/explained_variance``：解释方差
   -  ``train/critic/lr``：学习率
   -  ``train/critic/value_clip_ratio``：价值裁剪比率
   -  ``train/critic/value_loss``：价值损失

-  **Rollout 指标**

   -  ``rollout/advantages_max``：最大优势值
   -  ``rollout/advantages_mean``：平均优势值
   -  ``rollout/advantages_min``：最小优势值
   -  ``rollout/returns_max``：最大回合回报
   -  ``rollout/returns_mean``：平均回合回报
   -  ``rollout/returns_min``：最小回合回报
   -  ``rollout/rewards``：奖励

-  **环境指标**

   -  ``env/episode_len``：平均回合长度
   -  ``env/num_trajectories``：轨迹数量
   -  ``env/return``：平均回合回报
   -  ``env/reward``：平均步奖励
   -  ``env/success_once``：任务成功率

**3. 视频生成**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

**4. WandB 集成**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "isaaclab_franka_stack_cube_ppo_gr00t"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

强化学习结果
------------

下表汇总了不同训练阶段的任务成功率提升：

.. list-table::
   :header-rows: 1

   * - 模型阶段
     - 成功率
   * - 基础模型（无 SFT）
     - 0.0
   * - SFT 模型
     - 0.654
   * - RL 微调模型（SFT + RL）
     - 0.897

致谢
----
感谢 `许明辉 <https://github.com/smallcracker>`_ 和 `杨楠 <https://github.com/AquaSage18>`_ 对本示例的贡献和支持！