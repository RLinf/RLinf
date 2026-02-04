基于IsaacLab评测平台的强化学习训练
==============================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例提供了在 `IsaacLab <https://developer.nvidia.com/isaac/lab>`__ 环境中使用 **RLinf** 框架
通过强化学习微调gr00t算法的完整指南。它涵盖了整个过程——从环境设置和核心算法设计到训练配置、评估和可视化——以及可重现的命令和配置片段。

主要目标是开发一个能够执行机器人操作能力的模型：

1. **视觉理解**\ ：处理来自机器人相机的 RGB 图像。
2. **语言理解**\ ：理解自然语言的任务描述。
3. **动作生成**\ ：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**\ ：结合环境反馈，使用 PPO 优化策略。

环境
-------------------

**IsaacLab 环境**

- **环境**：高度客制化的仿真系统，基于isaacsim制作  
- **任务**：高度客制化适应多个智能体的任务
- **观测**：高度客制化输入
- **动作空间**：高度客制化动作

**数据结构**

- **任务描述**: 参考 `IsaacLab-Examples <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html>`__ 获取已有可用任务. 如果您想自定义任务请参考 `IsaacLab-Quickstart <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/own-project/index.html>`__ .

**添加自定义任务**

如果您想添加自定义任务请参考 `RLinf/rlinf/envs/isaaclab/tasks/stack_cube.py` , 并将您自定义的脚本放置在  `RLinf/rlinf/envs/isaaclab/tasks` 下,  同时在 `RLinf/rlinf/envs/isaaclab/__init__.py` 内添加相关代码

算法
--------------

**核心算法组件**

1. **PPO (近端策略优化)**

   - 使用 GAE (广义优势估计) 进行优势估计

   - 带比例限制的策略裁剪

   - 价值函数裁剪

   - 熵正则化

2. **GRPO (组相对策略优化)**

   - 对于每个状态/提示，策略生成 *G* 个独立动作

   - 通过减去组平均奖励来计算每个动作的优势

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
      rlinf/rlinf:agentic-rlinf0.1-isaaclab
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-isaaclab

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model gr00t --env isaaclab
   source .venv/bin/activate

ISAAC-SIM下载
--------------------

在使用IsaacLab之前，您需要下载并设置Isaac Sim。请按照以下说明操作：

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip

下载后，通过以下方式设置环境变量：

.. warning::

   每次打开新终端使用Isaac Sim时都必须执行此步骤。

.. code-block:: bash

   source ./setup_conda_env.sh

模型下载
----------------

.. code-block:: bash

   cd /workspace
   # 下载libero spatial少样本SFT模型（任选其一）
   # 方法1：使用git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

   # 方法2：使用huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir RLinf-Gr00t-SFT-Spatial

运行脚本
-------------------
.. note:: 因为现在暂时没有isaaclab的专家数据，所以我们现在的脚本都是可以跑通流程的demo

**1. 关键集群配置**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

您可以灵活配置 env、rollout 和 actor 组件的 GPU 数量。
此外，通过在配置中设置 ``pipeline_stage_num = 2``，
您可以实现 rollout 和 env 之间的管道重叠，提高 rollout 效率。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

您也可以重新配置布局以实现完全共享，
其中 env、rollout 和 actor 组件都共享所有 GPU。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

您也可以重新配置布局以实现完全分离，
其中 env、rollout 和 actor 组件各自使用自己的 GPU，无
干扰，消除了卸载功能的需要。

**2. 配置文件**

gr00t上测试isaaclab中的 `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0` 任务

- gr00t demo配置文件: ``examples/embodiment/config/isaaclab_ppo_gr00t_demo.yaml``

请将配置文件中的 `rollout.model.model_path` 参数修改为您本地下载的模型文件地址。

**3. 启动命令**

体验在isaaclab中训练gr00t:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_ppo_gr00t_demo

若想测试gt00t，请运行

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh isaaclab_ppo_gr00t_demo

可视化和结果
-------------------------------

**1. TensorBoard 日志记录**

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

-  **训练指标**

   -  ``actor/loss``: 策略损失
   -  ``actor/value_loss``: 价值函数损失 (PPO)
   -  ``actor/grad_norm``: 梯度范数
   -  ``actor/approx_kl``: 新旧策略之间的 KL 散度
   -  ``actor/pg_clipfrac``: 策略裁剪比例
   -  ``actor/value_clip_ratio``: 价值损失裁剪比例 (PPO)

-  **Rollout 指标**

   -  ``rollout/returns_mean``: 平均回合回报
   -  ``rollout/advantages_mean``: 平均优势值

-  **环境指标**

   -  ``env/episode_len``: 平均回合长度
   -  ``env/success_once``: 任务成功率

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
       experiment_name: "isaaclab_ppo_gr00t_demo"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab