基于MetaWorld模拟器的强化学习训练
==================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例提供了在 `MetaWorld <https://metaworld.farama.org/>`_ 环境中使用 **RLinf** 框架
通过强化学习微调 π\ :sub:`0`\和π\ :sub:`0.5` 算法的完整指南。它涵盖了整个过程——从环境设置和核心算法设计到训练配置、评估和可视化——以及可重现的命令和配置片段。

主要目标是开发一个能够执行机器人操作能力的模型：

1. **视觉理解**\ ：处理来自机器人相机的 RGB 图像。
2. **语言理解**\ ：理解自然语言的任务描述。
3. **动作生成**\ ：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**\ ：结合环境反馈，使用 PPO 优化策略。


环境
-----------

**MetaWorld 环境**

- **Environment**：基于 *MuJoCo* 的多任务仿真环境  
- **Task**：指挥一台 7 自由度机械臂完成多种操作
- **Observation**：工作区周围离屏相机采集的 RGB 图像
- **Action Space**：4 维连续动作  
  - 末端执行器三维位置控制（x, y, z）  
  - 夹爪控制（开/合）

**数据结构**

- **Images**：RGB 张量 ``[batch_size, 3, 480, 480]``  
- **Task Descriptions**：自然语言指令  
- **Actions**：归一化的连续值
- **Rewards**：基于任务完成的稀疏奖励

算法
-----------

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
-----------

如果您使用的是 Docker 镜像，请通过 `docker pull` 拉取最新镜像以获取所需的依赖项。

如果您已经手动安装了uv虚拟环境，请运行 `uv pip install metaworld` 来安装 MetaWorld 包及其依赖项。


模型下载
-----------

在开始训练之前，您需要下载相应的预训练模型：

.. code:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld
   git clone https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld

   # 方法 2: 使用 huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-MetaWorld
   hf download RLinf/RLinf-Pi05-MetaWorld

或者，您也可以使用 ModelScope 从 https://www.modelscope.cn/models/RLinf/RLinf-Pi0-MetaWorld 下载模型。

下载后，请确保在配置 yaml 文件中正确指定模型路径。

运行脚本
-----------

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

您可以灵活配置 env、rollout 和 actor 组件的 GPU 数量。使用上述配置，您可以实现
env 和 rollout 之间的管道重叠，以及与 actor 的共享。
此外，通过在配置中设置 ``pipeline_stage_num = 2``，
您可以实现 rollout 和 actor 之间的管道重叠，提高 rollout 效率。

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
MetaWorld MT50 多任务联合训练配置文件 （在该任务设定下，训练和推理阶段均在多任务环境当中进行）：

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/metaworld_50_ppo_openpi.yaml``

- π\ :sub:`0.5`\ + PPO:
  ``examples/embodiment/config/metaworld_50_ppo_openpi_pi05.yaml``

MetaWorld ML45 联合训练配置文件 （在该任务设定下，训练在45个任务中进行，推理在OOD的5个任务中进行：

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/metaworld_45_ppo_openpi.yaml``

**3. 启动命令**

要使用选定的配置开始训练，请运行以下
命令：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，要在 MetaWorld 环境中使用 PPO 算法训练 π\ :sub:`0`\ 模型，请运行：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi


可视化和结果
-------------------------

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
       experiment_name: "test_metaworld"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


MetaWorld 结果
-------------------------
下表Diffusion Policy, TinyVLA和SmolVLA的结果参考 `SmolVLA 论文 <https://arxiv.org/abs/2403.04880>`_ 论文得到。π\ :sub:`0`\ 和 π\ :sub:`0.5`\ 的SFT结果是通过LeRobot官方提供的 `数据集 <https://huggingface.co/datasets/lerobot/metaworld_mt50>`_ 重新训练所得。

.. list-table:: **MetaWorld-MT50 性能对比（Success Rate, %）**
   :widths: 15 10 10 10 10 10
   :header-rows: 1

   * - **Methods**
     - **Easy**
     - **Medium**
     - **Hard**
     - **Very Hard**
     - **Avg.**
   * - Diffusion Policy
     - 23.1
     - 10.7
     - 1.9
     - 6.1
     - 10.5
   * - TinyVLA
     - 77.6
     - 21.5
     - 11.4
     - 15.8
     - 31.6
   * - SmolVLA
     - 87.1
     - 51.8
     - 70.0
     - 64.0
     - 68.2
   * - π\ :sub:`0`\
     - 77.9
     - 51.8
     - 53.3
     - 20.0
     - 50.8
   * - π\ :sub:`0`\  + PPO
     - **92.1**
     - **74.6**
     - 61.7
     - **84.0**
     - **78.1**
   * - π\ :sub:`0.5`\
     - 68.2
     - 37.3
     - 41.7
     - 28.0
     - 43.8
   * - π\ :sub:`0.5`\  + PPO
     - 86.4
     - 55.5
     - **75.0**
     - 66.0
     - 70.7