基于Behavior模拟器的强化学习训练
====================================

本示例提供了在 `Behavior <https://behavior.stanford.edu/index.html>`_ 环境中使用 **RLinf** 框架
通过强化学习微调 Behavior 算法的完整指南。它涵盖了整个过程——从
环境设置和核心算法设计到训练配置、
评估和可视化——以及可重现的命令和
配置片段。

主要目标是开发一个能够执行
机器人操作能力的模型：

1. **视觉理解**\ ：处理来自机器人相机的 RGB 图像。
2. **语言理解**\ ：理解自然语言的任务描述。
3. **动作生成**\ ：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**\ ：结合环境反馈，使用 PPO 优化策略。


环境
-----------

**Behavior 环境**

- **环境**: 基于 *IsaacSim* 构建的 Behavior 仿真基准测试。
- **任务**: 控制双臂 R1 Pro 机器人执行各种家庭操作技能（抓取放置、堆叠、打开抽屉、空间重排）。
- **观察**: 由机器人搭载的传感器捕获的多相机 RGB 图像：
  - **头部相机**: 提供 224×224 RGB 图像用于全局场景理解
  - **手腕相机**: 左右 RealSense 相机提供 224×224 RGB 图像用于精确操作
- **动作空间**: 23 维连续动作（3-DOF (x,y,rz) 关节组、4-DOF 躯干、x2 7-DOF 手臂和 x2 1-DOF 平行夹爪）

**数据结构**

- **任务描述**: 从 `behavior-1k` 任务中选择
- **图像**: 多相机 RGB 张量
  - 头部图像: ``[batch_size, 3, 224, 224]``
  - 手腕图像: ``[batch_size, 2, 3, 224, 224]`` (左右相机)


算法
---------

**核心算法组件**

1. **PPO (近端策略优化)**

   - 使用 GAE (广义优势估计) 进行优势估计

   - 带比例限制的策略裁剪

   - 价值函数裁剪

   - 熵正则化

2. **GRPO (组相对策略优化)**

   - 对于每个状态/提示，策略生成 *G* 个独立动作

   - 通过减去组平均奖励来计算每个动作的优势

3. **视觉-语言-动作模型**

   - 具有多模态融合的 OpenVLA 架构

   - 动作标记化和去标记化

   - 用于批评函数的价值头

前置依赖（软件安装与数据集/资源下载）
-------------------------------------

依赖安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   请参考以下 ISAAC-SIM 的软硬件依赖文档确定自己的环境是否满足要求。

   https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html

   https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   尤其注意，如果你的GPU是Hopper及以上架构，请按照570及以上的NVIDIA驱动。

   另外，如果您的GPU没有Ray Tracing能力（例如A100、H100），BEHAVIOR的渲染质量会非常差，画面可能会出现严重的马赛克或模糊。
   
**选项 1：Docker 镜像**

使用我们的新 Docker 镜像 `rlinf/rlinf:agentic-rlinf0.1-behavior` 来运行BEHAVIOR实验。

**选项 2：自定义环境**

.. warning::

   **风险自负！**

   我们强烈建议不要构建自定义环境，因为 BEHAVIOR 和 ISAAC-SIM 的依赖关系非常复杂，一旦出错，可能会导致难以调试的问题。
   但我们仍然提供此选项，以防 Docker 在您的环境中不可用。

.. code:: bash

   pip install uv
   bash requirements/install.sh openvla-oft --enable-behavior

**资源文件和数据集**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ISAAC-SIM 4.5下载

.. warning::

   `ISAAC_PATH` 环境变量必须在每次运行实验前都进行设置。

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   mkdir -p $ISAAC_PATH && cd $ISAAC_PATH
   curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o isaac-sim.zip
   unzip isaac-sim.zip && rm isaac-sim.zip

* BEHAVIOR 数据集和资源下载

.. warning::

   `OMNIGIBSON_DATA_PATH` 环境变量必须在每次运行实验前都进行设置。

.. code:: bash

   # 将以下环境变量改到你希望存放Behavior资源和数据集的目录
   # 注意，相关数据集会占用超过30GB的存储空间
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   mkdir -p $OMNIGIBSON_DATA_PATH

   # 请确保您在运行下面的命令前已激活正确的 Python 虚拟环境（venv）
   # 如果您在使用 Docker 镜像，您需要通过`source switch_env openvla-oft`命令切换到`openvla-oft`环境
   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)" 
   python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"


模型下载
---------------

在开始训练之前，您需要下载相应的预训练模型。根据您要使用的算法类型，我们提供不同的模型选项：

**OpenVLA-OFT 模型下载**

OpenVLA-OFT 提供了一个适用于 Behavior 环境中所有任务类型的统一模型。

.. code:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior

   # 方法 2: 使用 huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior

或者，您也可以使用 ModelScope 从 https://www.modelscope.cn/models/RLinf/RLinf-OpenVLAOFT-Behavior 下载模型。

下载后，请确保在配置 yaml 文件中正确指定模型路径。

运行脚本
---------------

**1. 关键集群配置**

.. warning::

   注意，由于ISAAC-SIM的特殊行为，请尽量将env放置在从0开始的GPU上。
   否则，ISAAC-SIM可能会在某些GPU上卡住。

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

--------------

**2. 配置文件**

以 behavior 为例：

- OpenVLA-OFT + PPO:
  ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml``
- OpenVLA-OFT + GRPO:
  ``examples/embodiment/config/behavior_grpo_openvlaoft.yaml``

--------------

**3. 启动命令**

要使用选定的配置开始训练，请运行以下
命令：

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，要在 Behavior 环境中使用 PPO 算法训练 OpenVLA-OFT 模型，请运行：

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft


可视化和结果
-------------------------

**1. TensorBoard 日志记录**

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

--------------

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

--------------

**3. 视频生成**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

--------------

**4. WandB 集成**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "test_behavior"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


对于 Behavior 实验，我们受到了 
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_ 的启发， 
仅进行了少量修改。我们感谢作者发布开源代码。
