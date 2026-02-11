GR00T-N1.5模型强化学习训练
==================================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例提供了一份完整指南，介绍如何在LIBERO环境中使用RLinf框架，通过强化学习对GR00T-N1.5算法进行微调。内容涵盖从环境设置、核心算法设计到训练配置、评估和可视化的全过程，并提供可复现的命令和配置片段。

主要目标是开发一个能够执行机器人操作的模型，具体包括：

1. **视觉理解**：处理机器人摄像头拍摄的RGB图像。
2. **语言理解**：解读自然语言的任务描述。
3. **动作生成**：生成精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**：通过PPO算法结合环境反馈优化策略。

环境
-----------

**LIBERO环境**

- **环境**：基于robosuite（MuJoCo）构建的LIBERO仿真基准。
- **任务**：控制7自由度机械臂执行各种家庭操作技能（拾取放置、堆叠、打开抽屉、空间重排）。
- **观测**：由放置在工作区周围的离屏摄像头捕获的RGB图像（典型分辨率为128×128或224×224）。
- **动作空间**：7维连续动作——3D末端执行器位置控制（x、y、z）、3D旋转控制（横滚、俯仰、偏航）、夹爪控制（打开/关闭）

**任务描述格式**

GR00T-N1.5直接将环境提供的自然语言任务描述作为语言模型的输入。

**数据结构**

- **图像**：主视角和手腕视角的RGB张量，分别命名为“main_images”和“wrist_images”，形状为``[batch_size, 224, 224, 3]``
- **状态**：末端执行器的位置、姿态和夹爪状态
- **任务描述**：自然语言指令
- **奖励**：稀疏的成功/失败奖励

算法
---------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用GAE（广义优势估计）进行优势估计
   - 带比例限制的策略裁剪
   - 价值函数裁剪
   - 熵正则化

2. **GRPO（Group Relative Policy Optimization）**

   - 结合GR00T-N1.5的GRPO算法正在测试中，结果将在后续发布。

依赖安装
-----------------------

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

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env gr00t

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model gr00t --env maniskill_libero
   source .venv/bin/activate

模型下载
--------------

开始训练前，您需要下载相应的预训练模型。
目前我们支持四种libero任务：Spatial, Object, Goal, and Long。

**GR00T-N1.5少样本SFT模型下载**

.. code:: bash

   # 方法1：使用git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

   # 方法2：使用huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir RLinf-Gr00t-SFT-Spatial

其他任务的SFT模型下载: 
- `Libero-Object <https://huggingface.co/lixiang-95/RLinf-Gr00t-SFT-Object>`_
- `Libero-Goal <https://huggingface.co/lixiang-95/RLinf-Gr00t-SFT-Goal>`_
- `Libero-Long <https://huggingface.co/lixiang-95/RLinf-Gr00t-SFT-10>`_

--------------

GR00T-N1.5的预备知识
-----------------------------
此处介绍GR00T-N1.5的重要设计，以帮助用户更便捷地使用该模型。

**1. 模态配置（Modality Config）**

模态配置是GR00T-N1.5中一项关键且突出的设计特性。
通过定义统一的数据集接口，它使不同的机器人配置能够利用相同的数据集。例如，双臂数据集可通过这一创新设计用于训练单臂模型。为实现此功能，GR00T-N1.5采取了以下关键措施。


**1.1 增强的LeRobot数据集**

LeRobot数据集包含一个meta文件夹，其中详细记录了数据集的所有元数据。
GR00T-N1.5进一步定义了一个**modality.json**文件，用于确定数据集的数据接口。

**1.2 DataConfig类**

GR00T-N1.5引入了DataConfig类，用于描述模型训练所需的所有信息。
它将数据集和机器人配置解耦，使模型能够在不同机器人之间进行训练，而无需修改数据处理代码。该类还定义了所有数据模态的转换方式。

**1.3 具身标签（Embodiment Tag）**

具身标签是一个枚举值，用于确定训练过程中使用哪个DataConfig。模型还会根据此标签采用不同的状态和动作编码器/解码器。

---------------

微调后，GR00T-N1.5会生成一个``experiment_cfg/metadata.json``文件，其中包含所有模态配置和微调数据集的统计信息。
该文件对于GR00T-N1.5的推理和强化学习后训练至关重要。
更多细节请参考 `GR00T-N1.5官方仓库的getting_started/4_deeper_understanding.md <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/4_deeper_understanding.md>`__。

**2. 微调指南**

基于上述设计，除LIBERO外，在新环境中部署GR00T-N1.5之前，用户需要对其进行微调。
微调指南可在 `GR00T-N1.5官方仓库的getting_started/2_finetuning.ipynb <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/2_finetuning.ipynb>`_ 中找到。

---------------

运行脚本
---------------

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

在此处，您可以灵活配置env、rollout和actor组件的GPU数量。此外，通过在配置中设置``pipeline_stage_num = 2``，可以实现rollout与env之间的流水线重叠，提高rollout效率。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

您也可以重新配置放置方式以实现完全共享，即env、rollout和actor组件共享所有GPU。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

您还可以重新配置放置方式以实现完全分离，即env、rollout和actor组件各自使用独立的GPU，互不干扰，无需卸载功能。

--------------

**2. 模型关键参数配置**

**2.1 模型参数**

.. code:: yaml

  model:
     num_action_chunks: 5
     denoising_steps: 4
     rl_head_config:
       noise_method: "flow_sde"
       noise_level: 0.5
       disable_dropout: True

您可以调整noise_level和denoising_steps来控制噪声强度和流匹配步骤。
num_action_chunks决定了将用于前向仿真环境的未来步骤数量。
GR00T-N1.5的动作头包含dropout层，这会干扰对数概率的计算，因此需将disable_dropout设置为True，以将其替换为恒等层。
可通过noise_method选择不同的噪声注入方法。
我们提供两种选项：
`flow-sde <https://arxiv.org/abs/2505.05470>`__ 和
`flow-noise <https://arxiv.org/abs/2505.22094>`__。

**2.2 LoRA设置**

LoRA设置正在测试中，即将推出。

**3. 配置文件**

- GR00T-N1.5 + PPO + Libero-Spatial：
  ``examples/embodiment/config/libero_spatial_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Object：
  ``examples/embodiment/config/libero_object_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Goal：
  ``examples/embodiment/config/libero_goal_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Long：
  ``examples/embodiment/config/libero_10_ppo_gr00t.yaml``

--------------

**4. 启动命令**

要使用选定的配置开始训练，请运行以下命令之一：

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_object_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_goal_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_gr00t

--------------

可视化与结果
-------------------------

**1. TensorBoard日志**

.. code:: bash

   # 启动TensorBoard
   tensorboard --logdir ./logs --port 6006

--------------

**2. 关键监控指标**

- **训练指标**

  - ``actor/loss``：策略损失
  - ``actor/value_loss``：价值函数损失（PPO）
  - ``actor/grad_norm``：梯度范数
  - ``actor/approx_kl``：新旧策略之间的KL散度
  - ``actor/pg_clipfrac``：策略裁剪比例
  - ``actor/value_clip_ratio``：价值损失裁剪比例（PPO）

- **rollout指标**

  - ``rollout/returns_mean``：平均回合回报
  - ``rollout/advantages_mean``：平均优势值

- **环境指标**

  - ``env/episode_len``：平均回合长度
  - ``env/success_once``：任务成功率

--------------

**3. 视频生成**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

--------------

**4. WandB集成**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "libero_10_ppo_gr00t"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

--------------

**LIBERO结果**
~~~~~~~~~~~~~~~~~~

我们在LIBERO环境中使用PPO训练了GR00T-N1.5。其他结果（Flow-Noise的RL训练）将在近期发布。结果链接指向Hugging Face上的对应模型。
通过强化学习训练获得的结果如下：

.. list-table:: **GR00T-N1.5模型使用Flow-SDE方法在LIBERO上的结果**
   :header-rows: 1

   * - 模型
     - Spatial 
     - Object
     - Goal 
     - Long 
     - Average
     - Δ Avg.

   * - GR00T（少样本）
     - |huggingface| `41.4% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial>`_
     - |huggingface| `58.6% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Object>`_
     - |huggingface| `48.2% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Goal>`_
     - |huggingface| `61.9% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Long>`_
     - 52.5%
     - ---

   * - +PPO
     - |huggingface| `92.5% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Spatial-Step400>`_
     - |huggingface| `95.0% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Object-Step400>`_
     - |huggingface| `84.3% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Goal-Step500>`_
     - |huggingface| `86.3% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Long-Step300>`_
     - **89.5%**
     - **+37.0%**

我们想指出上述结果使用了与 :math:`\pi_0` 相同的超参数设置。这些发现主要展示了所提出RL训练框架的广泛适用性和鲁棒性。通过参数调优可以更进一步提升模型性能。