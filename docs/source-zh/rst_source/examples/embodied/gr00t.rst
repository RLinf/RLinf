GR00T模型强化学习训练
==================================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例提供了一份完整指南，介绍如何在LIBERO环境中使用RLinf框架，通过强化学习对GR00T模型进行微调。内容涵盖从环境设置、核心算法设计到训练配置、评估和可视化的全过程，并提供可复现的命令和配置片段。

.. note::

   RLinf 同时支持 GR00T-N1.5、GR00T-N1.6 和 GR00T-N1.7。N1.6 引入了流匹配动作头、FSDP 训练和更强的跨具身支持；N1.7 则进一步将官方 backbone 升级到 Cosmos-Reason2-2B / Qwen3-VL，并显著扩展了官方通用 state/action 空间。版本差异以 **N1.5** / **N1.6** / **N1.7** 标注区分。

环境
-----------

**LIBERO环境**

- **环境**：基于robosuite（MuJoCo）构建的LIBERO仿真基准。
- **任务**：控制7自由度机械臂执行各种家庭操作技能（拾取放置、堆叠、打开抽屉、空间重排）。

**N1.5:**

- **观测**：由放置在工作区周围的离屏摄像头捕获的RGB图像（典型分辨率为128×128或224×224）。
- **动作空间**：7维连续动作——3D末端执行器位置控制（x、y、z）、3D旋转控制（横滚、俯仰、偏航）、夹爪控制（打开/关闭）

**N1.6:**

- **观测**：由放置在工作区周围的离屏摄像头捕获的RGB图像（典型分辨率为128×128、224×224或256×256）。
- **动作空间**：环境原生提供7维连续动作。*注：GR00T-N1.6 在底层会通过具身标签将该7维动作统一零填充至128维的跨具身通用动作空间中。*

**N1.7:**

- **观测**：在当前 RLinf 的 LIBERO-N1.7 集成中，环境仍然提供主视角图像、手腕视角图像，以及一个原始 8 维的 LIBERO state 张量；RLinf 再把它们转换为官方 GR00T-N1.7 processor 所需的输入格式。
- **动作空间**：对 LIBERO 而言，环境侧动作仍然是7维连续控制；在官方 GR00T-N1.7 模型内部，动作会先按其动作表示进行解码，再映射回 LIBERO 的 7 维控制空间。

**任务描述格式**

GR00T 直接将环境提供的自然语言任务描述作为语言模型的输入。

**N1.5:**

**数据结构**

- **图像**：主视角和手腕视角的RGB张量，分别命名为"main_images"和"wrist_images"，形状为``[batch_size, 224, 224, 3]``
- **状态**：末端执行器的位置、姿态和夹爪状态
- **任务描述**：自然语言指令
- **奖励**：稀疏的成功/失败奖励

**N1.6:**

**数据结构**

- **图像**：主视角和手腕视角的连续RGB视频帧，通常命名为``main_images``和``wrist_images``。考虑到时间步历史，形状通常为``[batch_size, seq_len, 224, 224, 3]``。
- **状态**：末端执行器的位置、姿态和夹爪状态（在网络底层与视觉特征拼接作为状态表征）。
- **任务描述**：自然语言指令。
- **奖励**：用于PPO强化的稀疏奖励（成功为1，失败为0）。

**N1.7:**

**数据结构**

- **图像**：RLinf 当前从 LIBERO 读取 ``main_images`` 和 ``wrist_images``，并在调用官方 processor 之前，将它们转换为 GR00T-N1.7 的观测字段（例如 ``video.image`` 和 ``video.wrist_image``）。
- **状态**：RLinf 当前接收一个原始 8 维的 LIBERO state 张量，并将其拆分为 ``state.x``、``state.y``、``state.z``、``state.roll``、``state.pitch``、``state.yaw`` 和 ``state.gripper``。因此，当前 LIBERO 示例在环境边界仍沿用 RLinf/LIBERO 现有的 state 契约，而官方 N1.7 模型内部使用的是更大的通用 state/action 表示。
- **任务描述**：``task_descriptions`` 会映射为官方 processor 所需的 ``annotation.human.action.task_description``。
- **奖励**：用于 PPO 强化的稀疏奖励（成功为1，失败为0）。

算法
---------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用GAE（广义优势估计）进行优势估计
   - 带比例限制的策略裁剪
   - 价值函数裁剪
   - 熵正则化

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
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

**N1.5:**

.. code:: bash

   source switch_env gr00t

**N1.6:**

.. code:: bash

   source switch_env gr00t_n1d6

**N1.7:**

.. code:: bash

   source switch_env gr00t_n1d7

**选项 2：自定义环境**

**N1.5:**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加``--use-mirror``到下面的install.sh命令
   bash requirements/install.sh embodied --model gr00t --env maniskill_libero
   source .venv/bin/activate

**N1.6:**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加``--use-mirror``到下面的install.sh命令
   bash requirements/install.sh embodied --model gr00t_n1d6 --env maniskill_libero
   source .venv/bin/activate

**N1.7:**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加``--use-mirror``到下面的install.sh命令
   bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
   source .venv/bin/activate

模型下载
--------------

开始训练前，您需要下载相应的预训练模型。

**N1.5: GR00T-N1.5 少样本SFT模型下载**

目前我们支持四种libero任务：Spatial, Object, Goal, and Long。

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
- `Libero-Object <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Object>`_
- `Libero-Goal <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Goal>`_
- `Libero-Long <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-10>`_

**N1.6: GR00T-N1.6 SFT模型**

需要先运行RLinf提供的GR00T-N1.6的SFT，获得经过格式转换的模型，并将模型路径配置到指定的yaml文件中。

RLinf SFT的模型将会后续放出，敬请期待！

目前支持四种libero任务：Spatial, Object, Goal, 10。

**N1.7：当前临时使用官方 release checkpoint**

RLinf 目前还没有提供一个由本仓库产出的、专门用于这个 RL 示例的 GR00T-N1.7 SFT checkpoint。因此，在当前仓库状态下，维护中的 N1.7 RL 示例暂时直接使用官方发布的 LIBERO checkpoint 作为任务 checkpoint 起点。

换句话说：

- ``model_path`` 当前指向的是本地官方 ``nvidia/GR00T-N1.7-LIBERO`` checkpoint，而不是本仓库导出的 N1.7 SFT checkpoint。
- ``backbone_model_path`` 指向本地 ``Cosmos-Reason2-2B`` checkpoint，以便 actor、rollout 和 processor 全部离线运行。

对应官方 N1.7 文档里的核心升级方向：

- **Relative EEF Action Space**：N1.7 使用机器人与人类具身共享的 relative EEF 动作空间，以当前位姿增量而不是绝对目标来表达动作，这是其跨具身泛化能力的重要原因之一。
- **Human Video Pretraining**：N1.7 在多样机器人示范之外，还联合使用了 20K 小时 EgoScale 人类视频进行预训练，因此可以将从人类视频中学到的操作先验迁移到机器人控制。
- **Key Changes from N1.6**：N1.7 将 VLM backbone 升级为 ``Cosmos-Reason2-2B`` / Qwen3-VL，简化了数据处理流水线，并增强了 ONNX / TensorRT 导出支持。

在 RLinf 自己的 N1.7 SFT checkpoint 发布之前，可以先按下面的方式离线下载官方模型：

.. code:: bash

   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub

   # 下载 Cosmos-Reason2-2B backbone
   uv run hf download nvidia/Cosmos-Reason2-2B \
      --local-dir checkpoints/Cosmos-Reason2-2B

   # 下载 GR00T-N1.7-LIBERO 任务 checkpoint（仅 libero_spatial 所需文件）
   uv run hf download nvidia/GR00T-N1.7-LIBERO \
      --include "libero_spatial/config.json" \
                "libero_spatial/embodiment_id.json" \
                "libero_spatial/model-*.safetensors" \
                "libero_spatial/model.safetensors.index.json" \
                "libero_spatial/processor_config.json" \
                "libero_spatial/statistics.json" \
      --local-dir checkpoints/GR00T-N1.7-LIBERO

当前示例使用这组最小文件即可运行。如果你的本地 checkpoint 目录里还能保留 ``experiment_cfg/metadata.json``，建议一并保留；RLinf 在存在该文件时会优先使用它，但即使缺失也可以回退到 modality/config 推断。

当前维护中的 RLinf N1.7 RL 示例仅为 LIBERO Spatial。

--------------

GR00T 核心设计理念
-----------------------------

**N1.5:**

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

**2. 微调指南**

基于上述设计，除LIBERO外，在新环境中部署GR00T-N1.5之前，用户需要对其进行微调。
微调指南可在 `GR00T-N1.5官方仓库的getting_started/finetune_new_embodiment.md <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md>`_ 中找到。

微调后，GR00T-N1.5会生成一个``experiment_cfg/metadata.json``文件，其中包含所有模态配置和微调数据集的统计信息。
该文件对于GR00T-N1.5的推理和强化学习后训练至关重要。
更多细节请参考 `GR00T-N1.5官方仓库的getting_started/GR00T_inference.ipynb <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/GR00T_inference.ipynb>`__。

**N1.6:**

**1. 两阶段解耦训练范式**

RLinf 框架针对GR00T-N1.6采用了高度解耦的两阶段训练架构：

- **第一阶段（纯 SFT 预热）**：采用``Pure SFT Model``模式。模型完全脱离物理仿真环境，仅依赖离线专家数据集进行监督微调，专注拟合目标动作轨迹。
- **第二阶段（PPO 强化对齐）**：在SFT收敛的基础上，将模型载入基于FSDP的分布式Actor中，与仿真环境进行实时交互。

**2. 极简的局部微调策略**

为了在大幅节省显存的同时防止"灾难性遗忘"，框架默认采用"冻结主干"策略：

- **主干冻结**：在SFT和后续RL阶段中，视觉-语言主干网络将被严格锁定（``requires_grad=False``）。
- **专注动作头**：仅解冻动作输出头参与梯度更新。

**3. 流匹配动作生成（Flow-Matching Action Head）**

- 模型通过加噪与去噪的流匹配机制（Flow-SDE / Diffusion），直接在连续空间中生成高频动作块。
- 关键配置：通过 ``num_action_chunks`` 控制预测步长， ``denoising_steps`` 控制去噪深度。

**4. 跨具身泛化（Cross-Embodiment）**

- **具身标签（Embodiment Tag）**：依靠传入的配置标签（如``ROBOCASA_PANDA_OMRON``），系统能动态适配对应的状态编码器与动作空间。无论是单臂机械臂，还是四足机器人形态均可复用。

**5. FSDP 分布式并行架构**

- 底层系统针对Actor节点进行了重构（``EmbodiedFSDPActor``），能够跨GPU节点对模型权重、梯度与优化器状态进行分片切分（Sharding）。
- 鉴于GR00T-N1.6参数规模的显著增长，RLinf的Actor节点已全面重构，打破了传统DDP的单卡显存瓶颈，极大提升了吞吐量。

微调完成后，系统将在输出目录生成``metadata.json``等统计文件，保留推理和后续部署所需的关键模态信息。

**N1.7:**

**1. 官方 N1.7 的核心新变化**

- GR00T N1.7 建立在 N1.6 之上，引入了新的 VLM backbone 以及一系列代码层面的改进。
- **Relative EEF Action Space**：N1.7 采用机器人与人类具身共享的 relative end-effector 动作空间。相较于用绝对目标表达动作，使用相对当前位姿的增量表达能显著提升泛化能力，这也是其跨具身表现的重要原因之一。
- **Human Video Pretraining**：N1.7 联合使用了 20K 小时 EgoScale 人类视频和多样化机器人示范进行预训练。由于 relative EEF 动作表示在机器人和人类数据之间是一致的，模型能将从人类视频中学到的操作先验更直接地迁移到机器人控制。

**2. 相对 N1.6 的关键变化**

- 官方 backbone 升级为 ``nvidia/Cosmos-Reason2-2B``，采用 Qwen3-VL 风格架构，替代了 N1.6 所使用的 Eagle backbone。
- 官方 ``processing_gr00t_n1d7.py`` 路径相较 N1.6 简化了数据处理流水线。
- 官方 N1.7 还增强了完整的 ONNX / TensorRT 导出支持。
- 官方 N1.7 模型配置将默认通用上限提升为 ``max_state_dim=132``、``max_action_dim=132`` 和 ``action_horizon=40``。

**3. RLinf 当前的 checkpoint 策略**

- RLinf 目前还没有为这个 RL 示例提供一个由仓库自产出的 N1.7 SFT checkpoint。
- 因此，当前维护中的示例暂时使用官方发布的 ``GR00T-N1.7-LIBERO/libero_spatial`` checkpoint 作为 ``model_path``。

**4. RLinf 的 N1.7 接口适配**

- 在当前 RLinf 实现中，LIBERO 的原始 state 在转换前是 8 维，而官方 N1.7 模型内部使用的是更大的通用 state/action 表示。
- 当前 LIBERO 示例使用 ``embodiment_tag: libero_sim``，并在共享的环境动作工具中应用 LIBERO 的 gripper 约定。

**5. Checkpoint 与 processor 契约**

- RLinf 直接从 checkpoint 目录加载官方 processor。
- 在离线或镜像环境中，``backbone_model_path`` 可将官方 backbone id 重定向到本地 ``Cosmos-Reason2-2B`` snapshot。

**6. 本仓库中的 RL 训练契约**

- 当前维护中的 RLinf N1.7 RL 示例为 ``examples/embodiment/config/libero_spatial_ppo_gr00t_n1d7.yaml``。
- 当前 RL 设置使用 PPO 且 ``algorithm.loss_type: actor_critic``，因此训练时必须保证 ``actor.model.add_value_head`` 为 ``True``。
- 当前仓库中经过验证的 LIBERO 示例使用 ``num_action_chunks: 16`` 和 ``denoising_steps: 4``。

---------------

运行脚本
---------------

**1. 关键集群配置**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

您可以将env、rollout和actor组件的放置配置为共享所有GPU。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

您也可以灵活配置env、rollout和actor组件的GPU数量，并通过``pipeline_stage_num``实现rollout与env之间的流水线重叠。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

您还可以将组件完全分离，各自使用独立GPU，无需卸载功能。

--------------

**2. 模型关键参数配置**

**N1.5:**

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
可通过noise_method选择不同的噪声注入方法。我们提供两种选项：
`flow-sde <https://arxiv.org/abs/2505.05470>`__ 和
`flow-noise <https://arxiv.org/abs/2505.22094>`__。

**N1.6:**

**Actor 模型与动作头配置**

.. code:: yaml

   model:
      model_type: "gr00t_n1d6"
      add_value_head: True          # 强化学习关键：动态注入价值网络预测优势
      num_action_chunks: 16         # 每次推理预测的未来动作步数
      denoising_steps: 4            # 控制流匹配(Flow-Matching)去噪步数

**FSDP 分布式切片策略**

.. code:: yaml

   fsdp_config:
     wrap_policy:
       transformer_layer_cls_to_wrap:
         - "Qwen3DecoderLayer"
         - "Siglip2EncoderLayer"

**N1.7:**

**Actor 模型与动作头配置**

.. code:: yaml

   model:
      model_type: "gr00t_n1d7"
      add_value_head: True
      num_action_chunks: 16
      denoising_steps: 4

**运行时路径配置**

.. code:: yaml

   model:
      model_path: "/path/to/GR00T-N1.7-LIBERO/libero_spatial"
      backbone_model_path: "/path/to/Cosmos-Reason2-2B"

**PPO 与优化器超参数**

.. code:: yaml

   algorithm:
      adv_type: gae
      clip_ratio_high: 0.2
      gamma: 0.99
      gae_lambda: 0.95

   optim:
      lr: 5.0e-6
      value_lr: 1.0e-4
      clip_grad: 1.0

**3. 配置文件**

**N1.5:**

- GR00T-N1.5 + PPO + Libero-Spatial：
  ``examples/embodiment/config/libero_spatial_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Object：
  ``examples/embodiment/config/libero_object_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Goal：
  ``examples/embodiment/config/libero_goal_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Long：
  ``examples/embodiment/config/libero_10_ppo_gr00t.yaml``

**N1.6:**

- GR00T-N1.6 + PPO + Libero-Spatial：
  ``examples/embodiment/config/libero_spatial_ppo_gr00t_n1d6.yaml``

需要修改SFT后模型的路径：

.. code:: yaml

   model:
      model_path: "/path/to/RLinf-Gr00t-N1.6-RL-Spatial"

**N1.7:**

- GR00T-N1.7 + PPO + Libero-Spatial:
  ``examples/embodiment/config/libero_spatial_ppo_gr00t_n1d7.yaml``

需要修改模型路径和 backbone 路径：

.. code:: yaml

   model:
      model_path: "/path/to/GR00T-N1.7-LIBERO/libero_spatial"
      backbone_model_path: "/path/to/Cosmos-Reason2-2B"

--------------

**4. 启动命令**

**N1.5:**

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_object_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_goal_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_gr00t

**N1.6:**

.. code:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t_n1d6

**N1.7:**

.. code:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t_n1d7

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
       experiment_name: "libero_spatial_ppo_gr00t"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

--------------

**LIBERO结果**

**N1.5:**

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

**N1.6:**

.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
     <div style="flex: 0.5; text-align: center;">
       <img src="https://github.com/RLinf/misc/blob/main/pic/gr00t_1.6_ppo_success_rate.png?raw=true" style="width: 100%;"/>
       <p><em>GR00T-N1.6 SFT + PPO在 LIBERO_Spatial 的成功率曲线</em></p>
     </div>
   </div>

.. list-table:: **GR00T-N1.6 使用Flow-SDE方法在LIBERO Spatial上的结果**
   :header-rows: 1

   * - 模型
     - Spatial

   * - GR00T-N1.6 SFT
     - |huggingface| `70% <https://huggingface.co/RLinf/RLinf-Gr00t-N1.6-RL-Spatial>`_

   * - +PPO
     - |huggingface| `82% <https://huggingface.co/RLinf/RLinf-Gr00t-N1.6-RL-Spatial-Step500>`_


**N1.7:**

.. list-table:: **GR00T-N1.7 使用Flow-SDE方法在LIBERO Spatial上的结果**
   :header-rows: 1

   * - 模型
     - Spatial

   * - GR00T-N1.7 PPO
     - |huggingface| TODO
