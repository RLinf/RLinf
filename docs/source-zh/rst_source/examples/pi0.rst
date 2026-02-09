π\ :sub:`0`\和π\ :sub:`0.5`\ 模型强化学习训练
===============================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例展示 RLinf 框架对 π\ :sub:`0`\和π\ :sub:`0.5`
算法进行强化学习微调的完整指南。示例覆盖从环境输入、核心算法、训练脚本配置到评估与可视化的完整流程，并提供可复现的命令和配置片段。 

详细技术报告参考论文: `πRL: ONLINE RL FINE-TUNING FOR FLOW-BASED VISION-LANGUAGE-ACTION MODELS <https://arxiv.org/abs/2510.25889>`__.

主要目标是让模型具备以下能力：

1. **视觉理解**\ ：处理来自机器人相机的 RGB 图像。
2. **语言理解**\ ：理解自然语言的任务描述。
3. **动作生成**\ ：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**\ ：结合环境反馈，使用 PPO 优化策略。

环境
----

**LIBERO环境**

-  **Environment**\ ：基于 robosuite（MuJoCo）的 LIBERO 仿真基准
-  **Task**\ ：指挥一台 7 自由度机械臂完成多种家居操作技能（抓取放置、叠放、开抽屉、空间重排等）
-  **Observation**\ ：工作区周围离屏相机采集的 RGB 图像（常见分辨率 128×128 或 224×224）
-  **Action Space**\ ：7 维连续动作
   - 末端执行器三维位置控制（x, y, z）
   - 三维旋转控制（roll, pitch, yaw）
   - 夹爪控制（开/合）

**ManiSkill3 环境**

-  **Environment**\ ：ManiSkill3 仿真平台
-  **Task**\ ：控制机械臂抓取多种物体
-  **Observation**\ ：第三人称相机的 RGB 图像（224×224）
-  **Action Space**\ ：7 维连续动作
   - 三维位置控制（x, y, z）
   - 三维旋转控制（roll, pitch, yaw）
   - 夹爪控制（开/合）

**任务描述格式**

   π\ :sub:`0`\ 和 π\ :sub:`0.5`\ 使用环境给出的原始任务描述直接作为语言模型的输入。

**数据结构**

-  **Images**\ ：包含主视角图像和腕部视角图像，均为RGB 张量
   ``[batch_size, 224, 224, 3]``
-  **States**\ ：在LIBERO当中是末端执行器的位姿（位置 + 姿态）以及夹爪状态, 在ManiSkill3当中是机器人关节角度
-  **Task Descriptions**\ ：自然语言指令
-  **Rewards**\ ：任务成功/失败的稀疏奖励

算法
----

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   -  使用 GAE（Generalized Advantage Estimation）进行优势估计
   -  基于比率的策略裁剪
   -  价值函数裁剪
   -  熵正则化

2. **GRPO（Group Relative Policy Optimization）**

   -  对于每个状态/提示，策略生成 *G* 个独立动作
   -  以组内平均奖励为基线，计算每个动作的相对优势

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

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env openpi

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

模型下载
--------

在开始训练之前，您需要下载相应的预训练模型。例如，针对 LIBERO 环境的 Spatial、Object、Goal 类型的任务，您可以通过如下方式进行下载：

.. code:: bash

   # 下载 Spatial-Object-Goal 模型（选择以下任一方式）
   # 方式1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

   # 方式2：使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

或者，您可以从 ModelScope 下载该模型 https://www.modelscope.cn/models/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal。

当然，RLinf 也提供了针对其他环境的预训练模型。模型列表如下：

.. list-table:: **π**\ :sub:`0` **模型列表**
   :header-rows: 1
   :widths: 15 25 15 12 12

   * - 环境
     - 任务说明
     - SFT Model
     - Flow-SDE
     - Flow-Noise

   * - LIBERO
     - Spatial, Object, Goal
     - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT>`__
     - -
     - -

   * - LIBERO
     - Long
     - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Long-SFT>`__
     - -
     - -

   * - ManiSkill3
     - 多任务
     - |huggingface| `38.4% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-SFT>`__
     - |huggingface| `78.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowSDE>`__
     - |huggingface| `77.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowNoise>`__

   * - MetaWorld
     - MT50
     - |huggingface| `50.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT>`__
     - |huggingface| `78.1% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowSDE>`__
     - |huggingface| `85.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise>`__

   * - CALVIN
     - ABC-D
     - |huggingface| `57.5% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT>`__
     - |huggingface| `61.7% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowSDE>`__
     - |huggingface| `59.9% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowNoise>`__

.. list-table:: **π**\ :sub:`0.5` **模型列表**
   :header-rows: 1
   :widths: 15 25 15 12 12
   :align: left

   * - 环境
     - 任务说明
     - SFT Model
     - Flow-SDE
     - Flow-Noise

   * - LIBERO
     - Spatial, Object, Goal, Long
     - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__
     - -
     - -

   * - ManiSkill3
     - 多任务
     - |huggingface| `40.1% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT>`__
     - |huggingface| `90.9% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE>`__
     - |huggingface| `89.7% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise>`__

   * - MetaWorld
     - MT50
     - |huggingface| `43.8% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT>`__
     - |huggingface| `70.7% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE>`__
     - |huggingface| `66.1% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowNoise>`__

   * - CALVIN
     - ABC-D
     - |huggingface| `61.3% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT>`__
     - |huggingface| `87.0% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE>`__
     - |huggingface| `84.5% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowNoise>`__

下载完成后，请确保在配置文件中正确指定模型路径。

运行脚本
--------

**1. 运行关键参数配置**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

你可以灵活配置 env、rollout、actor 三个组件使用的 GPU 数量。
此外，在配置中设置 ``pipeline_stage_num = 2``\ ，可实现 **rollout 与
env** 之间的流水线重叠，从而提升 rollout 效率。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

你也可以重新配置 Placement，实现 **完全共享**\ ：env、rollout、actor
三个组件共享全部 GPU。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

你还可以重新配置 Placement，实现 **完全分离**\ ：env、rollout、actor
各用各的 GPU，互不干扰，这样就不需要 offload 功能。

**2. 模型关键参数配置**

**2.1 模型参数**

.. code:: yaml

   openpi:
     noise_level: 0.5 # flow_sde 的默认噪声强度
     noise_logvar_range: [0.08, 0.16] # flow_noise 的默认可学习噪声范围
     action_chunk: ${actor.model.num_action_chunks}
     num_steps: ${actor.model.num_steps}
     train_expert_only: True
     action_env_dim: ${actor.model.action_dim}
     noise_method: "flow_sde" # flow_sde, flow_noise
     add_value_head: False
     pi05: False 
     value_after_vlm: False

- 通过 ``num_steps`` 设置不同的流匹配步数。

- 通过修改 ``noise_method`` 使用不同的加噪方式。我们提供\ `flow_sde <https://arxiv.org/abs/2505.05470>`__\ 和\ `flow_noise <https://arxiv.org/abs/2505.22094>`__\ 两种方式。其中 ``noise_level`` 用于控制 ``flow_sde`` 的加噪强度，``noise_logvar_range`` 用于控制 ``flow_noise`` 的可学习噪声范围。

- 通过设置 ``pi05: True`` 启用 π\ :sub:`0.5`\ 模型。

- 通过 ``value_after_vlm`` 控制 critic 的位置：当该参数为 True 时，critic 接入到 VLM 模块的输出后；为 False 时，critic 的输入为 action expert 模块的输出。

**2.2 算法配置**

在论文中，我们提供 flow-noise 和 flow-sde 两种技术方案来微调 π\ :sub:`0`\ 和 π\ :sub:`0.5`\ 模型。具体而言，你可以通过切换如下配置来选择不同的技术方案：

.. code:: yaml
   
   algorithm:
      entropy_bonus: 0.0 # 熵正则化系数，flow-sde 设置为0.0，flow-noise 设置为0.005
   openpi:
     noise_method: "flow_sde" # [flow_sde,flow_noise] 噪声注入方式，flow-sde 通过ode-sde转换引入噪声，flow-noise 引入噪声网络注入噪声
     noise_level: 0.5 # flow-sde 的噪声强度
     noise_logvar_range: [0.08, 0.16] # 针对 flow-noise 的可学习噪声范围
     joint_logprob: False # 是否优化联合概率密度函数，对于flow-sde，请设置为False，对于flow-noise，请设置为True

例如，针对 flow-sde 的完整参数设置，可以参考 ``libero_spatial_ppo_openpi.yaml``；针对 flow-noise 的完整参数设置，可以参考 ``maniskill_ppo_openpi.yaml``。

**2.3 LoRA设置**

.. code:: yaml

   model:
     is_lora: True
     lora_rank: 8
     gradient_checkpointing: False

如果你想使用LoRA（Low-Rank Adaptation）对VLM部分进行参数高效微调，请设置 ``is_lora: True`` 并配置 ``lora_rank`` 参数。需要注意的是，当前\ **不支持**\ 启用梯度检查点，请保持该参数为 ``gradient_checkpointing: False``。

⭐ **2.4 最小测试案例** ⭐

如果你遇到OOM报错或者想用尽可能少的资源实现一个最小测试案例，可以参考 ``libero_spatial_ppo_openpi_quickstart.yaml``。
相比于标准的任务配置，我们主要做了以下修改：

.. code:: yaml

   rollout_epoch: 8 -> 2
   total_num_envs: 64 -> 32
   micro_batch_size: 128 -> 64
   global_batch_size: 2048 -> 256
   lr: 5e-6 -> 1e-6
   actor.enable_offload: False -> True
   rollout.enable_offload: False -> True

在4张H100 GPU上，我们对比了标准参数和最小测试参数的结果，可以发现它们在相同时间下的性能几乎是持平的：（最小测试参数尽管每一轮优化的时间更快，但是收敛速度更慢）

.. image:: https://github.com/user-attachments/assets/80d098f6-5286-4ff4-89be-547f43a4dc86
   :alt: 最小测试案例对比
   :width: 95%
   :align: center

同时，如果你在最小参数配置下仍然遇到了OOM问题，我们提供如下解决方案：

**如果是rollout阶段遇到OOM问题：**

- 可以尝试将渲染引擎从 ``egl`` 替换为 ``osmesa``
- 进一步减少 ``total_num_envs``，从32减少为16，但是增加 ``rollout_epoch`` 从2为4，以保证每轮rollout环境总数一致
- 检查actor的 ``enable_offload`` 是否开启，如果是 ``False`` 则设置为 ``True``

**如果是actor阶段遇到OOM问题：**

- 可以尝试减少 ``micro_batch_size``，从64减少为32，保持 ``global_batch_size`` 为256不变
- 检查rollout的 ``enable_offload`` 是否开启，如果是 ``False`` 则设置为 ``True``

.. note::

   如果遇到 ``micro_batch_size`` 和 ``global_batch_size`` 不匹配的问题，需要保证 ``global_batch_size`` 是 ``micro_batch_size`` × GPU数量 的整数倍。

**2.5 模型评估** 

针对SFT或RL训练后的模型，我们提供两种评估方式：

- 使用RLinf统一的评估脚本，参考 `VLA评估文档 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/vla-eval.html>`__ 进行评估，这种方式支持并行环境评估，速度快，但是只支持输出整个任务的成功率。

.. note::

   ``Metaworld`` 暂时不支持设置 ``env.eval.auto_reset=True`` 的评估模式，建议使用单个脚本文件进行模型评估。

- 使用单个脚本文件进行模型评估，参考示例 `README.md <https://github.com/RLinf/RLinf/blob/main/toolkits/eval_scripts_openpi/README.md>`__，这种方式的评估脚本和 ``openpi`` 官方提供的评估脚本一致，支持输出每个子任务的成功率，但是速度较慢。

**3. 配置文件**

   以libero-10为例，对应π\ :sub:`0`\ 和π\ :sub:`0.5`\ 的配置文件：

- π\ :sub:`0`\ + PPO:
   ``examples/embodiment/config/libero_10_ppo_openpi.yaml``
- π\ :sub:`0`\ + GRPO:
   ``examples/embodiment/config/libero_10_grpo_openpi.yaml``
- π\ :sub:`0.5`\ + PPO:
   ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml``
- π\ :sub:`0.5`\ + GRPO:
   ``examples/embodiment/config/libero_10_grpo_openpi_pi05.yaml``

**4. 启动命令**

选择配置后，运行以下命令开始训练：

::

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 LIBERO 环境中使用 PPO 训练 π\ :sub:`0`\ 模型：

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi_quickstart

可视化与结果
------------

**1. TensorBoard 日志**

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

-  **训练指标**\ ：

   -  ``actor/loss``\ ：策略损失
   -  ``actor/value_loss``\ ：价值函数损失（PPO）
   -  ``actor/grad_norm``\ ：梯度范数
   -  ``actor/approx_kl``\ ：更新前后策略 KL 值
   -  ``actor/pg_clipfrac``\ ：策略损失裁减比例
   -  ``actor/value_clip_ratio``\ ：价值损失裁剪比例（PPO）

-  **Rollout 指标**\ ：

   -  ``rollout/returns_mean``\ ：平均回合回报
   -  ``rollout/advantages_mean``\ ：平均优势值

-  **环境指标**\ ：

   -  ``env/episode_len``\ ：平均回合长度
   -  ``env/success_once``\ ：任务完成率

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
       experiment_name: "libero_10_ppo_openpi"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

LIBERO 结果
~~~~~~~~~~~

我们在 LIBERO 环境中使用 PPO 和 GRPO 训练了 π\ :sub:`0`\ 和 π\ :sub:`0.5`\。通过 RL 训练所获得的结果如下：

.. list-table:: **π**\ :sub:`0` **在 LIBERO 环境中的训练结果**
   :header-rows: 1

   * - Model
     - Spatial 
     - Object
     - Goal 
     - Long 
     - Average
     - Δ Avg.

   * - π\ :sub:`0`\ (few-shot)
     - 65.3%
     - 64.4%
     - 49.8%
     - 51.2%
     - 57.6%
     - ---

   * - +GRPO
     - 97.8%
     - 97.8%
     - 83.2%
     - 81.4%
     - 90.0%
     - +32.4

   * - +PPO
     - **98.4%**
     - **99.4%**
     - **96.2%**
     - **90.2%**
     - **96.0%**
     - **+38.4**

.. list-table:: **π**\ :sub:`0.5` **在 LIBERO 环境中的训练结果**
   :header-rows: 1

   * - Model
     - Spatial 
     - Object
     - Goal 
     - Long 
     - Average
     - Δ Avg.

   * - π\ :sub:`0.5`\ (few-shot)
     - 84.6%
     - 95.4%
     - 84.6%
     - 43.9%
     - 77.1%
     - ---

   * - +GRPO
     - 97.4%
     - 99.8%
     - 91.2%
     - 77.6%
     - 91.5%
     - +14.4

   * - +PPO
     - **99.6%**
     - **100%**
     - **98.8%**
     - **93.0%**
     - **97.9%**
     - **+20.8**

MetaWorld 结果
~~~~~~~~~~~~~~~~~
有关 MetaWorld 结果，请查看 `MetaWorld 页面 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/metaworld.html>`__。


CALVIN 结果
~~~~~~~~~~~
有关 CALVIN 结果，请查看 `CALVIN 页面 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/calvin.html>`__。
