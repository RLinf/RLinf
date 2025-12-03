π\ :sub:`0`\和π\ :sub:`0.5`\ 模型强化学习训练
===============================================

本示例展示如何在 LIBERO 环境中，使用 RLinf 框架对 π\ :sub:`0`\和π\ :sub:`0.5`
算法进行强化学习微调的完整指南。示例覆盖从环境输入、核心算法、训练脚本配置到评估与可视化的完整流程，并提供可复现的命令和配置片段。

主要目标是让模型具备以下能力：

1. **视觉理解**\ ：处理来自机器人相机的 RGB 图像。
2. **语言理解**\ ：理解自然语言的任务描述。
3. **动作生成**\ ：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**\ ：结合环境反馈，使用 PPO 优化策略。

环境
----

**LIBERO环境**

-  **Environment**\ ：基于 robosuite（MuJoCo）的 LIBERO 仿真基准
-  **Task**\ ：指挥一台 7
   自由度机械臂完成多种家居操作技能（抓取放置、叠放、开抽屉、空间重排等）
-  **Observation**\ ：工作区周围离屏相机采集的 RGB 图像（常见分辨率
   128×128 或 224×224）
-  **Action Space**\ ：7 维连续动作 - 末端执行器三维位置控制（x, y, z）
   - 三维旋转控制（roll, pitch, yaw） - 夹爪控制（开/合）

**任务描述格式**

   π\ :sub:`0`\ 和 π\ :sub:`0.5`\ 使用环境给出的原始任务描述直接作为语言模型的输入

**数据结构**

-  **Images**\ ：包含主视角图像和腕部视角图像，均为RGB 张量
   ``[batch_size, 3, 224, 224]``
-  **States**\ ：末端执行器的位姿（位置 + 姿态）以及夹爪状态
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

模型下载
--------

在开始训练之前，您需要下载相应的预训练模型。根据您要使用的算法类型，我们提供了不同的模型选择：

**π**\ :sub:`0`\ **模型下载**

π\ :sub:`0`\ 根据任务类型提供两个不同的模型选项：

**Option #1 RLinf-Pi0-SFT-Spatial-Object-Goal 模型**

该模型专门用于处理 object、goal、spatial 类型的任务。

.. code:: bash

   # 下载 Spatial-Object-Goal 模型（选择以下任一方式）
   # 方式1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal

   # 方式2：使用 huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal

或者，您可以从ModelScope下载该模型 https://www.modelscope.cn/models/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal。

**Option #2 RLinf-Pi0-SFT-Long 模型**

该模型专门用于处理 Long（libero10）类型任务。

.. code:: bash

   # 下载 Long 模型（选择以下任一方式）
   # 方式1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-SFT-Long

   # 方式2：使用 huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-SFT-Long

或者，您可以从ModelScope下载该模型 https://www.modelscope.cn/models/RLinf/RLinf-Pi0-SFT-Long。

**π**\ :sub:`0.5`\ **模型下载**

π\ :sub:`0.5`\ 提供一个统一的模型，该模型适用于所有类型的任务，包括 object、goal、spatial 和 Long 类型任务。

.. code:: bash

   # 方式1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-SFT

   # 方式2：使用 huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-SFT

或者，您可以从ModelScope下载该模型 https://www.modelscope.cn/models/RLinf/RLinf-Pi05-SFT。

**模型选择指南**

- 如果您要使用π\ :sub:`0`\ 模型训练**object、goal、spatial** 类型的任务，请使用 `RLinf-Pi0-SFT-Spatial-Object-Goal` 模型
- 如果您要使用π\ :sub:`0`\ 模型训练 **libero10** 的 Long 类型任务，请使用 `RLinf-Pi0-SFT-Long` 模型
- 如果您要使用π\ :sub:`0.5`\ 模型训练所有类型的任务，请使用 `RLinf-Pi05-SFT` 模型

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
使用上述配置，可以让 env 与 rollout 之间流水线重叠，并与 actor 共享。
此外，在配置中设置 ``pipeline_stage_num = 2``\ ，可实现 **rollout 与
actor** 之间的流水线重叠，从而提升 rollout 效率。

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
各用各的 GPU、互不干扰， 这样就不需要 offload 功能。

**2. 模型关键参数配置**

**2.1 模型参数**

.. code:: yaml

   openpi:
     noise_level: 0.5
     action_chunk: ${actor.model.num_action_chunks}
     num_steps: ${actor.model.num_steps}
     train_expert_only: True
     action_env_dim: ${actor.model.action_dim}
     noise_method: "flow_sde"
     add_value_head: False
     pi05: False 
     value_after_vlm: False

你可以通过配置 ``noise_level`` 以及 ``num_steps`` ，设置不同的加噪强度以及流匹配步数。

你可以通过修改 ``noise_method`` 使用不同的加噪方式。我们提供\ `flow_sde <https://arxiv.org/abs/2505.05470>`__\ 和\ `flow_noise <https://arxiv.org/abs/2505.22094>`__\ 两种方式。

你可以通过设置 ``pi05: True`` 启用π\ :sub:`0.5`\模式，通过 ``value_after_vlm`` 参数控制state输入路径：当该参数为 True 时，state 特征输入至 VLM 模块（为 π\ :sub:`0.5`\ 的默认配置）；为 False 时，state 特征输入至 action expert 模块（为 π\ :sub:`0`\ 的默认配置）。

**2.2 LoRA设置**

.. code:: yaml

   model:
     is_lora: True
     lora_rank: 8
     gradient_checkpointing: False

如果你想使用LoRA（Low-Rank Adaptation）对VLM部分进行参数高效微调，请设置 ``is_lora: True`` 并配置 ``lora_rank`` 参数。需要注意的是，当前\ **不支持**\ 启用梯度检查点，请保持该参数为 ``gradient_checkpointing: False``。



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

   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi

可视化与结果
------------

**1. TensorBoard 日志**

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

-  **训练指标**\ ：

   -  ``actor/loss``\ ：策略损失
   -  ``actor/value_loss``\ ：价值函数损失(PPO)
   -  ``actor/grad_norm``\ ：梯度范数
   -  ``actor/approx_kl``: 更新前后策略KL值
   -  ``actor/pg_clipfrac``: 策略损失裁减比例
   -  ``actor/value_clip_ratio``: 价值损失裁剪比例(PPO)

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
       experiment_name: "test_openpi"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

LIBERO 结果
~~~~~~~~~~~

我们在 LIBERO 环境中使用 PPO 和GRPO训练了π\ :sub:`0`\和π\ :sub:`0.5`\。通过 RL训练所获得的结果如下：

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
