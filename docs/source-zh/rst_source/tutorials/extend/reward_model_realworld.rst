Reward Model 使用指南（真机）
===============================

本文档介绍如何在真实世界的 Franka 机械臂上直接采集并预处理 reward model 训练数据集。
支持两种数据采集方式：**通用键盘标注方式** 和 **固定位姿方式** （通过预定的目标位姿驱动 episode 成功/失败）。

在开始前，强烈建议先阅读以下文档：

1. :doc:`../../examples/embodied/franka` 以熟悉 Franka 机械臂真机训练全流程。
2. :doc:`reward_model` 以了解 RLinf 中标准的 reward model 工作流（通过 ``pickle`` 采集数据、离线预处理、训练、RL 推理）。
3. :doc:`../../examples/embodied/franka_reward_model` 以了解在训练好 reward model 后如何接入真机 RL 流程。

工作流概览
-----------------

方式一将数据采集、标注和数据集生成整合为一次端到端运行；方式二采用简化的两步式流程。

.. code-block:: text

   真机数据集采集（本指南）
   ├── 方式一：键盘标注（通用）
   │   1. 使用 SpaceMouse / 键盘遥操作启动单个 RealWorld episode。
   │   2. 按 'c'（成功）或 'a'（失败）标注每一帧。
   │   3. 达到阈值或 max_steps 时停止。
   │   4. 对 fail:success 比例进行采样，并划分训练/验证集。
   │   5. 直接保存 train.pt / val.pt（无中间 .pkl 文件）。
   │
   └── 方式二：固定位姿（目标驱动）
       1. 配置目标末端执行器位姿（无需键盘标注）。
       2. 机器人到达目标位姿时 episode 自动终止。
       3. 保存 episode 轨迹为 .pkl 文件。
       4. 从 episode 轨迹中自动提取成功/失败帧。
       5. 通过 preprocess_reward_dataset.py 预处理并生成 train.pt / val.pt。

预备工作
------------

请根据 :doc:`../../examples/embodied/franka` 中的 **Prerequisites** 和 **Hardware Setup** 章节，
完成机器人连接和环境验证步骤。

数据采集
------------

方式一：键盘标注（通用）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

此方式通过键盘在实时 episode 中手动标注每一帧，适用于任何操作任务。

**配置文件** — ``examples/reward/config/realworld_collect_dataset.yaml``，
环境参数从 ``env/realworld_bin_relocation.yaml`` 继承：

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 1
     component_placement:
       env:
         node_group: franka
         placement: 0
     node_groups:
       - label: franka
         node_ranks: 0
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 0

   runner:
     task_type: embodied
     logger:
       log_path: null
       project_name: rlinf
       experiment_name: "collect-dataset"
       logger_backends: ["tensorboard"]
     num_success_frames: 50    # 目标采集的成功帧数
     num_fail_frames: 150      # 目标采集的失败帧数
     val_split: 0.2            # 用于验证集的帧比例
     fail_success_ratio: 2.0   # 训练集后处理时将失败帧下采样至 success * ratio
     random_seed: 42

   env:
     group_name: "EnvGroup"
     eval:
       no_gripper: False
       use_spacemouse: True
       max_episode_steps: 10000
       keyboard_reward_wrapper: single_stage
       override_cfg:
         target_ee_pose: TARGET_EE_POSE

**关键配置字段说明：**

- ``runner.num_success_frames`` / ``runner.num_fail_frames`` — 目标采集帧数。两个阈值均达到时停止采集。
- ``runner.val_split`` — 所有标注帧中用于验证集的比例。
- ``runner.fail_success_ratio`` — 训练集后处理阶段，失败帧会被下采样，使 ``num_fail = num_success * fail_success_ratio``。设为 ``0`` 可禁用下采样。
- ``env.eval.keyboard_reward_wrapper`` — 设为 ``single_stage``（或任务对应的 ``stage``）以启用键盘标注界面。
- ``env.eval.use_spacemouse`` — 是否使用 SpaceMouse 进行遥操作（step info 中的 ``intervene_action`` 会覆盖默认零动作）。
- ``env.eval.override_cfg.target_ee_pose`` — 任务的目标末端执行器位姿。

**启动命令：**

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh

或者显式指定配置名称：

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset

终端会实时打印进度条：

.. code-block:: text

   success: 12/50 [############----------------]  fail: 28/150 [#####################-----------]

在 episode 过程中使用以下按键：

- ``c`` — 将当前帧标注为**成功**。
- ``a`` — 将当前帧标注为**失败**。
- ``keyboard_reward_wrapper`` 中的键盘操作也会控制 episode 是否继续或重置。

当 ``num_success_frames`` 和 ``num_fail_frames`` 两个阈值均达到后，
脚本自动停止、划分数据并保存 ``.pt`` 文件。


方式二：固定位姿（目标驱动）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

此方式专为**固定目标位姿**的任务设计（例如到达预定箱体位置）。
无需手动键盘标注，episode 会根据机器人是否到达配置的 ``target_ee_pose`` 自动驱动成功/失败判定。
可以设置 ``success_hold_steps``，要求机器人在目标位姿保持一定步数后才判定为成功，
有助于采集更多样的成功样本。

此方式的数据采集流程同 :doc:`../../examples/embodied/franka_reward_model`，
但预处理步骤与方式一相同，使用同一脚本。


步骤 1：固定位姿 Reward 数据采集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了得到高质量的 reward model，需要采集更多的数据用来训练和评估。
在上述专家轨迹采集的基础上，进一步对采集脚本做以下修改：

将配置中的 ``success_hold_steps`` 字段增大，以便在有限的采集轮次内得到更多的成功数据。
机械臂末端在到达目标位姿后不会立刻判定为成功并重置，
而是需要到达目标位姿并保持一定步数（``success_hold_steps``）后才会判定为成功。
如果中途退出成功状态，会重新开始计数。

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

采集技巧：

- 请尽量缓慢移动机械臂，以便获得更多样的失败样本。
- 在到达目标位姿时，在保持目标位姿的前提下进行小范围移动，以便获得更多样的成功样本。

步骤 2：预处理为 Reward Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

采集好的 ``.pkl`` episode 通过 ``preprocess_reward_dataset.py`` 转换为 ``train.pt`` / ``val.pt``。
建议调高 ``fail-success-ratio`` 至 ``3``：

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data \
       --fail-success-ratio 3

生成文件如下：

.. code-block:: text

   logs/xxx/processed_reward_data/
   ├── train.pt
   └── val.pt

生成的 ``.pt`` 文件符合 ``RewardDatasetPayload`` 约定的标准格式：

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],
       "metadata": dict[str, Any],
   }

其中：

- ``images`` — 训练样本图像。
- ``labels`` — 二分类标签（1 = 成功，0 = 失败）。
- ``metadata`` — 原始数据路径、采样参数、划分比例等信息。


输出
~~~~~~

采集完成后（两种方式均适用），两个 ``.pt`` 文件会保存到 ``runner.logger.log_path``
（默认为 Hydra run dir）：

.. code-block:: text

   logs/<timestamp>-collect-dataset/
   ├── train.pt
   └── val.pt
   └── run_collect_process.log   # （仅方式一）

每个 ``.pt`` 文件遵循 ``RewardDatasetPayload`` 约定的标准格式：

.. code-block:: python

   {
       "images": list[torch.Tensor], 
       "labels": list[int],             # 1 = 成功，0 = 失败
       "metadata": dict,                # 采集统计信息和配置参数
   }

``metadata`` 字典包含以下字段：

- ``num_success_frames`` / ``num_fail_frames`` — 比例采样前的原始帧数。
- ``fail_success_ratio`` / ``val_split`` / ``random_seed`` — 采样参数。
- ``num_train_samples`` / ``num_val_samples`` — 最终数据集大小。

生成的 ``.pt`` 文件可直接用于 ``RewardBinaryDataset`` 进行训练，
具体用法与 :doc:`reward_model` 第 2 节描述一致。

数据采集方式对比
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * -
     - 键盘标注
     - 固定位姿（目标驱动）
   * - **标注方式**
     - 手动逐帧（``c`` / ``a``）
     - 自动（episode 成功/失败信号）
   * - **Episode 终止**
     - 由键盘封装器驱动
     - 由到达 ``target_ee_pose`` 驱动
   * - **成功保持**
     - 不适用
     - ``success_hold_steps`` 捕获多样成功样本
   * - **输出流程**
     - 直接生成 .pt（一个脚本）
     - ``.pkl`` episode → ``preprocess_reward_dataset.py`` → .pt
   * - **适用场景**
     - 任意操作任务
     - 具有固定目标位姿的任务

Reward Model 训练
-------------------

完成以上步骤后，继续参考 :doc:`reward_model` 第 2 节（**Reward Model 训练**）
使用生成的 ``train.pt`` / ``val.pt`` 文件进行 reward model 训练。

训练好 reward model 后，有两种方式在真机上使用：

- **真机遥操作 + 在线推理** （见下文）——使用 SpaceMouse 遥操作机械臂，
  同时让 reward model 在 GPU 节点上运行，实时向终端输出成功概率。
  无需启动完整 RL 训练循环。
- **真机 RL 训练** （参见 :doc:`../../examples/embodied/franka_reward_model`）——
  将 reward model 接入物理 Franka 上的完整 RL 训练循环。

真机遥操作 + 在线 Reward Model 推理
--------------------------------------------

获得 reward model checkpoint 后，``examples/reward/eval_realworld_teleop.py`` 提供了一种
遥操作模式：SpaceMouse 控制机器人运动，reward model 在 GPU 节点上运行，
实时在终端打印每步成功概率。

此功能的适用场景：

- 对 reward model 在真实机器人观测上的准确性进行冒烟测试（sanity check）。
- 采集符合人类判断的成功/失败数据，用于进一步扩充数据集。
- 定性评估 reward model 对当前场景的泛化能力。

集群配置
------------

遥操作脚本需要**两个节点**：一个用于 Franka 机器人，一个用于运行 reward model 推理的 GPU：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka
         placement: 0
       reward:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

Reward worker 被部署在 GPU 节点（``"4090"``）上，与机器人节点（``franka``）上的遥操作 worker 分离。
这是一种解聚式部署（disaggregated placement）。

配置文件
------------

默认配置为 ``examples/reward/config/realworld_teleop.yaml``，
环境参数从 ``env/realworld_bin_relocation.yaml`` 继承：

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka
         placement: 0
       reward:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   env:
     group_name: "EnvGroup"
     eval:
       no_gripper: True
       use_spacemouse: True
       max_episode_steps: 10000
       override_cfg:
         target_ee_pose: TARGET_EE_POSE
         camera_serials: ["0123456789"]

   reward:
     use_reward_model: True
     use_reward_prob: True    # 打印每步原始 sigmoid 概率到终端
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.2
     model:
       model_path: path/to/reward_model_checkpoint
       model_type: "resnet"
       arch: "resnet18"
       image_size: [3, 128, 128]

关键配置字段说明：

- ``reward.use_reward_model: True`` — 启用 reward model 推理。
- ``reward.use_reward_prob: True`` — 每步将原始 sigmoid 概率打印到终端。
- ``reward.standalone_realworld: True`` — 利用 reward model 直接判断成功/失败并触发重置。
- ``reward.reward_threshold`` — 概率阈值，低于该值的成功判定将被抑制。根据模型校准情况调整。
- ``reward.model.model_path`` — 指向训练好的 reward model checkpoint。

启动
------

设置环境变量并运行：

.. code-block:: bash

   bash examples/reward/run_realworld_teleop.sh

或显式指定配置名称：

.. code-block:: bash

   bash examples/reward/run_realworld_teleop.sh realworld_teleop

终端每步输出如下：

.. code-block:: text

   [TeleopWorker] Starting teleoperation loop.
   [TeleopWorker] EmbodiedRewardWorker ready: type=EmbodiedRewardWorker | reward_threshold=0.200
   Step 0      | rm_reward: 0 | success: False
   Step 1      | rm_reward: 0 | success: False
   Step 10     | rm_reward: 0 | success: False
   Step 123    | rm_reward: 1 | success: True
   Step 124    | rm_reward: 1 | success: True

SpaceMouse 控制说明：

- **移动** — 遥操作机械臂。
- **左键** — 合拢夹爪。
- **右键** — 张开夹爪。
- **Ctrl+C** — 停止。

工作原理
------------

``TeleopWorker`` 内部流程：

1. ``RealWorldEnv`` 以 ``use_spacemouse=True`` 初始化，包装了 ``SpacemouseIntervention``。
   当 SpaceMouse 输入非零（或按下按钮）时，用 SpaceMouse 动作覆盖零 dummy 动作，持续 0.5 秒。
2. ``EmbodiedRewardWorker`` 通过 ``EmbodiedRewardWorker.launch_for_realworld(...)``
   在 GPU 节点上启动，在启动时一次性完成初始化。
3. 每步遥操作中，从观测中提取腕部相机图像（``obs["main_images"]``）并发送给 reward worker 进行推理。
4. 原始 sigmoid 概率被打印到终端。当 ``standalone_realworld=True`` 时，
   reward model 还直接驱动成功/失败判定和环境重置。

与 :doc:`../../examples/embodied/franka_reward_model` 中的完整 RL 流程相比，
遥操作脚本不运行策略、actor 或 rollout worker——它纯粹是人在回路的 reward model 评估。
