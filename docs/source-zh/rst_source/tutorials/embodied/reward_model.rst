Reward Model 使用指南
======================

本文档介绍如何在 RLinf 中使用 reward model，涵盖仿真和真机两种场景，
覆盖 ``ResNetRewardModel`` 这类图像分类 reward，以及 QwenTrend / ``HistoryVLMRewardModel`` 这类 VLM reward。

.. contents::
   :depth: 2
   :local:

仿真场景 Reward Model
---------------------

仿真场景完整流程包括四个阶段：

1. 数据收集：在 RL 运行过程中采集原始 episode 数据。
2. 数据转换：将原始 episode 转成图像分类数据或 VLM SFT 数据。
3. Reward model 训练：训练 ResNet reward model，或微调 VLM reward model。
4. Reward model 在 RL 中推理：将训练好的模型接入在线 rollout，参与最终 reward 计算。

1. 数据收集
^^^^^^^^^^^

reward model 的训练数据通常来自 episode 级数据采集。RLinf 提供了统一的数据采集封装，
相关用法可参考 :doc:`数据采集教程 <data_collection>`。

对于 reward model 场景，建议先以 ``pickle`` 格式保存原始 episode 数据，再通过预处理脚本转换为训练集。

启用数据采集
""""""""""""

在 YAML 配置文件的 ``env`` 部分开启 ``data_collection``：

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: False

对于 QwenTrend VLM reward，RLinf 也提供了可直接运行的数据采集配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_collect

预处理为 ResNet Reward Dataset
""""""""""""""""""""""""""""""

使用 ``examples/reward/preprocess_reward_dataset.py`` 进行转换，将 ``.pkl`` episode 转为 ``.pt`` 文件：

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data

转换为 QwenTrend VLM Dataset
""""""""""""""""""""""""""""

QwenTrend 使用短时间双视角历史窗口。使用
``examples/reward/preprocess_qwentrend_reward_dataset.py`` 进行转换：

.. code-block:: bash

   python examples/reward/preprocess_qwentrend_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_qwentrend_reward_data \
       --window-size 5 \
       --stride 1 \
       --delta-threshold 0.05

2. Reward Model 训练
^^^^^^^^^^^^^^^^^^^^

RLinf 支持两条 reward 训练路径。

微调 ResNet Reward Model
""""""""""""""""""""""""

训练前修改 ``examples/reward/config/reward_training.yaml`` 中的数据路径：

.. code-block:: yaml

   data:
     train_data_paths: "logs/processed_reward_data/train.pt"
     val_data_paths: "logs/processed_reward_data/val.pt"

   actor:
     model:
       model_type: "resnet"
       arch: "resnet18"
       pretrained: False
       image_size: [3, 128, 128]

启动训练：

.. code-block:: bash

   bash examples/reward/run_reward_training.sh

微调 QwenTrend VLM Reward Model
"""""""""""""""""""""""""""""""

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/processed_qwentrend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_qwentrend

3. Reward Model 在 RL 中推理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RLinf 提供了多个 reward model 接入 RL 的示例配置。

基本配置项
""""""""""

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     reward_mode: "terminal"   # 或 "per_step" / "history_buffer"
     reward_threshold: 0.5
     reward_weight: 1.0
     env_reward_weight: 0.0
     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"    # 或 "vlm" / "history_vlm"

Rollout 阶段的 Worker 交互
""""""""""""""""""""""""""

在线 RL 阶段，``env``、``rollout``、``reward`` 三类 worker 协同工作：

.. code-block:: text

   Env worker
      | 1. 与环境交互，获得 obs / env reward / done
      | 2. 将 obs 发送给 Rollout worker 生成动作
      | 3. 当启用 reward model 时，将 reward input dict 发送给 Reward worker
      v
   Reward worker
      | 4. 执行 ``compute_reward(...)``，返回 reward model output
      v
   Env worker
      | 5. 接收 Rollout worker 的 bootstrap values
      | 6. 将 env reward 与 reward model output 组合
      v
   Final reward -> 写入 rollout 结果并参与后续 RL 更新

最终 Reward 的计算
""""""""""""""""""

.. code-block:: python

   reward = env_reward_weight * env_reward + reward_weight * reward_model_output

部署 QwenTrend 进行 MLP RL
""""""""""""""""""""""""""

安装依赖后启动训练：

.. code-block:: bash

   bash requirements/install.sh embodied --env maniskill_libero --vlm-reward
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_reward

真机场景 Reward Model
---------------------

本节介绍如何在真实世界的 Franka 机械臂上直接采集并预处理 reward model 训练数据集。
支持两种数据采集方式：**通用键盘标注方式** 和 **固定位姿方式**。

在开始前，强烈建议先阅读：

1. :doc:`../../examples/embodied/franka` 以熟悉 Franka 机械臂真机训练全流程。
2. :doc:`../../examples/embodied/franka_reward_model` 以了解训练好 reward model 后如何接入真机 RL 流程。

工作流概览
^^^^^^^^^^

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
^^^^^^^^

请根据 :doc:`../../examples/embodied/franka` 中的 **Prerequisites** 和 **Hardware Setup** 章节，
完成机器人连接和环境验证步骤。

方式一：键盘标注（通用）
^^^^^^^^^^^^^^^^^^^^^^^^

此方式通过键盘在实时 episode 中手动标注每一帧，适用于任何操作任务。

**配置文件** — ``examples/reward/config/realworld_collect_dataset.yaml``：

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
     num_success_frames: 50    # 目标采集的成功帧数
     num_fail_frames: 150      # 目标采集的失败帧数
     val_split: 0.2            # 用于验证集的帧比例
     fail_success_ratio: 2.0   # 训练集后处理时下采样比例
     random_seed: 42

   env:
     eval:
       no_gripper: False
       use_spacemouse: True
       max_episode_steps: 10000
       keyboard_reward_wrapper: single_stage

**启动命令：**

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh

按键说明：``c`` — 标注成功，``a`` — 标注失败。

方式二：固定位姿（目标驱动）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

此方式专为**固定目标位姿**的任务设计。通过配置 ``success_hold_steps`` 采集更多样成功样本：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

采集技巧：缓慢移动机械臂获得更多样失败样本；到达目标位姿时进行小范围移动获得更多样成功样本。

预处理为 Reward Dataset：

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data \
       --fail-success-ratio 3

数据采集方式对比
""""""""""""""""

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
^^^^^^^^^^^^^^^^^

完成以上步骤后，继续参考上方的 **仿真场景 Reward Model** → **2. Reward Model 训练**，
使用生成的 ``train.pt`` / ``val.pt`` 文件进行训练。

真机遥操作 + 在线 Reward Model 推理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

获得 reward model checkpoint 后，``examples/reward/eval_realworld_teleop.py`` 提供遥操作模式：
SpaceMouse 控制机器人运动，reward model 在 GPU 节点上实时推理。

遥操作脚本需要**两个节点**：一个用于 Franka 机器人，一个用于 GPU 推理。

关键配置：

.. code-block:: yaml

   reward:
     use_reward_model: True
     use_reward_prob: True    # 打印每步原始 sigmoid 概率到终端
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.2
     model:
       model_path: path/to/reward_model_checkpoint
       model_type: "resnet"

启动命令：

.. code-block:: bash

   bash examples/reward/run_realworld_teleop.sh

SpaceMouse 控制：移动遥操作、左键合拢夹爪、右键张开夹爪、Ctrl+C 停止。

总结
----

完整工作流：

1. 在环境配置中开启 ``data_collection``，将数据保存为 ``pickle`` 格式。
2. 对于 ResNet reward，使用 ``preprocess_reward_dataset.py`` 构建 ``train.pt`` / ``val.pt``，再用 ``run_reward_training.sh`` 训练。
3. 对于 QwenTrend VLM reward，使用 ``preprocess_qwentrend_reward_dataset.py`` 构建双视角历史窗口数据，再用 ``run_vlm_sft.sh`` 微调。
4. 在 RL YAML 中开启 ``reward.use_reward_model=True``，接入 reward worker 完成在线推理。
5. 对于真机场景，使用键盘标注或固定位姿方式采集数据，训练后在物理机器人上部署 reward model。
