Franka真机强化学习（基于 Reward Model ）
========================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 Franka 机械臂真机环境中的训练任务中使用 reward model，
重点介绍如何从零开始训练并部署基于 ResNet 的 reward model ，以辅助完成机器人操作任务。

在开始前，强烈建议先阅读以下文档：

1. :doc:`franka` 以熟悉 Franka 机械臂真机环境下训练全流程。
2. :doc:`../../tutorials/extend/reward_model` 以熟悉 RLinf 的仿真环境中使用 reward model 的完整流程。

预备工作
-----------------------
请根据 :doc:`franka` 中 ``运行实验`` 的 ``数据采集`` 之前的章节，完成数据采集之前的全部工作。

数据采集
-----------------------

需要采集两类数据：（1）用于 demo buffer 的专家轨迹数据；（2）用于 reward model 训练和评估的数据。

专家轨迹数据采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

首先需要采集专家轨迹数据，该数据会在训练中事先存储在样本缓冲区（demo buffer）中。
具体步骤同 :doc:`franka` 中 ``运行实验`` 的 ``数据采集`` 小节。
注意确认，配置文件 ``examples/embodiment/config/realworld_collect_data.yaml`` 中
``env`` 部分的 ``data_collection`` 已开启：

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: True

Reward Model 数据集采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

采集 reward model 训练和评估数据支持两种方式，详细说明请参考
:doc:`../../tutorials/extend/reward_model_realworld` 中的 **数据采集** 部分。
两种方式的核心区别在于标注方式：方式一为手动键盘标注，适用于任意操作任务；
方式二为基于位姿的自动标注，专为固定目标位姿的任务设计。

方式一：键盘标注（通用）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

此方式通过键盘在实时 episode 中手动标注每一帧，适用于任何操作任务。
此方式将数据采集、标注和数据集生成整合为一次端到端运行，无需繁琐的离线预处理步骤。

**关键配置：**

- ``runner.num_success_frames`` / ``runner.num_fail_frames`` — 目标采集帧数，两个阈值均达到时停止采集。
- ``runner.val_split`` — 所有标注帧中用于验证集的比例。
- ``runner.fail_success_ratio`` — 训练集后处理阶段失败帧下采样比例。
- ``env.eval.keyboard_reward_wrapper`` — 设为 ``single_stage`` 以启用键盘标注界面。
- ``env.eval.use_spacemouse`` — 是否使用 SpaceMouse 进行遥操作。
- ``env.eval.override_cfg.target_ee_pose`` — 任务的目标末端执行器位姿。

**启动命令：**

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset

**按键说明：**

- ``c`` — 将当前帧标注为**成功**。
- ``a`` — 将当前帧标注为**失败**。

达到目标帧数后，脚本自动停止、划分数据并保存 ``train.pt`` / ``val.pt`` 文件。
详细配置说明及完整示例请参见 :doc:`../../tutorials/extend/reward_model_realworld` 中的方式一。

方式二：固定位姿（目标驱动）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

此方式专为**固定目标位姿**的任务设计，无需手动键盘标注，episode 会根据机器人是否到达
配置的 ``target_ee_pose`` 自动驱动成功/失败判定。
可以设置 ``success_hold_steps``，要求机器人在目标位姿保持一定步数后才判定为成功，
有助于采集更多样的成功样本。
此方式采用简化的两步式流程。

**步骤 1：固定位姿 Reward 数据采集**

在上述专家轨迹采集的基础上，将配置中的 ``success_hold_steps`` 字段增大：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

采集技巧：

- 请尽量缓慢移动机械臂，以便获得更多样的失败样本。
- 在到达目标位姿时，在保持目标位姿的前提下进行小范围移动，以便获得更多样的成功样本。

**步骤 2：预处理为 Reward Dataset**

采集好的 ``.pkl`` episode 通过 ``preprocess_reward_dataset.py`` 转换为 ``train.pt`` / ``val.pt``，
建议将 ``fail-success-ratio`` 调高至 ``3``：

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data \
       --fail-success-ratio 3

生成的 ``.pt`` 文件符合 ``RewardDatasetPayload`` 约定的标准格式，包含 ``images``、
``labels``（1 = 成功，0 = 失败）和 ``metadata``。
详细说明及完整示例请参见 :doc:`../../tutorials/extend/reward_model_realworld` 中的方式二。

Reward Model 训练
-----------------------
本步骤同 :doc:`../../tutorials/extend/reward_model` 中的 ``2. Reward Model 训练`` 部分。

特别的，在真实世界场景中，建议降低 ``early_stop`` 的 ``min_delta``，例如：

.. code-block:: yaml

  runner:
    early_stop
      min_delta: 1e-6

如需在真机遥操作中进行在线 reward model 推理（SpaceMouse + GPU 节点，无需 RL 训练循环），
请参考 :doc:`../../tutorials/extend/reward_model_realworld` 中的 **真机遥操作 + 在线 Reward Model 推理** 部分。

集群配置
-----------------------
本步骤同 :doc:`franka` 中的 ``运行实验`` 下的 ``集群配置`` 部分。

配置文件
-----------------------
本步骤同 :doc:`franka` 中的 ``配置文件`` 小节，对 ``examples/embodiment/config/realworld_charger_sac_cnn_async_standalone_reward.yaml`` 进行配置。
特别的，还需要启用位于 ``reward`` 段的 reward model 相关参数：

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.8

     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"

其中：

- ``reward_mode`` 控制 reward model 在每一步推理，还是仅在终止帧推理。
- ``standalone_realworld`` 利用 reward model 直接判断任务是否成功，进而触发重置。
- ``reward_threshold`` 用于对 reward model 输出的成功概率做阈值过滤；低于阈值的项会被置为 ``0``。
- ``model_path`` 指向用于在线推理的 reward model 权重。

开始实验
-----------------------
启动训练后，reward model 会直接基于图像观测判定任务成功/失败，并驱动环境重置。
其余步骤请继续参照 :doc:`franka` 中 ``运行实验`` 章节执行。

Rollout 阶段的 worker 交互
----------------------------------------------
与 :doc:`../../tutorials/extend/reward_model` 中的 ``3.2 Rollout 阶段的 worker 交互`` 和 ``3.3 最终 reward 的计算`` 部分不同的是：
在真机系统中，由于启动了 ``standalone_realworld``，reward model 将不再 `将 env reward 与 reward model output 组合`。

换句话说，reward model 在 RL 中 `不会` 作为 env worker 中的附加 reward 来源参与最终 reward 的构造，
因为系统会直接绕过 ``env_reward`` 和 ``reward_model_output`` 加权求和的过程。
因此，reward_mode、reward_weight、env_reward_weight 均不生效，最终 reward 由 FrankaEnv 内部直接基于 reward model 判定成功/失败后生成。

从系统的角度看，真机系统中的实际行为可以看做：
直接替换 env worker 中的 env_reward，通过沿用原本 env_reward 的功能来实现奖励赋值和控制系统重置等目的，从根本上进行了 reward model 接入。