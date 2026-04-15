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

专家轨迹数据采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在数据采集方面，首先需要采集专家轨迹数据。
该数据会在训练中事先存储在样本缓冲区（ demo buffer ）中，
具体步骤同 :doc:`franka` 中 ``运行实验`` 的 ``数据采集`` 小节。
注意确认，配置文件 ``examples/embodiment/config/realworld_collect_data.yaml``中 ``env`` 部分的 ``data_collection`` 已开启：

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: True

启动数据采集脚本后，环境会自动将 episode 保存到 ``save_dir``。当 ``export_format="pickle"`` 时，
每个 episode 会被写入一个独立的 ``.pkl`` 文件，便于后续离线预处理。

reward model 训练和评估数据采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了得到高质量的 reward model ，需要采集更多的数据用来训练和评估 reward model。
在 :doc:`franka` 中 ``运行实验`` 的 ``数据采集`` 小节的基础上，进一步对采集脚本做以下修改。

将配置中的 ``success_hold_steps`` 字段增大，以便在有限的采集轮次内得到更多的成功数据。
机械臂末端在到达目标位姿后不会立刻判定为成功并重置，
而是需要到达目标位姿并保持一定步数（ ``success_hold_steps`` ）后才会判定为成功。
如果中途退出成功状态，会重新开始计数。

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

启动数据采集脚本后，环境会自动将 episode 保存到 ``save_dir``。当 ``export_format="pickle"`` 时，
每个 episode 会被写入一个独立的 ``.pkl`` 文件，便于后续离线预处理。

在采集过程中，请尽量缓慢移动，以便获得更多样的失败样本。
在到达目标位姿时，在保持目标位姿的前提下进行小范围移动，以便获得更多样的成功样本。

预处理为 reward dataset
-----------------------
本步骤同 :doc:`../../tutorials/extend/reward_model` 中的 ``1.2 预处理为 reward dataset`` 部分。

特别的，建议调高 ``fail-success-ratio`` 至 ``3``。

.. code-block:: bash

  Example:
      python examples/reward/preprocess_reward_dataset.py \
          --raw-data-path logs/xxx/collected_data \
          --output-dir logs/xxx/processed_reward_data \
          --fail-success-ratio 3

Reward Model 训练
-----------------------
本步骤同 :doc:`../../tutorials/extend/reward_model` 中的 ``2. Reward Model 训练`` 部分。

特别的，在真实世界场景中，建议降低 ``early_stop`` 的 ``min_delta``，例如：

.. code-block:: bash

  runner:
    early_stop
      min_delta: 1e-6
          
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