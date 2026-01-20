奖励模型集成
========================

本指南介绍如何将学习型奖励模型集成到 RLinf 中用于具身 RL 训练。
奖励模型可以替代或增强环境奖励，从视觉观测中学习。

概述
--------

RLinf 支持基于图像的奖励模型：

- 以 RGB 图像作为输入
- 输出成功/失败概率
- 可用于终止奖励（episode 结束时）或逐步奖励

组件
----------

奖励模型系统包括：

1. **ResNetRewardModel** - 基于 ResNet 的二分类器
2. **ImageRewardWorker** - RL 训练期间的推理 Worker
3. **FSDPRewardWorker** - 使用 FSDP 训练奖励模型的 Worker

环境变量
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 变量
     - 说明
   * - ``TORCHDYNAMO_DISABLE=1``
     - 禁用 torch dynamo 以避免 jinja2 兼容性问题
   * - ``DEBUG_IMAGE_SAVE_DIR``
     - 保存 ResNet 分类可视化调试图像的目录

配置
-------------

在 YAML 配置中添加奖励模型：

.. code-block:: yaml

    reward:
      use_reward_model: True
      group_name: "RewardGroup"
      
      # 奖励模式：
      # - "per_step": 每帧计算奖励
      # - "terminal": 仅在 episode 结束时计算奖励
      reward_mode: "terminal"
      
      # 非对称奖励映射
      success_reward: 1.0   # 成功奖励 (prob > threshold)
      fail_reward: 0.0      # 失败奖励 (prob < threshold)
      reward_threshold: 0.5 # 概率阈值
      
      # 与环境奖励的组合方式
      combine_mode: "replace"  # 或 "add", "weighted"
      reward_weight: 10.0
      env_reward_weight: 0.0
      
      model:
        model_type: "resnet_reward"
        arch: "resnet18"
        hidden_dim: 256
        dropout: 0.1
        image_size: [3, 128, 128]
        checkpoint_path: "path/to/checkpoint.safetensors"
        debug_save_dir: "${runner.logger.log_path}/${runner.logger.experiment_name}/debug_images"

训练奖励模型
-----------------------

1. **采集数据**

   可直接使用 ``maniskill_ppo_mlp_collect.yaml`` 配置，该配置已集成数据采集功能
   （详见 :doc:`data_collection`）：

   .. code-block:: bash

       bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_collect

   此配置同时完成策略预热训练和数据采集，采集的数据保存在 ``collected_data`` 目录。

2. **训练模型**

   使用奖励训练配置（注意修改 ``data.data_path`` 为实际数据路径）：

   .. code-block:: bash

       # 指定数据路径
       bash examples/reward/run_reward_training.sh --data /path/to/collected_data

3. **在 RL 训练中使用**

   指向训练好的 checkpoint：

   .. code-block:: bash

       bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

推荐训练策略
-----------------------------

为获得最佳效果，我们推荐 **两阶段训练方法**：

**阶段 1：使用环境奖励预热（100+ 步）**

首先使用环境原生奖励信号训练策略：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_collect

这将创建一个具有基本任务理解的 checkpoint，并收集用于训练奖励模型的 episode 数据。

**阶段 2：继续使用 ResNet 奖励**

从阶段 1 的 checkpoint 恢复训练，使用 ResNet 奖励模型：

.. code-block:: yaml

    runner:
      resume_dir: "logs/.../checkpoints/global_step_100"
    
    reward:
      use_reward_model: True
      model:
        checkpoint_path: "path/to/trained_resnet.safetensors"

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

**为什么需要两阶段？**

- 从头开始使用学习奖励往往会失败（冷启动问题）
- 阶段 1 为训练奖励模型提供多样化数据
- 阶段 1 的 checkpoint 让策略在精调前具备基本能力

奖励模式
------------

**终止模式** （推荐）

仅在每个 episode 结束时提供奖励。防止智能体利用中间状态进行奖励欺骗。

.. code-block:: yaml

    reward_mode: "terminal"
    success_reward: 1.0
    fail_reward: 0.0

**逐步模式**

为每帧提供奖励。可能导致奖励欺骗，但提供更密集的学习信号。

.. code-block:: yaml

    reward_mode: "per_step"

调试
---------

启用调试图像保存以可视化 ResNet 分类结果：

.. code-block:: yaml

    model:
      debug_save_dir: "logs/debug_images"

图像保存位置：

- ``resnet_success/`` - 被分类为成功的图像
- ``resnet_fail/`` - 被分类为失败的图像

文件名格式：``{id:06d}_prob{probability:.4f}.png``

故障排除
---------------

**奖励欺骗（成功率为 0 但奖励增加）**

- 切换到终止模式：``reward_mode: "terminal"``
- 增加阈值：``reward_threshold: 0.6``

**策略崩溃（全是负奖励）**

- 使用非对称奖励：``fail_reward: 0.0`` 而不是负值
- 不要使用 ``use_negative_reward: true``

**Checkpoint 加载错误**

- 确保 ``hidden_dim`` 与训练的 checkpoint 匹配
- 检查 ``image_size`` 与训练数据匹配

