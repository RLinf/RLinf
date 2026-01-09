ResNet 奖励模型训练指南
=======================

本指南介绍如何使用基于 ResNet 的奖励模型来训练 ManiSkill PickCube 任务。

概述
----

ResNet 奖励模型是一个基于图像的二分类器，用于预测机器人是否成功完成抓取任务。完整的训练流程包含四个阶段：

1. **数据收集**：在训练策略的同时收集带有成功/失败标签的 RGB 图像
2. **ResNet 训练**：在收集的数据上训练 ResNet 二分类器
3. **策略预训练**：使用环境的稠密奖励训练初始策略
4. **ResNet 奖励训练**：使用 ResNet 奖励替代环境奖励继续训练

前置条件
--------

- ManiSkill 环境已正确安装
- GPU 显存足够支持渲染和训练

阶段一：数据收集
----------------

在使用稠密奖励训练策略的同时，收集标记为成功/失败的 RGB 图像。

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_collect_reward_data

这将会：

- 使用稠密奖励训练策略（与 ``maniskill_ppo_mlp`` 相同）
- 在训练过程中渲染 RGB 图像
- 将成功/失败图像保存到 ``examples/embodiment/data/``

配置文件 (``maniskill_collect_reward_data.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    reward_data_collection:
      enabled: True
      save_dir: "${oc.env:EMBODIED_PATH}/data"
      target_success: 5000      # 收集的成功样本数量
      target_failure: 5000      # 收集的失败样本数量
      sample_rate_fail: 0.01    # 采样 1% 的失败帧
      sample_rate_success: 1.0  # 采样 100% 的成功帧

数据保存格式::

    examples/embodiment/data/
    ├── success/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    └── failure/
        ├── 0001.png
        ├── 0002.png
        └── ...

阶段二：训练 ResNet 奖励模型
----------------------------

在收集的数据上训练 ResNet 二分类器。

.. code-block:: bash

    python examples/embodiment/train_reward_model.py --config-name maniskill_train_reward_model

配置文件 (``maniskill_train_reward_model.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    reward_model_training:
      data_path: "${oc.env:EMBODIED_PATH}/data"
      epochs: 100
      batch_size: 64
      lr: 1.0e-4
      val_split: 0.1
      save_dir: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints"
      early_stopping_patience: 15

训练好的模型将保存到 ``logs/reward_checkpoints/best_model.pt``。

阶段三和四：使用 ResNet 奖励训练策略
------------------------------------

阶段三：使用稠密奖励预训练策略（可选）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用环境原生的稠密奖励训练初始策略（默认 100 步）：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

检查点将保存到 ``logs/<timestamp>-maniskill_ppo_mlp/maniskill_ppo_mlp/checkpoints/``。

训练将在第 100 步自动停止。

阶段四：使用 ResNet 奖励继续训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

更新 ``maniskill_ppo_mlp_resnet_reward.yaml`` 中的 ``resume_dir``，指向阶段三的检查点：

.. code-block:: yaml

    runner:
      # TODO: 设置为你的 maniskill_ppo_mlp 检查点路径
      resume_dir: "logs/<timestamp>-maniskill_ppo_mlp/maniskill_ppo_mlp/checkpoints/global_step_100"

然后运行：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

配置说明
--------

关键参数 (``maniskill_ppo_mlp_resnet_reward.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    env:
      train:
        reward_render_mode: "episode_end"  # 必须与数据收集时一致
        show_goal_site: True               # 显示绿色目标标记
        init_params:
          control_mode: "pd_joint_delta_pos"  # 必须与数据收集时一致

    reward:
      use_reward_model: True
      reward_model_type: "resnet"
      mode: "replace"  # 用 ResNet 奖励替代环境奖励
      alpha: 1.0
      
      resnet:
        checkpoint_path: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints/best_model.pt"
        threshold: 0.5
        use_soft_reward: False  # 二值 0/1 奖励

关键参数对齐
~~~~~~~~~~~~

以下参数 **必须** 与数据收集时使用的参数保持一致：

.. list-table::
   :header-rows: 1

   * - 参数
     - 值
     - 说明
   * - ``control_mode``
     - ``pd_joint_delta_pos``
     - 控制模式（8 维动作空间）
   * - ``reward_render_mode``
     - ``episode_end``
     - 仅在回合结束时渲染图像
   * - ``show_goal_site``
     - ``True``
     - 显示绿色目标标记
   * - ``image_size``
     - ``[3, 224, 224]``
     - 图像尺寸

预期结果
--------

- 经过约 500-1000 步后，``env/success_once`` 应接近 100%
- ``env/episode_len`` 应降低到约 15-20 步
- ``env/reward`` 会显示较低的值（这是稀疏二值奖励的正常现象）

具身智能奖励模型架构与 API
--------------------------

本模块为具身强化学习任务提供奖励模型实现，支持基于图像的（单帧）和基于视频的（多帧）奖励模型。

架构
~~~~

.. code-block:: text

    BaseRewardModel (Abstract Root)
    |
    +-- BaseImageRewardModel (Abstract)     # Single-frame reward
    |   +-- ResNetRewardModel               # Binary classifier (HIL-SERL style)
    |
    +-- BaseVideoRewardModel (Abstract)     # Multi-frame/video reward
        +-- Qwen3VLRewardModel              # VLM-based reward (placeholder)

文件结构
~~~~~~~~

.. code-block:: text

    rlinf/models/embodiment/reward/
    +-- __init__.py                    # Module exports
    +-- base_reward_model.py           # BaseRewardModel (root abstract)
    +-- base_image_reward_model.py     # BaseImageRewardModel (single-frame)
    +-- base_video_reward_model.py     # BaseVideoRewardModel (multi-frame)
    +-- resnet_reward_model.py         # ResNet binary classifier
    +-- qwen3_vl_reward_model.py       # Qwen3-VL (placeholder)

    rlinf/algorithms/rewards/embodiment/
    +-- reward_manager.py              # RewardManager with registry pattern

    examples/embodiment/config/reward/
    +-- resnet_binary.yaml             # ResNet configuration
    +-- qwen3_vl.yaml                  # Qwen3-VL configuration

快速开始
~~~~~~~~

使用 RewardManager（推荐）
^^^^^^^^^^^^^^^^^^^^^^^^^^

``RewardManager`` 为所有奖励模型提供统一接口：

.. code-block:: python

    from rlinf.algorithms.rewards.embodiment import RewardManager
    from omegaconf import OmegaConf

    # 加载配置
    cfg = OmegaConf.load("examples/embodiment/config/reward/resnet_binary.yaml")
    cfg.resnet.checkpoint_path = "/path/to/your/checkpoint.pt"

    # 初始化奖励管理器
    reward_manager = RewardManager(cfg)

    # 计算奖励
    observations = {
        "images": images_tensor,  # [B, C, H, W] 或 [B, H, W, C]
        "states": states_tensor,  # 可选 [B, state_dim]
    }
    rewards = reward_manager.compute_rewards(observations)

直接使用模型
^^^^^^^^^^^^

.. code-block:: python

    from rlinf.models.embodiment.reward import ResNetRewardModel
    from omegaconf import DictConfig

    cfg = DictConfig({
        "checkpoint_path": "/path/to/checkpoint.pt",
        "image_size": [3, 224, 224],
        "threshold": 0.5,
        "use_soft_reward": False,
    })

    model = ResNetRewardModel(cfg)
    rewards = model.compute_reward(observations)

API 参考
~~~~~~~~

BaseRewardModel
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 方法
     - 说明
   * - ``compute_reward(observations, task_descriptions)``
     - 从观测计算奖励
   * - ``load_checkpoint(path)``
     - 加载模型权重
   * - ``scale_reward(reward)``
     - 应用缩放因子
   * - ``to_device(device)``
     - 移动模型到设备

BaseImageRewardModel
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 方法
     - 说明
   * - ``preprocess_images(images)``
     - 归一化并重排通道
   * - ``apply_threshold(probabilities)``
     - 转换为二值奖励

BaseVideoRewardModel
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 方法
     - 说明
   * - ``sample_frames(images, strategy, k)``
     - 从视频中采样帧
   * - ``preprocess_video(images)``
     - 归一化视频张量
   * - ``format_prompt(task_description)``
     - 格式化 VLM 提示词

RewardManager
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 方法
     - 说明
   * - ``compute_rewards(observations, task_descriptions)``
     - 统一的奖励计算接口
   * - ``register_model(name, cls)``
     - 注册新模型类型
   * - ``get_available_models()``
     - 列出已注册模型
   * - ``to_device(device)``
     - 移动模型到设备


