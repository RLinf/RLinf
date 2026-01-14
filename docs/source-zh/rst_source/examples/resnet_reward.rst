ResNet Reward Model 训练指南
============================

本指南介绍如何使用 ResNet reward model 训练 ManiSkill PickCube 任务。

概述
----

ResNet reward model 是一个基于图像的二分类器，用于预测机器人是否成功完成抓取任务。完整流程包括三个阶段：

1. **数据采集**: 在 PPO 训练过程中采集带有成功/失败标签的 RGB 图像
2. **ResNet 训练**: 训练 ResNet 二分类器
3. **使用 ResNet Reward 训练策略**: 用训练好的 ResNet reward 替换环境奖励

架构
----

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                          训练流程                                │
    └─────────────────────────────────────────────────────────────────┘
    
    阶段 1: 数据采集（在 PPO 训练过程中）
    ┌───────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │   EnvWorker   │────▶│ DataCollectorWrapper│────▶│  pkl 文件    │
    │  (GPU 0)      │     │                     │     │ success/fail │
    └───────────────┘     └─────────────────────┘     └──────────────┘
    
    阶段 2: Reward Model 训练
    ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │  pkl 文件    │────▶│ train_reward_model  │────▶│ best_model.pt│
    │              │     │     .py             │     │              │
    └──────────────┘     └─────────────────────┘     └──────────────┘
    
    阶段 3: 使用 Reward Model 训练策略
    ┌───────────────┐                              ┌──────────────┐
    │   EnvWorker   │──── env_output ────────────▶│   Rollout    │
    │   (GPU 0)     │                             │   Worker     │
    └───────────────┘                             └──────────────┘
                                                         │
    ┌───────────────┐     ┌─────────────────────┐       │
    │ RewardWorker  │◀────│  EmbodiedRunner     │◀──────┘
    │   (GPU 1)     │     │  _apply_reward_model│
    └───────────────┘     └─────────────────────┘
            │                      │
            └──── model_rewards ───┘
                       │
                       ▼
              ┌──────────────┐
              │    Actor     │
              │  (training)  │
              └──────────────┘

前置条件
--------

- ManiSkill 环境已正确安装
- 至少 2 张 GPU（一张用于 env/rollout/actor，一张用于 reward model）

阶段 1: 数据采集
----------------

在 PPO 训练过程中采集带有成功/失败标签的 RGB 图像。

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_collect_reward_data

这将：

- 使用密集奖励训练策略（与 ``maniskill_ppo_mlp`` 相同）
- 通过 ``DataCollectorWrapper`` 自动采集 RGB 图像
- 将完整的 episode 轨迹保存到 pkl 文件

配置 (``maniskill_collect_reward_data.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    env:
      train:
        reward_render_mode: "always"  # 每帧都渲染 RGB
        init_params:
          obs_mode: "state"           # 策略使用状态
          render_mode: "rgb_array"    # 启用 RGB 渲染

    reward_data_collection:
      enabled: True
      save_dir: "${oc.env:EMBODIED_PATH}/data"
      target_success: 5000
      target_failure: 5000

数据格式
~~~~~~~~

数据保存为单独的 pkl 文件（每个 episode 一个）::

    save_dir/
    ├── success/
    │   ├── episode_000000.pkl
    │   └── ...
    └── failure/
        ├── episode_000001.pkl
        └── ...

每个 pkl 文件包含::

    {
        "frames": [  # 帧列表（最多 50 帧）
            {
                "obs": {"main_images": ..., "states": ...},
                "action": ...,
                "reward": ...,
                "done": ...,
                "grasp": ...,
                "success": ...,
            },
            ...
        ],
        "success": True/False,
        "label": 1/0,
    }

阶段 2: 训练 ResNet Reward Model
--------------------------------

使用集成的 ``RewardWorker`` 训练 ResNet 二分类器。

.. code-block:: bash

    ./run_embodiment.sh maniskill_train_resnet_reward

配置 (``maniskill_train_resnet_reward.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    reward_training:
      enabled: True
      only: True  # 跳过环境交互，仅训练 reward model
      data_path: "${oc.env:EMBODIED_PATH}/data"
      epochs: 100
      micro_batch_size: 64
      global_batch_size: 64
      lr: 1.0e-4
      save_dir: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints"

训练好的模型将保存到 ``logs/reward_checkpoints/reward_model_best.pt``。

阶段 3: 使用 ResNet Reward 训练策略
-----------------------------------

使用训练好的 ResNet reward model 替换环境奖励。

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

配置 (``maniskill_ppo_mlp_resnet_reward.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    cluster:
      component_placement:
        actor: 0-0
        env: 0-0
        rollout: 0-0
        reward: 0-1  # RewardWorker 在 GPU 1 上

    runner:
      resume_dir: null  # 设置为预训练 checkpoint 路径

    reward:
      use_reward_model: True
      reward_model_type: "resnet"
      mode: "replace"  # 用 model reward 替换 env reward
      resnet:
        checkpoint_path: "logs/reward_checkpoints/best_model.pt"

工作原理
~~~~~~~~

1. **EmbodiedRunner** 从 actor 接收 rollout batch
2. **EmbodiedRunner** 调用 ``RewardWorker.compute_batch_rewards()`` 计算 rewards
3. **RewardWorker** 使用 mini-batch（每批 256 张图片）计算 model rewards
4. **EmbodiedRunner** 通过 ``actor.update_rewards()`` 更新 actor 的 rewards
5. **Actor** 使用新的 rewards 进行训练

主要特性
~~~~~~~~

- **并行处理**: 所有时间步在一个 batch 中处理
- **Mini-batching**: 大 batch 拆分成 256 张图片的小批次，避免 OOM
- **独立 GPU**: RewardWorker 在 GPU 1 上运行，避免内存冲突

预期结果
--------

- 经过约 500-1000 步后，``env/success_once`` 应接近 90%+
- ``time/compute_model_rewards`` 每步约 10-50ms
- ``rollout/rewards`` 将显示 model reward 值

API 参考
--------

DataCollectorWrapper
~~~~~~~~~~~~~~~~~~~~

用于自动数据采集的环境 wrapper。

.. code-block:: python

    from rlinf.envs.wrappers import DataCollectorWrapper

RewardWorker
~~~~~~~~~~~~

用于 reward model 推理的 worker（在独立 GPU 上运行）。

.. code-block:: python

    from rlinf.workers.reward.reward_worker import RewardWorker

.. list-table::
   :header-rows: 1

   * - 方法
     - 描述
   * - ``init_worker()``
     - 使用 model 初始化 RewardManager
   * - ``compute_batch_rewards(observations)``
     - 使用 mini-batching 计算 rewards
   * - ``save_checkpoint(path, step)``
     - 保存 model checkpoint

RewardManager
~~~~~~~~~~~~~

统一的 reward 计算接口。

.. code-block:: python

    from rlinf.algorithms.rewards.embodiment import RewardManager

.. list-table::
   :header-rows: 1

   * - 方法
     - 描述
   * - ``compute_rewards(observations)``
     - 从图像计算 rewards
   * - ``to_device(device)``
     - 将模型移动到设备

文件结构
--------

.. code-block:: text

    rlinf/
    ├── envs/
    │   └── wrappers.py              # DataCollectorWrapper
    │
    ├── workers/
    │   ├── env/
    │   │   └── env_worker.py        # EnvWorker（数据采集）
    │   └── reward/
    │       └── reward_worker.py     # RewardWorker（推理）
    │
    ├── runners/
    │   └── embodied_runner.py       # _apply_reward_model()
    │
    └── models/embodiment/reward/
        └── resnet_reward_model.py   # ResNet 实现

    examples/embodiment/
    ├── config/
    │   ├── maniskill_collect_reward_data.yaml
    │   ├── maniskill_train_reward_model.yaml
    │   └── maniskill_ppo_mlp_resnet_reward.yaml
    ├── train_embodied_agent.py      # 主入口
    └── train_reward_model.py        # 独立 reward 训练
