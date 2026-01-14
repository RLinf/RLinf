ResNet Reward Model 训练指南
============================

本指南介绍如何使用 ResNet reward model 训练 ManiSkill PickCube 任务。

概述
----

ResNet reward model 是一个基于图像的二分类器，用于预测机器人是否成功完成抓取任务。完整流程包括四个阶段：

1. **数据采集**: 使用 ``DataCollectorWrapper`` 采集带有成功/失败标签的 RGB 图像
2. **ResNet 训练**: 使用支持 FSDP 的 ``RewardWorker`` 训练 ResNet 二分类器
3. **策略预训练**: 使用环境的密集奖励训练初始策略
4. **ResNet Reward 训练**: 使用 ResNet reward 替代环境奖励继续训练

架构
----

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                          训练流程                                │
    └─────────────────────────────────────────────────────────────────┘
    
    阶段 1: 数据采集
    ┌───────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │   EnvWorker   │────▶│ DataCollectorWrapper│────▶│  pkl 文件    │
    │               │     │   (Env Wrapper)      │     │ (原始数据)   │
    └───────────────┘     └─────────────────────┘     └──────────────┘
    
    阶段 2: Reward Model 训练
    ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │  pkl 文件    │────▶│    RewardWorker     │────▶│  checkpoint  │
    │              │     │   (FSDP 支持)        │     │   .pt 文件   │
    └──────────────┘     └─────────────────────┘     └──────────────┘
    
    阶段 3-4: 使用 Reward Model 训练策略
    ┌───────────────┐     ┌─────────────────────┐     ┌──────────────┐
    │EmbodiedRunner │────▶│    RewardWorker     │────▶│   Rewards    │
    │               │     │ compute_rewards()   │     │              │
    └───────────────┘     └─────────────────────┘     └──────────────┘

前置条件
--------

- ManiSkill 环境已正确安装
- GPU 具有足够的显存用于渲染和训练

阶段 1: 数据采集
----------------

使用 ``DataCollectorWrapper`` 采集带有成功/失败标签的 RGB 图像。

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_collect_reward_data

这将：

- 使用密集奖励训练策略（与 ``maniskill_ppo_mlp`` 相同）
- 通过 ``DataCollectorWrapper`` 自动采集数据
- 将成功/失败样本保存到 pkl 文件

配置 (``maniskill_collect_reward_data.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    reward_data_collection:
      enabled: True
      save_dir: "${oc.env:EMBODIED_PATH}/data"
      target_success: 5000      # 目标成功样本数
      target_failure: 5000      # 目标失败样本数
      sample_rate_fail: 0.1     # 采样 10% 的失败帧
      sample_rate_success: 1.0  # 采样 100% 的成功帧

直接使用 DataCollectorWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你也可以在代码中直接使用 ``DataCollectorWrapper``：

.. code-block:: python

    from rlinf.envs.wrappers import DataCollectorWrapper
    
    # 包装环境
    env = DataCollectorWrapper(
        env=your_env,
        save_dir="./reward_data",
        target_success=5000,
        target_failure=5000,
        sample_rate_success=1.0,
        sample_rate_failure=0.1,
    )
    
    # 运行 episodes - 数据在 step() 中自动采集
    obs, info = env.reset()
    while not env.is_full:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    
    # 保存采集的数据
    env.save("reward_data.pkl")

数据将保存为单独的 pkl 文件（每个 episode 一个）::

    save_dir/
    ├── success/
    │   ├── episode_000000.pkl
    │   ├── episode_000001.pkl
    │   └── ...
    └── failure/
        ├── episode_000002.pkl
        └── ...

每个 pkl 文件包含::

    {
        "frames": [  # 50 帧的列表
            {
                "obs": ...,       # 观测 (numpy array)
                "action": ...,    # 执行的动作
                "reward": ...,    # 步骤奖励
                "done": ...,      # Episode 结束标志
                "grasp": ...,     # 该步骤是否抓取成功
                "success": ...,   # 该步骤任务是否成功
                "info": {...},    # 额外信息
            },
            ...
        ],
        "num_frames": 50,
        "grasp": True/False,      # Episode 级别抓取是否成功
        "success": True/False,    # Episode 级别任务是否成功
        "done": True,
        "label": 1/0,             # 1=成功, 0=失败
        "metadata": {...},
    }

阶段 2: 训练 ResNet Reward Model
--------------------------------

使用支持 FSDP 分布式训练的 ``RewardWorker`` 训练 ResNet 二分类器。

选项 A: 独立训练脚本
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python examples/embodiment/train_reward_model.py --config-name maniskill_train_reward_model

选项 B: 使用 RewardWorker 集成训练 (FSDP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于分布式训练，在配置中设置 ``reward_training``：

.. code-block:: yaml

    reward_training:
      enabled: True
      data_path: "${oc.env:EMBODIED_PATH}/data"
      epochs: 100
      micro_batch_size: 32
      global_batch_size: 64
      lr: 1.0e-4
      weight_decay: 1.0e-5
      save_dir: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints"

然后使用分布式训练运行：

.. code-block:: bash

    torchrun --nproc_per_node=4 examples/embodiment/train_embodied_agent.py \
        --config-name maniskill_ppo_mlp_resnet_reward \
        reward_training.enabled=True

``RewardWorker`` 自动使用：

- **FSDP** 跨 GPU 模型分片
- **DistributedSampler** 数据并行加载
- **梯度累积** 实现大的有效 batch size
- **混合精度** 训练

配置 (``maniskill_train_reward_model.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

阶段 3 & 4: 使用 ResNet Reward 训练策略
---------------------------------------

阶段 3: 使用密集奖励预训练策略（可选）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用环境的原生密集奖励训练初始策略：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

Checkpoint 将保存到 ``logs/<timestamp>-maniskill_ppo_mlp/maniskill_ppo_mlp/checkpoints/``。

阶段 4: 使用 ResNet Reward 继续训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``maniskill_ppo_mlp_resnet_reward.yaml`` 中更新 ``resume_dir``：

.. code-block:: yaml

    runner:
      # 设置为你的 maniskill_ppo_mlp checkpoint 路径
      resume_dir: "logs/<timestamp>-maniskill_ppo_mlp/maniskill_ppo_mlp/checkpoints/global_step_100"

然后运行：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

配置
----

关键参数 (``maniskill_ppo_mlp_resnet_reward.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    env:
      train:
        reward_render_mode: "episode_end"  # 必须与数据采集时一致
        show_goal_site: True               # 显示绿色目标标记
        init_params:
          control_mode: "pd_joint_delta_pos"  # 必须与数据采集时一致

    reward:
      use_reward_model: True
      reward_model_type: "resnet"
      mode: "replace"  # 用 ResNet reward 替换环境 reward
      alpha: 1.0
      
      resnet:
        checkpoint_path: "${oc.env:EMBODIED_PATH}/../../logs/reward_checkpoints/best_model.pt"
        threshold: 0.5
        use_soft_reward: False  # 二值 0/1 奖励

关键参数对齐
~~~~~~~~~~~~

以下参数 **必须** 与数据采集时使用的参数一致：

.. list-table::
   :header-rows: 1

   * - 参数
     - 值
     - 描述
   * - ``control_mode``
     - ``pd_joint_delta_pos``
     - 控制模式 (8 维动作空间)
   * - ``reward_render_mode``
     - ``episode_end``
     - 仅在 episode 结束时渲染图像
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
- ``env/reward`` 将显示较低的值（稀疏二值奖励的预期行为）

API 参考
--------

DataCollectorWrapper
~~~~~~~~~~~~~~~~~~~~

用于自动数据采集的环境 wrapper。

.. code-block:: python

    from rlinf.envs.wrappers import DataCollectorWrapper

.. list-table::
   :header-rows: 1

   * - 方法/属性
     - 描述
   * - ``step(action)``
     - 执行步骤并自动采集数据
   * - ``save(filename)``
     - 将采集的数据保存到 pkl 文件
   * - ``is_full``
     - 检查是否达到采集目标
   * - ``success_count``
     - 当前成功样本数
   * - ``failure_count``
     - 当前失败样本数
   * - ``get_statistics()``
     - 获取采集统计信息字典

RewardWorker
~~~~~~~~~~~~

支持 FSDP 分布式 reward model 训练和推理的 worker。

.. code-block:: python

    from rlinf.workers.reward.reward_worker import RewardWorker

.. list-table::
   :header-rows: 1

   * - 方法
     - 描述
   * - ``init_worker()``
     - 初始化 reward model (RewardManager 或基于规则)
   * - ``build_dataloader()``
     - 构建带有 DistributedSampler 的分布式 DataLoader
   * - ``run_training()``
     - 运行一步 FSDP 训练（带梯度累积）
   * - ``compute_rewards(input_channel, output_channel)``
     - 计算 rollout 结果的 rewards
   * - ``compute_batch_rewards_with_model(observations)``
     - 使用训练好的模型计算 rewards
   * - ``save_checkpoint(path, step)``
     - 保存模型 checkpoint
   * - ``load_checkpoint(path)``
     - 加载模型 checkpoint

RewardManager
~~~~~~~~~~~~~

带有注册表模式的统一 reward 计算接口。

.. code-block:: python

    from rlinf.algorithms.rewards.embodiment import RewardManager

.. list-table::
   :header-rows: 1

   * - 方法
     - 描述
   * - ``compute_rewards(observations, task_descriptions)``
     - 统一的 reward 计算
   * - ``register_model(name, cls)``
     - 注册新的模型类型
   * - ``get_available_models()``
     - 列出已注册的模型
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
    │   └── reward/
    │       └── reward_worker.py     # RewardWorker (FSDP 支持)
    │
    ├── algorithms/rewards/embodiment/
    │   ├── reward_manager.py        # 带注册表的 RewardManager
    │   ├── reward_data_collector.py # 旧版采集器 (已弃用)
    │   └── reward_model_trainer.py  # 独立训练器
    │
    └── models/embodiment/reward/
        ├── base_reward_model.py     # 基类
        ├── resnet_reward_model.py   # ResNet 实现
        └── qwen3_vl_reward_model.py # VLM 实现

    examples/embodiment/
    ├── train_embodied_agent.py      # 主入口 (创建 RewardWorker)
    ├── train_async.py               # 异步入口 (创建 RewardWorker)
    └── train_reward_model.py        # 独立 reward 训练
