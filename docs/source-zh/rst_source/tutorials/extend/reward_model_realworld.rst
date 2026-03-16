奖励模型集成 (真实世界)
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
      reward_mode: "per_step"
      
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
        dropout: 0.0
        image_size: [3, 128, 128]
        normalize: true
        precision: "bf16"
        checkpoint_path: "path/to/checkpoint/best_model"
        debug_save_dir: null

训练奖励模型 (以 peg insertion 任务为例)
----------------------------------------------

1. **采集数据**
   
   修改真机数据采集配置 ``examples/embodiment/config/realworld_collect_data.yaml``：

   .. code-block:: yaml

      cluster:
        node_groups:
          - hardware:
              type: Franka
              configs:
                - robot_ip: ROBOT_IP # 替换成你的机器人IP

      runner:
        num_data_episodes: 20 # 推荐不少于采集20条数据

      env:
        eval:
          override_cfg:
            target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0] # 设置为标定得到的目标末端位姿
            success_hold_steps: 20  # 推荐每个episode需要在目标位姿保持20个step才算成功


   需要注意的是， ``success_hold_steps`` 建议设置为20，以便在一条成功轨迹能捕捉到不少于20帧成功时的图像，以便快速收集大量成功姿态的图像，有助于后续对奖励模型的训练。
   
   配置修改完成后，可直接使用 ``collect_data.sh`` 启动真机数据采集程序：

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh

   采集的数据保存在 ``logs/[running-timestamp]/data.pkl``。

   采集完成后，将 ``data.pkl`` 上传至 GPU 训练服务器，以便进行后续步骤。

2. **数据处理**

   在进行 **训练模型** 之前，需要将 **采集数据** 中得到的数据进行数据清洗并处理成可以用来训练的格式。
   
   .. code-block:: bash

    # 指定数据路径
    python examples/embodiment/process_realworld_data.py /path/to/your/input_data.pkl
   
   ``process_realworld_data.py`` 会在 ``input_data.pkl`` 同一目录下生成 ``collected_data`` 文件夹，
   该文件夹内存放了真正用来训练奖励模型的数据。   

3. **训练模型**

   使用奖励训练配置（注意修改 ``data_path`` 为实际数据路径），真实世界的奖励模型的训练配置已在下面重映射：

   .. code-block:: bash

       bash examples/reward/run_reward_training.sh \
            --data /path/to/collected_data \
            data.num_samples_per_episode=0 \
            data.fail_success_ratio=3.0 \
            actor.global_batch_size=32 \
            actor.micro_batch_size=16 \
            runner.max_epochs=100 \
            runner.val_check_interval=10 \
            runner.save_interval=10 \
            runner.early_stop.patience=10 \
            data.debug_save_dir="logs/training_data_debug_realworld" \
            runner.logger.log_path="logs/reward_model_realworld"
   
   训练好的奖励模型文件将会存放在 ``logs/reward_model_realworld``。

4. **在 RL 训练中使用**

   修改真机训练配置 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``：

   .. code-block:: yaml

       reward:
          use_reward_model: True
          reward_mode: "per_step"
          combine_mode: "replace"
          checkpoint_path: "logs/reward_model_realworld/checkpoints/best_model"  # 选择存放奖励模型的文件夹
   
   开始训练：

   .. code-block:: bash

       bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

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

