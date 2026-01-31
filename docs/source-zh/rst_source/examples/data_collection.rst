数据采集 (仿真)
===============

本文档介绍如何在 RLinf 中使用数据采集功能。
``DataCollectorWrapper`` 可以在 RL 训练过程中透明地采集环境交互数据，
适用于离线学习、奖励模型训练或行为分析等场景。

概述
----

数据采集系统会自动记录所有环境交互数据（观测、动作、奖励、终止状态），
并将其保存为 pickle 文件。每个并行环境的 episode 会保存为独立文件，
支持对成功和失败 episode 设置不同的采样率。

配置
----

在 YAML 配置文件的 ``env`` 部分添加 ``data_collection`` 配置：

.. code-block:: yaml

   env:
     group_name: "EnvGroup"
     enable_offload: False

     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       mode: "all"
       sample_rate_success: 1.0
       sample_rate_fail: 0.1

配置参数
~~~~~~~~

.. list-table:: 数据采集配置
   :header-rows: 1
   :widths: 25 15 60

   * - 参数
     - 默认值
     - 说明
   * - ``enabled``
     - ``False``
     - 是否启用数据采集
   * - ``save_dir``
     - 无
     - episode 数据保存目录
   * - ``mode``
     - ``"all"``
     - 采集模式：``"train"``、``"eval"`` 或 ``"all"``
   * - ``sample_rate_success``
     - ``1.0``
     - 成功 episode 的保存概率 (0.0-1.0)
   * - ``sample_rate_fail``
     - ``0.1``
     - 失败 episode 的保存概率 (0.0-1.0)

FullStateWrapper
~~~~~~~~~~~~~~~~

在 ``obs_mode: "rgb"`` 模式下，ManiSkill 默认返回的 ``states`` 是 29 维的部分状态（不含物体位姿）。
如果需要使用 MLP 策略同时采集图像数据，可启用 ``use_full_state`` 将状态替换为完整的 42 维。

.. list-table:: FullStateWrapper 配置
   :header-rows: 1
   :widths: 25 15 60

   * - 参数
     - 默认值
     - 说明
   * - ``use_full_state``
     - ``False``
     - 启用后，rgb 模式下的 states 会被替换为完整 42 维状态

数据格式
--------

文件命名
~~~~~~~~

每个 episode 保存为一个 pickle (``.pkl``) 文件：

.. code-block:: text

   rank_{worker_rank}_env_{env_idx}_episode_{episode_id}_{success|fail}.pkl

示例：``rank_0_env_5_episode_42_success.pkl``

数据结构
~~~~~~~~

pickle 文件包含一个字典：

.. code-block:: python

   {
       "mode": str,           # "train"、"eval" 或 "all"
       "rank": int,           # Worker 编号
       "env_idx": int,        # 环境索引
       "episode_id": int,     # Episode 编号
       "success": bool,       # 是否成功
       "observations": list,  # 观测列表 (长度 = num_steps + 1)
       "actions": list,       # 动作列表 (长度 = num_steps)
       "rewards": list,       # 奖励列表 (长度 = num_steps)
       "terminated": list,    # 终止标志列表 (长度 = num_steps)
       "truncated": list,     # 截断标志列表 (长度 = num_steps)
       "infos": list,         # info 字典列表 (长度 = num_steps)
   }

使用方法
--------

1. 在 YAML 中添加数据采集配置：

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       mode: "all"
       sample_rate_success: 1.0
       sample_rate_fail: 0.1

2. 正常运行训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_collect

3. 采集的数据会保存在指定的 ``save_dir`` 目录中。

加载数据
~~~~~~~~

.. code-block:: python

   import pickle

   with open("collected_data/rank_0_env_5_episode_42_success.pkl", "rb") as f:
       episode = pickle.load(f)

   print(f"Episode ID: {episode['episode_id']}")
   print(f"Success: {episode['success']}")
   print(f"步数: {len(episode['actions'])}")

配置示例
--------

.. code-block:: yaml

   defaults:
     - env/maniskill_pick_cube@env.train
     - env/maniskill_pick_cube@env.eval
     - model/mlp_policy@actor.model
     - training_backend/fsdp@actor.fsdp_config
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-0
       env: 0-0
       rollout: 0-0

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "maniskill_ppo_mlp_collect"
       logger_backends: ["tensorboard"]
     max_epochs: 1000

   env:
     group_name: "EnvGroup"
     enable_offload: False
     use_full_state: True

     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       mode: "all"
       sample_rate_success: 1.0
       sample_rate_fail: 0.1

     train:
       total_num_envs: 256
       init_params:
         obs_mode: "rgb"
     eval:
       total_num_envs: 32
       init_params:
         obs_mode: "rgb"

   actor:
     model:
       obs_dim: 42

最佳实践
--------

1. **磁盘空间**：RGB 图像占用大量存储空间，可调低 ``sample_rate_fail`` 减少数据量。

2. **数据平衡**：使用较高的 ``sample_rate_success`` (1.0) 和较低的 ``sample_rate_fail`` (0.01-0.1) 以聚焦成功轨迹。

3. **环境数量**：采集高分辨率图像时，可适当减少 ``total_num_envs`` 以控制存储占用。

4. **MLP + 图像采集**：使用 ``obs_mode: "rgb"`` + ``use_full_state: True`` + ``obs_dim: 42``，可在 MLP 策略训练时同步采集图像数据用于 Reward Model 训练。

