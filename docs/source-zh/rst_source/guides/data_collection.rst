数据采集
========

采集有两个入口，写出的目录相同。``save_dir`` 默认是
``${runner.logger.log_path}``。

- **真机遥操**：``examples/embodiment/collect_data.sh`` 启动
  ``collect_real_data.py``。``export_format`` 可以选一种，也可以让
  ``replay_buffer`` 和一种 episode 格式一起写。
- **训练或评估**：在 ``env`` 的 ``data_collection`` 里打开。env worker 一次只写一种
  episode 格式。

真机脚本只接受 ``replay_buffer``、``pickle_episode``、``lerobot_dataset``。
``pickle_episode`` 与 ``lerobot_dataset`` 不能同时选择。不写 ``export_format``
时，默认写出 ``replay_buffer`` 和 ``lerobot_dataset``。训练里的 env worker 还接受
``pickle`` 和 ``lerobot``，分别等同于后两个名字。

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - 数据格式
     - 保存目录
     - 谁来读
   * - ``replay_buffer``
     - ``{save_dir}/demo_buffer/``
     - RLPD。训练配置 ``algorithm.demo_buffer.load_path`` 指向该目录，例如 Franka 插孔和 DoSW1。
   * - ``lerobot_dataset``
     - ``{save_dir}/lerobot_dataset/``
     - OpenPI SFT、HG-DAgger，以及其他读取 LeRobot shard 的模仿学习。
   * - ``pickle_episode``
     - ``{save_dir}/pickle_episode/``
     - 奖励模型预处理。``examples/reward/preprocess_reward_dataset.py`` 生成 ResNet 二分类数据；``examples/reward/vlm_trend/`` 生成 VLM trend 标签。

真机同时保存 replay buffer 和 LeRobot：

.. code-block:: yaml

   env:
     eval:
       data_collection:
         enabled: True
         save_dir: ${runner.logger.log_path}
         export_format: [replay_buffer, lerobot_dataset]

流式 LeRobot（例如双臂 YAM）只设 ``export_format: lerobot_dataset`` 和
``streaming: true``。帧随录随写，内存不随 episode 变长。成功和失败的 episode
都会进入 dataset，并标上 ``is_success``；``only_success`` 在流式模式下不起作用。
操作员丢弃的录制和录制队列溢出不会进入正式 dataset。

训练中采集 pickle，供奖励模型使用：

.. code-block:: yaml

   env:
     eval:
       data_collection:
         enabled: True
         save_dir: ${runner.logger.log_path}
         export_format: pickle
         only_success: True

``maniskill_ppo_mlp_collect`` 就是这个用法。``.pkl`` 在
``{save_dir}/pickle_episode/``。把 ``export_format`` 改成 ``lerobot`` 时，shard 在
``{save_dir}/lerobot_dataset/rank_*/id_*/``。

真机遥操数采步骤
----------------

在 ``examples/embodiment/config/realworld_collect_data.yaml`` 里填写机器人、
遥操方式和目标位姿：

.. code-block:: yaml

   cluster:
     node_groups:
       hardware:
         configs:
           robot_ip: "192.168.1.100"
   env:
     eval:
       teleop: spacemouse          # spacemouse、gello、pico 或 none
       override_cfg:
         target_ee_pose: [0.5, 0.0, 0.3, 0.0, 3.14, 0.0]
         success_hold_steps: 1     # 到达目标后连续保持多少步算成功
   runner:
     num_data_episodes: 20         # 成功条数达到后退出

然后：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh
   bash examples/embodiment/collect_data.sh realworld_collect_data_gello

成功次数达到 ``runner.num_data_episodes`` 后退出。``save_dir`` 为日志目录时，目录是：

.. code-block:: text

   logs/{timestamp}/
   ├── demo_buffer/
   ├── lerobot_dataset/rank_0/id_0/
   └── pickle_episode/*.pkl

``demo_buffer`` 里每条成功轨迹是一个 ``.pt``，``intervene_flags`` 全为 1，表示专家数据。
追加数据时，让新的一次采集指向同一个 ``demo_buffer`` 目录。

检查数据：

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/{timestamp}/demo_buffer

   python toolkits/lerobot/visualize_lerobot_dataset.py \
       --dataset-path logs/{timestamp}/lerobot_dataset/rank_0/id_0 \
       --output-dir logs/{timestamp}/lerobot_visualized

数据格式细节
------------

写入由 ``CollectEpisode`` 在后台线程完成，``env.close()`` 时做最后一次刷盘，
不挡住训练的 step。多个 env worker 各自使用不同 ``rank``，文件名和 shard
目录都带这个编号。

**replay_buffer**

只保存成功 episode。真机采集里，``manual_episode_control_only`` 为真时只看
``manual_done``，否则 ``reward >= 0.5`` 或 ``manual_done`` 算成功。
``recording_invalid`` 和 ``episode_discarded`` 不写入、也不计入成功次数。
每条轨迹一个 ``trajectory_*.pt``，同目录还有 ``metadata.json`` 和
``trajectory_index.json``。``intervene_flags`` 全为 1，RLPD 用它区分专家数据和
在线策略数据。指向同一目录再次采集时增量写入。张量形状随机器人和相机变化，
不是格式的一部分。

.. code-block:: python

   {
       "transitions": {
           "obs": {"states", "main_images"},
           "next_obs": {"states", "main_images"},
           "action": "float32[T, action_dim]",
           "rewards": "float32[T, 1]",
           "dones": "bool[T, 1]",
           "terminations": "bool[T, 1]",
           "truncations": "bool[T, 1]",
       },
       "intervene_flags": "全 1",
   }

图像在 episode 结束前留在内存里，用量随长度增长。

**pickle_episode**

``CollectEpisode`` 在 episode 结束时写一个 ``.pkl``。非流式且
``only_success: true`` 时丢掉失败 episode。文件名：

.. code-block:: text

   rank_{rank}_env_{env_idx}_episode_{episode_id}_step_{global_step}_{success|fail}.pkl

内容：

.. code-block:: python

   {
       "rank": int,
       "env_idx": int,
       "episode_id": int,
       "step": int,
       "success": bool,
       "observations": list,  # 长度 = num_steps + 1，第 0 项来自 reset()
       "actions": list,
       "rewards": list,
       "terminated": list,
       "truncated": list,
       "infos": list,         # 奖励模型读取其中的主图像和逐步 success
   }

**lerobot_dataset**

``CollectEpisode`` 把一个或多个 episode 写成一个 shard：
``lerobot_dataset/rank_{rank}/id_{N}/``。``resume: true`` 时从已有 shard
往后编号，不改已写完的数据。

.. code-block:: text

   id_0/
   ├── meta/info.json          # fps、robot_type、维度
   ├── meta/episodes.jsonl
   ├── meta/tasks.jsonl
   ├── meta/stats.json
   └── data/chunk-000/episode_*.parquet

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 列
     - 内容
   * - ``image``
     - 主相机图像，uint8
   * - ``extra_view_image`` / ``extra_view_image-N``
     - 其他视角。没有额外视角时为空
   * - ``state`` / ``actions``
     - 状态向量和动作向量，``float32``
   * - ``timestamp`` / ``frame_index``
     - 帧时间（秒）和 episode 内序号
   * - ``episode_index`` / ``index`` / ``task_index``
     - 全局 episode 号、全局帧号、任务号
   * - ``done`` / ``is_success``
     - 该步是否为 episode 最后一帧；该 episode 是否成功

图像从观测里按 ``main_images``、``image``、``full_image`` 的顺序取主视角，按
``extra_view_images``、``extra_view_image`` 取其余视角；``[N, H, W, C]``
会拆成 ``extra_view_image-0``、``extra_view_image-1``。状态键优先
``states``，其次 ``state``。浮点图像若在 ``[0, 1]`` 会乘 255 变成 uint8。

在自己的脚本里包一层环境时，把父目录传给 ``save_dir``，格式子目录由 wrapper 添加：

.. code-block:: python

   env = CollectEpisode(
       env=base_env,
       save_dir="./logs/run",
       export_format="lerobot",   # 或 "pickle"
       robot_type="panda",
       fps=10,
       only_success=True,
   )
   env.close()   # 刷出最后一批 episode

``streaming: true`` 时每帧立刻写成 PNG，episode 结束再写 parquet。成功和失败
都保存，``is_success`` 在结束时写入。操作员丢弃会删掉当前未发布的 episode；
录制队列溢出的片段移到该 shard 的 ``invalid_episodes/``，不作为正式 episode。
流式 shard 旁还有 ``stream_frames.jsonl``，记录每帧的 ``state`` 和 ``actions``。

每步 info 先看 ``final_info`` 和 ``episode`` 里的 ``success_once``、
``success_at_end``、``success``，这些都没有再看 info 根字段。任一来源为真即记为成功。
全程都没有这些字段时，用采集过程中维护的成功标志。
