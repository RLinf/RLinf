真机数据采集
============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/data_formats_zh.svg
   :align: center
   :width: 100%
   :alt: RLinf 数据格式流转图：采集来源、三种存储格式与下游消费阶段

   数据采集来源、落盘格式与下游训练阶段。

采集真机示范或仿真 rollout 数据时，所有输出统一写入 ``save_dir``\ （默认为 ``${runner.logger.log_path}``）。RLinf 提供两个采集入口：

- **真机遥操作**：运行 ``examples/embodiment/collect_data.sh`` 录制示范。``export_format`` 可指定单个格式，也可将 ``replay_buffer`` 与一种 episode 格式组合写出；省略该字段时默认同时写出 ``replay_buffer`` 和 ``lerobot_dataset``。

- **训练或评估**：在训练或评估配置中开启 ``env`` 下的 ``data_collection``。每个 env worker 一次写出一种 episode 格式，可选 ``lerobot_dataset`` 或 ``pickle_episode``。

两个入口使用同一套格式名称。单次运行不能同时选择 ``pickle_episode`` 和 ``lerobot_dataset``，训练与评估中的 env worker 也不写出 ``replay_buffer``。

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - 数据格式
     - 保存目录
     - 下游用途
   * - ``replay_buffer``
     - ``{save_dir}/demo_buffer/``
     - RLPD 训练，通过 ``algorithm.demo_buffer.load_path`` 加载（如 Franka 插孔和 DOS-W1）。
   * - ``lerobot_dataset``
     - ``{save_dir}/lerobot_dataset/``
     - OpenPI SFT、HG-DAgger 及其他读取 LeRobot shard 的模仿学习管线。
   * - ``pickle_episode``
     - ``{save_dir}/pickle_episode/``
     - 奖励模型预处理：``examples/reward/preprocess_reward_dataset.py`` 构建 ResNet 二分类数据集，``examples/reward/vlm_trend/`` 生成 VLM trend 标签。

采集真机遥操作数据
------------------

在 ``examples/embodiment/config/realworld_collect_data.yaml`` 中配置机械臂 IP、遥操作设备、目标位姿、导出格式和目标条数：

.. code-block:: yaml

   cluster:
     node_groups:
       - label: franka
         hardware:
           configs:
             - robot_ip: "192.168.1.100"
   env:
     eval:
       teleop: spacemouse          # spacemouse、gello、pico 或 none
       override_cfg:
         target_ee_pose: [0.5, 0.0, 0.3, 0.0, 3.14, 0.0]
         success_hold_steps: 1     # 到达目标后连续保持多少步算成功
       data_collection:
         enabled: True
         save_dir: ${runner.logger.log_path}
         export_format: [replay_buffer, lerobot_dataset]
   runner:
     num_data_episodes: 20         # 成功条数达到后退出

在仓库根目录启动数据采集（可传入其他配置名作为第一个参数）：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh
   bash examples/embodiment/collect_data.sh realworld_collect_data_gello

成功条数达到 ``runner.num_data_episodes`` 后脚本自动退出。默认输出写入 ``logs/{timestamp}/demo_buffer/``\ （标记为专家数据的成功 ``.pt`` 轨迹）与 ``logs/{timestamp}/lerobot_dataset/rank_0/id_0/``\ （导出 pickle 格式时为 ``pickle_episode/*.pkl``）。

再次采集并指向同一个 ``demo_buffer`` 目录即可增量追加轨迹；LeRobot 数据集设置 ``resume: true`` 后会沿用已有分片编号，不覆盖已完成的数据。

在训练或评估中采集数据
----------------------

在训练或评估 YAML 中开启 ``env.eval``\ （或 ``env.train``）下的 ``data_collection``。例如，``maniskill_ppo_mlp_collect`` 采集成功的 pickle episode 供奖励模型训练：

.. code-block:: yaml

   env:
     eval:
       data_collection:
         enabled: True
         save_dir: ${runner.logger.log_path}
         export_format: pickle_episode
         only_success: True

pickle 格式的 ``.pkl`` 文件落在 ``{save_dir}/pickle_episode/``；将 ``export_format`` 设为 ``lerobot_dataset`` 时，分片落在 ``{save_dir}/lerobot_dataset/rank_*/id_*/``。

合并流式采集分片
----------------

录制长时任务（如双臂 YAM）时，设置 ``export_format: lerobot_dataset`` 与 ``streaming: true``。图像帧随录随落盘，内存占用不随 episode 变长。

每条保留的录制单独生成一个分片，成功与失败的录制都会落盘并标上 ``is_success``\ （流式模式下 ``only_success`` 不生效）。操作员丢弃的片段与录制队列溢出片段不会进入正式数据集。

训练前合并这些分片，并过滤掉失败的录制：

.. code-block:: bash

   python -m toolkits.lerobot.merge_lerobot_datasets \
       --source-dir logs/{timestamp}/lerobot_dataset \
       --output-dir logs/{timestamp}/lerobot_merged \
       --only-success

检查采集数据
------------

训练前先可视化 replay buffer 或 LeRobot 分片，核对相机画面、状态和动作：

.. code-block:: bash

   python -m toolkits.replay_buffer.visualize \
       --replay_dir logs/{timestamp}/demo_buffer

   python -m toolkits.lerobot.visualize_lerobot_dataset \
       --dataset-path logs/{timestamp}/lerobot_dataset/rank_0/id_0 \
       --output-dir logs/{timestamp}/lerobot_visualized

.. dropdown:: 落盘格式细节与自定义环境包装

   采集器在后台线程写盘，环境关闭时刷出剩余数据，不阻塞训练步。多个 env worker 各自使用独立的 ``rank``，文件名与分片目录都会带上该编号。

   **replay_buffer**

   只保存成功的 episode。真机采集开启 ``manual_episode_control_only`` 后仅以 ``manual_done`` 判成功，否则 ``reward >= 0.5`` 或 ``manual_done`` 均算成功。

   无效录制与丢弃片段（``recording_invalid``、``episode_discarded``）既不写入也不计入成功条数。每条轨迹保存为 ``trajectory_*.pt``，同目录包含 ``metadata.json`` 与 ``trajectory_index.json``。

   每步都带有专家干预标记，供 RLPD 区分专家示范与在线 policy 数据。张量形状随机器人与相机配置变化，图像在 episode 结束前留在内存中。

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

   **pickle_episode**

   每条 episode 结束时写入一个 ``.pkl`` 文件；非流式模式下开启 ``only_success: true`` 会丢弃失败 episode。

   .. code-block:: text

      rank_{rank}_env_{env_idx}_episode_{episode_id}_step_{global_step}_{success|fail}.pkl

   文件内容结构：

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
          "infos": list,         # 奖励模型预处理读取其中的主图像和逐步 success
      }

   **lerobot_dataset**

   采集器将一个或多个 episode 写入 ``lerobot_dataset/rank_{rank}/id_{N}/``，流式模式下每个分片只保存一条保留的录制。设置 ``resume: true`` 会接续已有分片编号，不改动已写完的数据。

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
        - 其他视角；无额外视角时为空
      * - ``state`` / ``actions``
        - 状态向量和动作向量，``float32``
      * - ``timestamp`` / ``frame_index``
        - 帧时间（秒）和 episode 内序号
      * - ``episode_index`` / ``index`` / ``task_index``
        - 全局 episode 号、全局帧号、任务号
      * - ``done`` / ``is_success``
        - 该步是否为 episode 最后一帧；该 episode 是否成功

   观测字段的回退查找顺序：

   - **主视角图像**：依次查找 ``main_images``、``image``、``full_image``。落在 ``[0, 1]`` 的浮点图像会乘 255 转为 uint8。

   - **额外视角图像**：依次查找 ``extra_view_images``、``extra_view_image``。堆叠的 ``[N, H, W, C]`` 张量拆分为 ``extra_view_image-0``、``extra_view_image-1`` 等列。

   - **状态向量**：优先读取 ``states``，其次 ``state``。

   流式存储与成功判定规则：

   - 开启 ``streaming: true`` 后，每帧各视角立即写成未压缩 PNG，episode 结束再生成 parquet。由于 parquet 原样保存这些 PNG 字节，单路 640x480 图像每帧约占 0.9 MB 磁盘空间。

   - 成功与失败的录制都会落盘并在结束时写入 ``is_success``，同目录的 ``stream_frames.jsonl`` 记录每帧状态与动作。操作员丢弃会删除当前未发布的 episode；录制队列溢出的片段会移入该分片的 ``invalid_episodes/`` 目录，不作为正式 episode 发布。

   - 每步成功状态优先检查 ``final_info`` 与 ``episode`` 子字典，再检查顶层 info 字段。

   - 只要 ``success_once``、``success_at_end`` 或 ``success`` 任一为真即判成功；全程缺少这些键时，回退使用采集器维护的成功标志。

   **在自定义脚本中包装环境**

   将父目录传给 ``save_dir``\ （包装器自动创建格式子目录），结束时关闭环境以刷出剩余 episode：

   .. code-block:: python

      env = CollectEpisode(
          env=base_env,
          save_dir="./logs/run",
          export_format="lerobot_dataset",   # 或 "pickle_episode"
          robot_type="panda",
          fps=10,
          only_success=True,
      )
      env.close()   # 刷出最后一批 episode
