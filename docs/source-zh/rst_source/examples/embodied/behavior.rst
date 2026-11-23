BEHAVIOR-1K 强化学习
====================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   BEHAVIOR 基准（图片来源：`BEHAVIOR <https://behavior.stanford.edu>`__）。

`BEHAVIOR <https://behavior.stanford.edu>`__ 是一个基于 NVIDIA IsaacSim / OmniGibson 的日常家居活动基准。双臂 R1 Pro 机器人执行长程操作任务；RLinf 借助它用强化学习微调视觉-语言-动作（VLA）policy。

概览
----------------------------------------

在 BEHAVIOR 家居任务上用 PPO 微调 VLA。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      50 个 BEHAVIOR-1K 任务

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 支持光追的 GPU

| **你将完成：** 安装 IsaacSim 依赖 → 下载资产与基座模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · IsaacSim 4.5 与 BEHAVIOR-1K 资产（>30 GB）· 基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 任务
     - 来自 BEHAVIOR-1K 的 50 个家居操作任务（通过 ``task_idx`` 0–49 选择）。
   * - 机器人
     - IsaacSim / OmniGibson 上的双臂 R1 Pro。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测
     - 头部相机 RGB（720×720）以及左右腕部 RealSense RGB（480×480）。
   * - 动作
     - 23 维连续动作：3 自由度底盘（x, y, rz）、4 自由度躯干、2×7 自由度手臂、2×1 自由度平行夹爪。

安装与运行
----------

.. warning::

   安装前先检查 IsaacSim 软硬件要求：

   - https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html
   - https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   Hopper 架构 GPU 需要 NVIDIA driver 570 或更高版本。A100、H100 等不支持光线追踪的 GPU 可能会让 BEHAVIOR 场景出现严重渲染伪影。优先使用 RTX 30/40 系列或更新的 GPU，以获得更稳定的视觉效果和训练体验。

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env behavior && source .venv/bin/activate

      **2. 下载资源与模型**

      下载 IsaacSim 4.5、BEHAVIOR-1K 资产（超过 30 GB，在已激活的虚拟环境中运行）与 OpenVLA-OFT 基座 checkpoint，并在启动 shell 中保留 ``ISAAC_PATH`` 与 ``OMNIGIBSON_DATA_PATH``：

      .. code-block:: bash

         export ISAAC_PATH="$PWD/isaac-sim"
         export OMNIGIBSON_DATA_PATH="$PWD/BEHAVIOR-1K-datasets"
         mkdir -p "$ISAAC_PATH" "$OMNIGIBSON_DATA_PATH"
         curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o "$ISAAC_PATH/isaac-sim.zip"
         unzip "$ISAAC_PATH/isaac-sim.zip" -d "$ISAAC_PATH" && rm "$ISAAC_PATH/isaac-sim.zip"
         python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
         python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
         python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
         hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir ./RLinf-OpenVLAOFT-Behavior

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 30 20 50

         * - 模型
           - 算法
           - 配置
         * - OpenVLA-OFT
           - PPO
           - ``behavior_ppo_openvlaoft.yaml``

      通过命令行覆盖 ``actor.model.model_path`` 启动训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-Behavior

   .. tab-item:: OpenPI
      :sync: openpi

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env behavior && source .venv/bin/activate

      **2. 下载资源与模型**

      下载 IsaacSim 4.5、BEHAVIOR-1K 资产（超过 30 GB，在已激活的虚拟环境中运行）与 OpenPI 基座 checkpoint，并在启动 shell 中保留 ``ISAAC_PATH`` 与 ``OMNIGIBSON_DATA_PATH``：

      .. code-block:: bash

         export ISAAC_PATH="$PWD/isaac-sim"
         export OMNIGIBSON_DATA_PATH="$PWD/BEHAVIOR-1K-datasets"
         mkdir -p "$ISAAC_PATH" "$OMNIGIBSON_DATA_PATH"
         curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o "$ISAAC_PATH/isaac-sim.zip"
         unzip "$ISAAC_PATH/isaac-sim.zip" -d "$ISAAC_PATH" && rm "$ISAAC_PATH/isaac-sim.zip"
         python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
         python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
         python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
         hf download RLinf/RLinf-Pi0-Behavior --local-dir ./RLinf-Pi0-Behavior

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 30 20 50

         * - 模型
           - 算法
           - 配置
         * - π₀
           - PPO
           - ``behavior_ppo_openpi.yaml``
         * - π₀.₅
           - PPO
           - ``behavior_ppo_openpi_pi05.yaml``

      通过命令行覆盖 ``actor.model.model_path`` 启动训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi \
           actor.model.model_path=./RLinf-Pi0-Behavior

.. warning::

   将 BEHAVIOR env worker 放在从 0 开始的 GPU 上。IsaacSim 在 env worker 从更靠后的 GPU rank 启动时可能卡住。

.. warning::

   已知问题：在当前 BEHAVIOR 设置下，OpenVLA-OFT / π₀ 的训练成功率（``env/success_once``）可能保持为 0。该问题会在后续版本修复。

.. admonition:: 进一步配置
   :class: note

   - BEHAVIOR 吞吐调优 → 先增加 env GPU 数量，再调 ``env.num_env_subprocess`` 和 ``env.train.total_num_envs``。
   - 每个 BEHAVIOR 进程可能占用约 10 GiB 显存；请按 GPU 显存调节 subprocess 数量。
   - 缓存任务实例 → 使用 BEHAVIOR 实例生成脚本（见下方折叠区）与 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 生成。
   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`

.. dropdown:: 进阶配置与缓存任务实例（可选）

   BEHAVIOR 环境由 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 驱动。RLinf 先加载 OmniGibson 的基础配置（``base_config_name``），再应用 ``omni_config`` 覆盖项。下表中的字段控制 reset 行为、场景加载、仿真器频率与吞吐，大多有合理默认值，仅在自定义任务或调优性能时需要修改。

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - 配置项
        - 含义
      * - ``base_config_name``
        - 在 ``omni_config`` 覆盖前加载的 OmniGibson 基础配置（如 ``r1pro_behavior``）。
      * - ``omni_config.task.type`` / ``omni_config.scene.type``
        - 显式保留 ``BehaviorTask`` / ``InteractiveTraversableScene``，以确保覆盖后选中预期的上游 OmniGibson 类。
      * - ``task_idx``
        - 当前任务 id（0–49），映射到 ``task.activity_name``。
      * - ``omni_config.task.instance_resample_mode``
        - reset 时的实例切换：``disabled``\ （加载固定的 ``activity_instance_id``）、``offline``\ （启动时扫描一次 ``activity_instance_dir``，每次 reset 采样一个缓存实例——``*_template.json`` 走较重的场景重载路径，``*_template-tro_state.json`` 走较轻的原地路径），或 ``online``\ （需 ``online_object_sampling: True`` 且 ``use_presampled_robot_pose: False``）。
      * - ``omni_config.task.activity_instance_dir``
        - 缓存实例 JSON 目录（``*_template.json`` / ``*_template-tro_state.json``），供 ``offline`` 模式以及 ``disabled`` 模式下的固定 id 加载使用。
      * - ``omni_config.task.instance_file_format``
        - 缓存实例格式：``template``\ （完整重载）或 ``tro_state``\ （仅任务相关、轻量）。RLinf 接受不含 ``robot_poses`` 的 ``tro_state`` 文件，此时会清除过期的缓存机器人位姿元数据，reset 回退到任务默认机器人位姿。
      * - ``omni_config.scene.partial_scene_load``
        - 为 ``true`` 时自动按 ``activity_name`` 填充 ``scene.load_room_types``\ （减少启动时间与内存），需要 ``activity_name`` 与 ``scene_model``。为 ``false`` 或省略时需显式设置 ``load_room_types``。
      * - ``camera.head_resolution`` / ``camera.wrist_resolution``
        - 头部 / 腕部相机分辨率（默认 720×720 / 480×480，应用到 R1Pro 传感器）。
      * - ``omni_config.env.action_frequency`` / ``rendering_frequency`` / ``physics_frequency``
        - 动作 / 渲染 / 物理步进频率（常用默认 30 / 30 / 120）。越高越慢。
      * - ``omni_config.env.automatic_reset``
        - 保持 ``False``——reset 由 RLinf 训练/评估循环显式控制。
      * - ``omni_config.env.flatten_obs_space`` / ``flatten_action_space``
        - 保持 ``False`` 以保留结构化的观测 / 动作空间。
      * - ``omni_config.macro.use_gpu_dynamics``
        - ``False`` 通常更快；仅在需要粒子 / 流体时启用。
      * - ``omni_config.macro.enable_flatcache``
        - ``True`` 通常提升大场景性能。
      * - ``omni_config.macro.enable_object_states``
        - 保持 ``True``——``BehaviorTask`` 依赖物体状态。
      * - ``omni_config.macro.enable_transition_rules``
        - ``True`` 启用基于转移规则的状态变化（如切割、烹饪）。
      * - ``omni_config.macro.use_numpy_controller_backend``
        - ``True`` 使用 numpy 控制器后端，在单进程 / 中等并行下通常更快。
      * - ``skip_intermediate_renders``
        - 为 ``True`` 时跳过动作 chunk 内中间观测的采集（显著提升环境速度）。此时保存的视频只显示 policy 在 chunk 边界观测到的帧。
      * - ``num_env_subprocess``
        - 将 ``num_envs`` 拆分到多个子进程，每个子进程承载独立的 Isaac/OmniGibson 仿真。默认 ``1``。**约束：** ``num_envs`` 必须能被 ``num_env_subprocess`` 整除。提高该值可缓解环境步进瓶颈，但会成倍增加进程数与内存。

   **生成缓存任务实例**

   在仓库根目录运行 BEHAVIOR 实例生成脚本，直接从 ``behavior_r1pro.yaml`` 生成 ``*_template.json`` 与 ``*_template-tro_state.json`` 文件（它会读取 ``scene_model``、``activity_name``、``activity_definition_id``、机器人配置与房间加载设置，并临时切换为在线物体采样）。若设置了 ``activity_instance_dir`` 则写入该目录，否则写入 ``OMNIGIBSON_DATA_PATH`` 默认的 ``2025-challenge-task-instances`` 目录；可用 ``--output-dir`` 覆盖：

   .. code-block:: bash

      python rlinf/envs/sim/behavior/instance_generator.py \
        --config examples/embodiment/config/env/behavior_r1pro.yaml \
        --output-format template \
        --start-idx 1 --end-idx 50

      python rlinf/envs/sim/behavior/instance_generator.py \
        --config examples/embodiment/config/env/behavior_r1pro.yaml \
        --output-format tro_state \
        --start-idx 1 --end-idx 50

   生成的文件名遵循 ``<scene_model>_task_<activity_name>_<activity_definition_id>_<activity_instance_id>_template(.json|-tro_state.json)``，因此 ``--start-idx`` / ``--end-idx`` 决定 ``activity_instance_id`` 范围。``tro_state`` 输出仅在任务元数据提供时包含顶层 ``robot_poses``，否则省略该键，reset 回退到任务默认机器人位姿。BEHAVIOR-1K 上游的 ``multiply_b1k_tasks.py`` 仍可使用，但推荐 RLinf 的生成器，因为它直接读取 RLinf YAML 并保留 ``activity_definition_id``。

独立评测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

任何已转换为 PyTorch 格式的 ``pi05`` checkpoint（例如使用 `openpi-comet <https://github.com/mli0603/openpi-comet>`__ 转换的 `OpenPI-Comet <https://huggingface.co/sunshk/openpi_comet/tree/main>`__，或来自 :doc:`sft_openpi` 与 OpenPI checkpoint 转换器的权重）都可以使用 ``evaluations/behavior/behavior_openpi_pi05_eval.yaml`` 评估。归一化统计直接从 checkpoint 打包的 ``physical-intelligence/behavior/norm_stats.json`` 读取。如需更长轨迹，调高 ``env.eval.max_episode_steps`` 和 ``env.eval.max_steps_per_rollout_epoch``\ （例如 ``4096``）。

独立评测请走 :doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>`，其中包含 ``ISAAC_PATH`` / ``OMNIGIBSON_DATA_PATH`` 设置、评估 checkpoint 路径配置、``behavior_openpi_pi05_eval`` 启动命令与结果解读。

.. note::

   评估时流匹配动作头使用随机采样噪声，重复运行之间单次轨迹与成功次数会有轻微差异。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 关注 ``env/success_once``；如需保存评测视频，设置 ``env.eval.video_cfg.save_video=True``，更多日志后端与指标见 :doc:`训练指标 <../../reference/metrics>`。

对于 BEHAVIOR 实验，我们参考了 `Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_ 并做了少量修改。感谢作者发布开源代码。
