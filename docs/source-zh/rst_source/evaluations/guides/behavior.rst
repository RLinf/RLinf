BEHAVIOR-1K 评测
================

概览
----

BEHAVIOR-1K 是基于 OmniGibson 与 Isaac Sim 4.5 的大规模家居场景仿真基准，控制双臂 R1 Pro 机器人在 50 个任务中完成抓取、放置与整理等操作。RLinf 支持在 BEHAVIOR 环境中并行评测 OpenPI 等 VLA policy，并输出 ``eval/success_once`` 等任务指标。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      BEHAVIOR-1K（50 个家居任务，``behavior_r1pro``）

   .. grid-item-card:: 支持模型
      :text-align: center

      OpenPI π₀.₅（示例） · OpenVLA-OFT（派生配置）

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/behavior/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`BEHAVIOR-1K 训练 <../../examples/embodied/behavior>`

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: OpenPI π₀.₅

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env behavior && source .venv/bin/activate

      **2. 下载 Isaac Sim 4.5、BEHAVIOR-1K 资产（超 30 GB）与模型**

      .. code-block:: bash

         export ISAAC_PATH="$(pwd)/isaac-sim"
         export OMNIGIBSON_DATA_PATH="$(pwd)/BEHAVIOR-1K-datasets"
         mkdir -p "$ISAAC_PATH" "$OMNIGIBSON_DATA_PATH"
         curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o "$ISAAC_PATH/isaac-sim.zip"
         unzip "$ISAAC_PATH/isaac-sim.zip" -d "$ISAAC_PATH" && rm "$ISAAC_PATH/isaac-sim.zip"
         python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
         python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
         python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
         hf download RLinf/RLinf-Pi0-Behavior --local-dir ./RLinf-Pi0-Behavior

      **3. 启动评测**

      .. code-block:: bash

         bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi0-Behavior

``run_eval.sh`` 会根据 ``ISAAC_PATH`` 与 ``OMNIGIBSON_DATA_PATH`` 自动补全 ``OMNIGIBSON_DATASET_PATH``、``OMNIGIBSON_KEY_PATH``、``OMNIGIBSON_ASSET_PATH``、``EXP_PATH`` 与 ``CARB_APP_PATH``。使用第三方 OpenPI 权重（如 OpenPI-Comet）前须转换为 PyTorch 格式；OpenVLA-OFT 可从 ``examples/embodiment/config/`` 中的训练配置派生评测 YAML（见 :doc:`../reference/configuration`）。

.. dropdown:: 硬件要求、Isaac Sim 与示例配置

   BEHAVIOR 依赖 Isaac Sim 4.5（`依赖说明 <https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html>`_）：

   - 使用支持 Ray Tracing 的 GPU（RTX 30/40 系列或更新）；A100、H100 等无 RT 能力的卡渲染画面会出现马赛克或模糊。
   - Hopper 及以上架构 GPU 须使用 570 及以上版本 NVIDIA 驱动。
   - 预构建容器镜像请访问 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

   .. list-table::
      :header-rows: 1
      :widths: 40 30 30

      * - 配置文件
        - 环境预设
        - 模型
      * - ``behavior_openpi_pi05_eval.yaml``
        - ``behavior_r1pro``
        - π₀.₅（``openpi``）

   若 ``evaluations/behavior/<config>.yaml`` 不存在，``run_eval.sh`` 仅对 ``examples/embodiment/config/`` 中已定义 ``rollout.model``\ （含 ``model_type`` 与 ``num_action_chunks``）的评测就绪配置生效；纯训练配置须按 :doc:`../reference/configuration` 将 ``@actor.model`` 重绑定到 ``@rollout.model``。保持 ``action_dim: 23``、``num_action_chunks: 32`` 与 ``openpi.config_name: pi05_behavior`` 与训练一致。通用 ``env.eval`` 字段见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`。

.. _behavior-eval-config:

.. dropdown:: 评测配置详解

   每次启动评测 **单个任务**\ （由 ``omni_config.task.activity_name`` 指定，默认 ``house_double_floor_lower`` 场景下的 ``turning_on_radio``）。评测多个任务时可在脚本中循环修改 ``activity_name``；评测多个实例时使用 ``instance_resample_mode: offline`` 并通过 ``rollout_epoch`` 取平均。

   **实例采样模式**\ （``env.eval.omni_config.task.instance_resample_mode``）

   - ``disabled``\ （默认）：每次 reset 加载 ``activity_instance_id`` 对应的固定实例（若设置了 ``activity_instance_dir`` 则从中读取）。
   - ``offline``：每次 reset 从 ``activity_instance_dir`` 随机选取缓存实例。
   - ``online``：reset 时在线重采样物体布局（需 ``online_object_sampling: True`` 与 ``use_presampled_robot_pose: False``，启动较慢）。

   **核心评测字段**

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - 字段
        - 说明
      * - ``total_num_envs``
        - 全局并行环境数。单个环境约占 **10 GiB** 显存，``behavior_openpi_pi05_eval.yaml`` 默认为 ``4``。
      * - ``rollout_epoch``
        - 评测轮次，多轮指标取平均；``behavior_openpi_pi05_eval.yaml`` 默认为 ``4``。
      * - ``max_episode_steps``
        - 单条轨迹步数上限（π₀.₅ 示例为 ``4096``，预设默认 ``2000`` 对长程任务偏短）。
      * - ``max_steps_per_rollout_epoch``
        - 每轮 rollout 的总交互步数，**必须能被** ``rollout.model.num_action_chunks`` **整除**。
      * - ``num_env_subprocess``
        - 单个 env worker 内的 Isaac Sim 子进程数（默认 ``1``）。``total_num_envs`` 须能被 ``num_env_subprocess × pipeline_stage_num`` 整除。
      * - ``skip_intermediate_renders``
        - 设为 ``True`` 时跳过动作块内中间渲染以加速步进；视频仅保留块边界帧。

   **任务、场景与集群布局配置**

   .. code-block:: yaml

      cluster:
        component_placement:
          rollout,env: all                   # 共享全部 GPU，或将 env 与 rollout 分卡部署
      env:
        eval:
          omni_config:
            task:
              activity_name: turning_on_radio
              activity_definition_id: 0
              activity_instance_id: 0
              activity_instance_dir: null
              instance_file_format: tro_state      # template | tro_state
              instance_resample_mode: disabled     # disabled | offline | online
            scene:
              scene_model: house_double_floor_lower
              partial_scene_load: true             # 仅加载任务相关房间

   缓存实例生成与多卡布局详情见 :doc:`../../examples/embodied/behavior`。

.. dropdown:: 进阶用法

   **切换评测任务或调整并行规模**

   .. code-block:: bash

      bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
        rollout.model.model_path=./RLinf-Pi0-Behavior \
        env.eval.omni_config.task.activity_name=picking_up_trash

      bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
        rollout.model.model_path=./RLinf-Pi0-Behavior \
        env.eval.total_num_envs=4 \
        env.eval.num_env_subprocess=2

   **使用离线 instance 随机采样**

   .. code-block:: yaml

      env:
        eval:
          omni_config:
            task:
              activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2025-challenge-task-instances
              instance_file_format: tro_state
              instance_resample_mode: offline
          rollout_epoch: 5

结果与常见问题
--------------

终端输出 ``eval/success_once``，日志与视频保存在 ``logs/<timestamp>-<config>/``\ （详见 :doc:`../reference/results`）。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - 缺少 BEHAVIOR 资源或 license 报错
     - 按 :doc:`../../examples/embodied/behavior` 完成 Isaac Sim、OmniGibson 资产与 license 配置后再评测。
   * - ``ISAAC_PATH`` 未设置
     - 将 ``ISAAC_PATH`` 导出为 Isaac Sim 安装目录；``run_eval.sh`` 默认设置 ``OMNIGIBSON_HEADLESS=1``。
   * - 显存不足
     - 降低 ``total_num_envs`` 或 ``num_env_subprocess``；单个环境约占 10 GiB 显存。
   * - 渲染马赛克或模糊
     - 换用支持 Ray Tracing 的 GPU（RTX 30/40 系列或更新）。
   * - 启动极慢
     - 保持 ``partial_scene_load: true``，只加载任务相关房间。
   * - 视频帧数少于预期
     - ``skip_intermediate_renders: True`` 会跳过块内中间帧，仅保留 policy 消费的观测帧。
   * - instance 加载失败
     - 核对 ``activity_instance_dir`` 中的 JSON 文件名与 ``activity_name``、``activity_definition_id``、``scene_model`` 一致。
   * - 步数校验失败
     - 将 ``max_steps_per_rollout_epoch`` 设为 ``rollout.model.num_action_chunks`` 的整数倍。
