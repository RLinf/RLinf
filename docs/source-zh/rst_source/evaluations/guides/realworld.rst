真机评测
========

概览
----

在 Franka 机械臂上评测与部署 VLA policy，涵盖 Bin-relocation（pick-and-place）任务、基于 YAML 配置的通用真机环境（``FrankaEnv-v1``，用于自定义任务）以及 Dual Franka 双臂场景。默认硬件为 Franka Emika Panda 机械臂配 Intel RealSense 相机，支持双节点异构布局（同一局域网内的 GPU head 节点 + 机器人控制节点）或单机工作站部署。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      Franka PnP · 自定义（``FrankaEnv-v1``） · Dual Franka

   .. grid-item-card:: 支持模型
      :text-align: center

      OpenPI π₀ / π₀.₅ · DreamZero

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/realworld/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`Franka SFT <../../examples/embodied/franka_pi0_sft_deploy>` · :doc:`Franka <../../examples/embodied/franka>` · :doc:`DreamZero <../../examples/embodied/sft_dreamzero>`

使用 ZED 相机或 Robotiq 夹爪见 :doc:`../../examples/embodied/franka_zed_robotiq`；Ray 集群完整搭建、固件版本与 libfranka 兼容性见 :doc:`../../examples/embodied/franka` 与 :doc:`../../guides/realworld_robot`。

一键上手
--------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: 机器人控制节点（安装与检查）

      在直连 Franka 与相机的控制节点上安装依赖（无需 GPU，默认使用 Franky 后端）：

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

         # 2. 检查相机序列号与目标位姿
         python -m toolkits.realworld_check.test_franka_camera
         export FRANKA_ROBOT_IP=<robot_ip>
         python -m toolkits.realworld_check.test_franka_controller

   .. tab-item:: GPU / Rollout 节点（PnP / π₀）

      在 GPU head 节点安装 OpenPI，在双节点启动 Ray（见下方折叠区），修改 ``evaluations/realworld/realworld_pnp_eval.yaml`` 中的 ``robot_ip``、``camera_serials`` 与 ``target_ee_pose``，再在 head 节点启动评测：

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. 启动评测（传入 Pi0 基座模型与 SFT full_weights.pt）
         bash evaluations/run_eval.sh realworld_pnp_eval \
           rollout.model.model_path=/path/to/pi0-model \
           runner.ckpt_path=/path/to/full_weights.pt

   .. tab-item:: 单机工作站（自定义任务）

      GPU 与控制合署一机时，替换 ``evaluations/realworld/realworld_eval.yaml`` 中的 ``ROBOT_IP`` 与 ``MODEL_PATH`` 后直接运行：

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. 启动评测
         bash evaluations/run_eval.sh realworld_eval

如需改用旧版 ROS 后端控制机械臂，见 :doc:`../../examples/embodied/franka`。DreamZero 真机评测还需在 GPU 节点安装 DreamZero 依赖（见 :doc:`../../examples/embodied/sft_dreamzero`）。

.. warning::

   真机评测前务必确认工作空间限位（``ee_pose_limit_min`` / ``ee_pose_limit_max``）与急停功能正常；首次评测将 ``env.eval.rollout_epoch`` 设为较小值。

.. dropdown:: 节点拓扑与示例配置

   双节点 PnP 评测（``realworld_pnp_eval.yaml`` 与 ``realworld_pnp_eval_dreamzero.yaml``）采用 **1 个 GPU 节点 + 1 个 Franka 控制节点** 的异构布局；``realworld_eval.yaml`` 在单机上同时运行 ``env`` 与 ``rollout``。

   .. list-table::
      :header-rows: 1
      :widths: 18 30 52

      * - ``RLINF_NODE_RANK``
        - 角色
        - 说明
      * - rank 0
        - GPU / head
        - 运行 ``rollout``，在此节点提交 ``run_eval.sh``。
      * - rank 1
        - 机器人控制
        - 运行 ``env`` worker，直连 Franka 与相机。

   ``evaluations/realworld/`` 已有示例配置：

   .. list-table::
      :header-rows: 1
      :widths: 42 34 24

      * - 配置文件
        - 任务
        - 模型
      * - ``realworld_pnp_eval.yaml``
        - Bin-relocation（PnP）
        - π₀
      * - ``realworld_pnp_eval_dreamzero.yaml``
        - Bin-relocation（PnP）
        - DreamZero
      * - ``realworld_eval.yaml``
        - 自定义任务（``FrankaEnv-v1``）
        - π₀
      * - ``realworld_dual_franka_pi05_RTC.yaml``
        - 单机 Dual Franka RTC
        - π₀.₅

   若 ``evaluations/realworld/<config>.yaml`` 不存在，``run_eval.sh`` 会回退到 ``examples/embodiment/config/`` 下同名配置（例如双节点 ``realworld_eval_dual_franka``，见 :doc:`../reference/cli`）。

.. dropdown:: 启动前检查与 Ray 集群启动

   **启动前检查**

   1. **相机序列号**\ （控制节点）：运行 ``python -m toolkits.realworld_check.test_franka_camera``，记录序列号填入 ``env.eval.override_cfg.camera_serials``。
   2. **目标位姿**\ （PnP 任务，控制节点）：导出 ``FRANKA_ROBOT_IP``，运行 ``python -m toolkits.realworld_check.test_franka_controller`` 并输入 ``getpos_euler``，将结果填入 ``env.eval.override_cfg.target_ee_pose``\ （运动空间最低点，用于成功判定与工作空间截断，见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`）。
   3. **Ray 连通性**\ （各节点）：运行 ``ray status``，确认双节点在线。
   4. **Dummy 模式（可选）：** 在 ``override_cfg`` 中设 ``is_dummy: True``\ （参考 ``examples/embodiment/config/realworld_dummy_franka_sac_cnn.yaml``），无需机械臂运动即可验证集群通路。

   **启动 Ray 集群**

   在每个节点执行 ``ray start`` **之前** 统一环境变量并激活 ``.venv``\ （Ray 启动时会冻结当时的 Python 解释器与环境变量）：

   .. code-block:: bash

      source ray_utils/realworld/setup_before_ray.sh
      source .venv/bin/activate
      export RLINF_NODE_RANK=<0|1>          # 集群内唯一
      # 多网卡时指定对外可达网卡：
      # export RLINF_COMM_NET_DEVICES=<network_interface>

   然后启动 Ray（记 GPU head 节点 IP 为 ``<head_ip>``）：

   .. code-block:: bash

      # GPU 节点（rank 0，head）
      export RLINF_NODE_RANK=0
      ray start --head --port=6379 --node-ip-address=<head_ip>

      # Franka 控制节点（rank 1）
      export RLINF_NODE_RANK=1
      ray start --address=<head_ip>:6379

.. _realworld-eval-config:

.. dropdown:: 评测配置详解

   ``evaluations/realworld/realworld_pnp_eval.yaml`` 至少需要更新以下字段（``node_ranks`` 与 ``component_placement`` 须与各机器的 ``RLINF_NODE_RANK`` 一致）：

   .. code-block:: yaml

      cluster:
        node_groups:
          - label: "4090"
            node_ranks: 0          # GPU 节点 rank
          - label: franka
            node_ranks: 1          # Franka 控制节点 rank
            hardware:
              type: Franka
              configs:
                - robot_ip: ROBOT_IP
                  node_rank: 1

      runner:
        ckpt_path: /path/to/full_weights.pt

      env:
        eval:
          override_cfg:
            target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
            camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
            task_description: "pick up the object and place it into the container"

      rollout:
        model:
          model_path: /path/to/pi0-model

   **必填字段**

   .. list-table::
      :header-rows: 1
      :widths: 28 32 40

      * - 字段
        - 位置
        - 说明
      * - ``robot_ip``
        - ``cluster.node_groups[].hardware.configs``
        - Franka 机器人 IP。
      * - ``node_ranks`` / ``component_placement``
        - ``cluster``
        - 与 ``RLINF_NODE_RANK`` 对齐（``env`` 在 Franka 节点，``rollout`` 在 GPU 节点）。
      * - ``target_ee_pose``
        - ``env.eval.override_cfg``
        - PnP 目标末端位姿 ``[x,y,z,rx,ry,rz]``，影响成功判定与运动截断。
      * - ``camera_serials``
        - ``env.eval.override_cfg``
        - RealSense 序列号列表（位于 ``override_cfg`` 下）。
      * - ``task_description``
        - ``env.eval.override_cfg``
        - 语言指令，须与 SFT 训练一致。
      * - ``model_path``
        - ``rollout.model``
        - Pi0 基座模型目录（含 ``<repo_id>/norm_stats.json``）。
      * - ``ckpt_path``
        - ``runner``
        - SFT 导出的 ``full_weights.pt``。
      * - ``config_name``
        - ``rollout.model.openpi``
        - PnP 任务填 ``"pi0_realworld"``。

   **关键评测字段**

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - 字段
        - 说明
      * - ``rollout_epoch``
        - 评测轮数（默认 ``20``）。
      * - ``max_episode_steps``
        - 单条轨迹最大步数（默认 ``200``）。
      * - ``max_steps_per_rollout_epoch``
        - 每轮步数上限，**必须能被** ``rollout.model.num_action_chunks`` **整除**\ （PnP 默认 ``10``）。
      * - ``total_num_envs``
        - 并行环境数（真机通常为 ``1``）。
      * - ``teleop``
        - 接管设备，评测时填 ``none``。

   ``realworld`` 评测不调用 ``setup_sim_env``。DreamZero 真机评测（``realworld_pnp_eval_dreamzero.yaml``）完整流程见 :doc:`../../examples/embodied/sft_dreamzero`。

.. dropdown:: 自定义真机任务评测

   通过 YAML 定义自定义任务（无需编写 Python 环境类）时，使用 ``FrankaEnv-v1``、环境模板 ``examples/embodiment/config/env/realworld_franka_sft_env.yaml`` 与评测配置 ``evaluations/realworld/realworld_eval.yaml``\ （单机布局，``num_nodes: 1``）：

   .. code-block:: yaml

      override_cfg:
        task_description: "pick up the object and place it into the container"
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # 目标位姿
        reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]   # 复位位姿（应高于目标）
        max_num_steps: 300
        reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
        action_scale: [1.0, 1.0, 1.0]
        ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
        ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

   自定义任务的示教采集、SFT 训练与部署见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`。

.. dropdown:: Dual Franka 部署

   双臂评测提供两份配置：

   - **双节点回退配置**\ （``examples/embodiment/config/realworld_eval_dual_franka.yaml``）：将 ``rollout.model.model_path`` 设为同步后的 checkpoint 目录，将 ``actor.model.openpi_data.repo_id`` 设为包含 ``norm_stats.json`` 的 repo id。
   - **单机 RTC 评测**\ （``evaluations/realworld/realworld_dual_franka_pi05_RTC.yaml``）：GPU、两台机械臂与三路相机在同一台机器上（20 维 ``tcp_rot6d`` 动作，``openpi.config_name: pi05_dualfranka_tcp_rot6d``，主相机 ``left_wrist_0_rgb``）。替换 ``MODEL_PATH``、``norm_stats`` 路径、机械臂 IP、相机序列号、夹爪串口、``keyboard_device`` 与 ``TASK_DESCRIPTION``。复位完成后在 ``keyboard_device``\ （或 ``RLINF_KEYBOARD_DEVICE``）指向的 evdev USB 键盘或脚踏上按 ``a`` 启动 policy、``c`` 记成功、``b`` 记失败（SSH 会话按键不会到达该设备）。无图形界面时保持 ``enable_camera_player: False``，并保持 ``max_steps_per_rollout_epoch`` 为 ``20`` 的整数倍。

   .. code-block:: bash

      bash evaluations/run_eval.sh realworld_eval_dual_franka \
        rollout.model.model_path=/path/to/deploy/global_step_<N> \
        actor.model.openpi_data.repo_id=<repo_id>/tcp_rot6d_v1 \
        env.eval.override_cfg.task_description="handover the object"

      bash evaluations/run_eval.sh realworld_dual_franka_pi05_RTC

   完整采集、SFT、checkpoint 同步与按键流程见 :doc:`../../examples/embodied/dual_franka` 与 :doc:`../../examples/embodied/dual_franka_pico_dagger`。

结果与常见问题
--------------

任务指标打印在终端，日志保存在 ``logs/<时间戳>-<config_name>/eval_embodiment.log``，开启 ``env.eval.video_cfg.save_video: True`` 时视频保存在 ``video_base_dir`` 或 ``<log_path>/video/eval/``\ （详见 :doc:`../reference/results`）。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - 运动越界或安全风险
     - 评测前确认 ``ee_pose_limit_min`` / ``ee_pose_limit_max`` 与急停功能正常，首次评测减小 ``rollout_epoch``。
   * - 节点拓扑或 rank 不匹配
     - ``env`` worker 须运行在直连 Franka 的节点，且 ``node_ranks`` 与 ``RLINF_NODE_RANK`` 一致（PnP 与 ``realworld_eval_dual_franka`` 为双节点；``realworld_eval`` 与 ``realworld_dual_franka_pi05_RTC`` 为单机）。
   * - 相机未找到
     - 在控制节点运行 ``python -m toolkits.realworld_check.test_franka_camera``，核对 ``env.eval.override_cfg.camera_serials``。
   * - 机械臂动作异常
     - 检查 ``model_path/<repo_id>/norm_stats.json`` 是否存在，并核对 ``openpi.config_name`` 与训练一致。
   * - ``ray status`` 只见一个节点
     - 检查防火墙、``RLINF_COMM_NET_DEVICES`` 与其他节点到 ``<head_ip>`` 的网络连通性。
   * - 步数校验报错
     - 将 ``max_steps_per_rollout_epoch`` 设为 ``rollout.model.num_action_chunks`` 的整数倍。
   * - ZED 相机或 Robotiq 夹爪配置
     - 参阅 :doc:`../../examples/embodied/franka_zed_robotiq`。
