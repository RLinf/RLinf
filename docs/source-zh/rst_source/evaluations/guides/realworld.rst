真机评测
========

概览
----

在 Franka 机械臂上评测与部署 VLA policy，覆盖 Bin-relocation（pick-and-place）、YAML 自定义任务（``FrankaEnv-v1``）与 Dual Franka RTC 场景。硬件接线、实时内核、相机与夹爪配置见 :doc:`../../examples/embodied/franka`、:doc:`../../examples/embodied/franka_zed_robotiq` 与 :doc:`../../examples/real_world_index`。

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

一键上手
--------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: 双节点 PnP（π₀）

      在 GPU head 节点（rank 0）安装 ``--model openpi --env franka``，在机器人控制节点（rank 1，无需 GPU）仅安装 ``--env franka``。双节点启动 Ray（见下方折叠区），修改 ``evaluations/realworld/realworld_pnp_eval.yaml`` 中的 ``robot_ip``、``camera_serials`` 与 ``target_ee_pose`` 后，从 head 节点启动评测：

      .. code-block:: bash

         # 1. 在 GPU head 节点安装（控制节点去掉 --model openpi）
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. 在 GPU head 节点启动评测（替换为本地模型与权重路径）
         bash evaluations/run_eval.sh realworld_pnp_eval \
           rollout.model.model_path=./pi0-model \
           runner.ckpt_path=./full_weights.pt

   .. tab-item:: 单机自定义任务

      GPU 推理与 Franka 控制合署同一工作站时，更新 ``evaluations/realworld/realworld_eval.yaml``\ （``FrankaEnv-v1``）中的 ``robot_ip`` 与任务空间参数后直接启动（单机配置会自动启动 Ray）：

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. 启动评测
         bash evaluations/run_eval.sh realworld_eval \
           rollout.model.model_path=./pi0-sft-model

   .. tab-item:: 单机双臂 RTC（π₀.₅）

      单机双臂 RTC 在一台机器上运行 GPU 推理、两台 Franka 与三路 RealSense 相机（``evaluations/realworld/realworld_dual_franka_pi05_RTC.yaml``，自动启动 Ray）。填入双臂 IP、相机序列号、夹爪串口、``keyboard_device``、``norm_stats_path`` 与 ``task_description``；复位后在 USB 键盘或脚踏上按 ``a`` 启动、``c`` 记成功、``b`` 记失败：

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. 启动评测
         bash evaluations/run_eval.sh realworld_dual_franka_pi05_RTC \
           rollout.model.model_path=./pi05-dual-franka

.. warning::

   真机评测前务必核对工作空间限位（``ee_pose_limit_min`` / ``ee_pose_limit_max``）与急停开关，首次运行先用较小的 ``env.eval.rollout_epoch``。

.. dropdown:: 启动前检查与 Ray 集群启动

   **步骤 1（硬件检查）：** 评测前在机器人控制节点保持 ``.venv`` 激活并检查相机与机械臂。

   **步骤 2（启动 Ray 集群）：** 仅适用于双节点配置（``双节点 PnP（π₀）``、``realworld_dual_yam_openpi_pi05_eval`` 等）。单机配置（``单机自定义任务`` 与 ``单机双臂 RTC（π₀.₅）``）会自动启动 Ray，无需导出 ``RLINF_NODE_RANK`` 或手动执行 ``ray start``。

   双节点运行时，在两台机器上保持 ``.venv`` 已激活（也可按需修改并 source ``ray_utils/realworld/setup_before_ray.sh``），在 ``ray start`` 前导出 ``RLINF_NODE_RANK``\ （Ray 会在启动时冻结环境变量），并按实际替换示例 IP（GPU head 节点 ``192.168.1.10``、Franka 控制节点 ``192.168.1.20``）：

   .. code-block:: bash

      # 1. 查询 RealSense 序列号与目标位姿（输入 getpos_euler 获取 target_ee_pose）
      python -m toolkits.realworld_check.test_franka_camera
      export FRANKA_ROBOT_IP=192.168.1.20
      python -m toolkits.realworld_check.test_franka_controller

      # 2a. 仅双节点需要 —— GPU head 节点（rank 0；多网卡主机额外导出 RLINF_COMM_NET_DEVICES=eth0）
      export RLINF_NODE_RANK=0
      ray start --head --port=6379 --node-ip-address=192.168.1.10

      # 2b. 仅双节点需要 —— Franka 控制节点（rank 1；多网卡主机额外导出 RLINF_COMM_NET_DEVICES=eth0）
      export RLINF_NODE_RANK=1
      ray start --address=192.168.1.10:6379

   运行 ``ray status`` 确认两个节点均在线。可在 ``override_cfg`` 中设 ``is_dummy: True``，在不移动机械臂的情况下验证集群链路。

.. _realworld-eval-config:

.. dropdown:: 评测配置参考

   ``evaluations/realworld/`` 已有配置如下（``realworld`` 跳过仿真环境初始化；缺省配置会回退到 ``examples/embodiment/config/``，例如双节点 ``realworld_eval_dual_franka``）：

   .. list-table::
      :header-rows: 1
      :widths: 42 36 22

      * - 配置文件
        - 任务与拓扑
        - 模型
      * - ``realworld_pnp_eval.yaml``
        - Bin-relocation PnP（双节点）
        - π₀
      * - ``realworld_pnp_eval_pi05_sft_RTC.yaml``
        - Bin-relocation PnP RTC（双节点）
        - π₀.₅
      * - ``realworld_pnp_eval_dreamzero.yaml``
        - Bin-relocation PnP（双节点）
        - DreamZero
      * - ``realworld_eval.yaml``
        - 自定义 ``FrankaEnv-v1``\ （单机）
        - π₀
      * - ``realworld_dual_franka_pi05_RTC.yaml``
        - Dual Franka RTC（单机）
        - π₀.₅
      * - ``realworld_dual_yam_openpi_pi05_eval.yaml``
        - Dual YAM（双节点）
        - π₀.₅

   真机配置需重点核对以下字段：

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - 字段
        - 说明
      * - ``cluster.node_groups`` / ``component_placement``
        - 填写 ``robot_ip`` 并让 ``node_ranks`` 与各节点的 ``RLINF_NODE_RANK`` 对齐（``env`` 在控制节点，``rollout`` 在 GPU 节点）。
      * - ``env.eval.override_cfg``
        - 设置 ``camera_serials``、``task_description``\ （须与 SFT 一致）与 ``target_ee_pose``\ （``FrankaEnv-v1`` 还需 ``reset_ee_pose`` 与位姿限位）。
      * - ``rollout.model.model_path`` / ``runner.ckpt_path``
        - 含 ``norm_stats.json`` 的基座模型目录与可选 SFT ``full_weights.pt``；``rollout.model.openpi.config_name`` 须与训练一致。
      * - ``env.eval.rollout_epoch`` / ``max_steps_per_rollout_epoch``
        - 评测轮数与每轮步数上限；``max_steps_per_rollout_epoch`` 须为 ``rollout.model.num_action_chunks`` 的整数倍。
      * - ``env.eval.teleop``
        - 评测时设为 ``none``，真机 ``total_num_envs`` 通常保持 ``1``。

   自定义 ``FrankaEnv-v1`` 的采集与 SFT 见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`；双节点 Dual Franka 回退配置与采集流程见 :doc:`../../examples/embodied/dual_franka` 与 :doc:`../../examples/embodied/dual_franka_pico_dagger`；DreamZero 真机依赖与权重见 :doc:`../../examples/embodied/sft_dreamzero`。

结果与常见问题
--------------

任务指标输出在终端，日志保存在 ``logs/<时间戳>-<config_name>/eval_embodiment.log``，开启 ``env.eval.video_cfg.save_video: True`` 时视频保存在 ``<log_path>/video/eval/``\ （详见 :doc:`../reference/results`）。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - 节点拓扑或 rank 不匹配
     - ``env`` worker 须运行在直连 Franka 的节点，且 ``node_ranks`` 与 ``RLINF_NODE_RANK`` 一致。
   * - 相机未检测到
     - 在控制节点运行 ``python -m toolkits.realworld_check.test_franka_camera``，核对 ``env.eval.override_cfg.camera_serials``。
   * - 机械臂动作异常
     - 确认 ``norm_stats.json`` 与训练一致，并核对 ``rollout.model.openpi.config_name`` 匹配对应权重。
   * - ``ray status`` 只显示一个节点
     - 检查防火墙、``RLINF_COMM_NET_DEVICES`` 与到 GPU head IP 的网络连通性。
   * - 步数校验报错
     - 将 ``max_steps_per_rollout_epoch`` 设为 ``rollout.model.num_action_chunks`` 的整数倍。
   * - Dual Franka 按键无响应
     - 须在 ``keyboard_device``\ （或 ``RLINF_KEYBOARD_DEVICE``）指向的本地 USB evdev 设备上按键，SSH 终端按键不会传入。
