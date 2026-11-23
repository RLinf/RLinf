Franka π₀ SFT 与部署
====================

概览
----

使用 OpenPI π₀ 在 Franka 机械臂上端到端跑通 Bin-relocation 流程：通过 SpaceMouse 采集 LeRobot 格式示教数据，计算 OpenPI 归一化统计量，执行 π₀ 微调（SFT），并将训练好的 checkpoint 部署到真机。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      Franka Emika Panda（含夹爪）

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      Intel RealSense · SpaceMouse · 键盘

   .. grid-item-card:: 流程
      :text-align: center

      采集 → 归一化统计 → SFT → 真机部署

   .. grid-item-card:: 硬件
      :text-align: center

      1× NVIDIA GPU（或控制机 + GPU）

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - 阶段
     - 配置 / 格式
     - 说明
   * - 数据采集
     - ``realworld_collect_data`` + ``pi0_realworld``
     - 采集 Franka Bin-relocation 轨迹并导出为 OpenPI 可读取的 LeRobot 格式。
   * - SFT 训练
     - ``realworld_bin_relocation_sft_openpi``
     - 在真机采集的数据集上微调 OpenPI π₀。
   * - 真机部署
     - ``realworld_pnp_eval`` / ``realworld_eval``
     - 在物理 Franka 机械臂上运行仅评测（eval-only）自主部署。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 腕部 RealSense 相机帧（映射至 ``pi0_realworld`` OpenPI 字段）与机器人状态。
   * - 动作
     - 按 ``pi0_realworld`` 格式编码的 Franka 笛卡尔动作与夹爪控制。
   * - 奖励
     - 采集阶段按键盘 ``c`` 标记成功；部署阶段由操作员观察任务完成情况。
   * - 提示词
     - 保存在 ``override_cfg.task_description`` 与数据集元数据中的任务描述。

硬件与安全准备
--------------

准备一台带夹爪的 Franka Emika Panda 机械臂、一台 Intel RealSense 腕部相机、一台用于 SFT 与 rollout 的 NVIDIA GPU 主机，以及用于遥操作采集的 SpaceMouse（可选）。单机模式下该 GPU 主机同时控制机械臂；多机模式下由独立无 GPU 控制机运行 Franky 与数据采集。先按 :doc:`franka` 完成固件检查、实时内核配置与 FCI 激活，并按 :doc:`sft_openpi` 准备 OpenPI π₀ 基础模型权重。

.. warning::

   每次采集与真机部署均须由操作员全程监控并保持急停触手可及。在 Bin-relocation 任务中，``target_ee_pose`` 定义为运动空间中点处的最低位置，环境会围绕该位姿裁剪工作空间以防止末端撞击容器边缘；请在当前机械臂上实测该位姿并清空工作区后再启动自主控制。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. 安装与数据采集

      **安装依赖**

      在 GPU 主机上安装 OpenPI 与 Franka 环境（加上 ``--env franka`` 可直接用于单机部署；独立控制机按 :doc:`franka` 安装即可）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **测量目标位姿**

      在引导模式下将机械臂移至 Bin-relocation 运动空间最低中点，在 Franka Desk 中激活 FCI，运行控制器检查脚本并输入 ``getpos_euler`` 记录 ``[x, y, z, roll, pitch, yaw]``：

      .. code-block:: bash

         export FRANKA_ROBOT_IP=192.168.1.10  # 替换为机械臂 IP
         python -m toolkits.realworld_check.test_franka_controller

      **配置并采集示教数据**

      数据采集、训练与部署共用环境模板 ``examples/embodiment/config/env/realworld_bin_relocation.yaml``。在 ``examples/embodiment/config/realworld_collect_data.yaml`` 中按 :doc:`franka` 填好机械臂 IP 与相机序列号后，应用以下 Bin-relocation 覆盖项（切换环境、开启夹爪、启用键盘标注、填写任务文本并导出 LeRobot 格式）：

      .. code-block:: yaml

         defaults:
           - env/realworld_bin_relocation@env.eval
           - override hydra/job_logging: stdout

         env:
           eval:
             no_gripper: False
             keyboard_reward_wrapper: single_stage
             override_cfg:
               target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
               task_description: "pick up the object and place it into the container"
             data_collection:
               enabled: True
               export_format: lerobot_dataset

      运行数据采集：

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_collect_data

      用 SpaceMouse 遥操作完成任务，每条轨迹结束后按 ``c`` 标记成功并复位机械臂，同时将目标物体放回起点。默认采集 20 条 episode 后自动结束（可通过 ``num_data_episodes`` 调整），LeRobot 数据保存在 ``logs/<running-timestamp>/lerobot_dataset/rank_*/id_*/``。

   .. tab-item:: 2. 统计量与 SFT

      **准备数据集并计算归一化统计量**

      将采集到的 LeRobot 数据复制到训练节点（单机模式跳过复制），放在例如 ``/path/to/lerobot_data/realworld_franka_bin_relocation/`` 目录下（内含 ``data/`` 与 ``meta/``）。将 ``HF_LEROBOT_HOME`` 指向该根目录并计算 ``pi0_realworld`` 归一化统计量：

      .. code-block:: bash

         export HF_LEROBOT_HOME=/path/to/lerobot_data
         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi0_realworld \
             --repo-id realworld_franka_bin_relocation

      ``config-name`` 须与训练使用的 OpenPI 数据配置一致，``repo-id`` 须与数据集目录名一致。脚本会生成 ``./assets/pi0_realworld/realworld_franka_bin_relocation/norm_stats.json``。与 :doc:`sft_openpi` 一致，训练和评测都不会在 SFT ``full_weights.pt`` 同级目录查找统计文件：推荐通过 ``openpi_data.norm_stats_path`` 显式指定路径，或将其复制到 ``<model_path>/realworld_franka_bin_relocation/norm_stats.json`` 以走默认回退路径。

      **启动 OpenPI SFT**

      编辑 ``examples/sft/config/realworld_bin_relocation_sft_openpi.yaml``，或直接通过 Hydra CLI 参数传入数据集、基座模型与归一化统计路径（更多 OpenPI SFT 选项见 :doc:`sft_openpi`）：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi \
           data.train_data_paths=./lerobot_data/realworld_franka_bin_relocation \
           actor.model.model_path=./pi0-model \
           +actor.model.openpi_data.norm_stats_path=./assets/pi0_realworld/realworld_franka_bin_relocation/norm_stats.json

   .. tab-item:: 3. 真机部署

      编辑 ``evaluations/realworld/realworld_pnp_eval.yaml``，填入机械臂 IP 与实测目标位姿（RealSense 相机自动发现，仅需筛选子集或固定顺序时才填写 ``camera_serials``）：

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: franka
               hardware:
                 configs:
                   - robot_ip: ROBOT_IP

         env:
           eval:
             rollout_epoch: 20
             override_cfg:
               target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
               task_description: "pick up the object and place it into the container"

      ``evaluations/realworld/realworld_pnp_eval.yaml`` 默认为双节点集群（``cluster.num_nodes: 2``，``franka`` 位于 ``node_rank: 1``）。多机部署先按 :doc:`franka` 启动 Ray 集群；单机部署须将 ``cluster.num_nodes`` 改为 ``1``，并将 ``franka`` 组的 ``node_ranks`` 与 ``node_rank`` 均改为 ``0``\ （或改用下方单机配置 ``realworld_eval``），否则 Ray 会一直等待第二个节点。在仓库根目录通过 Hydra CLI 参数传入 checkpoint、基座模型与 ``norm_stats.json`` 路径并启动部署（更多选项见 :doc:`真机评测指南 <../../evaluations/guides/realworld>`）：

      .. code-block:: bash

         bash evaluations/run_eval.sh realworld_pnp_eval \
           runner.ckpt_path=./logs/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt \
           rollout.model.model_path=./pi0-model \
           +rollout.model.openpi_data.norm_stats_path=./assets/pi0_realworld/realworld_franka_bin_relocation/norm_stats.json

      policy 会按 ``env.eval.rollout_epoch`` 指定的轮数自主控制机械臂。

.. dropdown:: 通用真机 SFT 环境（FrankaEnv-v1）

   如需仅通过 YAML 定义新任务（无需编写自定义环境类）以采集 SFT 数据或部署 policy，可修改环境模板 ``examples/embodiment/config/env/realworld_franka_sft_env.yaml``：

   .. code-block:: yaml

      override_cfg:
        task_description: "pick up the object and place it into the container"
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # 目标位姿 [x, y, z, rx, ry, rz]
        reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]   # 复位位姿（高于目标）
        max_num_steps: 300
        reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
        action_scale: [1.0, 1.0, 1.0]                       # [xyz, rpy, 夹爪]
        ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
        ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

   使用该通用环境单机部署 π₀ 时，替换 ``evaluations/realworld/realworld_eval.yaml`` 中的 ``ROBOT_IP`` 与 ``MODEL_PATH``，然后运行（详见 :doc:`真机评测指南 <../../evaluations/guides/realworld>`）：

   .. code-block:: bash

      bash evaluations/run_eval.sh realworld_eval

结果与常见问题
--------------

在 TensorBoard（``tensorboard --logdir ./logs``）中查看 SFT 损失曲线并在真机上观察自主执行效果；日志配置见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 现象
     - 解决方法
   * - ``calculate_norm_stats.py`` 找不到数据集
     - 将 ``HF_LEROBOT_HOME`` 指向包含 ``<repo_id>/{data,meta}`` 的父目录，并确保 ``--repo-id`` 与目录名一致。
   * - SFT 或部署时提示缺少 ``norm_stats.json``
     - 通过 ``openpi_data.norm_stats_path`` 显式传入生成的 ``norm_stats.json``\ （推荐，与 :doc:`sft_openpi` 一致），或将其复制到 ``<model_path>/<repo_id>/norm_stats.json``。
   * - 数据采集时夹爪无法开合
     - 在 ``realworld_collect_data.yaml`` 中设置 ``env.eval.no_gripper: False``。
   * - 机械臂碰触容器边缘或触发限位
     - 在运动空间安全最低中点重新测量 ``target_ee_pose``，并核对 ``ee_pose_limit_min`` 与 ``ee_pose_limit_max``。
