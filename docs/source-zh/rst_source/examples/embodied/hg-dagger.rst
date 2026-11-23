Franka HG-DAgger
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/hg-dagger.jpg
   :align: center
   :width: 80%

   用于采集干预数据并在线训练 Franka policy 的 Human-Gated DAgger 流程。

概览
----

使用 Human-Gated DAgger（HG-DAgger）在线提升 Franka 真机 OpenPI policy：先用 PICO 采集示教，再计算归一化统计量并运行 SFT 获得学生模型初始化，最后启动带实时人工接管的在线 HG-DAgger。在线阶段由 LeRobot 归档完整的成功 episode（未接管帧保留 policy 实际执行动作，接管帧保存人工动作与 ``intervene_flag``）；启用 ``only_save_expert: True`` 后，训练仅采样非 padding 帧全部由人工接管的 action chunk。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      Franka Emika Panda（Franky）

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      双 RealSense · PICO VR

   .. grid-item-card:: 流程
      :text-align: center

      PICO 数据采集 → SFT → 在线 HG-DAgger

   .. grid-item-card:: 硬件
      :text-align: center

      控制节点 + GPU 节点

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - 阶段
     - 配置
     - 说明
   * - 示教采集
     - ``realworld_collect_data_pico``
     - 使用 PICO 遥操作采集真机抓放示教数据。
   * - SFT 训练
     - ``realworld_bin_relocation_sft_openpi``
     - 训练 OpenPI 学生模型初始化权重。
   * - HG-DAgger
     - ``realworld_pnp_dagger_openpi``
     - 用在线 LeRobot 归档完整成功轨迹，并仅在全接管 action chunk 上在线训练。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 双 RealSense 相机帧与机器人状态。
   * - 动作
     - OpenPI 动作块解码后的 Franka 笛卡尔与夹爪控制指令。
   * - 奖励
     - PICO 人工门控接管信号与任务成功结果。
   * - 提示词
     - OpenPI 数据集与配置元数据中的任务文本。

硬件与安全准备
--------------

本流程由一台 Franka 控制节点（Franka 机械臂、两台 RealSense 相机、PICO VR 遥操作数据流）与一台运行 OpenPI π₀ / π₀.₅ 的 GPU 训练 / rollout 节点组成。开始前按 :doc:`franka` 完成机械臂固件检查、实时内核配置与 FCI 激活，按 :doc:`franka_vr` 配置 PICO 数据流，并按 :doc:`sft_openpi` 准备 OpenPI 基础权重。

.. warning::

   在线训练期间须由操作员全程监控并保持急停按钮触手可及。开启人工接管前先确认 PICO 数据流延迟与按键标定正常、清空工作区域，并在每个节点执行 ``ray start`` 前激活对应虚拟环境并导出 ``RLINF_NODE_RANK``\ （Ray 会捕获启动时的终端环境）。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 机器人 / 环境节点

      **1. 安装依赖**

      安装基于 Franky 的 ``franka`` 环境，并将 ``LIBFRANKA_VERSION`` 匹配到机械臂固件版本：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         LIBFRANKA_VERSION=0.19.0 bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. 加入 Ray 集群**

      在 GPU 节点启动 Ray head 后，本节点作为 worker 节点 1 加入（将 ``HEAD_NODE_IP`` 替换为 GPU 节点 IP）：

      .. code-block:: bash

         export RLINF_NODE_RANK=1
         export HEAD_NODE_IP=192.168.1.10
         ray start --address="${HEAD_NODE_IP}:6379"

      **3. 采集 PICO 示教数据**

      复制 ``examples/embodiment/config/realworld_collect_data_pico.yaml`` 为 ``my_realworld_pnp_collect.yaml``，将环境切换为 bin relocation，并填写机械臂 IP、目标位姿与相机序列号：

      .. code-block:: yaml

         defaults:
           - env/realworld_bin_relocation@env.eval
           - override hydra/job_logging: stdout

         cluster:
           node_groups:
             - label: franka
               hardware:
                 configs:
                   - robot_ip: ROBOT_IP
                     node_rank: 0

         env:
           eval:
             override_cfg:
               target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
               camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

      按 :doc:`franka_vr` 启动并验证 PICO 数据流（默认的 ``ipc:///tmp/vr_data.ipc`` 要求 publisher 与 env worker 在同一台机器；跨机部署时在 ``env.eval.pico.zmq_addr`` 填入 ``tcp://<publisher_ip>:<port>``），再启动采集：

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh my_realworld_pnp_collect

      采集会同时写出 replay-buffer 轨迹到 ``logs/{timestamp}/demo_buffer/``，以及 LeRobot 数据到 ``logs/{timestamp}/lerobot_dataset/``\ （数据格式见 :doc:`../../guides/data_collection`）。

      .. dropdown:: 进阶配置（可选）

         仓库自带采集配置已预设 PICO 按键映射、标定与 LeRobot 导出默认项：

         .. code-block:: yaml

            env:
              eval:
                teleop: pico
                pico:
                  zmq_addr: "ipc:///tmp/vr_data.ipc"
                  hand: "right"
                  control_trigger: "grip"
                  control_threshold: 0.85
                  gripper_close_button: "A"
                  gripper_open_button: "B"
                  position_scale: 1.0
                  rotation_scale: 1.0
                  max_stale_s: 0.2
                  calibration:
                    enabled: True
                    required: True
                    auto_calibrate_on_start: True
                    button: "trigger"
                override_cfg:
                  success_hold_steps: 1
                data_collection:
                  enabled: True
                  save_dir: ${runner.logger.log_path}
                  export_format: [replay_buffer, lerobot_dataset]
                  only_success: True
                  robot_type: "panda"
                  fps: 10

   .. tab-item:: 训练与 Rollout 节点

      **1. 安装依赖**

      在 GPU 节点安装 OpenPI 与 Franka 环境：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **2. 启动 Ray head 节点**

      .. code-block:: bash

         export RLINF_NODE_RANK=0
         export HEAD_NODE_IP=192.168.1.10
         ray start --head --port=6379 --node-ip-address="$HEAD_NODE_IP"

      **3. 计算归一化统计量并运行 SFT**

      为采集得到的 LeRobot 数据集计算 ``pi0_realworld`` 归一化统计量（数据集根目录与 ``repo_id`` 须与 SFT 保持一致，详见 :doc:`sft_openpi`）：

      .. code-block:: bash

         export HF_LEROBOT_HOME=/path/to/lerobot_root
         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi0_realworld \
             --repo-id realworld_franka_bin_relocation

      编辑 ``examples/sft/config/realworld_bin_relocation_sft_openpi.yaml`` 并运行 SFT，产出学生模型初始权重：

      .. code-block:: yaml

         data:
           train_data_paths: "/path/to/realworld-franka-bin-relocation-dataset"

         actor:
           model:
             model_path: "/path/to/pi0-model"
             openpi:
               config_name: "pi0_realworld"

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

      **4. 启动真机异步 HG-DAgger**

      编辑 ``examples/embodiment/config/realworld_pnp_dagger_openpi.yaml``，填写 SFT checkpoint、基础模型路径（rollout 自动沿用 ``actor.model.model_path``）、``repo_id``、``robot_ip``、``target_ee_pose`` 与 ``camera_serials``：

      .. code-block:: yaml

         runner:
           ckpt_path: "/path/to/sft_checkpoint/full_weights.pt"

         actor:
           model:
             model_path: "/path/to/pi0-model"
             openpi_data:
               repo_id: "realworld_franka_bin_relocation"

      在 Ray head 节点启动 HG-DAgger：

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_pnp_dagger_openpi

      .. dropdown:: 进阶配置（可选）

         仓库自带 HG-DAgger 配置预设了双机节点分组、在线 LeRobot 归档与 PICO 接管参数：

         .. code-block:: yaml

            cluster:
              num_nodes: 2
              node_groups:
                - label: "train"
                  node_ranks: 0
                - label: franka
                  node_ranks: 1
                  hardware:
                    type: Franka
                    configs:
                      - robot_ip: ROBOT_IP
                        node_rank: 1

            algorithm:
              dagger:
                only_save_expert: True
                online_lerobot:
                  enabled: True
                  only_success: True
                  robot_type: "panda"
                  fps: 10
                  finalize_interval: 1
                  data_path: ${runner.logger.log_path}/online_lerobot
                  rolling_lerobot_window_size: 50000
                  min_frames: 1
                  lerobot_num_workers: 0

            env:
              train:
                smooth_intervene: True
                teleop: pico
                pico:
                  zmq_addr: "ipc:///tmp/vr_data.ipc"
                  hand: "right"
                  control_trigger: "grip"
                  control_threshold: 0.85
                  gripper_close_button: "A"
                  gripper_open_button: "B"
                  position_scale: 1.0
                  rotation_scale: 1.0
                  max_stale_s: 0.2
                  calibration:
                    enabled: True
                    required: True
                    auto_calibrate_on_start: True
                    button: "trigger"
                override_cfg:
                  target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
                  camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
              eval:
                teleop: none
                override_cfg:
                  target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
                  camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

            actor:
              model:
                openpi:
                  config_name: "pi0_realworld"

         关键配置行为：

         - ``online_lerobot.enabled: True`` 开启在线 LeRobot 数据链路，过滤后的 episode 直接进入 actor 的滚动数据集，不再使用 trajectory replay buffer。
         - ``smooth_intervene: True`` 会在 PICO ``grip`` 接管持续到 action chunk 末尾时跳过下一次 policy 推理，松开 ``grip`` 或 episode 结束后恢复正常推理（仅支持 PICO，且每个 env worker pipeline stage 仅运行一个环境）。``env.eval.teleop: none`` 让评测保持纯 policy 执行。
         - ``only_success: True`` 丢弃失败 episode；``only_save_expert: True`` 在磁盘归档完整成功 episode，同时仅采样非 padding 帧全为 ``intervene_flag=True`` 的 chunk 起点（``num_action_chunks: 1`` 时退化为逐帧过滤）。
         - ``finalize_interval: 1`` 每完成一个成功 episode 立即写入一个 LeRobot shard；``rolling_lerobot_window_size: 50000`` 将在线采样限制在最近 50,000 个有效专家 chunk 起点，旧 shard 仍保留在磁盘。
         - 真机配置省略 beta 混合字段，因为人工接管由 PICO wrapper 控制，不配置 ``rollout.expert_model``。

结果与常见问题
--------------

在 TensorBoard（``tensorboard --logdir ./logs``）中监控在线训练，日志配置见 :doc:`训练指标 <../../reference/metrics>`。成功 episode 以 LeRobot shard 形式写入 ``logs/<timestamp>-realworld_pnp_dagger_openpi/online_lerobot/rank_0/id_*/``，失败 episode 不会写入。

重点关注指标：``train/dagger/actor_loss``、``train/lerobot_dataset/total_episodes``、``train/lerobot_dataset/physical_frames``、``train/lerobot_dataset/logical_samples``、``train/lerobot_dataset/num_sub_datasets``、``train/actor/lr`` 与 ``train/actor/grad_norm``。

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 现象
     - 解决方法
   * - Env worker 收不到 PICO 位姿更新
     - 按 :doc:`franka_vr` 确认 PICO publisher 正常运行；跨机部署时将 ``pico.zmq_addr`` 从 ``ipc:///tmp/vr_data.ipc`` 改为 ``tcp://<publisher_ip>:<port>``。
   * - Worker 启动报缺包或节点编号错误
     - 执行 ``ray stop``，激活对应节点的虚拟环境并导出 ``RLINF_NODE_RANK``\ （head 为 ``0``，控制机为 ``1``）后再重启 Ray。
   * - ``smooth_intervene`` 启动校验失败
     - 确认 ``env.train.teleop`` 为 ``pico``，且每个 env worker pipeline stage 只运行一个环境。
   * - ``train/lerobot_dataset/logical_samples`` 始终为 ``0``
     - 开启 ``only_save_expert: True`` 与 ``only_success: True`` 时，episode 须成功且至少包含一个非 padding 帧全部由人工接管的 action chunk。
