双臂 Franka OpenPI PyTorch
==========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-deploy-rlinf.jpg
   :align: center
   :width: 80%
   :alt: RLinf PyTorch 双 Franka 部署

   使用 RLinf PyTorch 完成双 Franka policy 微调与部署。

用 GELLO 采集双臂 Franka 示教数据，清洗合并 shard 并回填为 ``tcp_rot6d``，在 RLinf 原生 OpenPI PyTorch 中微调 π₀.₅，再以双节点常规评测或单机 RTC 方式部署到真机。

概览
----

本页在双臂 Franka GELLO 闭环（:doc:`dual_franka`）基础上，补齐数据集清洗合并、RLinf 原生 OpenPI PyTorch 微调、``sft2deploy`` 权重转换以及单机 RTC 部署流程。单臂配置与 GELLO 组装见 :doc:`franka` 与 :doc:`franka_gello`。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      2× Franka（Franky）· 2× Robotiq

   .. grid-item-card:: 遥操作与相机
      :text-align: center

      2× GELLO · 脚踏 · RealSense + 2× Lumos

   .. grid-item-card:: 工作流
      :text-align: center

      数据采集 → 清洗合并 → 回填 → PyTorch SFT → 部署 / RTC

   .. grid-item-card:: 硬件
      :text-align: center

      2 台机器人节点（或单机 GPU 跑 RTC）· GPU 训练节点

.. include:: _dual_franka_setup.rst

使用单机 RTC 部署时，将 GPU、两台机械臂和三路相机接在同一台工作站上。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. 安装与硬件自检

      **机器人节点**\ （``node 0`` 与 ``node 1``）：预编译 Franky wheel 在 x86_64 上内置 libfranka ``0.15.0`` 或 ``0.19.0``\ （默认）；按 Franka 官方 `兼容性表 <https://frankarobotics.github.io/docs/compatibility.html>`_ 选择对应的 ``LIBFRANKA_VERSION``\ （其他固件版本可通过 ``FRANKY_WHEEL`` 传入自编译 wheel）。在 ``node 0`` 上按 :doc:`franka_gello` 安装 GELLO 依赖。

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export LIBFRANKA_VERSION=0.19.0
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **训练节点**\ （远端 GPU 服务器）：安装 OpenPI PyTorch 依赖。

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **在** ``node 0`` **完成硬件自检**\ （启动 Ray 前）：为脚踏授予权限（将 ``event12`` 替换为实际设备编号），核对相机 USB-3 速率（``lsusb -t`` 显示 ``5000M``），并验证两台 GELLO 主手输出。

      .. code-block:: bash

         # 脚踏设备（将 event12 替换为实际设备编号）
         ls -l /dev/input/by-id/*-event-kbd
         sudo chmod 666 /dev/input/event12
         export RLINF_KEYBOARD_DEVICE=/dev/input/event12

         # 相机检查（确认 RealSense 序列号、两路 Lumos 与 5000M 速率）
         rs-enumerate-devices | grep -E "Name|Serial|USB Type"
         ls /dev/v4l/by-id/
         lsusb -t

         # GELLO 主手（逐台接入以区分 FTDI 路径）
         ls /dev/serial/by-id/ | grep -i ftdi
         python -m rlinf.robotics.parts.teleop.gello_joint \
             --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0  # 替换为左臂 GELLO 设备

      正常输出会持续刷新 ``joints=[+0.012 -0.604 ...]  gripper=[0.035]``。若数值停止更新或跳变约 ``2π``，先执行 ``align-sequential``\ （见上方“GELLO 首次标定”）。

   .. tab-item:: 2. 数据采集与 SFT 训练

      **启动双节点 Ray 集群**\ （在 RLinf 仓库根目录且 ``.venv`` 已激活；``ray start`` 前导出 ``RLINF_NODE_RANK`` 与 ``RLINF_KEYBOARD_DEVICE``）：

      .. code-block:: bash

         # node 0（按工作站替换 IP 与脚踏设备）
         export RLINF_NODE_RANK=0
         export RLINF_KEYBOARD_DEVICE=/dev/input/event12
         ray start --head --port=6379 --node-ip-address=192.168.1.10

         # node 1（按工作站替换 IP）
         export RLINF_NODE_RANK=1
         ray start --address=192.168.1.10:6379 --node-ip-address=192.168.1.20

      在 ``node 0`` 运行 ``ray status``，确认两个节点均为 ``ALIVE``。

      **采集示教数据**\ ：确认 ``align-sequential``\ （见上方“GELLO 首次标定”） 通过后，在 ``node 0`` 启动采集：

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh \
             realworld_collect_data_gello_joint_dual_franka 2>&1 | tee logs/collect.log

      在另一终端运行 ``python -m toolkits.realworld_check.collect_monitor logs/collect.log`` 监控进度。脚踏按键：``a`` 开始录制（录制中再按中止并丢弃缓存），``b`` 递增 ``segment_id``，``c`` 标记成功并写入 LeRobot shard。需要续采时，设置 ``data_collection.resume: true`` 并保持相同的 ``data_collection.save_dir``。

      **清洗合并、回填 tcp_rot6d 并同步至训练节点：**

      .. code-block:: bash

         # 可选：先试跑删脏数据与合并多 shard（去掉 --dry-run 生效）
         python -m toolkits.dual_franka.delete_lerobot \
             --data-dir ./logs/collect_run/lerobot_dataset/rank_0 --delete "id_0:3,5" --dry-run
         python -m toolkits.dual_franka.merge_lerobot \
             --rank-dir ./logs/collect_run/lerobot_dataset --out-dir ./logs/collect_run/lerobot_dataset \
             --log-file ./logs/collect.log --dry-run

         # 回填 tcp_rot6d 并同步到训练节点
         export HF_LEROBOT_HOME="$HOME/lerobot_data"     # 替换为 LeRobot 数据根目录
         export DATA_REPO_ID="my_org/dual_franka_demo"   # 替换为数据集 repo ID
         export SFT_REPO_ID="$DATA_REPO_ID/tcp_rot6d_v1"
         export TRAINER_IP="192.168.1.50"                # 替换为训练节点 IP

         python -m toolkits.dual_franka.backfill_tcp_rot6d \
             --src "$HF_LEROBOT_HOME/$DATA_REPO_ID/joint_v1" \
             --dst "$HF_LEROBOT_HOME/$SFT_REPO_ID"

         ssh "$TRAINER_IP" "mkdir -p $HF_LEROBOT_HOME/$SFT_REPO_ID"
         rsync -av "$HF_LEROBOT_HOME/$SFT_REPO_ID/" \
             "$TRAINER_IP:$HF_LEROBOT_HOME/$SFT_REPO_ID/"

      **在训练节点运行 OpenPI PyTorch SFT**\ （在 RLinf 仓库根目录且 ``.venv`` 已激活）：先按 :doc:`sft_openpi` 将基础模型从 ``openpi-jax`` 转换为 PyTorch 格式，并在 ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml`` 中更新 ``train_data_paths``、``model_path`` 和 ``openpi_data.norm_stats_path``，然后执行：

      .. code-block:: bash

         export HF_LEROBOT_HOME="$HOME/lerobot_data"            # 替换为 LeRobot 数据根目录
         export SFT_REPO_ID="my_org/dual_franka_demo/tcp_rot6d_v1"

         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi05_dualfranka_tcp_rot6d \
             --repo-id "$SFT_REPO_ID"

         bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_tcp_rot6d

      Checkpoint 保存在 ``<log_path>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``。

   .. tab-item:: 3. 真机部署与评测

      **将 SFT checkpoint 转换为部署格式**\ （同步到 ``node 0`` 之前执行）：

      .. code-block:: bash

         python -m rlinf.utils.ckpt_convertor.openpi.convert \
             --mode sft2deploy \
             --ckpt ./logs/sft_run/checkpoints/global_step_10000 \
             --output "$HOME/deploy/global_step_10000" \
             --reference-model "$HOME/models/pi05_base_openpi" \
             --dtype-reference "$HOME/models/deploy_ref/full_weights.pt"  # 按实际路径替换

      **同步部署权重与归一化统计量** 回 ``node 0``，确保 ``$DEPLOY_CKPT`` 下同时包含 ``actor/model_state_dict/full_weights.pt`` 与 ``<repo_id>/tcp_rot6d_v1/norm_stats.json``：

      .. code-block:: bash

         export TRAINER_IP="192.168.1.50"                       # 替换为训练节点 IP
         export TRAINER_DEPLOY_CKPT="$HOME/deploy/global_step_10000"  # 训练节点 sft2deploy 的 --output 目录
         export TRAIN_LOG="./logs/sft_run"                      # 替换为训练节点 SFT 日志目录
         export DEPLOY_CKPT="$HOME/deploy/global_step_10000"    # 替换为 node 0 目标部署目录
         export SFT_REPO_ID="my_org/dual_franka_demo/tcp_rot6d_v1"

         mkdir -p "$DEPLOY_CKPT/actor/model_state_dict" "$DEPLOY_CKPT/$SFT_REPO_ID"

         rsync -av \
             "$TRAINER_IP:$TRAINER_DEPLOY_CKPT/actor/model_state_dict/full_weights.pt" \
             "$DEPLOY_CKPT/actor/model_state_dict/full_weights.pt"
         rsync -av \
             "$TRAINER_IP:$TRAIN_LOG/checkpoints/global_step_10000/$SFT_REPO_ID/norm_stats.json" \
             "$DEPLOY_CKPT/$SFT_REPO_ID/norm_stats.json"

      在 ``examples/embodiment/config/realworld_eval_dual_franka.yaml`` 中将 ``rollout.model.model_path`` 设为 ``$DEPLOY_CKPT``，将 ``actor.model.openpi_data.repo_id`` 设为 ``<repo_id>/tcp_rot6d_v1``。

      **启动真机评测**\ ：在 ``node 0`` 运行（更多选项见 :doc:`真机评测指南 <../../evaluations/guides/realworld>`）。双节点部署使用 ``realworld_eval_dual_franka``；GPU 与两台机械臂同机并需要推理与动作重叠执行时，使用 ``realworld_dual_franka_pi05_RTC``：

      .. code-block:: bash

         # 双节点部署
         bash evaluations/run_eval.sh realworld_eval_dual_franka

         # 单工作站 RTC 部署
         bash evaluations/run_eval.sh realworld_dual_franka_pi05_RTC

      复位后在键盘或脚踏上按 ``a`` 启动 policy（``b`` 标记失败并复位，``c`` 标记成功并复位）。

.. dropdown:: 配置文件、统计量与权重转换说明

   直接编辑仓库预置配置，替换所有带 ``# Replace:`` 标记的占位符：

   .. list-table::
      :header-rows: 1
      :widths: 52 48

      * - 配置文件
        - 用途
      * - ``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
        - GELLO 关节空间采集
      * - ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
        - OpenPI PyTorch π₀.₅ SFT
      * - ``examples/embodiment/config/realworld_eval_dual_franka.yaml``
        - 双节点真机部署
      * - ``evaluations/realworld/realworld_dual_franka_pi05_RTC.yaml``
        - 单工作站 RTC 部署
      * - ``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
        - 关节空间硬件默认配置
      * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
        - ``tcp_rot6d`` 硬件默认配置

   重点字段说明：

   * ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP``：各控制节点可见的 FCI IP。
   * ``BASE_CAMERA_SERIAL``、``LEFT_CAMERA_SERIAL``、``RIGHT_CAMERA_SERIAL``：RealSense 序列号与稳定的 ``/dev/v4l/by-id`` 路径。
   * ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION`` 与 ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT``：稳定的 ``/dev/serial/by-id`` 路径。
   * ``TASK_DESCRIPTION``：采集、SFT 与部署共用的自然语言任务描述。
   * ``openpi_data.norm_stats_path``：指向 ``calculate_norm_stats.py`` 生成的 ``norm_stats.json``\ （通常为 ``./assets/pi05_dualfranka_tcp_rot6d/<repo_id>/norm_stats.json``）；省略时按 OpenPI TrainConfig 默认从 ``model_path`` 读取。
   * 基础 ``openpi-jax`` 转换步骤见 :doc:`sft_openpi`；如需将 RLinf SFT ``full_weights.pt`` 转换为 OpenPI safetensors 格式，见 :doc:`../../guides/convertor` 或查看命令行参数：

   .. code-block:: bash

      python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi --help

结果与常见问题
--------------

在训练节点运行 ``tensorboard --logdir ./logs`` 观察训练损失与学习率，在 ``node 0`` 运行 ``python -m toolkits.realworld_check.collect_monitor logs/collect.log`` 监控实时采集进度。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理方法
   * - Ray worker 导入依赖失败
     - 在执行 ``ray start`` 的同一 shell 中检查 ``which python`` 与 ``python -c "import franky, gello, gello_teleop"``，并查看 ``/tmp/ray/session_latest/logs/worker-*.err``。
   * - 脚踏设备权限不足
     - 重新执行 ``sudo chmod 666 /dev/input/eventXX``，并确认 ``ray start`` 前已导出指向同一设备的 ``RLINF_KEYBOARD_DEVICE``。
   * - RealSense 降级为 USB 2
     - 更换线缆或接口，确保 ``lsusb -t`` 显示 ``5000M`` 而非 ``480M``。
   * - GELLO 停止输出或跳变 ``2π``
     - 重启主手电源、重插 FTDI 转接器，用 ``gello_joint --port ...`` 检查后重新执行 ``align-sequential``\ （见上方“GELLO 首次标定”）。
   * - 单侧机械臂复位无响应
     - 在对应控制节点运行 ``ping -c 100 <robot_ip>``；若存在丢包，先修复网卡/FCI 链路或重启机器人。
   * - 部署找不到 ``norm_stats.json``
     - 确认文件位于 ``<model_path>/<actor.model.openpi_data.repo_id>/norm_stats.json``。
   * - 部署启动后停留在 idle
     - 检查脚踏路径与权限后按下 ``a``；eval wrapper 会在回合间保持 idle 等待人工触发。
