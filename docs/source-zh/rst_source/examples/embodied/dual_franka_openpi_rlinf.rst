使用 OpenPI_RLinf π0 与 π0.5 完成双 Franka 训练与部署
=======================================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-deploy-rlinf.jpg
   :align: center
   :width: 80%
   :alt: OpenPI_RLinf 双 Franka 部署

   使用 OpenPI_RLinf 完成双 Franka π0 与 π0.5 策略训练与部署。

本指南介绍 Dual Franka 上 OpenPI_RLinf π0 与 π0.5 的统一训练和部署流程。
两种模型共用相同的数据格式、tcp_rot6d action representation、SFT 入口、
checkpoint 保存结构、权重转换工具、真机环境和 evaluation 入口，因此也可
使用这套管线开展 π0/π0.5 对比实验。

概览
----------------------------------------

构建一套双臂数据集，训练 π0 或 π0.5，并通过 legacy real-world PT 后端或
OpenPI_RLinf safetensors 后端部署到双节点 Franka rig。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenPI_RLinf π0 · π0.5

   .. grid-item-card:: 算法
      :text-align: center

      SFT · eval-only deployment

   .. grid-item-card:: 任务
      :text-align: center

      Dual-arm manipulation

   .. grid-item-card:: 硬件
      :text-align: center

      2× Franka · 2 robot nodes · GELLO

| **你将完成:** 安装 franky 依赖 → 采集 GELLO 示教 → 转换 rot6d 数据 → 运行 SFT → 部署 eval 配置.
| **前置条件:** :doc:`franka` · :doc:`franka_gello` · 两台 Franka · OpenPI assets.

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - 任务
     - 配置 / 入口
     - 说明
   * - Collection
     - ``realworld_collect_data_gello_joint_dual_franka``
     - 采集双臂关节轨迹。
   * - SFT
     - ``realworld_sft_openpi_rlinf_pi0_dual_franka_tcp_rot6d`` / ``realworld_sft_openpi_rlinf_pi05_dual_franka_tcp_rot6d``
     - 使用对应的 tcp_rot6d 配置微调 π0 或 π0.5。
   * - Deployment（OpenPI PyTorch）
     - ``realworld_eval_dual_franka_openpi_pi0`` / ``realworld_eval_dual_franka_openpi_pi05``
     - 使用 legacy FP32 ``full_weights.pt`` 在机器人节点上部署。
   * - Deployment（OpenPI_RLinf）
     - ``realworld_eval_dual_franka_openpi_pi0_rlinf`` / ``realworld_eval_dual_franka_openpi_pi05_rlinf``
     - 使用新版 FP32 ``model.safetensors`` 在机器人节点上部署。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - 字段
     - 说明
   * - Observation
     - 腕部/全局相机视角加双臂机器人状态。
   * - Action
     - 双臂 tcp_rot6d：``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``。
   * - Reward
     - 评测成功信号或人工门控部署结果。
   * - Prompt
     - OpenPI 数据/配置 metadata 中的任务文本。

π0 与 π0.5 共用三路相机、prompt、20D tcp_rot6d action 和 20 帧 action
horizon。π0 还会通过 ``state_proj`` 将 normalization 后的当前 TCP
pose/gripper state 注入模型 core；π0.5 不向模型 core 注入该连续 state。

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - 项目
     - π0
     - π0.5
   * - 连续 state 输入模型 core
     - 是
     - 否
   * - SFT 入口与 checkpoint 结构
     - 共用
     - 共用
   * - Legacy PT converter
     - ``openpi_rlinf_to_openpi_pytorch``
     - ``openpi_rlinf_to_openpi_pytorch``
   * - OpenPI_RLinf converter
     - ``sft_to_openpi_rlinf``
     - ``sft_to_openpi_rlinf``

安装
----------------------------------------

机器人节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``node 0`` 和 ``node 1`` 上分别执行机器人节点安装。根据 Franka 官方 `compatibility matrix
<https://frankarobotics.github.io/docs/compatibility.html>`_ 选择
``LIBFRANKA_VERSION``；避免使用 libfranka ``0.18.0``。

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   export LIBFRANKA_VERSION=0.15.0       # 替换为与固件兼容的版本
   bash requirements/install.sh embodied --env franka-franky --use-mirror
   source .venv/bin/activate

按照 :doc:`franka_gello` 在 ``node 0`` 安装 GELLO 依赖。两台 GELLO 主手应
保留在 ``node 0`` 本机，不应通过 LAN 转发 1 kHz 数据流。

实时性前提
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``franka-franky`` 通过 franky/libfranka 与每台 Franka 进行 1 kHz 通信。
RLinf 安装脚本只安装运行依赖；PREEMPT_RT 内核与实时权限请按 Franka 官方
`实时内核文档
<https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_
配置。

启动 Ray 前，在每台直接与 Franka 通信的工作站上执行以下示例。将
``<FRANKA_NIC>`` 替换为机器人专用网卡；``<ROBOT_IP>`` 在 ``node 0`` 上使用
``LEFT_ROBOT_IP``，在 ``node 1`` 上使用 ``RIGHT_ROBOT_IP``。

.. code-block:: bash

   # 每次开机后重新执行。
   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
       echo performance > "$g"
   done'
   sudo sysctl -w kernel.sched_rt_runtime_us=-1
   sudo ethtool -C <FRANKA_NIC> rx-usecs 0 tx-usecs 0 2>/dev/null || true

   # 可选：让 RT 调度预算设置跨重启持久化。
   echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf

   # 启动 RLinf 前检查实时权限和机器人链路。
   uname -a | grep -o PREEMPT_RT
   ulimit -r
   ulimit -l
   sudo cyclictest -p 80 -t 4 -i 1000 -l 300000 -m
   ping -c 1000 -i 0.001 <ROBOT_IP> | tail -3

``ulimit -r`` 应为 ``99`` 或 ``unlimited``；``ulimit -l`` 应为
``unlimited``。每次重启机器人工作站后，都需要重新执行每次开机后的调优命令。

训练节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在执行 SFT 的远端 GPU 训练集群上安装 OpenPI 依赖：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openpi --env maniskill_libero --use-mirror
   source .venv/bin/activate


配置
----------------------------------------

基于仓库预置配置进行参数替换，无需新增独立配置文件：

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - 配置
     - 用途
   * - ``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
     - GELLO 关节空间采集
   * - ``examples/sft/config/realworld_sft_openpi_rlinf_pi05_dual_franka_tcp_rot6d.yaml``
     - 在转换后的 tcp_rot6d 数据上执行 π₀.₅ SFT
   * - ``examples/sft/config/realworld_sft_openpi_rlinf_pi0_dual_franka_tcp_rot6d.yaml``
     - 在转换后的 tcp_rot6d 数据上执行 π₀ SFT
   * - ``evaluations/realworld/realworld_eval_dual_franka_openpi_pi0.yaml``
     - legacy OpenPI PyTorch π0 FP32 真机部署
   * - ``evaluations/realworld/realworld_eval_dual_franka_openpi_pi0_rlinf.yaml``
     - OpenPI_RLinf π0 FP32 真机部署
   * - ``evaluations/realworld/realworld_eval_dual_franka_openpi_pi05.yaml``
     - legacy OpenPI PyTorch π0.5 FP32 真机部署
   * - ``evaluations/realworld/realworld_eval_dual_franka_openpi_pi05_rlinf.yaml``
     - OpenPI_RLinf π0.5 FP32 真机部署
   * - ``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
     - 关节空间硬件默认配置
   * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
     - tcp_rot6d 硬件默认配置

替换以下带 ``# Replace:`` 标记的占位符：

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP``：各控制节点可见的 FCI IP。
* ``BASE_CAMERA_SERIAL``、``LEFT_CAMERA_SERIAL``、``RIGHT_CAMERA_SERIAL``：
  相机 serial 或稳定的 ``/dev/v4l/by-id`` 路径。
* ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION``：Robotiq 转接器
  的稳定 ``/dev/serial/by-id`` 路径。
* ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT``：两台 GELLO 主手的稳定
  ``/dev/serial/by-id`` 路径。
* ``TASK_DESCRIPTION``：采集、SFT 和部署使用的自然语言任务描述。
* ``SFT_DATASET_REPO_ID``：转换后的数据集 ID，通常是
  ``<repo_id>/tcp_rot6d_v1``。
* ``MODEL_PATH``：``node 0`` 上的部署 checkpoint 目录。


硬件检查
----------------------------------------

启动 Ray 前完成以下检查。

脚踏
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用厂商工具将 PCsensor FootSwitch 的按键配置为 ``a`` / ``b`` / ``c``。然后在
``node 0`` 执行：

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   sudo chmod 666 /dev/input/eventXX
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

.. note::

   将所有 ``eventXX`` 替换为第一条命令显示的实际 ``eventNN``，例如
   ``event7``。必须在 ``ray start`` 前导出 ``RLINF_KEYBOARD_DEVICE``。

相机
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   rs-enumerate-devices | grep -E "Name|Serial|USB Type"
   ls /dev/v4l/by-id/
   lsusb -t

预期输出应包含 RealSense serial、两个 Lumos 设备，以及类似 ``5000M`` 的
USB-3 速度。``480M`` 表示设备已降级为 USB 2。

GELLO 主手
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每次仅连接一台主手，并使用以下命令识别两个 FTDI 路径：

.. code-block:: bash

   ls /dev/serial/by-id/ | grep -i ftdi

验证每台主手能够稳定输出关节值：

.. code-block:: bash

   cd /path/to/RLinf
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python -m rlinf.envs.realworld.common.gello.gello_joint_expert \
       --port /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0

该命令会持续刷新输出，例如：

.. code-block:: text

   joints=[+0.012 -0.604 +0.031 -2.184 +0.019 +1.571 +0.781]  gripper=[0.035]

如果数值停止更新或出现约 ``2π`` 的跳变，请执行以下标定流程。

GELLO 标定
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _dual-franka-gello-calibration-zh:

每台 GELLO 需完成一次标定，并使用 ``align-sequential`` 验证。两台主手均可在
``node 0`` 上对左臂完成标定。

.. code-block:: bash

   cd /path/to/RLinf
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export GELLO_PORT=/dev/serial/by-id/usb-FTDI_..._<ID>-if00-port0

   python toolkits/realworld_check/test_gello.py calibrate
   python toolkits/realworld_check/test_gello.py align-sequential

成功时，``align-sequential`` 输出如下：

.. code-block:: text

   ALL JOINTS ALIGNED
     per-joint Δ (rad): ['+0.012', '-0.008', '+0.005', '+0.021', '-0.041', '+0.009', '-0.003']
     max |Δ| = 0.041 rad on J5 (stream gate threshold = 0.5 rad — well under)
   You can now Ctrl-C and start collect_data.sh.

将 ``GELLO_PORT`` 替换为第二台主手路径后，重复上述两条命令。


运行
----------------------------------------

启动 Ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray 在 ``ray start`` 时捕获环境变量。启动集群前导出节点 rank 和脚踏设备。

.. code-block:: bash

   # node 0
   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1
   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1

   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

在 ``node 0`` 运行 ``ray status``，确认两个节点均为 ALIVE。

采集演示数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

确认 :ref:`align-sequential <dual-franka-gello-calibration-zh>` 报告
``ALL JOINTS ALIGNED`` 后，在 ``node 0`` 执行采集：

.. code-block:: bash

   cd /path/to/RLinf
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   bash examples/embodiment/collect_data.sh \
       realworld_collect_data_gello_joint_dual_franka 2>&1 | tee logs/collect.log

在另一个 ``node 0`` 终端监控进度：

.. code-block:: bash

   cd /path/to/RLinf
   python toolkits/realworld_check/collect_monitor.py logs/collect.log

脚踏按键：

* ``a``：开始录制；录制过程中再次按下将中止录制并丢弃当前 buffer。
* ``b``：递增 ``segment_id``，用于标记子任务边界。
* ``c``：标记成功，写入 LeRobot shard，并结束当前 episode。

如需继续采集，设置 ``data_collection.resume: true`` 并保持相同的
``data_collection.save_dir``，新数据将追加为新的 ``id_*`` shard。

数据集处理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
采集结果可能存在多id、以及脏数据情况，建议在 SFT 前进行数据清洗以及将数据合并到一个id下。可使用RLinf/toolkits/dual_franka
``delete_lerobot.py`` 删除指定 id/多id 下的数据，使用 ``merge_lerobot.py`` 将多个 id 合并为一个 id。

代码文件中有使用说明

.. code-block:: bash

   python delete_lerobot.py \
        --data-dir /path/to/dataset/rank_0 \
        --delete "id_0:3,5" \
        --dry-run

   python merge_lerobot.py \
        --rank-dir /path/to/dataset \
        --out-dir  /path/to/dataset \
        --log-file /path/to/log \
        --dry-run

回填 tcp_rot6d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

采集结果为关节空间数据。SFT 前需要转换为 tcp_rot6d：

.. code-block:: bash

   cd /path/to/RLinf
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export HF_LEROBOT_HOME=/path/to/lerobot_root
   export DATA_REPO_ID=<repo_id>
   export SFT_REPO_ID=$DATA_REPO_ID/tcp_rot6d_v1

   python toolkits/dual_franka/backfill_tcp_rot6d.py \
       --src $HF_LEROBOT_HOME/$DATA_REPO_ID/joint_v1 \
       --dst $HF_LEROBOT_HOME/$SFT_REPO_ID

运行 SFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先将转换后的数据集同步至训练节点，然后在训练节点执行 SFT：

.. code-block:: bash

   export TRAINER_IP=<trainer_ip>
   export HF_LEROBOT_HOME=/path/to/lerobot_root
   export SFT_REPO_ID=<repo_id>/tcp_rot6d_v1

   ssh $TRAINER_IP "mkdir -p $HF_LEROBOT_HOME/$SFT_REPO_ID"
   rsync -av $HF_LEROBOT_HOME/$SFT_REPO_ID/ \
       $TRAINER_IP:$HF_LEROBOT_HOME/$SFT_REPO_ID/

在训练节点执行：

.. code-block:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export HF_LEROBOT_HOME=/path/to/lerobot_root
   export SFT_REPO_ID=<repo_id>/tcp_rot6d_v1

   # π0.5
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_tcp_rot6d \
       --repo-id $SFT_REPO_ID

   # π0
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_dualfranka_tcp_rot6d \
       --repo-id $SFT_REPO_ID

π0 与 π0.5 使用相同的 SFT 启动入口，并分别传入对应的实验配置：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh SFT_CONFIG_NAME

并在对应的 ``examples/sft/config/realworld_sft_openpi_rlinf_pi0_dual_franka_tcp_rot6d.yaml``
或 ``examples/sft/config/realworld_sft_openpi_rlinf_pi05_dual_franka_tcp_rot6d.yaml``
中更新 ``train_data_paths``、``model_path``、``assets_dir``、``asset_id``、``logger`` 设置和集群放置。
Checkpoint 保存到
``<log_path>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``。

其中 ``assets_dir`` 为 ``norm_stats.json`` 所在目录，``asset_id`` 为
``norm_stats.json`` 所在的 ``repo_id``，``model_path`` 为训练时指定的模型
路径。本文使用的 base model 需要从 ``openpi-jax`` 格式转换，转换接口请参考
``rlinf/utils/ckpt_convertor/openpi/README.md``。

使用代码为：

.. code-block:: bash

   rlinf/utils/ckpt_convertor/openpi/jax_to_openpi_rlinf.py

转换 SFT Checkpoint 用于部署
----------------------------------------

π0 与 π0.5 的 SFT checkpoint 都支持两种部署格式。legacy real-world PT
部署使用 ``openpi_rlinf_to_openpi_pytorch`` 的 PT 输出，其中 reference model
必须与 SFT variant
匹配，即对应的原始 OpenPI PyTorch π0 或 π0.5 base 权重。该转换器要求 SFT
checkpoint 为 FP32，并在 PT 到 PT 的直接转换中始终保持 FP32。程序会根据
匹配的 reference 自动选择 π0 的标准 RMSNorm 映射或 π0.5 的自适应 RMSNorm
映射，不需要额外传入 variant 参数。
部署精度可通过 YAML 中的 ``rollout.model.precision`` 控制，其行为取决于
所使用的 backend：

不同 Backend 的部署精度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Backend
     - ``fp32``
     - ``bf16``
     - ``null``
   * - Legacy OpenPI
     - 纯 FP32
     - 纯 BF16
     - OpenPI 原有的混合精度
   * - OpenPI_RLinf
     - 纯 FP32
     - 纯 BF16
     - 保持 ``Pi0Config`` 创建时的 BF16；不是独立的精度模式

OpenPI PyTorch 部署使用 ``openpi_rlinf_to_openpi_pytorch``。将
``INPUT_CHECKPOINT`` 设置为 RLinf SFT checkpoint，将
``OPENPI_PYTORCH_REFERENCE`` 设置为与 SFT variant 匹配的原始 OpenPI
PyTorch π0 或 π0.5 base model，并通过 ``OUTPUT_DIRECTORY`` 指定输出目录。

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert \
     --mode            openpi_rlinf_to_openpi_pytorch \
     --input-model     INPUT_CHECKPOINT \
     --reference-model OPENPI_PYTORCH_REFERENCE \
     --output-model    OUTPUT_DIRECTORY \
     --output-format   pt \
     --dtype           fp32

OpenPI PyTorch 输出目录为：

.. code-block:: text

   OUTPUT_DIRECTORY/
   ├── actor/model_state_dict/full_weights.pt
   ├── actor/model_state_dict/full_weights.pt.report.json
   └── DATASET_REPO_ID/norm_stats.json  # 需要单独复制

OpenPI_RLinf 部署使用 ``sft_to_openpi_rlinf``。π0 的 ``CONFIG_NAME`` 为
``pi0_dualfranka_tcp_rot6d``，π0.5 的 ``CONFIG_NAME`` 为
``pi05_dualfranka_tcp_rot6d``。

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert \
     --mode sft_to_openpi_rlinf \
     --ckpt INPUT_CHECKPOINT \
     --input-norm-stats NORM_STATS_JSON \
     --output-model OUTPUT_MODEL \
     --output-norm-stats OUTPUT_MODEL/DATASET_REPO_ID/norm_stats.json \
     --config-name CONFIG_NAME \
     --dtype fp32

OpenPI_RLinf 输出目录为：

.. code-block:: text

   OUTPUT_MODEL/
   ├── model.safetensors
   ├── config.json
   └── DATASET_REPO_ID/norm_stats.json

评估与部署
----------------------------------------

准备 checkpoint 文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``node 0`` 上的部署 checkpoint 目录必须符合上面两种结构之一。将
``rollout.model.model_path`` 指向对应的 ``OUTPUT_DIRECTORY`` 或
``OUTPUT_MODEL``。

将转换后的 checkpoint 和匹配的 normalization stats
同步回 ``node 0``：

.. code-block:: bash

   export TRAINER_IP=<trainer_ip>
   export DEPLOY_CKPT=/path/to/deploy/global_step_<N>
   export SFT_REPO_ID=<repo_id>/tcp_rot6d_v1

   mkdir -p $DEPLOY_CKPT/actor/model_state_dict
   mkdir -p $DEPLOY_CKPT/$SFT_REPO_ID

   rsync -av \
       $TRAINER_IP:<train_log>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt \
       $DEPLOY_CKPT/actor/model_state_dict/full_weights.pt
   rsync -av $TRAINER_IP:<train_log>/checkpoints/global_step_<N>/$SFT_REPO_ID/norm_stats.json \
       $DEPLOY_CKPT/$SFT_REPO_ID/norm_stats.json

在 ``evaluations/realworld`` 下选用的配置中将
``rollout.model.model_path`` 设为 ``$DEPLOY_CKPT``，将
``actor.model.openpi_data.repo_id`` 设为 ``<repo_id>/tcp_rot6d_v1``。

启动策略部署
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可复用采集阶段的 Ray 集群，也可使用相同环境变量重新启动。策略启动由
:doc:`真机评测指南 <../../evaluations/guides/realworld>` 统一维护，使用与
模型 variant 和部署格式匹配的配置。
请根据上表中对应 backend 的规则设置 ``rollout.model.precision``。
OpenPI_RLinf 如需显式指定精度，请使用 ``fp32`` 或 ``bf16``。

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_eval_dual_franka_openpi_pi0
   bash evaluations/run_eval.sh realworld_eval_dual_franka_openpi_pi0_rlinf
   bash evaluations/run_eval.sh realworld_eval_dual_franka_openpi_pi05
   bash evaluations/run_eval.sh realworld_eval_dual_franka_openpi_pi05_rlinf

部署阶段脚踏按键：

* ``a``：从 idle 状态启动策略执行。
* ``b``：标记失败并 reset。
* ``c``：标记成功并 reset。

每次 reset 后，wrapper 会再次等待 ``a``，便于在下一次 episode 前重新布置
场景。


故障排查
----------------------------------------

**Ray worker 导入失败**
   在运行 ``ray start`` 的同一个 shell 中检查 ``which python`` 和
   ``python -c "import franky, gello, gello_teleop"``。worker 日志位于
   ``/tmp/ray/session_latest/logs/worker-*.err``。

**脚踏设备权限不足**
   重新执行 ``sudo chmod 666 /dev/input/eventXX``，并确认
   ``RLINF_KEYBOARD_DEVICE`` 指向同一个设备。

**RealSense 显示为 USB 2**
   更换线缆或接口。``lsusb -t`` 应显示 ``5000M``，而非 ``480M``。

**GELLO 输出停止**
   重启主手电源，重新连接 FTDI 转接器，并使用
   ``python -m rlinf.envs.realworld.common.gello.gello_joint_expert --port ...``
   验证输出。

**某一机械臂 reset 过程无响应**
   在对应 controller 节点运行 ``ping -c 100 <robot_ip>``。如果出现丢包，先修复
   NIC/FCI 连接或重启机器人。

**部署时无法找到 ``norm_stats.json``**
   确认文件路径为
   ``<model_path>/<actor.model.openpi_data.repo_id>/norm_stats.json``。

**部署持续停留在 idle**
   确认脚踏路径和权限后按下 ``a``。eval wrapper 会在两个 episode
   之间主动停留在 idle 状态。
