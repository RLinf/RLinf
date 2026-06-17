双 Franka 真机：GELLO 数据采集、π₀.₅ SFT 与部署
====================================================

本示例介绍 RLinf 双臂 Franka 真机流程的最小可执行工作流：启动双节点真机
集群，使用两台 GELLO 主手采集双臂演示数据，将数据转换为 tcp_rot6d 后对
π₀.₅ 执行 SFT，并通过脚踏控制将训练后的策略部署至机器人。

本文包含硬件布局、依赖安装、必须替换的配置项、硬件检查、数据采集、
tcp_rot6d 回填与 SFT、部署，以及常见故障排查。

首次搭建硬件时，建议先阅读：

* :doc:`franka`：单臂 Franka 基础、Ray 集群、FCI 与
  ``RLINF_NODE_RANK``。
* :doc:`franka_gello`：GELLO 安装、Dynamixel 权限和 ``gello-teleop``。


概览
----

流程包含五个阶段：

1. 在两个机器人节点上安装 ``franka-franky`` 依赖。
2. 在 head 节点设置 ``RLINF_NODE_RANK=0``，在 worker 节点设置
   ``RLINF_NODE_RANK=1``，然后启动 Ray。
3. 使用 ``realworld_collect_data_gello_joint_dual_franka`` 采集关节空间演示。
4. 将数据转换为 tcp_rot6d，并使用
   ``realworld_sft_openpi_dual_franka_tcp_rot6d`` 执行 π₀.₅ SFT。
5. 使用 ``realworld_eval_dual_franka`` 部署策略。

仓库预置配置已指定对应环境。通常仅需替换硬件路径、任务文本、数据集
ID 和模型 checkpoint。


环境
----

硬件布局
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - 节点
     - 角色
     - 硬件
   * - ``node 0`` （head）
     - Ray head、env worker、左臂 Franka controller、相机、GELLO 输入、
       采集和部署入口
     - GPU 机器；左 Franka FR3；左 Robotiq gripper；base 相机；左右腕相机；
       两台 GELLO 主手；PCsensor 脚踏
   * - ``node 1`` （worker）
     - Ray worker 和右臂 Franka controller
     - 右 Franka FR3；右 Robotiq gripper；GPU 可选

两台 Franka 通常分别通过专用网卡连接到本地控制节点。所有相机、两台 GELLO
主手和脚踏都连接到 ``node 0``。

数据与动作空间
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 22 20 34

   * - 阶段
     - 环境
     - 形状
     - 用途
   * - 采集
     - ``DualFrankaJointEnv-v1``
     - ``state=[68]``，``actions=[16]``
     - GELLO 关节空间演示
   * - SFT / 部署
     - ``DualFrankaTCPEnv-v1``
     - ``state=[20]``，``actions=[20]``
     - π₀.₅ tcp_rot6d 策略

每条 tcp_rot6d 动作为每只手臂一组 ``[xyz(3), rot6d(6), gripper(1)]``。
主图像键为 ``left_wrist_0_rgb``；额外视角顺序为 ``base_0_rgb`` 和
``right_wrist_0_rgb``。


依赖安装
--------

机器人节点
~~~~~~~~~~

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
~~~~~~~~~~

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
~~~~~~~~

在执行 SFT 的远端 GPU 训练集群上安装 OpenPI 依赖：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openpi --env maniskill_libero --use-mirror
   source .venv/bin/activate


配置
----

基于仓库预置配置进行参数替换，无需新增独立配置文件：

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - 配置
     - 用途
   * - ``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
     - GELLO 关节空间采集
   * - ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
     - 在转换后的 tcp_rot6d 数据上执行 π₀.₅ SFT
   * - ``examples/embodiment/config/realworld_eval_dual_franka.yaml``
     - 真机策略部署
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
--------

启动 Ray 前完成以下检查。

脚踏
~~~~

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
~~~~

.. code-block:: bash

   rs-enumerate-devices | grep -E "Name|Serial|USB Type"
   ls /dev/v4l/by-id/
   lsusb -t

预期输出应包含 RealSense serial、两个 Lumos 设备，以及类似 ``5000M`` 的
USB-3 速度。``480M`` 表示设备已降级为 USB 2。

GELLO 主手
~~~~~~~~~~

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
~~~~~~~~~~

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


快速开始
--------

启动 Ray
~~~~~~~~

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
~~~~~~~~~~~~

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

回填 tcp_rot6d
~~~~~~~~~~~~~~

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
~~~~~~~~

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
   export DUAL_FRANKA_DATA_ROOT=/path/to/lerobot_root
   export PI05_BASE_CKPT=/path/to/pi05/torch
   export SFT_REPO_ID=<repo_id>/tcp_rot6d_v1

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_tcp_rot6d \
       --repo-id $SFT_REPO_ID

   mkdir -p $PI05_BASE_CKPT/$SFT_REPO_ID
   cp <openpi_assets_dirs>/pi05_dualfranka_tcp_rot6d/$SFT_REPO_ID/norm_stats.json \
      $PI05_BASE_CKPT/$SFT_REPO_ID/norm_stats.json

   bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_tcp_rot6d

并在 ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
中更新 ``SFT_DATASET_REPO_ID``、``PI05_BASE_CKPT``、logger 设置和集群放置。
Checkpoint 保存到
``<log_path>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``。


评估与部署
----------

准备 checkpoint 文件
~~~~~~~~~~~~~~~~~~~~

``node 0`` 上的部署 checkpoint 目录必须包含：

.. code-block:: text

   <model_path>/
   ├── actor/model_state_dict/full_weights.pt
   └── <repo_id>/tcp_rot6d_v1/norm_stats.json

将 SFT checkpoint 和匹配的 normalization stats 同步回 ``node 0``：

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

在 ``examples/embodiment/config/realworld_eval_dual_franka.yaml`` 中将
``rollout.model.model_path`` 设为 ``$DEPLOY_CKPT``，将
``actor.model.openpi_data.repo_id`` 设为 ``<repo_id>/tcp_rot6d_v1``。

启动策略部署
~~~~~~~~~~~~

可复用采集阶段的 Ray 集群，也可使用相同环境变量重新启动。随后在
``node 0`` 执行：

.. code-block:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}

   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka

Hydra override 示例：

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka \
       rollout.model.model_path=/path/to/deploy/global_step_<N> \
       actor.model.openpi_data.repo_id=<repo_id>/tcp_rot6d_v1 \
       env.eval.override_cfg.task_description="handover the object"

部署阶段脚踏按键：

* ``a``：从 idle 状态启动策略执行。
* ``b``：标记失败并 reset。
* ``c``：标记成功并 reset。

每次 reset 后，wrapper 会再次等待 ``a``，便于在下一次 episode 前重新布置
场景。


故障排查
--------

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
