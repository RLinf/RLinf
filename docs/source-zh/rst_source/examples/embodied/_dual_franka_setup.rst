.. This file is a reusable include, not a standalone page.
   Shared by Dual-Arm Franka recipes (dual_franka.rst and dual_franka_openpi_pytorch.rst).

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 观测
     - 底座 RealSense 全局视角、左右 Lumos 腕部视角，以及双臂机器人状态。
   * - 动作
     - 20 维双臂 ``tcp_rot6d`` 目标：``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``。
   * - 奖励
     - 来自 USB 脚踏（``a`` / ``b`` / ``c``）的人工回合标记。
   * - 提示词
     - 保存在数据集与 OpenPI 元数据中的自然语言任务描述 ``TASK_DESCRIPTION``。

硬件与安全准备
--------------

将左臂 Franka、三路相机、两台 GELLO 主手和 PCsensor FootSwitch 接到 ``node 0``，右臂 Franka 接到 ``node 1``。两台 GELLO 主手必须直连 ``node 0`` 本机，避免 1 kHz 串口数据流经局域网转发。

.. warning::

   启用机械臂前，把急停开关放在触手可及的位置，清空双臂共享运动空间，并确认双臂处于安全初始姿态。核对 ``target_ee_pose`` 落在 ``realworld_dual_franka_tcp_rot6d`` 的工作空间限位盒内：``x`` 在 ``[0.3, 0.9]``，``y`` 在 ``[-0.4, 0.4]``，``z`` 在 ``[0.02, 0.7]``。切勿跨网段转发 1 kHz GELLO 数据流。

先用厂商工具将 PCsensor FootSwitch 按键设为 ``a`` / ``b`` / ``c``，并整理工作站标识：左右臂 FCI IP、RealSense 序列号与 Lumos 的 ``/dev/v4l/by-id`` 路径、Robotiq 与 GELLO 的 ``/dev/serial/by-id`` 路径。

.. dropdown:: 实时内核前提与系统调优（推荐）

   ``franka`` 环境通过 Franky 与 libfranka 以 1 kHz 频率与机械臂通信。请按 Franka 官方 `实时内核文档 <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_ 配置 ``PREEMPT_RT`` 内核与实时权限。

   双臂硬件配置默认 ``realtime_config: ignore``，在标准内核上也会启动并输出警告；高负载下错过 1 kHz 控制周期可能触发机器人 reflex。如需强制要求实时内核，可改为 ``realtime_config: enforce``。

   启动 Ray 前，在每台直连 Franka 的机器人节点执行调优与链路检查（将 ``eth0`` 替换为专用网卡，``172.16.0.2`` 在 ``node 0`` 填 ``LEFT_ROBOT_IP``，在 ``node 1`` 填 ``RIGHT_ROBOT_IP``）：

   .. code-block:: bash

      # 每次开机后重新执行
      sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
          echo performance > "$g"
      done'
      sudo sysctl -w kernel.sched_rt_runtime_us=-1
      sudo ethtool -C eth0 rx-usecs 0 tx-usecs 0 2>/dev/null || true  # 替换为 Franka 专用网卡

      # 可选：跨重启保留 RT 调度预算设置
      echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf

      # 检查实时权限与机器人链路时延
      uname -a | grep -o PREEMPT_RT
      ulimit -r
      ulimit -l
      sudo cyclictest -p 80 -t 4 -i 1000 -l 300000 -m
      ping -c 1000 -i 0.001 172.16.0.2 | tail -3  # 替换为机械臂 IP

   确认 ``ulimit -r`` 返回 ``99`` 或 ``unlimited``，``ulimit -l`` 返回 ``unlimited``。

.. dropdown:: GELLO 首次标定

   在 RLinf 仓库根目录（保持 ``.venv`` 已激活），两台 GELLO 主手均可在 ``node 0`` 上对左臂完成一次标定，并用 ``align-sequential`` 验证对齐。更换 ``GELLO_PORT`` 后对第二台主手重复执行：

   .. code-block:: bash

      export GELLO_PORT=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0  # 替换为实际 FTDI 设备

      python -m toolkits.realworld_check.test_gello calibrate
      python -m toolkits.realworld_check.test_gello align-sequential

   对齐成功时的预期输出：

   .. code-block:: text

      ALL JOINTS ALIGNED
        per-joint Δ (rad): ['+0.012', '-0.008', '+0.005', '+0.021', '-0.041', '+0.009', '-0.003']
        max |Δ| = 0.041 rad on J5 (stream gate threshold = 0.5 rad — well under)
      You can now Ctrl-C and start collect_data.sh.
