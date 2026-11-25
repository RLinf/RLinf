XSquare Turtle2 在线强化学习
============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg
   :align: center
   :width: 80%

   用于 SAC/CNN 真机按键训练的 XSquare Turtle2 双臂机器人。

在 XSquare Turtle2 双臂机器人上部署控制节点与 GPU 训练节点，训练视觉 SAC/CNN 按键 policy。

概览
----

控制主机在厂商 ROS 容器内驱动机器人，GPU 节点加载预训练 ResNet-10 主干网络，运行异步 SAC、Cross-Q 或 RLPD 训练。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      XSquare Turtle2 · 单臂 / 双臂

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      最多 3 路 RGB 相机（ID 0–2）

   .. grid-item-card:: 工作流
      :text-align: center

      Dummy 自检 · SAC / Cross-Q / RLPD · CNN

   .. grid-item-card:: 硬件
      :text-align: center

      控制主机（XSquare 容器）· GPU 节点

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - 任务
     - 配置 / 入口
     - 说明
   * - Dummy 自检
     - ``realworld_dummy_turtle2_sac_cnn``
     - 在无硬件运动情况下验证配置与集群连通性。
   * - 真机训练
     - ``realworld_button_turtle2_sac_cnn``
     - 使用单臂或双臂训练按键任务。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 最多 3 路 RGB 相机流，以及每条活动机械臂的 TCP 位姿。
   * - 动作
     - 按键任务使用每臂 6 维连续增量位姿（双臂 12 维，夹爪固定闭合；机械臂硬件本身支持含夹爪的 7 维控制）。
   * - 奖励
     - 按键成功时返回 ``1.0``，可选指数稠密奖励。
   * - 指令
     - ``Press the button with the end-effector.``

硬件与安全准备
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 组件
     - 要求
   * - 机器人与相机
     - XSquare Turtle2 双臂机器人，机身最多 3 路 RGB 相机（ID 0–2）。
   * - 控制节点
     - 同一局域网下的小型 Linux 主机（不需要 GPU），运行 XSquare 官方 Docker 容器。
   * - 训练 / Rollout 节点
     - 同一局域网下的 Linux GPU 服务器，运行 CNN policy。

.. warning::

   训练节点与机器人控制节点须处于同一局域网。关闭 ``is_dummy`` 前，确保物理急停开关触手可及并清空双臂活动范围。按键任务复位时会将机械臂移至 ``target_ee_pose`` 沿 Z 轴上方略高处，启动运动前先核对标定位姿。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 机器人控制节点

      **1. 进入 XSquare 容器并安装 RLinf**

      先进入 XSquare 官方 Docker 容器（联系 `XSquare <https://x2robot.com>`__ 获取镜像与启动说明），再安装 RLinf：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env xsquare_turtle2 && source .venv/bin/activate

      **2. 记录目标末端位姿**

      通过 XSquare 控制界面将机械臂移到按键目标位姿，按 XSquare 欧拉角顺序 ``[x, y, z, rz, ry, rx]`` 读取并记录。双臂模式下分别为两条机械臂记录。

      **3. 加入 Ray 集群**

      Ray 在执行 ``ray start`` 时锁定当前 Python 解释器与环境变量。在仓库根目录激活 ``.venv`` 并导出环境变量后启动 worker（将 ``192.168.1.10`` 替换为 head 节点 IP，多网卡机器按需修改 ``eth0``）：

      .. code-block:: bash

         source .venv/bin/activate
         export RLINF_NODE_RANK=1
         export RLINF_COMM_NET_DEVICES=eth0  # 仅多网卡机器需要设置
         ray start --address='192.168.1.10:6379'

   .. tab-item:: 训练与 Rollout 节点

      **1. 安装 RLinf 并下载 ResNet-10 权重**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env xsquare_turtle2 && source .venv/bin/activate
         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./RLinf-ResNet10-pretrained

      **2. 启动 Ray head**

      在仓库根目录导出环境变量并启动 Ray head（将 ``192.168.1.10`` 替换为本机 IP），用 ``ray status`` 确认双节点已连接：

      .. code-block:: bash

         export RLINF_NODE_RANK=0
         export RLINF_COMM_NET_DEVICES=eth0  # 仅多网卡机器需要设置
         ray start --head --port=6379 --node-ip-address=192.168.1.10
         ray status

      **3. 修改机械臂、目标位姿与权重路径**

      编辑 ``examples/embodiment/config/realworld_button_turtle2_sac_cnn.yaml`` 中的关键字段：

      .. code-block:: yaml

         env:
           train:
             override_cfg:
               is_dummy: False
               use_arm_ids: [1]          # 0=左臂，1=右臂；双臂用 [0, 1]
               use_camera_ids: [2]       # 使用的相机 ID（0、1 或 2）
               target_ee_pose:           # [[左臂], [右臂]]，欧拉角 [x, y, z, rz, ry, rx]
                 - [0, 0, 0, 0, 0, 0]
                 - [0.3, 0.0, 0.15, 0.0, 1.0, 0.0]

         actor:
           model:
             model_path: "./RLinf-ResNet10-pretrained"
             state_dim: 6    # 单臂 6，双臂 12
             action_dim: 6   # 单臂 6，双臂 12

      .. dropdown:: 完整配置参考（可选）

         .. code-block:: yaml

            cluster:
              num_nodes: 2  # 1 个训练/rollout 节点 + 1 个控制节点
              component_placement:
                actor:
                  node_group: "gpu"
                  placement: 0
                rollout:
                  node_group: "gpu"
                  placement: 0
                env:
                  node_group: turtle2
                  placement: 0
              node_groups:
                - label: "gpu"
                  node_ranks: 0
                - label: turtle2
                  node_ranks: 1
                  hardware:
                    type: Turtle2
                    configs:
                      - node_rank: 1

            env:
              train:
                override_cfg:
                  is_dummy: False
                  use_arm_ids: [1]          # 0=左臂，1=右臂；双臂使用 [0,1]
                  use_camera_ids: [2]       # 要使用的相机 ID（0、1 或 2）
                  target_ee_pose:           # [[左臂目标位姿], [右臂目标位姿]]，欧拉角 [x,y,z,rz,ry,rx]
                    - [0, 0, 0, 0, 0, 0]
                    - [0.3, 0.0, 0.15, 0.0, 1.0, 0.0]

            actor:
              model:
                model_path: "./RLinf-ResNet10-pretrained"
                state_dim: 6    # 单臂 6（xyz+euler），双臂 12
                action_dim: 6   # 单臂 6（xyz_delta+rpy_delta），双臂 12

      **4. 先做 Dummy 自检，再启动真机训练**

      在 head 节点先用 dummy 配置验证集群与模型管线，再启动真机训练：

      .. code-block:: bash

         # 无硬件运动的管线自检
         bash examples/embodiment/run_realworld_async.sh realworld_dummy_turtle2_sac_cnn

         # 启动真机训练
         bash examples/embodiment/run_realworld_async.sh realworld_button_turtle2_sac_cnn

结果与常见问题
--------------

在 Ray head 节点运行 ``tensorboard --logdir ./logs --port 6006``，重点观察 ``env/success_once``、``env/return``、``train/sac/*`` 与 ``train/replay_buffer/*``，完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 现象
     - 解决方法
   * - 控制节点找不到 XSquare SDK 或 ROS 包
     - 在 XSquare 官方 Docker 容器内安装 RLinf 并执行 ``ray start``，确保 SDK 位于 Python 搜索路径中。
   * - Ray worker 缺少 ``RLINF_NODE_RANK`` 或虚拟环境
     - 先激活 ``.venv`` 并导出 ``RLINF_NODE_RANK``，再执行 ``ray start``。
   * - 双臂模式下状态或动作维度不匹配
     - 设置 ``use_arm_ids: [0, 1]`` 时，将 ``actor.model.state_dim`` 和 ``actor.model.action_dim`` 均改为 ``12``。

如需在 Turtle2 上开发其他真机任务，参见 :doc:`新增真机任务 <../../extending/new_task>`。
