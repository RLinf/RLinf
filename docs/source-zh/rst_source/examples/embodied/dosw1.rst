Dexmal DOS-W1 在线强化学习
==========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png
   :align: center
   :width: 80%

   用于 Flow Matching 单臂抓取流程的 Dexmal DOS-W1 双臂机器人。

在 Dexmal DOS-W1 双臂机器人上接入 AirBot gRPC 服务、RealSense 相机与键盘门控，运行 SAC / RLPD Flow Matching 真机抓取训练。

概览
----

机器人控制节点运行 AirBot 服务与环境，GPU 节点运行同步或异步 SAC / RLPD Flow Matching 训练。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      Dexmal DOS-W1 · 双臂 + 主动臂

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      最多 3 路 RealSense · 主动臂 · 键盘门控

   .. grid-item-card:: 工作流
      :text-align: center

      目标关节标定 · 遥操作数据采集 · SAC / RLPD Flow

   .. grid-item-card:: 硬件
      :text-align: center

      AirBot gRPC · 控制主机 · GPU（≥ 4090）

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - 任务
     - 配置 / 入口
     - 说明
   * - Dummy 自检
     - ``dosw1_dummy_sac_mlp_pick.yaml``
     - 无需相机与真机，验证配置和集群连通性。
   * - 数据采集
     - ``dosw1_collect_data``
     - 可选采集主动臂遥操作示教，用于 RLPD 热启动。
   * - 真机训练
     - ``dosw1_pick_sac_flow(_async)``
     - 在抓取任务上用 SAC 训练 ResNet-10 Flow Matching policy。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - ``cam_front``、``cam_left``、``cam_right`` 三路 RGB 图像，加双臂关节与夹爪状态。
   * - 动作
     - 14 维关节空间动作：左右各 6 个关节角加夹爪宽度。
   * - 奖励
     - 接近、抓取、抬升阶段的稠密奖励，到达 ``target_lift_joint`` 时判定成功。
   * - 指令
     - ``Perform the DOSW1 dual-arm manipulation task.``

硬件与安全准备
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 组件
     - 要求
   * - 机器人
     - DOS-W1 双臂机器人及配套主动臂。
   * - 相机
     - 最多 3 路 Intel RealSense 相机（序列号填入 ``camera_serials``；单路相机可将 ``image_num`` 设为 ``1``）。
   * - 机器人控制节点
     - DOS-W1 本机或同一局域网主机，运行 AirBot gRPC 服务（默认端口：左主动臂 ``50050``、左从动臂 ``50051``、右主动臂 ``50052``、右从动臂 ``50053``）。
   * - 训练 / Rollout 节点
     - 同一局域网下的 Linux GPU 服务器（建议 RTX 4090 及以上）。

.. warning::

   训练节点与 DOS-W1 须在同一局域网互通。真机运行时须有安全操作员守在急停开关旁，并清空双臂活动空间。请始终从完整的 ``dosw1_pick_sac_flow.yaml`` 模板修改配置（不要拼接局部片段），核对 ``cluster.num_nodes`` 与各机 ``RLINF_NODE_RANK`` 一致，并保持 ``keyboard_intervention_wrapper: True`` 与 ``enable_human_in_loop: True`` 开启。

.. dropdown:: 安全限幅与末端安全盒机制

   向 AirBot SDK 下发指令前，DOS-W1 环境会对 policy 与 ``TELEOP`` 模式统一施加三层防护：

   1. **单步关节钳位**：将目标关节裁剪到 ``current ± max_joint_delta``\ （默认 ``0.1`` 弧度，约 5.7°/步）。
   2. **绝对关节限幅**：将目标裁剪到 ``[joint_limit_min, joint_limit_max]``\ （默认 ``±π``，可按工况收紧）。
   3. **末端安全盒**：沿关节空间插值做二分搜索，取正运动学结果仍落在 ``left_ee_pose_limit_min/max`` 和 ``right_ee_pose_limit_min/max`` 内的最大比例，自动截断越界动作。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 机器人控制节点

      **1. 安装 RLinf 与 AirBot SDK**

      AirBot SDK 通常预装在 ``~/dos_w1/airbot/5.1.6/airbot_py-5.1.6-py3-none-any.whl`` 与 ``~/dos_w1/airbot/airbot_api``\ （路径不同时先导出 ``DOSW1_SDK_WHEEL`` 与 ``DOSW1_API_PATH``）。请执行完整安装脚本，不要仅运行 ``uv pip install -e .``，否则会缺少 ``evdev``、``opencv-python`` 和 RealSense 绑定：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env dosw1 && source .venv/bin/activate

      **2. 启动 AirBot 服务并标定目标关节**

      启动从动臂与主动臂服务（例如 ``sh ~/dos_w1/airbot/whole_start.sh``）并确认各面板无报错，然后运行标定脚本（将 ``192.168.1.20`` 替换为 DOS-W1 IP）：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_dosw1_controller \
             --robot-url 192.168.1.20 \
             --print-hz 5

      用左主动臂将左从动臂拖至目标抓取位，复制 ``left_joint`` 的前 6 个弧度值（第 7 位为夹爪宽度）填入 ``target_grasp_joint``；再抬至安全抬升位，将新的 ``left_joint[:6]`` 填入 ``target_lift_joint``。

      **3. 开启键盘门控权限**

      授予当前用户访问 ``/dev/input`` 的权限并重新登录（多键盘主机如何指定 ``RLINF_KEYBOARD_DEVICE`` 见 :doc:`franka`）：

      .. code-block:: bash

         sudo usermod -aG input $USER

      .. list-table::
         :header-rows: 1
         :widths: 14 86

         * - 按键
           - 作用
         * - ``s`` / ``r`` / ``d``
           - 从自由遥操作开始新 episode / 中止并丢弃当前 episode / 标记完成并保存。
         * - ``p`` / ``t`` / ``m``
           - 从 ``MODEL`` 或 ``TELEOP`` 暂停 / 切入 ``TELEOP`` / 切回 ``MODEL``\ （开启 ``manual_episode_control_only`` 时忽略）。

      **4. 加入 Ray 集群**

      保持 ``.venv`` 已激活，导出 ``RLINF_NODE_RANK`` 并加入 Ray 集群（将 ``192.168.1.10`` 替换为 GPU 节点 IP）：

      .. code-block:: bash

         export RLINF_NODE_RANK=1
         ray start --address=192.168.1.10:6379

      .. dropdown:: 采集遥操作示教用于 RLPD（可选）

         先修改 ``examples/embodiment/config/dosw1_collect_data.yaml`` 中的 ``robot_url``、四个 gRPC 端口与 ``camera_serials``\ （单节点配置，``cluster.num_nodes: 1``，会自动启动 Ray），再在机器人节点运行：

         .. code-block:: bash

            bash examples/embodiment/collect_data.sh dosw1_collect_data

         用主动臂摆好初始姿态后按 ``s`` 开始，遥操作完成抓取与抬起，按 ``d`` 保存或按 ``r`` 放弃重来。达到 ``runner.num_data_episodes``\ （默认 ``20``）条后自动退出，数据写入 ``<log_path>/demo_buffer/``\ （原始回合位于 ``<log_path>/pickle_episode/``）。在训练配置中指向该目录即可启用 RLPD：

         .. code-block:: yaml

            algorithm:
              demo_buffer:
                load_path: "/path/to/logs/dosw1-collect/<timestamp>/demo_buffer"

   .. tab-item:: 训练与 Rollout 节点

      **1. 安装 RLinf 并下载 ResNet-10 权重**

      GPU 节点通常不含 ``~/dos_w1/airbot``，安装脚本会打印 warning 并跳过 AirBot SDK，其余训练依赖照常安装：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env dosw1 && source .venv/bin/activate
         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./RLinf-ResNet10-pretrained

      **2. 启动 Ray head**

      先导出 ``RLINF_NODE_RANK=0`` 再启动 Ray head（将 ``192.168.1.10`` 替换为本机 IP），并用 ``ray status`` 确认双节点已加入：

      .. code-block:: bash

         export RLINF_NODE_RANK=0
         ray start --head --port=6379 --node-ip-address=192.168.1.10
         ray status

      **3. 修改硬件、目标关节与权重路径**

      编辑 ``examples/embodiment/config/dosw1_pick_sac_flow.yaml``\ （异步解耦版本为 ``dosw1_pick_sac_flow_async.yaml``）：

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: dosw1
               node_ranks: 1
               hardware:
                 type: DOSW1
                 configs:
                   - robot_url: "192.168.1.20"
                     camera_serials: ["SERIAL_1", "SERIAL_2", "SERIAL_3"]
                     node_rank: 1

         env:
           train:
             override_cfg:
               is_dummy: False
               target_grasp_joint: [...]   # 标定得到的 6 个弧度值
               target_lift_joint:  [...]   # 标定得到的 6 个弧度值

         actor:
           model:
             model_path: "./RLinf-ResNet10-pretrained"

      .. dropdown:: 完整配置参考（可选）

         .. code-block:: yaml

            cluster:
              num_nodes: 2
              component_placement:
                actor:   { node_group: "gpu",   placement: 0 }
                rollout: { node_group: "gpu",   placement: 0 }
                env:     { node_group: dosw1,   placement: 0 }
              node_groups:
                - label: "gpu"
                  node_ranks: 0
                - label: dosw1
                  node_ranks: 1
                  hardware:
                    type: DOSW1
                    configs:
                      - robot_url: "192.168.1.20"
                        left_arm_port: 50051
                        right_arm_port: 50053
                        left_lead_port: 50050
                        right_lead_port: 50052
                        camera_serials:
                          - "SERIAL_1"
                          - "SERIAL_2"
                          - "SERIAL_3"
                        node_rank: 1

            env:
              train:
                keyboard_intervention_wrapper: True
                override_cfg:
                  is_dummy: False
                  use_dense_reward: True
                  target_grasp_joint: [...]
                  target_lift_joint:  [...]
                  max_joint_delta: 0.1
                  action_scale: 1.0
                  left_ee_pose_limit_min: [0.1, -0.35, 0.02]
                  left_ee_pose_limit_max: [0.4,  0.08, 0.40]
                  right_ee_pose_limit_min: [0.28, -0.01, 0.16]
                  right_ee_pose_limit_max: [0.30,  0.01, 0.17]
                  enable_human_in_loop: True

            actor:
              model:
                model_path: "./RLinf-ResNet10-pretrained"
                state_dim: 14        # 双臂：每臂 6 关节 + 1 夹爪
                action_dim: 14
                image_num: 3         # 1 仅用 cam_left；3 使用全部三路相机

      **4. 运行 Dummy 自检（可选）并启动训练**

      可先在单机运行纯状态 dummy 配置验证流程（自动启动 Ray），再从 GPU 节点启动双节点真机训练（日志保存在 ``logs/<timestamp>-<config>/``）：

      .. code-block:: bash

         # 可选：单机无硬件自检（无需相机与真机）
         PYTHONPATH=$PWD python examples/embodiment/train_embodied_agent.py \
             --config-path "$PWD/tests/e2e_tests/embodied" \
             --config-name dosw1_dummy_sac_mlp_pick \
             runner.max_epochs=1

         # 同步真机训练
         bash examples/embodiment/run_realworld.sh dosw1_pick_sac_flow

         # 异步真机训练
         bash examples/embodiment/run_realworld_async.sh dosw1_pick_sac_flow_async

结果与常见问题
--------------

在 Ray head 节点运行 ``tensorboard --logdir ./logs --port 6006``，主要观察 ``env/success_once``、``env/return``、``train/sac/*`` 与 ``train/replay_buffer/*``，完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 现象
     - 解决方法
   * - 机器人节点报 ``ImportError: airbot_sdk is not installed``
     - 导出指向实际 SDK 文件的 ``DOSW1_SDK_WHEEL`` 与 ``DOSW1_API_PATH`` 后重跑 ``install.sh``；仅验证管线时可设 ``is_dummy: true``。
   * - 5 秒内未读到 DOS-W1 状态（``TimeoutError``）
     - 确认 AirBot 服务已运行（``sh ~/dos_w1/airbot/whole_start.sh``），核对 ``robot_url`` 与 50050–50053 端口，并检查 GPU 节点到机器人的 TCP 连通性。
   * - ``RuntimeError: DOSW1Connection is not connected``
     - 回看前文连接日志中的具体报错原因。
   * - 相机不出帧
     - 在机器人节点运行 ``rs-enumerate-devices`` 核对 ``camera_serials`` 并检查 USB 接线；无显示器服务器可设 ``override_cfg.enable_camera_player: false`` 关闭预览窗。
   * - 奖励恒为 0 或卡在 ``reach``
     - 确认 ``is_dummy: false``，用 ``test_dosw1_controller.py`` 重新标定可达的 ``target_grasp_joint``，并将 ``joint_reward_sharpness`` 尝试降到 ``1.0``。
   * - ``Missing key runner`` 或 Hydra 报错
     - 从完整的 ``dosw1_pick_sac_flow.yaml``\ （或 ``dosw1_pick_sac_flow_async.yaml``）模板修改，不要拼接局部片段。
   * - 训练不稳定或发散
     - 调低 ``actor.optim.lr`` 与 ``actor.critic_optim.lr``\ （如 ``1e-4``），增大 ``algorithm.replay_buffer.min_buffer_size``，将 ``algorithm.gamma`` 尝试调至 ``0.8–0.9``，或通过 ``algorithm.demo_buffer.load_path`` 启用 RLPD。

如需在 DOS-W1 上开发自定义操作任务，参见 :doc:`新增真机任务 <../../extending/new_task>`。
