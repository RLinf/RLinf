Dexmal DOS-W1 真机强化学习
==========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 框架中，针对 **DOS-W1** 双臂机器人运行真机强化学习。
当前示例为 **单臂抓取** 任务（``DOSW1PickEnv-v1``）：左臂从 home 位出发，
先到达目标抓取关节位，闭合夹爪抓住物体，再到达目标抬起关节位；右臂在整个
过程中保持 home 位。

该示例使用 **Flow Matching 策略 + ResNet-10 视觉编码器**，通过 **SAC**
进行训练，整体流程与 :doc:`franka` / :doc:`xsquare_turtle2` 的真机 RL 管线一致。

环境
-----

**DOS-W1 Pick**

- **机器人**：DOS-W1 双臂机器人（基于 AirBot），配有主动臂用于遥操作。
- **任务**：单臂抓取，在关节空间分三阶段执行：

  1. **Reach**：左臂从 home 位移动到 ``target_grasp_joint``。
  2. **Grasp**：夹爪闭合（宽度 ≤ ``gripper_closed_max_width``）。
  3. **Lift**：抓住物体后，继续移动到 ``target_lift_joint``。

- **观测（Observation）**：

  - 最多 3 路 RGB 图像（128 × 128）：``cam_front``、``cam_left``、``cam_right``
    （由 Intel RealSense 相机采集）。
  - 本体状态：左右臂各 6 维关节角，以及左右夹爪宽度。

- **动作空间（Action Space）**：14 维连续动作，每只手臂 6 维关节目标
  （弧度）+ 1 维夹爪宽度（米），排列顺序为
  ``[left_joint(6), left_gripper(1), right_joint(6), right_gripper(1)]``。

**数据结构**

- **Images**：``frames/{cam_front, cam_left, cam_right}``，uint8
  ``[H, W, 3] = [128, 128, 3]``。
- **State**：``state/{left_joint_positions(6), left_gripper(1),
  right_joint_positions(6), right_gripper(1)}``。
- **Actions**：14 维浮点，关节空间 + 夹爪宽度。
- **Rewards**：默认使用密集奖励：

  - ``reach`` 阶段：``exp(-sharpness * ||q - q_grasp||^2)``
  - 成功抓取时额外加 ``grasp_bonus``（默认 ``0.3``）
  - ``lift`` 阶段：``exp(-sharpness * ||q - q_lift||^2)``
  - 当 ``||q - q_lift|| <= lift_threshold`` 时，reward 置为 ``1.0``，
    episode ``terminated = True``。
  - 额外惩罚右臂偏离 home 位的距离（保持非活动臂不动）。

算法
-----

**核心算法组件**

1. **SAC (Soft Actor-Critic)** + 自适应熵调节（``loss_type:
   embodied_sac``、``adv_type: embodied_sac``），与 :doc:`franka` /
   :doc:`sac_flow` 共用同一套 embodied RL 栈。
2. **Flow Matching 策略**\ （``flow_policy``）。一个轻量的去噪 Transformer
   （4 步去噪、2 层、256 隐藏维），以视觉 + 本体特征为条件，输出关节空间动作。
3. **ResNet-10 视觉编码器**\ （共享 backbone）。预训练权重来自
   ``RLinf/RLinf-ResNet10-pretrained``。
4. **RLPD（可选）**。若有遥操示教数据，可把
   ``algorithm.demo_buffer.load_path`` 指向 demo 目录，用先验数据做
   warm start。

如果只想做 **state-only MLP 烟雾测试**\ （无相机，用于 CI），请参考
``tests/e2e_tests/embodied/dosw1_dummy_sac_mlp_pick.yaml``。

硬件环境搭建
-------------

- **机器人**：DOS-W1 双臂机器人（含主动臂）。
- **相机**：最多 3 路 Intel RealSense 相机，序列号填入
  ``cluster.node_groups[<dosw1>].hardware.configs[*].camera_serials``。
  若只用 1 路相机，可将 ``image_num: 1``，并只填一个序列号。
- **训练 / Rollout 节点**：至少 1 块 CUDA GPU（建议 RTX 4090 或更高）。
- **机器人控制节点**：DOS-W1 自身（或与之在同一局域网的小型计算机），
  运行 AirBot 的 gRPC 服务。

.. warning::

   训练节点与 DOS-W1 必须在 **同一局域网** 内互通。AirBot gRPC 默认端口：

   - ``left_arm_port = 50051``\ （左从动臂）
   - ``left_lead_port = 50050``\ （左主动臂）
   - ``right_arm_port = 50053``\ （右从动臂）
   - ``right_lead_port = 50052``\ （右主动臂）

依赖安装
---------

机器人节点与训练 / Rollout 节点用同一条命令安装依赖，但机器人端额外需要
官方 **AirBot SDK**\ （``airbot_py`` wheel + ``airbot_api`` 源码）；GPU 端
只通过 gRPC 与机器人通信，不需要 SDK。

机器人节点
~~~~~~~~~~

AirBot SDK 通常已在 DOS-W1 机器上预部署，以下命令可自动检测并安装。

a. 克隆 RLinf 仓库
^^^^^^^^^^^^^^^^^^

.. code:: bash

   # 中国大陆用户可用下面这条加速下载：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. 安装依赖
^^^^^^^^^^^

.. code:: bash

   # 中国大陆用户可加 --use-mirror 提高下载速度
   bash requirements/install.sh embodied --env dosw1
   source .venv/bin/activate

安装脚本默认从以下路径读取 AirBot SDK：

- ``~/dos_w1/airbot/5.1.6/airbot_py-5.1.6-py3-none-any.whl``
- ``~/dos_w1/airbot/airbot_api``\ （以 editable 模式安装）

如果 SDK 文件不在默认路径，通过环境变量覆盖：

.. code:: bash

   export DOSW1_SDK_WHEEL=/path/to/airbot_py-5.1.6-py3-none-any.whl
   export DOSW1_API_PATH=/path/to/airbot_api
   bash requirements/install.sh embodied --env dosw1

训练 / Rollout 节点
~~~~~~~~~~~~~~~~~~~

a. 克隆 RLinf 仓库
^^^^^^^^^^^^^^^^^^

.. code:: bash

   # 中国大陆用户可用下面这条加速下载：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. 安装依赖
^^^^^^^^^^^

.. code:: bash

   # 中国大陆用户可加 --use-mirror 提高下载速度
   bash requirements/install.sh embodied --env dosw1
   source .venv/bin/activate

GPU 节点通常没有 ``~/dos_w1/airbot`` 目录，安装脚本会打印 warning 并
**跳过** AirBot SDK 安装 —— 这是预期行为，其余依赖（``embodied`` extra、
``evdev``、``opencv-python``、RLinf 本身）都会正常安装。

.. note::

   **不要** 只执行 ``uv pip install -e .``，这样不会安装 ``embodied``
   extra，会缺少环境侧依赖（``evdev``、``opencv-python``、RealSense
   Python 绑定等）。

模型下载
---------

Flow Matching 策略使用的 ResNet-10 预训练权重（配置里的
``actor.model.encoder_config.ckpt_name: resnet10_pretrained.pt``）：

.. code:: bash

   # 方式 1：git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # 方式 2：huggingface-hub（中国大陆：export HF_ENDPOINT=https://hf-mirror.com）
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

下载完成后，把 YAML 中的 ``actor.model.model_path`` 与
``rollout.model.model_path`` 都指向下载目录。

运行实验
---------

目标关节标定
~~~~~~~~~~~~

``target_grasp_joint`` 与 ``target_lift_joint`` 必须反映你工作台的实际
布局。推荐的标定方式是：用主动臂把从动臂遥操到目标位姿，然后读取当前关节角。

1. 在机器人节点启动从动臂服务（例如
   ``sh ~/dos_w1/airbot/whole_start.sh``），确认弹出的 tmux 各 panel
   没有报错。

2. 激活虚拟环境，运行遥操检查脚本：

.. code-block:: bash

   source .venv/bin/activate

   python toolkits/realworld_check/test_dosw1_controller.py \
       --robot-url <ROBOT_IP> \
       --print-hz 5

3. 用左主动臂把左从动臂拖到期望的 **抓取** 位。终端会持续打印类似
   下面的内容::

     [1713600000.000] left_joint=[-0.4725 -1.1332  0.6510  1.4082 -0.5987 -1.0904  0.0700]
     [1713600000.000] left_eef=[...] right_eef=[...]

   ``left_joint`` 共 **7** 个值：前 6 个是关节角（弧度），最后 1 个是
   夹爪宽度。将 **前 6 个** 值填入 ``target_grasp_joint``。

4. 继续把主动臂抬到安全的 lift 位，读取新的 ``left_joint[:6]`` 并填入
   ``target_lift_joint``。

.. tip::

   ``left_eef`` / ``right_eef`` 可以辅助确认末端是否到达预期位置，但
   ``target_grasp_joint`` / ``target_lift_joint`` 需要的是 **关节角**，
   不是 ee pose。

Ray 集群启动
~~~~~~~~~~~~

真机训练采用双节点 Ray 集群：GPU 节点跑 ``actor`` / ``rollout``，机器人
节点跑 ``env``。

1. 在 **每一台** 节点上先 ``source`` 虚拟环境、再 ``export RLINF_NODE_RANK``
   才能启动 Ray。Ray 会在 ``ray start`` 时锁定当前的 Python 解释器与
   环境变量，之后再设置的变量对 Ray worker 不可见。

2. 启动 Ray：

.. code-block:: bash

   # GPU 节点（node rank 0，Ray head）
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<GPU_SERVER_IP>

   # 机器人节点（node rank 1，Ray worker）
   export RLINF_NODE_RANK=1
   ray start --address=<GPU_SERVER_IP>:6379

3. 用 ``ray status`` 确认两个节点都已加入。

配置文件
~~~~~~~~

直接基于 ``examples/embodiment/config/dosw1_pick_sac_flow.yaml`` 模板修改；
带权重同步解耦的异步版本为 ``dosw1_pick_sac_flow_async.yaml``。

常见需要改动的字段：

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
         # 机器人连接信息（gRPC URL / 端口 / 相机序列号）由 scheduler 管理，
         # 不再写在 env 配置里。
         hardware:
           type: DOSW1
           configs:
             - robot_url: "<ROBOT_IP>"            # DOS-W1 gRPC 地址
               left_arm_port: 50051
               right_arm_port: 50053
               left_lead_port: 50050
               right_lead_port: 50052
               camera_serials:                    # RealSense 相机序列号
                 - "<SERIAL_1>"
                 - "<SERIAL_2>"
                 - "<SERIAL_3>"
               node_rank: 1

   env:
     train:
       keyboard_intervention_wrapper: True
       override_cfg:
         is_dummy: False
         use_dense_reward: True
         target_grasp_joint: [...]                # 来自上文标定
         target_lift_joint:  [...]                # 来自上文标定
         max_joint_delta: 0.1                     # 每步最大关节变化（弧度，约 5.7°）
         action_scale: 1.0
         # 末端安全盒（xyz 单位米，含端点）
         left_ee_pose_limit_min: [0.1, -0.35, 0.02]
         left_ee_pose_limit_max: [0.4,  0.08, 0.40]
         right_ee_pose_limit_min: [0.28, -0.01, 0.16]
         right_ee_pose_limit_max: [0.30,  0.01, 0.17]
         enable_human_in_loop: True

   actor:
     model:
       model_path: "/path/to/RLinf-ResNet10-pretrained"
       state_dim: 14        # 双臂：每只手臂 6 关节 + 1 夹爪
       action_dim: 14
       image_num: 3         # 1 = 仅 cam_left；3 = 三路相机全用
   rollout:
     model:
       model_path: "/path/to/RLinf-ResNet10-pretrained"

.. warning::

   ``cluster.num_nodes`` 必须与实际节点数一致，每个 ``node_ranks``
   必须等于该机器上的 ``RLINF_NODE_RANK``。不要从 diff 片段里手拼
   一份"精简"配置 —— 真机训练请始终以
   ``dosw1_pick_sac_flow.yaml`` 完整模板为起点进行修改。

键盘干预
~~~~~~~~

DOS-W1 的 episode 进度由机器人节点上的键盘监听器控制。只要在 YAML 中
保持下面两项打开即可（提供的模板默认已打开）：

.. code-block:: yaml

   env:
     train:
       keyboard_intervention_wrapper: True
       override_cfg:
         enable_human_in_loop: True

训练开始前，需要授予当前用户访问 ``/dev/input`` 设备的权限
（关于 ``RLINF_KEYBOARD_DEVICE`` 的详细配置请参阅 :doc:`franka`
的 *无显示器键盘奖励包装器* 章节）：

.. code-block:: bash

   sudo usermod -aG input $USER
   # 退出当前登录并重新登录使组权限生效

运行时支持的按键：

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - 按键
     - 功能
   * - ``s``
     - 在 free-teleop / reset 阶段请求开始新 episode。
   * - ``r``
     - 中止当前 episode，回到 free-teleop（不保存）。
   * - ``d``
     - 将当前 episode 标记为 "manual done" 并保存。
   * - ``p``
     - 从 ``MODEL`` 或 ``TELEOP`` 切到 ``PAUSE``。
   * - ``t``
     - 从 ``PAUSE`` 切到 ``TELEOP``\ （策略让位于主动臂）。
   * - ``m``
     - 从 ``PAUSE`` 切回 ``MODEL``\ （由策略驱动机械臂）。

当 ``manual_episode_control_only`` 打开时，``p`` / ``t`` / ``m`` 被忽略，
系统保持在主动臂遥操模式下，仅 ``s`` / ``r`` / ``d`` 生效。

数据采集（可选，用于 RLPD）
~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果希望用遥操示教数据 warm start 训练，可以用下面的脚本采集。
``dosw1_collect_data.yaml`` 已经打开了 ``enable_human_in_loop`` 和
键盘包装器，且只用 **单节点** 运行（``cluster.num_nodes: 1``，直接跑在
机器人节点上）。

在机器人节点执行：

.. code-block:: bash

   source .venv/bin/activate

   bash examples/embodiment/collect_data.sh dosw1_collect_data

执行前请先编辑 ``examples/embodiment/config/dosw1_collect_data.yaml``
里 ``cluster.node_groups[<dosw1>].hardware.configs`` 下的
``robot_url``、四个 gRPC 端口以及 ``camera_serials``。

典型的采集流程：

1. 环境进入 free-teleop，用主动臂把从动臂摆到初始姿态。
2. 按 ``s`` 开始一个 episode。
3. 通过主动臂完成抓取与抬起。
4. 成功后按 ``d`` 保存；如果这条不要了，按 ``r`` 放弃并重来。
5. 达到 ``runner.num_data_episodes``（默认 ``20``）条后脚本自动退出。

成功的轨迹会保存在 ``<log_path>/demos/``。把它填回训练配置：

.. code-block:: yaml

   algorithm:
     demo_buffer:
       load_path: "/path/to/logs/dosw1-collect/<timestamp>/demos"

若不启用 RLPD，直接去掉 ``demo_buffer`` 段即可。

Dummy 自检（可选）
~~~~~~~~~~~~~~~~~~

接真机之前，推荐先跑一遍 CI 中使用的最小 dummy 配置 —— 它设置
``is_dummy: true``\ （完全不调用硬件）且用 state-only MLP，可以单机、无
相机运行，用于验证配置树与集群管线：

.. code-block:: bash

   export REPO_PATH=$(pwd)
   ray start --head

   python examples/embodiment/train_embodied_agent.py \
       --config-path $REPO_PATH/tests/e2e_tests/embodied/ \
       --config-name dosw1_dummy_sac_mlp_pick \
       runner.max_epochs=1

这一步 **不是** 真机训练 recipe —— 图像相关路径被禁用，只用于流程自检。

启动训练
~~~~~~~~

标定、demo、YAML 都就位后，在 GPU 节点发起训练：

.. code-block:: bash

   # 同步管线
   bash examples/embodiment/run_realworld.sh dosw1_pick_sac_flow

   # 异步管线（rollout / learner 权重同步解耦）
   bash examples/embodiment/run_realworld_async.sh dosw1_pick_sac_flow_async

日志写到 ``logs/<timestamp>-<config>/`` 下。

关键安全机制
-------------

``DOSW1Env._execute_model_action`` 在把命令发到 SDK 之前，会做三层防护：

1. **单步关节钳位**：目标关节会被 clip 到
   ``current ± max_joint_delta``（默认 ``0.1`` 弧度，约 5.7°/步）。
2. **绝对关节限幅**：再 clip 到
   ``[joint_limit_min, joint_limit_max]``（默认 ``±π``，可根据实际工况收紧）。
3. **末端安全盒**：沿着"当前关节 → 目标关节"在关节空间做二分搜索，找到
   经正运动学换算后仍落在
   ``left_ee_pose_limit_min/max`` / ``right_ee_pose_limit_min/max``
   之内的最大插值比例；超出安全盒的动作会自动被截断。

这三层在 TELEOP 模式下同样生效，因此即便操作者用主动臂做剧烈运动，从动臂
也不会越过安全盒。

可视化与结果
------------

**TensorBoard**

在 Ray head 节点上：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**关键指标**

- **环境**：``env/episode_len``、``env/return``、``env/reward``、
  ``env/success_once``\ （未归一化的 episode 成功率，主要监控指标）。
- **训练（SAC）**：``train/sac/critic_loss``、``train/sac/actor_loss``、
  ``train/sac/alpha_loss``、``train/sac/alpha``、``train/actor/entropy``、
  ``train/actor/grad_norm``、``train/critic/grad_norm``。
- **Replay buffer**：``train/replay_buffer/size``、
  ``train/replay_buffer/{mean,max,min,std}_reward``、
  ``train/replay_buffer/utilization``。

常见问题
---------

**``ImportError: airbot_sdk is not installed``（机器人节点）**
  默认路径下找不到 AirBot wheel。用 ``DOSW1_SDK_WHEEL`` /
  ``DOSW1_API_PATH`` 指向实际 SDK 文件后重跑 ``install.sh``。
  如果只是想做流程自检，改用 ``env.train.override_cfg.is_dummy: true``。

**``TimeoutError: Timed out waiting for DOSW1 state from AirbotRobot``**
  5 秒内没有从 gRPC 读到状态。排查：

  - 机器人是否上电，从动/主动臂服务是否正常（
    ``sh ~/dos_w1/airbot/whole_start.sh``）。
  - ``cluster.node_groups[<dosw1>].hardware.configs`` 下的 ``robot_url``
    与四个端口是否与机器配置一致。
  - GPU 节点能否 ``ping`` 到机器人，并能在 50050–50053 建立 TCP 连接。

**``RuntimeError: DOSW1SDKAdapter is not connected``**
  connect 没有成功执行。回看此前日志中
  ``[DOSW1SDK] Connecting via AirbotRobot`` 一行的错误原因。

**相机问题 / ``Camera ... is not producing frames``**
  在机器人节点执行 ``rs-enumerate-devices``，确认
  ``cluster.node_groups[<dosw1>].hardware.configs[*].camera_serials``
  中每个序列号都能被识别，USB 线牢固。
  若跑在无显示器服务器上，可设
  ``override_cfg.enable_camera_player: false`` 关闭预览窗口（不影响训练）。

**reward 恒为 0 / phase 卡在 ``reach``**
  常见原因：

  - 当前配置中 ``is_dummy`` 还是 ``true``。
  - ``target_grasp_joint`` 不可达 —— 重新用
    ``test_dosw1_controller.py`` 标定，确认机械臂能物理到达。
  - ``joint_reward_sharpness`` 过大（密集奖励在远离目标时迅速衰减到 0），
    可尝试降到 ``1.0``。

**``Missing key runner`` / Hydra 报错**
  配置不是从完整模板来的，而是拼接片段。真机训练请始终以
  ``examples/embodiment/config/dosw1_pick_sac_flow.yaml``
  （或 ``dosw1_pick_sac_flow_async.yaml``）为起点覆盖字段；快速自检可用
  ``tests/e2e_tests/embodied/dosw1_dummy_sac_mlp_pick.yaml``。

**训练不稳定 / 发散**
  几个常用的 SAC 调节旋钮：

  - 降低 ``actor.optim.lr`` 与 ``actor.critic_optim.lr``（如 ``1e-4``）。
  - 提高 ``algorithm.replay_buffer.min_buffer_size``，让 buffer
    积累更多数据再开始更新。
  - 若任务 horizon 较短，可适当调低 ``algorithm.gamma``
    （pick 任务通常 ``0.8 – 0.9`` 是不错的起点）。
  - 采一小批遥操示教数据，通过 ``algorithm.demo_buffer.load_path``
    启用 RLPD。
