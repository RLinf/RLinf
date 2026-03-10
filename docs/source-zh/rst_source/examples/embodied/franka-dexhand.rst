Franka + 灵巧手真机强化学习
============================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 框架中为 Franka 机械臂配置 **灵巧手末端执行器**
（傲意手、睿研手），使用 **数据手套 + 空间鼠标** 进行遥操作数据采集和人类干预训练，
以及如何通过 **视觉奖励分类器** 为灵巧手任务提供自动化的成功/失败判定。

如果你还没有阅读基础的 Franka 真机环境搭建指南，请先参考 :doc:`franka`。

.. contents:: 目录
   :local:
   :depth: 2

总览
-----------

在默认的 Franka 真机场景中，末端执行器是 Franka 平行夹爪，动作空间为 7 维
（6 维臂 + 1 维夹爪）。灵巧手集成后，动作空间扩展为 **12 维**
（6 维臂 + 6 维手指），使 Franka 能够完成更复杂的灵巧操作任务。

**主要功能：**

1. **末端执行器抽象层** — 统一的 ``EndEffector`` 接口，支持在 Franka 夹爪、
   傲意灵巧手、睿研灵巧手之间通过配置文件一键切换。
2. **数据手套遥操作** — ``GloveExpert`` 读取 PSI 数据手套的 6 维手指角度，
   与 ``SpaceMouseExpert`` 组合形成 12 维人类专家动作。
3. **灵巧手干预包装器** — ``DexHandIntervention`` 自动替换
   ``SpacemouseIntervention``，在人类干预时提供完整的 12 维专家动作。
4. **视觉奖励分类器** — 对于灵巧手任务，单纯依靠末端位置难以判定成功/失败，
   因此提供了基于 ResNet-10 的视觉二分类器，从相机图像自动判断任务是否完成。

环境
-----------

- **Task**: 灵巧手操作任务（如抓取、精细装配等）
- **Observation**: 腕部或第三人称相机的 RGB 图像（128×128）+ 灵巧手 6 维状态
- **Action Space**: 12 维连续动作：

  - 三维位置控制（x, y, z）
  - 三维旋转控制（roll, pitch, yaw）
  - 六维手指控制（拇指旋转、拇指弯曲、食指、中指、无名指、小指），归一化 ``[0, 1]``

算法
-----------

灵巧手场景沿用与 Franka 夹爪相同的算法组件（SAC / Cross-Q / RLPD），
区别在于策略网络输出 12 维连续动作，以及可使用视觉分类器提供奖励信号。
具体算法介绍请参考 :doc:`franka`。


硬件环境搭建
----------------

除 :doc:`franka` 中列出的标准硬件外，灵巧手场景还需要以下额外组件：

- **灵巧手** — 傲意灵巧手（Modbus RTU 串口）或 睿研灵巧手（自定义串口协议）
- **数据手套** — PSI 数据手套，USB 串口连接（通常挂载为 ``/dev/ttyACM0``）

控制节点硬件连接
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在控制节点上需要连接以下硬件：

1. **Franka 机械臂** — 通过以太网连接
2. **灵巧手** — 通过 USB 串口连接（傲意：Modbus，睿研：自定义协议）
3. **空间鼠标（SpaceMouse）** — USB 连接
4. **数据手套** — USB 串口连接
5. **Realsense 相机** — USB 连接

**串口权限设置：**

.. code-block:: bash

   # 将用户添加到 dialout 组以获取串口权限
   sudo usermod -a -G dialout $USER
   # 重新登录后生效

   # 或者临时修改设备权限
   sudo chmod 666 /dev/ttyUSB0  # 灵巧手串口
   sudo chmod 666 /dev/ttyACM0  # 数据手套串口

**检查设备连接：**

.. code-block:: bash

   # 检查串口设备
   ls -la /dev/ttyUSB* /dev/ttyACM*

   # 检查 SpaceMouse（HID 设备）
   lsusb | grep -i 3dconnexion


依赖安装
-------------------------

灵巧手场景的依赖安装基于 :doc:`franka` 中的标准安装流程，
需要额外安装灵巧手和数据手套的驱动包。

控制节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

按照 :doc:`franka` 中的步骤完成基础依赖安装后，在控制节点的虚拟环境中执行：

.. code-block:: bash

   # 灵巧手 + 数据手套驱动（包含串口通信等全部依赖）
   pip install "rlinf_dexhand[all]"

``rlinf_dexhand`` 包含睿研灵巧手、傲意灵巧手和 PSI 数据手套的驱动，
以及所需的串口通信库（pyserial、pymodbus、pyyaml 等）。
如果只需要部分组件，可以使用更细粒度的可选依赖：

- ``pip install rlinf_dexhand`` — 基础（仅 pyserial + numpy）
- ``pip install "rlinf_dexhand[glove]"`` — 加装数据手套依赖（pyyaml）
- ``pip install "rlinf_dexhand[aoyi]"`` — 加装傲意灵巧手依赖（pymodbus）
- ``pip install "rlinf_dexhand[all]"`` — 全部依赖

训练 / Rollout 节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

与 :doc:`franka` 相同，不需要额外安装。


模型下载
---------------

灵巧手场景使用与 :doc:`franka` 相同的 ResNet-10 预训练 backbone 作为策略网络的视觉编码器：

.. code-block:: bash

   # 下载模型（两种方式二选一）
   # 方式 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # 方式 2：使用 huggingface-hub
   # 为了提高国内下载速度，可以添加以下环境变量：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

下载完成后，请在对应的配置 YAML 文件中正确填写模型路径。

.. note::

   灵巧手任务的预训练模型尚在训练中，后续将在 |huggingface| `HuggingFace <https://huggingface.co/RLinf>`_ 上发布。
   目前可以使用上述 ResNet-10 backbone 从零开始训练。


运行实验
-----------------------

前置准备
~~~~~~~~~~~~~~~

**1. 获取目标位姿**

使用诊断工具获取目标末端位姿和灵巧手状态。

设置环境变量后运行诊断脚本：

.. code-block:: bash

   export FRANKA_ROBOT_IP=<your_robot_ip>
   export FRANKA_END_EFFECTOR_TYPE=ruiyan_hand  # 或 aoyi_hand
   export FRANKA_HAND_PORT=/dev/ttyUSB0
   python -m toolkits.realworld_check.test_franka_controller

在交互界面中：

- 输入 ``getpos_euler`` 获取当前末端执行器位姿（欧拉角形式）
- 输入 ``gethand`` 查看灵巧手当前手指位置
- 输入 ``handinfo`` 确认灵巧手连接状态
- 输入 ``help`` 查看所有可用命令

**2. 测试硬件连接**

.. code-block:: bash

   # 测试相机
   python -m toolkits.realworld_check.test_franka_camera

数据采集
~~~~~~~~~~~~~~~~~

灵巧手场景的数据采集使用空间鼠标控制机械臂、数据手套控制手指，
``DexHandIntervention`` 会自动将两者输入合并为 12 维动作。

**遥操作控制方式：**

.. list-table::
   :widths: 18 30 52
   :header-rows: 1

   * - 输入设备
     - 按键 / 操作
     - 效果
   * - SpaceMouse
     - 6D 摇杆
     - 控制机械臂位姿增量（x, y, z, roll, pitch, yaw）
   * - SpaceMouse
     - **物理左键** （按住）
     - 激活相对手套控制：按下瞬间锁定手套基准，之后只应用增量到手指
   * - SpaceMouse
     - **物理左键** （松开）
     - 手指冻结在当前位置
   * - 数据手套
     - 手指弯曲
     - （仅在左键按住时生效）手指角度增量叠加到手的当前位姿

.. note::

   数据手套工作在 **相对控制模式** ：只有在按住 SpaceMouse 左键时才会应用手套读数的
   **变化量** （delta），松开后手指保持不动。这避免了绝对模式下手指突然跳跃的问题，
   适合需要精细控制的操作。

1. 激活虚拟环境：

.. code-block:: bash

   source /opt/venv/franka-0.15.0/bin/activate
   # 如有 ROS 依赖：source <your_catkin_ws>/devel/setup.bash

2. 修改配置文件 ``examples/embodiment/config/realworld_collect_data.yaml``：

.. code-block:: yaml

   # examples/embodiment/config/realworld_collect_data.yaml
   defaults:
     - env/realworld_dex_pnp@env.eval      # 引用 env 子配置
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 1
     component_placement:
       env:
         node_group: franka
         placement: 0
     node_groups:
       - label: franka
         node_ranks: 0
         hardware:
           type: Franka
           configs:
             - robot_ip: 172.16.0.2          # ← 替换为你的机器人 IP
               node_rank: 0

   runner:
     task_type: embodied
     num_data_episodes: 20

   env:
     group_name: "EnvGroup"
     eval:
       ignore_terminations: False            # 普通数据采集为 False
       auto_reset: False                     # 手动控制 reset
       use_spacemouse: True
       glove_config:
         left_port: "/dev/ttyACM0"           # 数据手套串口
         frequency: 30                       # 手套轮询频率 (Hz)
       override_cfg:
         target_ee_pose: [0.8188, 0.1384, 0.1188, -3.1331, -1.1213, -0.0676]
         end_effector_type: "ruiyan_hand"    # 或 aoyi_hand
         end_effector_config:
           port: "/dev/ttyUSB0"              # 灵巧手串口
           baudrate: 460800
           motor_ids: [1, 2, 3, 4, 5, 6]
           default_velocity: 2000
           default_current: 800
           default_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
         hand_target_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
         hand_reset_state: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]  # reset 时手指位姿
         hand_action_scale: 1.0
         joint_reset_qpos: [0, 0, 0, -1.9, 0, 2, 0]

其中各字段含义：

- ``target_ee_pose`` — 机械臂目标 TCP 位姿（通过 ``test_controller`` 的 ``getpos_euler`` 获取）
- ``end_effector_type`` — 灵巧手类型，``ruiyan_hand`` 或 ``aoyi_hand``
- ``hand_reset_state`` — 每个 episode 开始时手指复位到的位姿（6 维，``[0, 1]``）
- ``hand_target_state`` — 奖励计算用的目标手指位姿
- ``default_velocity`` / ``default_current`` — 灵巧手电机速度和电流限制

3. 运行数据采集脚本：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

在采集过程中，使用空间鼠标控制机械臂移动，按住左键后通过数据手套控制灵巧手手指动作。
采集到的数据会保存在 ``logs/[running-timestamp]/demos/`` 路径下。

.. tip::

   如果你的任务需要视觉分类器来判定成功/失败（推荐用于灵巧手任务），
   请跳过此节，按照 `视觉奖励分类器`_ 章节的完整 workflow 操作：

   1. 采集分类器训练数据 → 2. 训练分类器 → 3. 用分类器采集 demo → 4. 训练 RL

集群配置
~~~~~~~~~~~~~~~~~

集群配置步骤与 :doc:`franka` 完全一致。
在每个节点上运行 ``ray start`` 之前，确保已正确设置环境变量（参考 ``ray_utils/realworld/setup_before_ray.sh``）。

配置文件
~~~~~~~~~~~~~~~~~~~~~~

配置文件通常由两部分组成：一个主配置 YAML 和一个 ``env/`` 子配置 YAML。

**env 子配置** ``examples/embodiment/config/env/realworld_dex_pnp.yaml``：

.. code-block:: yaml

   env_type: realworld
   auto_reset: True
   ignore_terminations: True
   max_episode_steps: 100          # 每 episode 最大步数
   use_spacemouse: True
   main_image_key: wrist_1         # 主相机观测 key

   init_params:
     id: "DexpnpEnv-v1"

**睿研手完整配置示例** （主配置 ``override_cfg`` 部分）：

.. code-block:: yaml

   override_cfg:
     end_effector_type: "ruiyan_hand"
     end_effector_config:
       port: "/dev/ttyUSB0"
       baudrate: 460800
       motor_ids: [1, 2, 3, 4, 5, 6]
       default_velocity: 2000          # 电机速度（越大越快）
       default_current: 800            # 电机电流限制
       default_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_reset_state: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_target_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_action_scale: 1.0
     target_ee_pose: [0.8188, 0.1384, 0.1188, -3.1331, -1.1213, -0.0676]
     joint_reset_qpos: [0, 0, 0, -1.9, 0, 2, 0]

**傲意手完整配置示例：**

.. code-block:: yaml

   override_cfg:
     end_effector_type: "aoyi_hand"
     end_effector_config:
       port: "/dev/ttyUSB0"
       node_id: 2
       baudrate: 115200
     hand_reset_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_target_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_action_scale: 1.0

**手套配置** （在 ``env.eval`` 或 ``env.train`` 中）：

.. code-block:: yaml

   use_spacemouse: True
   glove_config:
     left_port: "/dev/ttyACM0"        # 左手手套串口
     frequency: 30                     # 轮询频率 (Hz)

同时，将 ``rollout`` 和 ``actor`` 部分的 ``model_path`` 字段设置为前面下载的预训练模型路径。

运行实验
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 head 节点上启动实验（灵巧手 dex_pnp 任务为例）：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async

该配置已包含分类器奖励和 demo 路径。完整配置参考
``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml``。


视觉奖励分类器
-----------------------

灵巧手任务中，末端执行器的位姿不足以判定任务是否成功
（例如抓取任务中物体是否被稳定握持），因此需要一个视觉分类器来自动判定奖励。

概述
~~~~~~~~~~~~~~~

视觉奖励分类器使用冻结的 ResNet-10 backbone 提取图像特征，
通过 ``SpatialLearnedEmbeddings`` 进行空间池化，最终经过一个二分类头输出
成功概率。

**硬件拓扑**

在典型的灵巧手场景中，有两个节点（可以是两台物理机，也可以是同一台机器上的两个 Docker 容器）：

- **训练 / GPU 节点** （node_rank=0）— 装有 GPU，运行 actor、rollout，以及 **奖励分类器推理**
- **控制节点** （node_rank=1）— 连接 Franka 机械臂和灵巧手，运行 env worker，**无 GPU**

分类器推理通过 ``ClassifierRewardServer`` （一个 Ray actor）运行在 GPU 节点上，
控制节点的 env worker 通过 Ray 远程调用获取分类结果。这样既利用了 GPU 的推理速度，
又避免了在无 GPU 的控制节点上加载 CUDA 模型的问题。

.. note::

   如果你的控制节点有 GPU（或希望在单节点上运行），也可以在配置中设置
   ``classifier_reward_wrapper.remote: false`` 并指定 ``device: cuda``，
   分类器将直接在 env worker 进程内加载推理。

**完整 workflow（按顺序执行）：**

1. **采集分类器训练数据** — 在控制节点上单节点运行，遥操作机器人，
   用 SpaceMouse 右键实时标记成功/失败帧
2. **人工审核** — 在有显示器的节点上用 OpenCV 窗口逐帧审核标注
3. **训练分类器** — **在 GPU 节点上运行**，训练 ResNet-10 二分类器
4. **启动 Ray 集群** — 在两个节点上分别启动 Ray，组成集群
5. **采集 demo 数据** — 在 2 节点集群上运行，分类器在 GPU 节点推理，
   env worker 在控制节点遥操作
6. **训练 RL** — 在 2 节点集群上运行，指定 demo 路径和分类器 ckpt 路径

步骤 1：采集分类器训练数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   此步骤**只需要控制节点**（连接机器人的节点），不需要 GPU 节点参与。
   在控制节点上单独启动 Ray 即可。

分类器数据采集使用与普通数据采集相同的遥操作环境（SpaceMouse + 数据手套 + Franka），
但通过 SpaceMouse 的 **物理右键** 来实时标记帧类别：

- **按住右键** → 当前帧标记为 **成功** （success）
- **松开右键** → 当前帧标记为 **失败** （failure，默认）

.. important::

   采集分类器数据时，**不能** 依靠末端位姿接近 ``target_pos`` 来判定成功/失败——
   因为视觉分类器的目的恰恰是替代位姿判定。因此脚本会自动设置
   ``ignore_terminations=True``，使每个 episode **始终运行到** ``max_episode_steps``
   才复位，操作者通过右键手动标注哪些帧属于成功状态。

**在控制节点上操作：**

.. code-block:: bash

   # 1. 激活虚拟环境
   source /opt/venv/franka-0.15.0/bin/activate
   # 如有 ROS 依赖：source <your_catkin_ws>/devel/setup.bash

   # 2. 在控制节点上单独启动 Ray（单节点模式）
   ray start --head --port=6379

   # 3. 运行分类器数据采集脚本
   bash examples/embodiment/collect_classifier_data.sh

   # 4. 采集完成后停止 Ray
   ray stop

数据保存在 ``logs/<时间戳>-reward-classifier-<env_name>/`` 目录下。

**采集流程：**

1. 脚本启动后，机器人复位到初始位姿，终端显示 Episode 进度条
2. 每个 Episode 开始时终端会打印醒目的提示信息，告知操作者可以开始遥操作
3. 操作者通过遥操作完成任务，在认为任务成功的时刻 **按住** SpaceMouse 右键
4. Episode 运行满 ``max_episode_steps`` 步后自动复位，开始下一个 Episode
5. 收集到足够的成功帧后（默认 200 帧），自动进入审核阶段

**终端输出示例：**

.. code-block:: text

   ##################################################
     Episode 3  成功: 42/200  失败: 158
     >>> 开始遥操作 — 右键标记成功帧 <<<
   ##################################################
   Ep3 [S:42/200 F:158]:  35%|████████          | 35/100

步骤 2：人工审核
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

数据采集完成后，脚本自动弹出 OpenCV 审核窗口。如果窗口未弹出（如在无显示器的
控制节点上），可以将数据目录拷贝到有显示器的节点上，手动运行审核脚本：

.. code-block:: bash

   bash examples/embodiment/review_classifier_data.sh \
       logs/<timestamp>-reward-classifier-dex_pnp

**审核控制键：**

.. list-table::
   :widths: 10 50
   :header-rows: 1

   * - 按键
     - 操作
   * - ``n``
     - 下一帧
   * - ``p``
     - 上一帧
   * - ``g``
     - 标记为 **保留** （good）
   * - ``b``
     - 标记为 **丢弃** （bad）
   * - ``1``
     - 只显示成功帧
   * - ``2``
     - 只显示失败帧
   * - ``0``
     - 显示全部
   * - ``s``
     - 保存
   * - ``q`` / ``ESC``
     - 完成审核并保存

审核完成后，最终数据以图像和 pickle 两种格式保存在同一目录下：

.. code-block:: text

   logs/<timestamp>-reward-classifier-dex_pnp/
   ├── raw_frames.pkl               # 原始采集数据
   ├── success/                     # 成功帧图像
   │   ├── 2026-03-03_04-41-36_00000.png
   │   └── ...
   ├── failure/                     # 失败帧图像
   │   ├── 2026-03-03_04-41-36_00000.png
   │   └── ...
   ├── success_202_2026-03-03_04-41-36.pkl   # 训练用 pickle
   └── failure_741_2026-03-03_04-41-36.pkl

步骤 3：训练分类器（在 GPU 节点上运行）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   分类器训练应在 **GPU 节点** 上运行以利用 GPU 加速。
   如果采集数据的控制节点和 GPU 节点是不同机器，需要先将数据目录拷贝到 GPU 节点上
   （如果两者通过共享卷挂载了同一目录，则不需要拷贝）。

**在 GPU 节点上操作：**

.. code-block:: bash

   # 确保已激活对应的虚拟环境(如果使用Docker，这一步可能不需要，因为环境已在容器内设置好)
   source /opt/venv/openvla/bin/activate

   # 训练分类器
   python examples/embodiment/train_reward_classifier.py \
       --log_dir logs/<timestamp>-reward-classifier-dex_pnp \
       --pretrained_ckpt RLinf-ResNet10-pretrained/resnet10_pretrained.pt \
       --image_keys global wrist_1 \
       --num_epochs 200 \
       --device cuda

脚本会自动检测 GPU 并使用 CUDA 加速训练。如果没有指定 ``--device``，
默认行为是有 GPU 时用 ``cuda``，无 GPU 时回退到 ``cpu``。

**训练参数：**

.. list-table::
   :widths: 22 12 50
   :header-rows: 1

   * - 参数
     - 默认值
     - 说明
   * - ``--log_dir``
     - （必须）
     - 数据目录（由 ``collect_classifier_data.sh`` 创建），同时也是模型保存目录
   * - ``--pretrained_ckpt``
     - ``RLinf-ResNet10-pretrained/resnet10_pretrained.pt``
     - ResNet-10 预训练权重路径
   * - ``--image_keys``
     - ``wrist_1``
     - 相机观测 key（与 env 配置中相机的 ``name`` 对应）
   * - ``--image_size``
     - ``128``
     - 输入图像尺寸
   * - ``--num_epochs``
     - ``200``
     - 训练轮数
   * - ``--batch_size``
     - ``64``
     - 批大小
   * - ``--lr``
     - ``1e-4``
     - 学习率
   * - ``--device``
     - 自动检测
     - 训练设备，建议显式设置 ``cuda``

训练过程中会打印每个 epoch 的 loss 和准确率，最佳模型保存为
``<log_dir>/reward_classifier.pt``。

.. code-block:: text

   Loading data from logs/...-reward-classifier-dex_pnp ...
     success: 202   failure: 741   total: 943
   Building classifier ...
   Training for 200 epochs ...
   Epoch  10/200  loss=0.4123  acc=0.8125
   ...
   Epoch 200/200  loss=0.0512  acc=0.9688

   Best accuracy: 0.9844
   Checkpoint saved to logs/...-reward-classifier-dex_pnp/reward_classifier.pt

步骤 4：启动 Ray 集群（2 节点）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从此步骤开始需要两个节点协作。请确保两个节点（或两个 Docker 容器）在同一网络下。

.. warning::

   这一步非常关键！在启动 Ray **之前** 必须在每个节点上正确设置环境变量，
   因为 Ray 会在启动时快照当前环境变量，之后 Ray 创建的所有进程都会继承该快照。

**在 GPU 节点（node_rank=0，head 节点）上：**

.. code-block:: bash

   # 激活虚拟环境
   source /opt/venv/openvla/bin/activate

   # 设置 RLinf 环境变量
   export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
   export RLINF_NODE_RANK=0

   # 启动 Ray head
   ray start --head --port=6380 --node-ip-address=<head_ip>

**在控制节点（node_rank=1，worker 节点）上：**

.. code-block:: bash

   # 激活虚拟环境（franka 环境）
   source /opt/venv/franka-0.15.0/bin/activate

   # 设置 RLinf 环境变量
   export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
   export RLINF_NODE_RANK=1

   # 加入 Ray 集群
   ray start --address='<head_ip>:6380'

用 ``ray status`` 确认两个节点已就绪。

.. tip::

   如果你使用的是同一台机器上的两个 Docker 容器且使用了 ``--network host``，
   ``<head_ip>`` 可以直接用宿主机的局域网 IP（如 ``172.16.0.1``）。

步骤 5：采集 demo 数据（2 节点，分类器在 GPU 推理）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

启动 Ray 集群后，在 **head 节点（GPU 节点）** 上运行 demo 采集脚本。
脚本会自动：

1. 在 GPU 节点上启动 ``ClassifierRewardServer``，加载分类器模型到 GPU
2. 在控制节点上启动 ``DataCollector``（env worker），通过 Ray 远程调用分类器

demo 采集流程中：

- **分类器判定成功** → episode 终止，保存为成功 demo
- **episode 到达** ``max_episode_steps`` → 超时截断，保存为失败 demo 并自动复位

**配置文件** ``examples/embodiment/config/realworld_collect_dexpnp_demo.yaml``
关键字段：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka     # env worker 在控制节点
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0          # GPU 节点
       - label: franka
         node_ranks: 1          # 控制节点
         env_configs:
           - node_ranks: 1
             env_vars:
               - PYTHONPATH: "..."
               - LD_LIBRARY_PATH: "..."
         hardware:
           type: Franka
           configs:
             - robot_ip: 172.16.0.2
               node_rank: 1

   env:
     eval:
       ignore_terminations: False   # 让分类器触发的 termination 生效
       classifier_reward_wrapper:
         checkpoint_path: "/path/to/reward_classifier.pt"
         device: cuda               # 分类器在 GPU 节点上用 CUDA 推理
         remote: true               # 关键：通过 Ray 远程调用 GPU 节点
         threshold: 0.75

修改配置中的 ``checkpoint_path`` 为步骤 3 训练得到的分类器路径，
以及 ``robot_ip``、``env_vars`` 等为实际值后，在 **head 节点** 运行：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_dexpnp_demo

.. important::

   ``ignore_terminations`` 必须为 ``False``，否则分类器触发的终止信号会被忽略。
   这与步骤 1（采集分类器训练数据）的 ``ignore_terminations: True`` **恰好相反**。

采集结果保存在 ``logs/<timestamp>/demos/`` 目录下。

**终端输出示例：**

.. code-block:: text

   [collect_real_data] ClassifierRewardServer 'ClassifierRewardServer' ready on GPU node.
   ...
   ✅ SUCCESS  classifier_reward=0.912
       成功: 5/20  总 episodes: 8
   ⏱️  TRUNCATED (max steps)
       成功: 5/20  总 episodes: 9

步骤 6：训练 RL（使用 demo 和分类器奖励）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

采集完 demo 后，**确保 Ray 集群仍在运行**（或重新启动），
然后在训练配置 YAML 中指定：

1. ``classifier_reward_wrapper.remote: true`` — 在线训练时分类器也在 GPU 节点推理
2. ``demo_buffer.load_path`` — 步骤 5 采集的 demo 数据路径

**训练配置**
``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` 关键字段：

.. code-block:: yaml

   env:
     train:
       classifier_reward_wrapper:
         checkpoint_path: /path/to/reward_classifier.pt
         device: cuda
         remote: true              # 分类器在 GPU 节点推理
         threshold: 0.75

   algorithm:
     demo_buffer:
       load_path: "/path/to/logs/<timestamp>/demos"

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       env:
         node_group: franka
         placement: 0
       rollout:
         node_group: "4090"
         placement: 0

修改 ``checkpoint_path`` 和 ``demo_buffer.load_path`` 后，在 **head 节点** 运行：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async

.. note::

   使用 ``classifier_reward_wrapper`` 后，无需再设置 ``keyboard_reward_wrapper``，
   两者是互斥的奖励来源。


完整操作速查
-----------------------

以下表格汇总了整个 workflow 中每一步的运行位置和命令：

.. list-table::
   :widths: 6 26 18 50
   :header-rows: 1

   * - 步骤
     - 操作
     - 运行位置
     - 命令
   * - 1
     - 采集分类器训练数据
     - 控制节点（单节点 Ray）
     - ``ray start --head`` → ``bash examples/embodiment/collect_classifier_data.sh`` → ``ray stop``
   * - 2
     - 人工审核
     - 有显示器的节点
     - ``bash examples/embodiment/review_classifier_data.sh logs/<dir>``
   * - 3
     - 训练分类器
     - GPU 节点
     - ``python examples/embodiment/train_reward_classifier.py --log_dir ... --device cuda``
   * - 4
     - 启动 Ray 集群
     - 两个节点
     - head: ``ray start --head`` / worker: ``ray start --address=...``
   * - 5
     - 采集 demo（分类器 GPU 推理）
     - head 节点（GPU）
     - ``bash examples/embodiment/collect_data.sh realworld_collect_dexpnp_demo``
   * - 6
     - 训练 RL
     - head 节点（GPU）
     - ``bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async``


诊断工具
---------------------------

``test_controller`` 是一个交互式命令行诊断工具，用于实时查询机械臂和灵巧手状态。

.. code-block:: bash

   export FRANKA_ROBOT_IP=<your_robot_ip>
   export FRANKA_END_EFFECTOR_TYPE=ruiyan_hand
   python -m toolkits.realworld_check.test_franka_controller

.. list-table:: 可用命令
   :widths: 20 60
   :header-rows: 1

   * - 命令
     - 说明
   * - ``getpos``
     - 获取机械臂 TCP 位姿（四元数表示，7 维）
   * - ``getpos_euler``
     - 获取机械臂 TCP 位姿（欧拉角表示，6 维）
   * - ``getjoints``
     - 获取机械臂关节位置和速度（各 7 维）
   * - ``getvel``
     - 获取机械臂 TCP 速度（6 维）
   * - ``getforce``
     - 获取机械臂 TCP 力和力矩（各 3 维）
   * - ``gethand``
     - 获取灵巧手各手指位置 [0, 1]
   * - ``gethand_detail``
     - 获取每个电机的详细状态（位置、速度、电流、状态码）
   * - ``handinfo``
     - 显示灵巧手配置信息（类型、串口、波特率等）
   * - ``state``
     - 显示完整机器人状态
   * - ``help``
     - 显示帮助信息
   * - ``q``
     - 退出

手指 DOF 映射
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 8 20 20 42
   :header-rows: 1

   * - #
     - DOF 名称
     - 中文名称
     - 说明
   * - 1
     - ``thumb_rotation``
     - 拇指旋转
     - 拇指侧摆（内收/外展）
   * - 2
     - ``thumb_bend``
     - 拇指弯曲
     - 拇指屈曲/伸展
   * - 3
     - ``index``
     - 食指
     - 食指屈曲/伸展
   * - 4
     - ``middle``
     - 中指
     - 中指屈曲/伸展
   * - 5
     - ``ring``
     - 无名指
     - 无名指屈曲/伸展
   * - 6
     - ``pinky``
     - 小指
     - 小指屈曲/伸展

所有位置值归一化至 ``[0, 1]``：``0`` = 全开，``1`` = 全闭。


末端执行器架构
---------------------------

所有末端执行器实现统一的 ``EndEffector`` 抽象基类：

.. code-block:: python

   class EndEffectorType(str, Enum):
       FRANKA_GRIPPER = "franka_gripper"   # 7 维动作
       AOYI_HAND      = "aoyi_hand"        # 12 维动作
       RUIYAN_HAND    = "ruiyan_hand"      # 12 维动作

工厂函数 ``create_end_effector(end_effector_type, **kwargs)`` 根据类型字符串
创建对应的末端执行器实例。切换末端执行器后，``FrankaEnv`` 会自动调整动作空间和观测空间。

**支持的灵巧手：**

- **傲意灵巧手** — Modbus RTU 串口协议，6 DOF，``[0, 1]`` 连续控制
- **睿研灵巧手** — 自定义串口协议，6 DOF，``[0, 1]`` 连续控制

遥操作架构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

灵巧手遥操作使用 **空间鼠标 + 数据手套** 的组合：

- **空间鼠标** — 6 维末端位姿增量（x, y, z, roll, pitch, yaw）
- **数据手套** — 6 维手指角度（相对模式）

两者由 ``DexHandIntervention`` 合并为 12 维人类专家动作。
系统根据配置中的 ``end_effector_type`` 自动选择对应的干预包装器，无需手动修改代码。

**相对手套控制（Relative Glove Control）**

手套工作在相对控制模式，通过 SpaceMouse 的左键切换：

1. **按下左键瞬间** — 捕获当前手套读数作为"基准"，同时记录手指当前位置
2. **持续按住左键** — 手套读数与基准的差值（delta）叠加到手指位置上
3. **松开左键** — 手指冻结在最后位置不动

这确保了手套不会导致手指突然跳变，操作者可以分多次按住左键来精细调节手指位姿。

**按键映射**

``pyspacemouse`` 库中 ``buttons[0]`` 对应物理右键，``buttons[1]`` 对应物理左键
（与直觉相反）。``DexHandIntervention`` 内部已做了映射纠正：

.. code-block:: python

   # pyspacemouse: buttons[0] = 物理右键, buttons[1] = 物理左键
   self.left, self.right = bool(buttons[1]), bool(buttons[0])

两个按键均通过 ``info["left"]`` / ``info["right"]`` 暴露给下游组件使用。

**Episode 间手指状态同步**

``DexHandIntervention.reset()`` 会在每个 Episode 开始时将内部手指追踪状态
同步为配置中的 ``hand_reset_state``，确保操作者从复位位姿（而非上一个 Episode
的结束位姿）开始新的操作。


可视化与结果
-------------------------

**TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**关键监控指标**

- ``env/success_once``：推荐关注的训练性能指标，直接反映回合成功率
- ``env/return``：回合总回报
- ``env/reward``：step-level 奖励

完整指标列表请参考 :doc:`franka`。

.. note::

   灵巧手任务的训练结果和演示视频将在后续更新中提供。
