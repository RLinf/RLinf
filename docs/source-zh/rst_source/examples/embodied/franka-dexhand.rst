Franka + 灵巧手真机强化学习
============================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 框架中为 Franka 机械臂配置 **灵巧手末端执行器**
（傲意手、锐研手），使用 **数据手套 + 空间鼠标** 进行遥操作数据采集和人类干预训练，
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
   傲意灵巧手、锐研灵巧手之间通过配置文件一键切换。
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

- **灵巧手** — 傲意灵巧手（Modbus RTU 串口）或 锐研灵巧手（自定义串口协议）
- **数据手套** — PSI 数据手套，USB 串口连接（通常挂载为 ``/dev/ttyACM0``）

控制节点硬件连接
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在控制节点上需要连接以下硬件：

1. **Franka 机械臂** — 通过以太网连接
2. **灵巧手** — 通过 USB 串口连接（傲意：Modbus，锐研：自定义协议）
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
需要额外安装灵巧手和数据手套的串口通信依赖。

控制节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

按照 :doc:`franka` 中的步骤完成基础依赖安装后，在控制节点的虚拟环境中执行：

.. code-block:: bash

   # 串口通信依赖（灵巧手 + 数据手套）
   pip install pyserial pymodbus pyyaml

   # 数据手套驱动
   pip install psi_glove_driver

.. note::

   如果 ``psi_glove_driver`` 无法通过 pip 安装，
   请联系数据手套供应商获取驱动包并手动安装。

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
   python -m toolkits.realworld_check.test_controller

在交互界面中：

- 输入 ``getpos_euler`` 获取当前末端执行器位姿（欧拉角形式）
- 输入 ``gethand`` 查看灵巧手当前手指位置
- 输入 ``handinfo`` 确认灵巧手连接状态
- 输入 ``help`` 查看所有可用命令

**2. 测试硬件连接**

.. code-block:: bash

   # 测试相机
   python -m toolkits.realworld_check.test_camera

数据采集
~~~~~~~~~~~~~~~~~

灵巧手场景的数据采集使用空间鼠标控制机械臂、数据手套控制手指，
``DexHandIntervention`` 会自动将两者输入合并为 12 维动作。

1. 激活虚拟环境并 source ROS 设置脚本：

.. code-block:: bash

   source <path_to_your_venv>/bin/activate
   source <your_catkin_ws>/devel/setup.bash

2. 修改配置文件 ``examples/embodiment/config/realworld_collect_data.yaml``：

.. code-block:: yaml

   env:
     train:
       override_cfg:
         end_effector_type: ruiyan_hand   # 或 aoyi_hand
         robot_ip: YOUR_ROBOT_IP
         target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
       use_spacemouse: true
       glove_config:
         left_port: "/dev/ttyACM0"
         frequency: 60

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
             - robot_ip: YOUR_ROBOT_IP
               node_rank: 0

3. 运行数据采集脚本：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh

在采集过程中，使用空间鼠标控制机械臂移动，同时佩戴数据手套控制灵巧手的手指动作。
采集到的数据会保存在 ``logs/[running-timestamp]/data.pkl`` 路径下。

集群配置
~~~~~~~~~~~~~~~~~

集群配置步骤与 :doc:`franka` 完全一致。
在每个节点上运行 ``ray start`` 之前，确保已正确设置环境变量（参考 ``ray_utils/realworld/setup_before_ray.sh``）。

配置文件
~~~~~~~~~~~~~~~~~~~~~~

在配置 YAML 中设置末端执行器类型，并根据使用的灵巧手型号填写对应参数：

**锐研手示例：**

.. code-block:: yaml

   env:
     train:
       override_cfg:
         end_effector_type: ruiyan_hand
         ruiyan_port: "/dev/ttyUSB0"
         ruiyan_baudrate: 460800
       use_spacemouse: true
       glove_config:
         left_port: "/dev/ttyACM0"

**傲意手示例：**

.. code-block:: yaml

   env:
     train:
       override_cfg:
         end_effector_type: aoyi_hand
         aoyi_port: "/dev/ttyUSB0"
         aoyi_node_id: 2
         aoyi_baudrate: 115200
       use_spacemouse: true
       glove_config:
         left_port: "/dev/ttyACM0"

同时，将 ``rollout`` 和 ``actor`` 部分的 ``model_path`` 字段设置为前面下载的预训练模型路径。

运行实验
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 head 节点上启动实验：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh <your_dexhand_config_name>


视觉奖励分类器
-----------------------

灵巧手任务中，末端执行器的位姿不足以判定任务是否成功
（例如抓取任务中物体是否被稳定握持），因此需要一个视觉分类器来自动判定奖励。

概述
~~~~~~~~~~~~~~~

视觉奖励分类器使用冻结的 ResNet-10 backbone 提取图像特征，
通过 ``SpatialLearnedEmbeddings`` 进行空间池化，最终经过一个二分类头输出
成功概率。训练流程：

1. 收集成功和失败的图像数据
2. 训练二分类器
3. 在环境中用分类器输出替代人工奖励

步骤 1：收集分类器训练数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用数据采集工具收集成功/失败图像，按空格键标记成功帧：

.. code-block:: bash

   python -m toolkits.realworld_check.record_classifier_data \
       --save_dir /path/to/classifier_data \
       --successes_needed 200

采集完成后，数据以 pickle 格式保存在指定目录中。

步骤 2：训练分类器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m toolkits.realworld_check.train_reward_classifier \
       --data_dir /path/to/classifier_data \
       --save_dir /path/to/classifier_ckpt \
       --pretrained_ckpt RLinf-ResNet10-pretrained/resnet10_pretrained.pt \
       --image_keys wrist_1 \
       --num_epochs 200

其中 ``--pretrained_ckpt`` 为 ResNet-10 策略 backbone 的预训练权重，
用于初始化分类器的冻结视觉编码器。训练好的分类器会保存在 ``--save_dir`` 目录下。

步骤 3：在环境中使用分类器奖励
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在配置 YAML 中添加 ``classifier_reward_wrapper`` 配置项，
系统会自动用分类器输出替代键盘人工奖励：

**单阶段任务：**

.. code-block:: yaml

   env:
     train:
       classifier_reward_wrapper:
         checkpoint_path: /path/to/classifier_ckpt/best_classifier.pt
         image_keys: [wrist_1]
         device: cuda
         threshold: 0.75   # sigmoid 输出超过此阈值判定为成功

**多阶段任务：**

.. code-block:: yaml

   env:
     train:
       classifier_reward_wrapper:
         multi_stage: true
         checkpoint_paths:
           - /path/to/stage1_classifier.pt
           - /path/to/stage2_classifier.pt
         image_keys: [wrist_1]
         device: cuda

.. note::

   使用 ``classifier_reward_wrapper`` 后，无需再设置 ``keyboard_reward_wrapper``，
   两者是互斥的奖励来源。


诊断工具
---------------------------

``test_controller`` 是一个交互式命令行诊断工具，用于实时查询机械臂和灵巧手状态。

.. code-block:: bash

   export FRANKA_ROBOT_IP=<your_robot_ip>
   export FRANKA_END_EFFECTOR_TYPE=ruiyan_hand
   python -m toolkits.realworld_check.test_controller

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
- **锐研灵巧手** — 自定义串口协议，6 DOF，``[0, 1]`` 连续控制

遥操作架构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

灵巧手遥操作使用 **空间鼠标 + 数据手套** 的组合：

- **空间鼠标** — 6 维末端位姿增量（x, y, z, roll, pitch, yaw）
- **数据手套** — 6 维手指角度

两者由 ``DexHandIntervention`` 合并为 12 维人类专家动作。
系统根据配置中的 ``end_effector_type`` 自动选择对应的干预包装器，无需手动修改代码。


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
