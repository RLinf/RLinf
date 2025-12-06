基于RoboCasa的强化学习
========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档提供了在RLinf框架中使用RoboCasa基准测试进行强化学习训练任务的全面指南。
RoboCasa是一个大规模的机器人学习仿真框架，专注于厨房环境中的操作任务，具有多样化的厨房布局、物体和操作任务。

RoboCasa将真实的厨房环境与多样化的操作挑战相结合，使其成为开发可泛化机器人策略的理想基准。
主要目标是训练能够执行以下任务的视觉-语言-动作模型:

1. **视觉理解**: 处理来自多个摄像头视角的RGB图像。
2. **语言理解**: 解释自然语言任务指令。
3. **操作技能**: 执行复杂的厨房任务,如拾取-放置、开关门和电器控制。

环境介绍
--------

**RoboCasa仿真平台**

- **环境**: RoboCasa Kitchen厨房仿真环境(基于robosuite构建)
- **机器人**: Panda机械臂带底座(PandaOmron),配备平行夹爪
- **任务**: 24个原子厨房任务,涵盖多个类别
- **观测**: 多视角RGB图像(机器人视角+腕部相机) + 本体感知状态
- **动作空间**: 7维连续动作(尽管机器人是PandaOmron,但底座和躯干保持固定)

  - 3D机械臂位置增量 (x, y, z)
  - 3D机械臂旋转增量 (rx, ry, rz)
  - 1D夹爪控制 (开/关)

  **注**: PandaOmron实际接受12维动作 ``[3D底座, 1D躯干, 3D臂位置, 3D臂旋转, 2D夹爪]``,
  但在RoboCasa Kitchen任务中,底座和躯干保持固定(设为0),策略只需输出7维动作控制机械臂和夹爪

**任务类别**

RoboCasa提供了组织成多个类别的多样化原子任务:

*门操作任务*:

- ``OpenSingleDoor``: 打开柜门或微波炉门
- ``CloseSingleDoor``: 关闭柜门或微波炉门
- ``OpenDoubleDoor``: 打开双开门柜子
- ``CloseDoubleDoor``: 关闭双开门柜子
- ``OpenDrawer``: 打开抽屉
- ``CloseDrawer``: 关闭抽屉

*拾取和放置任务*:

- ``PnPCounterToCab``: 从柜台拾取并放置到柜子中
- ``PnPCabToCounter``: 从柜子拾取并放置到柜台上
- ``PnPCounterToSink``: 从柜台拾取并放置到水槽中
- ``PnPSinkToCounter``: 从水槽拾取并放置到柜台上
- ``PnPCounterToStove``: 从柜台拾取并放置到炉灶上
- ``PnPStoveToCounter``: 从炉灶拾取并放置到柜台上
- ``PnPCounterToMicrowave``: 从柜台拾取并放置到微波炉中
- ``PnPMicrowaveToCounter``: 从微波炉拾取并放置到柜台上

*电器控制任务*:

- ``TurnOnMicrowave``: 打开微波炉
- ``TurnOffMicrowave``: 关闭微波炉
- ``TurnOnSinkFaucet``: 打开水龙头
- ``TurnOffSinkFaucet``: 关闭水龙头
- ``TurnSinkSpout``: 旋转水槽喷嘴
- ``TurnOnStove``: 打开炉灶
- ``TurnOffStove``: 关闭炉灶

*咖啡制作任务*:

- ``CoffeeSetupMug``: 放置咖啡杯
- ``CoffeeServeMug``: 将咖啡倒入杯中
- ``CoffeePressButton``: 按下咖啡机按钮

**观测结构**

- **主相机图像** (``base_image``): 机器人左侧视角 (128×128 RGB)
- **腕部相机图像** (``wrist_image``): 末端执行器视角相机 (128×128 RGB)
- **本体感知状态** (``state``): 16维向量,包含:

  - ``[0:2]`` 机器人底座位置 (x, y)
  - ``[2:5]`` 填充零值
  - ``[5:9]`` 末端执行器相对于底座的四元数
  - ``[9:12]`` 末端执行器相对于底座的位置
  - ``[12:14]`` 夹爪关节速度
  - ``[14:16]`` 夹爪关节位置

**数据结构**

- **图像**: 主相机RGB张量 ``[batch_size, 3, 128, 128]`` 和腕部相机 ``[batch_size, 3, 128, 128]``
- **状态**: 本体感知状态张量 ``[batch_size, 16]``
- **任务描述**: 自然语言指令
- **动作**: 7维连续动作(位置、四元数、夹爪)
- **奖励**: 基于任务完成的稀疏奖励