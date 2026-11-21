真实机器人强化学习
==================

按实体机器人平台选择对应指南，覆盖各平台支持的硬件配置、遥操作采数、SFT 部署与在线 RL 流程。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld_flow_zh.svg
   :align: center
   :width: 100%
   :alt: RLinf 真实机器人流程

   真机端到端流程：硬件配置、遥操作与数据采集、SFT、policy 部署与在线 RL。

机器人能力矩阵
--------------

按支持环节对比各机器人平台，再进入对应指南。

.. list-table::
   :header-rows: 1
   :widths: 22 10 16 14 12 26

   * - 机器人
     - 硬件配置
     - 遥操作与采数
     - SFT / 部署
     - 在线 RL
     - 指南
   * - 单臂 Franka
     - ✓
     - ✓
     - ✓
     - ✓
     - :doc:`单臂 Franka（9 篇） <embodied/franka_index>`
   * - 双臂 Franka
     - ✓
     - ✓
     - ✓
     - ✓
     - :doc:`双臂 Franka（3 篇） <embodied/dual_franka_index>`
   * - ARX X5
     - ✓
     - ✓
     - —
     - —
     - :doc:`ARX X5 指南 <embodied/arx5>`
   * - YAM
     - ✓
     - ✓
     - ✓
     - —
     - :doc:`YAM（2 篇） <embodied/yam_index>`
   * - GimArm
     - ✓
     - ✓
     - —
     - ✓
     - :doc:`GimArm 指南 <embodied/gim_arm>`
   * - XSquare Turtle2
     - ✓
     - —
     - —
     - ✓
     - :doc:`Turtle2 指南 <embodied/xsquare_turtle2>`
   * - Dexmal DOS-W1
     - ✓
     - ✓
     - —
     - ✓
     - :doc:`DOS-W1 指南 <embodied/dosw1>`
   * - Robbyant R2
     - ✓
     - ✓
     - —
     - ✓
     - :doc:`Robbyant R2 指南 <embodied/r2>`
   * - AgileX Piper
     - ✓
     - —
     - —
     - —
     - :doc:`Piper 指南 <embodied/piper>`
   * - SO-101
     - ✓
     - ✓
     - ✓
     - —
     - :doc:`SO-101 指南 <embodied/so101>`
   * - LeKiwi
     - ✓
     - ✓
     - —
     - —
     - :doc:`LeKiwi 指南 <embodied/lekiwi>`

工业与科研机械臂
----------------

适用于硬件配置、遥操作、数据采集、SFT 部署和在线 RL 的单臂与双臂机械臂平台。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 单臂 Franka
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
      :img-alt: 单臂 Franka
      :link: embodied/franka_index
      :link-type: doc

      真机强化学习、遥操作与数据采集、奖励模型、SFT、部署与硬件扩展。

   .. grid-item-card:: 双臂 Franka
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
      :img-alt: 双臂 Franka
      :link: embodied/dual_franka_index
      :link-type: doc

      GELLO 与 PICO 数据采集、OpenPI π₀.₅ SFT、policy 部署与在线 HG-DAgger。

   .. grid-item-card:: ARX X5
      :img-top: https://arx-global.us/assets/img/x5.webp
      :img-alt: ARX X5 机械臂
      :link: embodied/arx5
      :link-type: doc

      用 PICO 双手柄和三路 RGB 相机采集 LeRobot 格式双臂示教数据。

   .. grid-item-card:: YAM
      :link: embodied/yam_index
      :link-type: doc

      用 PICO 采集双臂示教数据，微调 OpenPI π₀.₅ 并在双臂上评估 checkpoint。

   .. grid-item-card:: GimArm
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png
      :img-alt: GimArm 六自由度机械臂
      :link: embodied/gim_arm
      :link-type: doc

      在 SocketCAN 6-DoF 机械臂上结合 VR 遥操作与 SAC / RLPD 训练插孔任务。

   .. grid-item-card:: AgileX Piper
      :img-top: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
      :img-alt: AgileX Piper 机械臂
      :link: embodied/piper
      :link-type: doc

      配置 SocketCAN 并检查 Piper 关节反馈、小幅运动与夹爪控制。

双臂与人形平台
--------------

支持全身遥操作与真机强化学习的一体化双臂和轮式人形机器人。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: XSquare Turtle2
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg
      :img-alt: XSquare Turtle2 双臂机器人
      :link: embodied/xsquare_turtle2
      :link-type: doc

      在 XSquare Turtle2 双臂机器人上用 SAC + CNN policy 训练按键任务。

   .. grid-item-card:: Dexmal DOS-W1
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png
      :img-alt: Dexmal DOS-W1 双臂机器人
      :link: embodied/dosw1
      :link-type: doc

      用主臂采集示教数据，并在 DOS-W1 上训练 Flow Matching + SAC 抓取 policy。

   .. grid-item-card:: Robbyant R2
      :link: embodied/r2
      :link-type: doc

      在 Robbyant R2 轮式人形机器人上运行硬件检查、全身 PICO VR 遥操作与关节到达 RL。

低成本与移动机器人
------------------

兼容 LeRobot 舵机生态的桌面机械臂与移动底盘。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: SO-101
      :img-top: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Follower.webp
      :img-alt: SO-101 从动机械臂
      :link: embodied/so101
      :link-type: doc

      配置并标定 SO-101，采集 leader 示教数据，完成 OpenPI SFT 与本地评估。

   .. grid-item-card:: LeKiwi
      :link: embodied/lekiwi
      :link-type: doc

      配置轮电机、检查前置摄像头，并用键盘本地或远程驱动全向移动底盘。
