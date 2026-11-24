Franka（单臂）
==============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: 单臂 Franka

   用于真机强化学习、数据采集和 policy 部署的单臂 Franka 平台。

先从核心流程完成单臂 Franka 配置与 policy 训练或部署，再按需求扩展遥操作设备、奖励模型、相机和末端执行器。

1. 核心流程
-----------

按训练或部署目标选择单臂 Franka 的主线指南。

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: 在线 RL
      :link: franka
      :link-type: doc

      配置单臂 Franka，采集演示数据，并运行在线强化学习训练。

   .. grid-item-card:: π₀ SFT 与部署
      :link: franka_pi0_sft_deploy
      :link-type: doc

      采集演示数据，微调 π₀，并在 Franka 上部署 policy。

   .. grid-item-card:: HG-DAgger
      :link: hg-dagger
      :link-type: doc

      采集人工干预数据，并用 Human-Gated DAgger 在线改进 Franka policy。

2. 遥操作与奖励模型
-------------------

按需为主线流程接入遥操作设备或学习型奖励模型。

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: GELLO 遥操作
      :link: franka_gello
      :link-type: doc

      用 GELLO 采集关节级遥操作示教数据。

   .. grid-item-card:: PICO VR 遥操作
      :link: franka_vr
      :link-type: doc

      用 VR 或 PICO 设备遥操作 Franka。

   .. grid-item-card:: 奖励模型
      :link: franka_reward_model
      :link-type: doc

      用学习到的奖励模型训练 Franka。

3. 相机与末端执行器
-------------------

为 Franka 接入可选深度相机、夹爪和灵巧手硬件。

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: ZED 与 Robotiq
      :link: franka_zed_robotiq
      :link-type: doc

      配置 ZED 相机与 Robotiq 夹爪。

   .. grid-item-card:: Orbbec 相机
      :link: franka_orbbec
      :link-type: doc

      读取 Orbbec USB 相机的彩色图像及对齐深度。

   .. grid-item-card:: 灵巧手
      :link: franka_dexhand
      :link-type: doc

      为 Franka 配置灵巧手末端执行器。

.. toctree::
   :hidden:
   :maxdepth: 1

   在线 RL <franka>
   π₀ SFT 与部署 <franka_pi0_sft_deploy>
   HG-DAgger <hg-dagger>
   GELLO 遥操作 <franka_gello>
   PICO VR 遥操作 <franka_vr>
   奖励模型 <franka_reward_model>
   ZED 与 Robotiq <franka_zed_robotiq>
   Orbbec 相机 <franka_orbbec>
   灵巧手 <franka_dexhand>
