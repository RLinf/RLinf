单臂 Franka
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: 单臂 Franka

   用于真机强化学习、数据采集和 policy 部署的单臂 Franka 平台。

从真机强化学习示例开始，完成单臂 Franka 的配置与 policy 训练，再按需要选择遥操作、数据采集、部署或硬件扩展指南。

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 真机强化学习
      :link: franka
      :link-type: doc

      配置单臂 Franka，采集演示数据，并运行在线强化学习训练。

   .. grid-item-card:: GELLO 数据采集
      :link: franka_gello
      :link-type: doc

      使用 GELLO 进行关节级遥操作数据采集。

   .. grid-item-card:: VR 遥操作
      :link: franka_vr
      :link-type: doc

      使用 VR / PICO 设备进行遥操作。

   .. grid-item-card:: 采集、SFT 与部署
      :link: franka_pi0_sft_deploy
      :link-type: doc

      采集演示数据，微调 π₀，并在 Franka 上部署 policy。

   .. grid-item-card:: HG-DAgger
      :link: hg-dagger
      :link-type: doc

      采集人工干预数据，并通过 Human-Gated DAgger 在线改进 Franka policy。

   .. grid-item-card:: 奖励模型
      :link: franka_reward_model
      :link-type: doc

      使用学习到的奖励模型训练 Franka。

   .. grid-item-card:: ZED + Robotiq
      :link: franka_zed_robotiq
      :link-type: doc

      使用 ZED 相机与 Robotiq 夹爪。

   .. grid-item-card:: 灵巧手
      :link: franka_dexhand
      :link-type: doc

      为 Franka 配置灵巧手末端执行器。

.. toctree::
   :hidden:
   :maxdepth: 1

   真机强化学习 <franka>
   GELLO 数据采集 <franka_gello>
   VR 遥操作 <franka_vr>
   采集、SFT 与部署 <franka_pi0_sft_deploy>
   HG-DAgger <hg-dagger>
   奖励模型 <franka_reward_model>
   ZED + Robotiq <franka_zed_robotiq>
   灵巧手 <franka_dexhand>
