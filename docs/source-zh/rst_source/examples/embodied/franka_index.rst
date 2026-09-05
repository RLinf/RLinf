Single-Arm Franka
=================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: Single-Arm Franka

   用于真机强化学习、数据采集和策略部署的单臂 Franka 平台。

本节汇总 RLinf 在单臂 Franka 上的真机强化学习工作流。

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: ZED + Robotiq
      :link: franka_zed_robotiq
      :link-type: doc

      使用 ZED 相机与 Robotiq 夹爪。

   .. grid-item-card:: Dexterous Hand
      :link: franka_dexhand
      :link-type: doc

      为 Franka 配置灵巧手末端执行器。

   .. grid-item-card:: Collect-GELLO
      :link: franka_gello
      :link-type: doc

      使用 GELLO 进行关节级遥操作数据采集。

   .. grid-item-card:: Collect-VR
      :link: franka_vr
      :link-type: doc

      使用 VR / PICO 设备进行遥操作。

   .. grid-item-card:: HG-DAgger
      :link: hg-dagger
      :link-type: doc

      采集人工干预数据，并通过 Human-Gated DAgger 在线提升 Franka 策略。

   .. grid-item-card:: Collect-SFT-Deploy
      :link: franka_pi0_sft_deploy
      :link-type: doc

      在 Franka 上部署 π₀ SFT 策略。

   .. grid-item-card:: Real-World RL
      :link: franka_single_machine
      :link-type: doc

      在一台带 GPU 的主机上通过显式实时配置运行真机强化学习。

   .. grid-item-card:: Real-World RL (Archived)
      :link: franka
      :link-type: doc

      配置 Franka 真机环境，采集示教数据，并运行在线强化学习训练（已归档）。

   .. grid-item-card:: Reward Model
      :link: franka_reward_model
      :link-type: doc

      使用学习到的奖励模型训练 Franka。

.. toctree::
   :hidden:
   :maxdepth: 1

   ZED + Robotiq <franka_zed_robotiq>
   Dexterous Hand <franka_dexhand>
   Collect-GELLO <franka_gello>
   Collect-VR <franka_vr>
   HG-DAgger <hg-dagger>
   Collect-SFT-Deploy <franka_pi0_sft_deploy>
   Real-World RL <franka_single_machine>
   Real-World RL (Archived) <franka>
   Reward Model <franka_reward_model>
