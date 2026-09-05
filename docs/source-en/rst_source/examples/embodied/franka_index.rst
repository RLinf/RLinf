Single-Arm Franka
=================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: Single-Arm Franka

   A single-arm Franka platform for real-world RL, data collection, and policy deployment.

This section collects RLinf real-world reinforcement learning workflows for a
single-arm Franka setup.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: ZED + Robotiq
      :link: franka_zed_robotiq
      :link-type: doc

      Use ZED cameras and Robotiq grippers.

   .. grid-item-card:: Dexterous Hand
      :link: franka_dexhand
      :link-type: doc

      Drive a Franka with a dexterous hand end-effector.

   .. grid-item-card:: Collect-GELLO
      :link: franka_gello
      :link-type: doc

      Collect joint-level teleoperation data with GELLO.

   .. grid-item-card:: Collect-VR
      :link: franka_vr
      :link-type: doc

      Use VR / PICO devices for teleoperation.

   .. grid-item-card:: HG-DAgger
      :link: hg-dagger
      :link-type: doc

      Collect interventions and improve a Franka policy with online human-gated DAgger.

   .. grid-item-card:: Collect-SFT-Deploy
      :link: franka_pi0_sft_deploy
      :link-type: doc

      Deploy a π₀ SFT policy on Franka.

   .. grid-item-card:: Real-World RL
      :link: franka_single_machine
      :link-type: doc

      Run real-world RL on one GPU host with explicit real-time configuration.

   .. grid-item-card:: Real-World RL (Archived)
      :link: franka
      :link-type: doc

      Configure a Franka setup, collect demonstrations, and run online RL training (archived).

   .. grid-item-card:: Reward Model
      :link: franka_reward_model
      :link-type: doc

      Train Franka with a learned reward model.

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
