Franka (Single-Arm)
===================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: Single-Arm Franka

   A single-arm Franka platform for real-world RL, data collection, and policy deployment.

Start with **Core Workflows** to set up a single-arm Franka and train or deploy a policy, then pick teleoperation, reward-model, or hardware guides as needed.

1. Core Workflows
-----------------

Choose the training or deployment loop you want to run on a single-arm Franka.

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: Online RL
      :link: franka
      :link-type: doc

      Set up a single-arm Franka, collect demonstrations, and run online RL training.

   .. grid-item-card:: π₀ SFT & Deploy
      :link: franka_pi0_sft_deploy
      :link-type: doc

      Collect demonstrations, fine-tune π₀, and deploy the policy on Franka.

   .. grid-item-card:: HG-DAgger
      :link: hg-dagger
      :link-type: doc

      Collect interventions and improve a Franka policy with online human-gated DAgger.

2. Teleoperation and Reward Models
----------------------------------

Add teleoperation input devices or learned reward models to any core workflow.

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: GELLO Teleop
      :link: franka_gello
      :link-type: doc

      Collect joint-level teleoperation data with GELLO.

   .. grid-item-card:: PICO VR Teleop
      :link: franka_vr
      :link-type: doc

      Use VR / PICO devices for teleoperation.

   .. grid-item-card:: Reward Models
      :link: franka_reward_model
      :link-type: doc

      Train Franka with a learned reward model.

3. Cameras and End-Effectors
----------------------------

Configure optional depth cameras, grippers, and dexterous hands on your Franka rig.

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: ZED & Robotiq
      :link: franka_zed_robotiq
      :link-type: doc

      Use ZED cameras and Robotiq grippers.

   .. grid-item-card:: Orbbec Cameras
      :link: franka_orbbec
      :link-type: doc

      Capture color images and aligned depth from Orbbec USB cameras.

   .. grid-item-card:: Dexterous Hand
      :link: franka_dexhand
      :link-type: doc

      Drive a Franka with a dexterous hand end-effector.

.. toctree::
   :hidden:
   :maxdepth: 1

   Online RL <franka>
   π₀ SFT & Deploy <franka_pi0_sft_deploy>
   HG-DAgger <hg-dagger>
   GELLO Teleop <franka_gello>
   PICO VR Teleop <franka_vr>
   Reward Models <franka_reward_model>
   ZED & Robotiq <franka_zed_robotiq>
   Orbbec Cameras <franka_orbbec>
   Dexterous Hand <franka_dexhand>
