Single-Arm Franka
=================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: Single-Arm Franka

   A single-arm Franka platform for real-world RL, data collection, and policy deployment.

Start with Real-World RL to set up a single-arm Franka and train a policy.
Then choose a guide for teleoperation, data collection, deployment, or hardware
extensions.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Real-World RL
      :link: franka
      :link-type: doc

      Set up a single-arm Franka, collect demonstrations, and run online RL training.

   .. grid-item-card:: GELLO Collection
      :link: franka_gello
      :link-type: doc

      Collect joint-level teleoperation data with GELLO.

   .. grid-item-card:: VR Teleoperation
      :link: franka_vr
      :link-type: doc

      Use VR / PICO devices for teleoperation.

   .. grid-item-card:: Collection / SFT / Deployment
      :link: franka_pi0_sft_deploy
      :link-type: doc

      Collect demonstrations, fine-tune π₀, and deploy the policy on Franka.

   .. grid-item-card:: HG-DAgger
      :link: hg-dagger
      :link-type: doc

      Collect interventions and improve a Franka policy with online human-gated DAgger.

   .. grid-item-card:: Reward Model
      :link: franka_reward_model
      :link-type: doc

      Train Franka with a learned reward model.

   .. grid-item-card:: ZED + Robotiq
      :link: franka_zed_robotiq
      :link-type: doc

      Use ZED cameras and Robotiq grippers.

   .. grid-item-card:: Dexterous Hand
      :link: franka_dexhand
      :link-type: doc

      Drive a Franka with a dexterous hand end-effector.

.. toctree::
   :hidden:
   :maxdepth: 1

   Real-World RL <franka>
   GELLO Collection <franka_gello>
   VR Teleoperation <franka_vr>
   Collection / SFT / Deployment <franka_pi0_sft_deploy>
   HG-DAgger <hg-dagger>
   Reward Model <franka_reward_model>
   ZED + Robotiq <franka_zed_robotiq>
   Dexterous Hand <franka_dexhand>
