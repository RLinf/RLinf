RL with Real-World Robots
=========================

Use this section when your starting point is physical robot hardware. Each robot page covers the steps supported for that platform, from hardware checks and teleoperation to SFT, deployment, and online RL.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld_flow.svg
   :align: center
   :width: 100%
   :alt: RLinf real-world robot workflow

   End-to-end real-world workflow: hardware setup, teleoperation and data collection, SFT, policy deployment, and online RL.

Robot Capability Matrix
-----------------------

Compare what each robot guide covers before choosing a platform.

.. list-table::
   :header-rows: 1
   :widths: 22 10 16 14 12 26

   * - Robot
     - Setup
     - Teleop & Collect
     - SFT / Deploy
     - Online RL
     - Guide
   * - Single-Arm Franka
     - ✓
     - ✓
     - ✓
     - ✓
     - :doc:`Single-Arm Franka (9 guides) <embodied/franka_index>`
   * - Dual-Arm Franka
     - ✓
     - ✓
     - ✓
     - ✓
     - :doc:`Dual-Arm Franka (3 guides) <embodied/dual_franka_index>`
   * - ARX X5
     - ✓
     - ✓
     - —
     - —
     - :doc:`ARX X5 Guide <embodied/arx5>`
   * - YAM
     - ✓
     - ✓
     - ✓
     - —
     - :doc:`YAM (2 guides) <embodied/yam_index>`
   * - GimArm
     - ✓
     - ✓
     - —
     - ✓
     - :doc:`GimArm Guide <embodied/gim_arm>`
   * - XSquare Turtle2
     - ✓
     - —
     - —
     - ✓
     - :doc:`Turtle2 Guide <embodied/xsquare_turtle2>`
   * - Dexmal DOS-W1
     - ✓
     - ✓
     - —
     - ✓
     - :doc:`DOS-W1 Guide <embodied/dosw1>`
   * - Robbyant R2
     - ✓
     - ✓
     - —
     - ✓
     - :doc:`Robbyant R2 Guide <embodied/r2>`
   * - AgileX Piper
     - ✓
     - —
     - —
     - —
     - :doc:`Piper Guide <embodied/piper>`
   * - SO-101
     - ✓
     - ✓
     - ✓
     - —
     - :doc:`SO-101 Guide <embodied/so101>`
   * - LeKiwi
     - ✓
     - ✓
     - —
     - —
     - :doc:`LeKiwi Guide <embodied/lekiwi>`

Industrial & Research Arms
--------------------------

Single-arm and dual-arm manipulators for hardware setup, teleoperation, data collection, SFT, deployment, and online RL.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Single-Arm Franka
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
      :img-alt: Single-Arm Franka
      :link: embodied/franka_index
      :link-type: doc

      Real-world RL, teleoperation, data collection, reward models, SFT, deployment, and hardware extensions.

   .. grid-item-card:: Dual-Arm Franka
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
      :img-alt: Dual-Arm Franka
      :link: embodied/dual_franka_index
      :link-type: doc

      GELLO and PICO data collection, OpenPI π₀.₅ SFT, policy deployment, and online HG-DAgger.

   .. grid-item-card:: ARX X5
      :img-top: https://arx-global.us/assets/img/x5.webp
      :img-alt: ARX X5 robot arms
      :link: embodied/arx5
      :link-type: doc

      Collect dual-arm demonstrations with PICO controllers and three RGB cameras in LeRobot format.

   .. grid-item-card:: YAM
      :link: embodied/yam_index
      :link-type: doc

      Collect dual-arm PICO demonstrations, fine-tune OpenPI π₀.₅, and evaluate checkpoints on both arms.

   .. grid-item-card:: GimArm
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png
      :img-alt: GimArm 6-DoF arm
      :link: embodied/gim_arm
      :link-type: doc

      Train a 6-DoF GimArm peg-insertion policy over SocketCAN with VR teleoperation and SAC / RLPD.

   .. grid-item-card:: AgileX Piper
      :img-top: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
      :img-alt: AgileX Piper arm
      :link: embodied/piper
      :link-type: doc

      Configure SocketCAN and verify Piper joint feedback, motion, and gripper control.

Dual-Arm & Humanoid Platforms
-----------------------------

Integrated bimanual and wheeled humanoid systems for whole-body teleoperation and real-world RL.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: XSquare Turtle2
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg
      :img-alt: XSquare Turtle2 dual-arm robot
      :link: embodied/xsquare_turtle2
      :link-type: doc

      Train a visual SAC CNN policy for button pressing on the XSquare Turtle2 dual-arm robot.

   .. grid-item-card:: Dexmal DOS-W1
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png
      :img-alt: Dexmal DOS-W1 dual-arm robot
      :link: embodied/dosw1
      :link-type: doc

      Collect leader-arm demonstrations and train a Flow Matching + SAC pick policy on DOS-W1.

   .. grid-item-card:: Robbyant R2
      :link: embodied/r2
      :link-type: doc

      Run hardware checks, whole-body PICO VR teleoperation, and joint-reach RL on the Robbyant R2 wheeled humanoid.

Low-Cost & Mobile Robots
------------------------

Desktop manipulators and mobile bases built on LeRobot-compatible servo hardware.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: SO-101
      :img-top: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Follower.webp
      :img-alt: SO-101 follower arm
      :link: embodied/so101
      :link-type: doc

      Set up and calibrate an SO-101 arm, collect leader demonstrations, run OpenPI SFT, and evaluate locally.

   .. grid-item-card:: LeKiwi
      :link: embodied/lekiwi
      :link-type: doc

      Configure wheel motors, check the front webcam, and drive the omnidirectional mobile base by keyboard.
