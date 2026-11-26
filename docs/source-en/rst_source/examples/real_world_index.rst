Real-World Robots
=================

Use this section when your starting point is physical robot hardware. Each robot page covers the steps supported for that platform, from hardware checks and teleoperation to SFT, deployment, and online RL.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld_flow.svg
   :align: center
   :width: 100%
   :alt: RLinf real-world robot workflow

   End-to-end real-world workflow: hardware setup, teleoperation and data collection, SFT, policy deployment, and online RL.

Shared Real-World Guides
------------------------

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: Cluster Setup

      Connect robot control nodes and GPU training nodes in one Ray cluster: :doc:`Two-Node Robot + GPU Ray Setup <../guides/realworld_robot>`.

   .. grid-item-card:: Teleoperation

      Configure SpaceMouse, GELLO, leader arms, or VR rigs: :doc:`Teleoperation Devices <../guides/teleoperation>` (for PICO VR software setup, see :ref:`PICO VR software setup <pico-software-setup>`).

   .. grid-item-card:: Data & Replay

      Record replay buffers and LeRobot shards during teleoperation or rollouts: :doc:`Data Collection & Replay <../guides/data_collection>`.

Robot Capability Matrix
-----------------------

Compare what each robot guide covers before choosing a platform.

.. list-table::
   :header-rows: 1
   :widths: 18 8 9 13 12 18 22

   * - Robot
     - Setup
     - Teleop
     - Data Collection
     - SFT / Deploy
     - Online RL
     - Guide
   * - AgileX Piper
     - ✓
     - —
     - —
     - —
     - —
     - :doc:`AgileX Piper Guide <embodied/piper>`
   * - ARX X5
     - ✓
     - ✓
     - ✓
     - —
     - —
     - :doc:`ARX X5 Guide <embodied/arx5>`
   * - Dexmal DOS-W1
     - ✓
     - ✓
     - ✓
     - —
     - ✓
     - :doc:`Dexmal DOS-W1 Guide <embodied/dosw1>`
   * - Franka (Dual-Arm)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - :doc:`Franka (Dual-Arm, 3 guides) <embodied/dual_franka_index>`
   * - Franka (Single-Arm)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - :doc:`Franka (Single-Arm, 9 guides) <embodied/franka_index>`
   * - GimArm
     - ✓
     - ✓
     - ✓
     - —
     - Env only (custom config)
     - :doc:`GimArm Guide <embodied/gim_arm>`
   * - LeKiwi
     - ✓
     - ✓
     - —
     - —
     - —
     - :doc:`LeKiwi Guide <embodied/lekiwi>`
   * - Robbyant R2
     - ✓
     - ✓
     - ✓
     - —
     - Env only (custom config)
     - :doc:`Robbyant R2 Guide <embodied/r2>`
   * - SO-101
     - ✓
     - ✓
     - ✓
     - ✓
     - —
     - :doc:`SO-101 Guide <embodied/so101>`
   * - XSquare Turtle2
     - ✓
     - —
     - —
     - —
     - ✓
     - :doc:`XSquare Turtle2 Guide <embodied/xsquare_turtle2>`
   * - YAM
     - ✓
     - ✓
     - ✓
     - ✓
     - —
     - :doc:`YAM (2 guides) <embodied/yam_index>`

Industrial & Research Arms
--------------------------

Single-arm and dual-arm manipulators for hardware setup, teleoperation, data collection, SFT, deployment, and online RL.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: AgileX Piper
      :img-top: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
      :img-alt: AgileX Piper arm
      :link: embodied/piper
      :link-type: doc

      Configure SocketCAN and verify Piper joint feedback, motion, and gripper control.

   .. grid-item-card:: ARX X5
      :img-top: https://arx-global.us/assets/img/x5.webp
      :img-alt: ARX X5 robot arms
      :link: embodied/arx5
      :link-type: doc

      Collect dual-arm demonstrations with PICO controllers and three RGB cameras in LeRobot format.

   .. grid-item-card:: Franka (Dual-Arm)
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
      :img-alt: Franka (Dual-Arm)
      :link: embodied/dual_franka_index
      :link-type: doc

      GELLO and PICO data collection, OpenPI π₀.₅ SFT, policy deployment, and online HG-DAgger.

   .. grid-item-card:: Franka (Single-Arm)
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
      :img-alt: Franka (Single-Arm)
      :link: embodied/franka_index
      :link-type: doc

      Real-world RL, teleoperation, data collection, reward models, SFT, deployment, and hardware extensions.

   .. grid-item-card:: GimArm
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png
      :img-alt: GimArm 6-DoF arm
      :link: embodied/gim_arm
      :link-type: doc

      Train a 6-DoF GimArm peg-insertion policy over SocketCAN with VR teleoperation and SAC / RLPD.

   .. grid-item-card:: YAM
      :link: embodied/yam_index
      :link-type: doc

      Collect dual-arm PICO demonstrations, fine-tune OpenPI π₀.₅, and evaluate checkpoints on both arms.

Dual-Arm & Humanoid Platforms
-----------------------------

Integrated bimanual and wheeled humanoid systems for whole-body teleoperation and real-world RL.

.. grid:: 1 2 3 3
   :gutter: 3

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

   .. grid-item-card:: XSquare Turtle2
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg
      :img-alt: XSquare Turtle2 dual-arm robot
      :link: embodied/xsquare_turtle2
      :link-type: doc

      Train a visual SAC CNN policy for button pressing on the XSquare Turtle2 dual-arm robot.

Low-Cost & Mobile Robots
------------------------

Desktop manipulators and mobile bases built on LeRobot-compatible servo hardware.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: LeKiwi
      :link: embodied/lekiwi
      :link-type: doc

      Configure wheel motors, check the front webcam, and drive the omnidirectional mobile base by keyboard.

   .. grid-item-card:: SO-101
      :img-top: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Follower.webp
      :img-alt: SO-101 follower arm
      :link: embodied/so101
      :link-type: doc

      Set up and calibrate an SO-101 arm, collect leader demonstrations, run OpenPI SFT, and evaluate locally.
