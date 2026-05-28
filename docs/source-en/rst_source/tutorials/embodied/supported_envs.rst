Supported Environments, Robots, and Models
==========================================

This page summarizes the simulators, real-world robotic platforms, and VLA/WAM models
supported by RLinf for embodied reinforcement learning.

Supported Simulators
--------------------

RLinf supports a wide range of GPU and CPU-based simulators through standardized RL interfaces:

* **ManiSkill3** — GPU-parallelized robotic manipulation simulator with diverse tasks and object sets.
* **LIBERO** — Lifelong Robot Learning benchmark with 130 language-conditioned manipulation tasks.
* **IsaacLab** — NVIDIA Isaac Lab for high-fidelity robot simulation, including GR00T workflows.
* **MetaWorld** — Classic robotic manipulation benchmark with 50 distinct tabletop tasks.
* **CALVIN** — Long-horizon language-conditioned benchmark with 4-DOF manipulation.
* **RoboCasa** — Large-scale simulation of daily household manipulation tasks.
* **RoboTwin 2.0** — Dual-arm manipulation benchmark with 50 diverse tasks.
* **RoboVerse** — Unified simulation platform integrating multiple environments and embodiments.
* **FrankaSim** — Franka arm simulation environment with MLP/CNN policy support.
* **Behavior** — Interactive simulation benchmark with complex household activities.
* **EmbodiChain** — Gym-style environment for chain-based manipulation tasks.

Real-World Robotics
-------------------

RLinf supports real-world RL training on the following robotic platforms:

* **Franka Arm** — Franka Research 3 robotic arm with RealSense/ZED cameras and standard or Robotiq grippers.
* **XSquare Turtle2** — Dual-arm robot with SAC + CNN policy training.
* **GimArm** — 6-DOF robotic arm with SocketCAN communication and Pinocchio-based forward kinematics.
* **Dexmal DOS-W1** — Dual-arm robot with flow-matching + SAC pick tasks.
* **Franka + Dexterous Hand** — Franka arm combined with Ruiyan five-finger dexterous hand for complex manipulation.

For detailed setup guides, see :doc:`realworld_robot` and the :doc:`Franka examples <../examples/embodied/franka>`.

Supported Embodied Models
--------------------------

RLinf supports the following VLA and embodied policy models:

Vision-Language-Action (VLA) Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **OpenVLA** — 7B-parameter open-source VLA model for general-purpose robot manipulation.
* **OpenVLA-OFT** — Fine-tuned OpenVLA with LoRA adaptation for improved task-specific performance.
* **π₀ / π₀.₅** — Flow-matching based VLA models from Physical Intelligence with state-of-the-art dexterity.
* **GR00T-N1.5** — NVIDIA's large-scale VLA model for generalist robot control.
* **StarVLA** — Vision-language-action model with spatial-temporal reasoning.
* **Dexbotic** — Dexterous manipulation model based on π₀.₅ architecture.
* **Lingbot-VLA** — VLA model optimized for language-conditioned manipulation.

World Action Models (WAMs)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **OpenSora** — Video generation world model used as a simulator for RL training.
* **Wan** — Large-scale video generation model for world-model-based RL.

Policy Networks
~~~~~~~~~~~~~~~

* **MLP** — Simple multi-layer perceptron policy for state-based RL.
* **CNN** — Convolutional neural network policy for visual RL tasks.
* **ResNet** — Pretrained ResNet models for image-based reward modeling.

For model-specific training examples, see the :doc:`Embodied Examples gallery <../examples/embodied/index>`.
