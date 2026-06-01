Embodied Intelligence
=====================

This section focuses on embodied RL training with RLinf, covering supported environments
and models, real-world robot deployment, data management, and reward model workflows.

- :doc:`supported_envs`
   Overview of supported simulators (ManiSkill, LIBERO, IsaacLab, etc.),
   real-world robotic platforms (Franka, XSquare Turtle2, etc.), and
   VLA/WAM models (OpenVLA, π₀, GR00T, etc.).

- :doc:`realworld_robot`
   Connect multiple Franka robots and GPU training nodes to one Ray cluster,
   configure YAML, and launch real-world RL training.

- :doc:`cloud_edge`
   Build a cloud-edge training setup with EasyTier, connect cloud and edge nodes
   on one overlay network, and run RLinf on top of it.

- :doc:`replay_buffer`
   Usage, sampling workflow, and storage practices for TrajectoryReplayBuffer.

- :doc:`data_collection`
   Data collection configuration, output formats, and usage for both simulation
   and real-robot scenarios.

- :doc:`reward_model`
   Complete reward model guide covering both simulation and real-world workflows:
   data collection, preprocessing, training, RL inference, and teleoperation with
   live reward feedback.


.. toctree::
   :hidden:
   :maxdepth: 1

   supported_envs
   realworld_robot
   cloud_edge
   replay_buffer
   data_collection
   reward_model
