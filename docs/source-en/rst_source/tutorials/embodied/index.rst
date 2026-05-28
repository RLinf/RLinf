Embodied Intelligence
=====================

This section focuses on embodied RL training with RLinf, covering data management,
real-world robot deployment, cloud-edge training, and reward model workflows.

- :doc:`replay_buffer`
   Usage, sampling workflow, and storage practices for TrajectoryReplayBuffer.

- :doc:`data_collection`
   Data collection configuration, output formats, and usage for both simulation
   and real-robot scenarios.

- :doc:`realworld_robot`
   Connect multiple Franka robots and GPU training nodes to one Ray cluster,
   configure YAML, and launch real-world RL training.

- :doc:`cloud_edge`
   Build a cloud-edge training setup with EasyTier, connect cloud and edge nodes
   on one overlay network, and run RLinf on top of it.

- :doc:`reward_model`
   Complete reward model workflow in RLinf's simulated environments,
   including data collection, offline preprocessing, training, and RL inference.

- :doc:`reward_model_realworld`
   Real-world reward model data collection and integration with the Franka pipeline,
   covering teleoperation, live inference, and RL training with learned rewards.


.. toctree::
   :hidden:
   :maxdepth: 1

   replay_buffer
   data_collection
   realworld_robot
   cloud_edge
   reward_model
   reward_model_realworld
