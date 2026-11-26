Workflows
=========

Pick these guides when setting up hardware workflows shared across every supported robot: connecting robot control and GPU nodes, configuring teleoperation devices, and recording demonstration datasets.

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Multi-Node Robot Launch
      :link: ../guides/realworld_robot
      :link-type: doc

      Connect a robot control machine and GPU training machine in one Ray cluster.

   .. grid-item-card:: Teleoperation
      :link: ../guides/teleoperation
      :link-type: doc

      Configure SpaceMouse, GELLO, leader arms, or VR controllers for human control and intervention.

   .. grid-item-card:: Data Collection
      :link: ../guides/data_collection
      :link-type: doc

      Record replay buffers and LeRobot shards during teleoperation or policy rollouts.

.. toctree::
   :hidden:

   Multi-Node Robot Launch <../guides/realworld_robot>
   Teleoperation <../guides/teleoperation>
   Data Collection <../guides/data_collection>
