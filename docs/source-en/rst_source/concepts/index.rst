Concepts
========

Use Concepts to understand how RLinf executes work, schedules components, and
organizes robot hardware. Start with the area closest to your current question;
follow its architecture links only when you need to extend or debug that layer.

Choose a Concept Area
---------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Execution
      :link: execution-model/index
      :link-type: doc

      Understand job flow, workers, clusters, channels, and collectives.

   .. grid-item-card:: Scheduling
      :link: scheduling-model/index
      :link-type: doc

      Understand placement strategies, execution modes, and replay buffers.

   .. grid-item-card:: Robotics Interface
      :link: robotics
      :link-type: doc

      Read and control a robot through named parts.

   .. grid-item-card:: Robotics Architecture
      :link: robotics_architecture
      :link-type: doc

      Trace shared connections, lifecycle, and remote placement.

   .. grid-item-card:: Real-World Tasks and Environments
      :link: realworld_envs
      :link-type: doc

      Understand how tasks, teleoperation, and wrappers fit around a robot.

.. toctree::
   :hidden:

   Execution <execution-model/index>
   Scheduling <scheduling-model/index>
   Robotics Interface <robotics>
   Robotics Architecture <robotics_architecture>
   Real-World Tasks and Environments <realworld_envs>
