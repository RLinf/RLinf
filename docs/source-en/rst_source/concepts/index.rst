Concepts
========

Use Concepts to build a mental model before you tune placement, workers,
communication, or hardware. Start with the overview for the layer you use;
follow its architecture link only when you need to extend or debug that layer.

Choose a Concept Area
---------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Execution Model
      :link: execution-model/index
      :link-type: doc

      Understand job flow, workers, clusters, channels, and collectives.

   .. grid-item-card:: Scheduling Model
      :link: scheduling-model/index
      :link-type: doc

      Understand placement strategies, execution modes, and replay buffers.

   .. grid-item-card:: Robotics Model
      :link: robotics
      :link-type: doc

      Use a robot as a tree of named observation and action parts.

   .. grid-item-card:: Robotics Architecture
      :link: robotics_architecture
      :link-type: doc

      Trace shared connections, lifecycle, and remote placement.

   .. grid-item-card:: Real-World Environment Model
      :link: realworld_envs
      :link-type: doc

      Understand how tasks, teleoperation, and wrappers fit around a robot.

.. toctree::
   :hidden:

   Execution Model <execution-model/index>
   Scheduling Model <scheduling-model/index>
   Robotics Model <robotics>
   Robotics Architecture <robotics_architecture>
   Real-World Environment Model <realworld_envs>
