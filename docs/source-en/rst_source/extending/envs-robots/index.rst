Add an Environment or Robot
===========================

Start at the smallest layer that fits: if RLinf already supports your physical robot, add a real-world task without writing robotics drivers; for a new simulator or new hardware, follow the corresponding integration guide below.

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Simulator Environment
      :link: ../new_env
      :link-type: doc

      Wrap a simulator backend, register its environment type, and map action formats.

   .. grid-item-card:: Real-World Robot Hardware
      :link: ../new_robot
      :link-type: doc

      Verify one local hardware part, compose a robot tree, and configure remote placement.

   .. grid-item-card:: Real-World Task
      :link: ../new_task
      :link-type: doc

      Define a new manipulation task on physical hardware RLinf already supports.

.. toctree::
   :hidden:

   Simulator Environment <../new_env>
   Real-World Robot Hardware <../new_robot>
   Real-World Task <../new_task>
