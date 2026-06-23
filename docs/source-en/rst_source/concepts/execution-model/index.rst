Execution Model
===============

Use these concepts when you need to understand how an RLinf job runs and how
workers exchange work.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Concept
     - What you get
   * - :doc:`RLinf Execution Flow <../execution_flow>`
     - End-to-end job flow across code, processes, and core abstractions.
   * - :doc:`M2Flow Programming Flow <../flow>`
     - The macro-to-micro model that separates logic from scheduling.
   * - :doc:`Worker and WorkerGroup <../worker>`
     - The computation unit and the handle that drives worker groups.
   * - :doc:`Cluster <../cluster>`
     - The cluster abstraction and resource model.
   * - :doc:`Channel <../channel>`
     - Asynchronous channels for inter-worker data exchange.
   * - :doc:`Collective Communication <../collective>`
     - Collective operations and asynchronous work handles.

.. toctree::
   :hidden:

   RLinf Execution Flow <../execution_flow>
   M2Flow Programming Flow <../flow>
   Worker and WorkerGroup <../worker>
   Cluster <../cluster>
   Channel <../channel>
   Collective Communication <../collective>
