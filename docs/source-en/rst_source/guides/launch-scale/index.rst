Scale to a Cluster
==================

Pick these guides when scaling a working single-node run across multiple machines, heterogeneous CPU/GPU nodes, cloud-edge links, 5D model parallelism, or dynamic cluster scheduling.

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Multi-Node Training
      :link: ../multi_node
      :link-type: doc

      Launch a multi-node Ray cluster and configure node ranks and network devices.

   .. grid-item-card:: Heterogeneous Clusters
      :link: ../hetero
      :link-type: doc

      Combine different GPU types or CPU simulator nodes in one cluster.

   .. grid-item-card:: Cloud-Edge Training
      :link: ../cloud_edge
      :link-type: doc

      Connect cloud GPU training nodes to edge machines behind NAT via overlay networking.

   .. grid-item-card:: Auto Placement
      :link: ../auto_placement
      :link-type: doc

      Let RLinf search for an optimal hardware placement across actor, rollout, and env workers.

   .. grid-item-card:: 5D Parallelism
      :link: ../5D
      :link-type: doc

      Combine tensor, pipeline, context, data, and expert parallelism for large LLMs.

   .. grid-item-card:: Dynamic Scheduling
      :link: ../dynamic_scheduling
      :link-type: doc

      Handle elastic scaling and worker rescheduling across dynamic cluster resources.

.. toctree::
   :hidden:

   Multi-Node Training <../multi_node>
   Heterogeneous Clusters <../hetero>
   Cloud-Edge Training <../cloud_edge>
   Auto Placement <../auto_placement>
   5D Parallelism <../5D>
   Dynamic Scheduling <../dynamic_scheduling>
