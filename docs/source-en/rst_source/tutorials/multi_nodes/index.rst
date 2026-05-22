Multi-Node and Heterogeneous Clusters
======================================

This chapter explains how to run Ray clusters across multiple machines with RLinf,
how to launch real-world robot training, and how to configure heterogeneous hardware
and cloud-edge setups.

- :doc:`multi_node`
   Start a multi-machine Ray cluster, configure environment variables and code sync,
   and launch RLinf training tasks through the Ray cluster.

- :doc:`realworld_robot`
   Connect multiple Franka robots and GPU training nodes to one Ray cluster,
   configure YAML, and launch real-world RL training.

- :doc:`hetero`
   Configure heterogeneous software and hardware clusters to use different
   compute resources and devices efficiently.

- :doc:`cloud-edge`
   Build a cloud-edge training setup with EasyTier, connect cloud and edge nodes
   on one overlay network, and run RLinf on top of it.

.. toctree::
   :hidden:
   :maxdepth: 1

   multi_node
   realworld_robot
   hetero
   cloud-edge
