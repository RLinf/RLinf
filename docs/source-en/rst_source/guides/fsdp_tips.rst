FSDP Practical Tips
===================

Use these settings to improve FSDP efficiency for multi-node embodied training.

Hybrid Sharding
---------------

Enable ``hybrid_shard`` to shard model state within each node and replicate it
across nodes:

.. code-block:: yaml

   cluster:
     num_nodes: 2

   actor:
     fsdp_config:
       strategy: fsdp
       sharding_strategy: hybrid_shard

RLinf uses the actor ranks on each node as the intra-node FSDP group. Keep the
same number of actor ranks on every participating node. Follow the
:doc:`multi-node setup <multi_node>` guide and set ``RLINF_NODE_RANK`` before
starting Ray on each node.

.. warning::

   Use at least two actor ranks per node. With one actor rank per node, the
   intra-node shard group has size one; use ``full_shard`` for that topology.
