FSDP Practical Tips
===================

Use these settings to improve FSDP efficiency for multi-node embodied training.

Hybrid Sharding
---------------

On a single node, sharding a model over every rank is the cheapest way to fit
it in memory. Across nodes the all-gather that rebuilds each parameter has to
cross the slower inter-node link on every forward and backward pass. Hybrid
sharding trades memory for that bandwidth: it shards the model state within each
node and replicates it across nodes, so parameter traffic stays on NVLink and
only gradients cross the network.

Enable it with ``sharding_strategy``:

.. code-block:: yaml

   cluster:
     num_nodes: 2

   actor:
     fsdp_config:
       strategy: fsdp
       sharding_strategy: hybrid_shard

The setting works the same for ``strategy: fsdp2``, which reads the sharding
layout from the device mesh rather than from an FSDP sharding-strategy enum.

RLinf sizes the intra-node shard group from the ranks the component has on each
node, so every participating node must host the same number of ranks. A
placement that gives nodes unequal shares fails at startup with an error listing
what each rank reported. Follow the :doc:`multi-node setup <multi_node>` guide
and set ``RLINF_NODE_RANK`` before starting Ray on each node.

.. note::

   Hybrid sharding pays off from two ranks per node upwards. With a single rank
   per node the intra-node shard group holds one rank, nothing is sharded, and
   RLinf logs a warning at startup — use ``full_shard`` for that topology.
