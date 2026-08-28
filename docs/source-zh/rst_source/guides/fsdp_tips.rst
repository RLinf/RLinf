FSDP 实用技巧
=============

使用这些配置提升多节点具身训练的 FSDP 效率。

Hybrid Sharding
---------------

启用 ``hybrid_shard``，在每个节点内分片模型状态，并在节点间复制：

.. code-block:: yaml

   cluster:
     num_nodes: 2

   actor:
     fsdp_config:
       strategy: fsdp
       sharding_strategy: hybrid_shard

RLinf 使用每个节点上的 actor rank 组成节点内 FSDP group。请确保所有参与节点
具有相同数量的 actor rank。按照 :doc:`多节点配置 <multi_node>` 指南操作，并在
每个节点启动 Ray 前设置 ``RLINF_NODE_RANK``。

.. warning::

   每个节点应至少使用两个 actor rank。若每个节点只有一个 actor rank，节点内
   shard group 的大小为 1；此时请使用 ``full_shard``。
