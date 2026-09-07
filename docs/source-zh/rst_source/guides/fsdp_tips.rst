FSDP 实用技巧
=============

使用这些配置提升多节点具身训练的 FSDP 效率。

Hybrid Sharding
---------------

在单节点上，把模型分片到每一个 rank 是最省显存的做法。跨节点时，每次前向和反向都要通过较慢的节点间链路重新 all-gather 出完整参数。Hybrid sharding 用显存换带宽：在节点内分片模型状态，在节点间复制，于是参数通信留在 NVLink 上，只有梯度需要跨网络。

通过 ``sharding_strategy`` 启用：

.. code-block:: yaml

   cluster:
     num_nodes: 2

   actor:
     fsdp_config:
       strategy: fsdp
       sharding_strategy: hybrid_shard

该配置对 ``strategy: fsdp2`` 同样有效：FSDP2 从 device mesh 读取分片布局，而不是从 FSDP 的 sharding strategy 枚举读取。

RLinf 根据该组件在每个节点上的 rank 数量决定节点内 shard group 的大小，因此所有参与的节点必须有相同的 rank 数量。若 placement 给各节点分配的份额不一致，启动时会报错，并列出各个 rank 上报的数值。请按照 :doc:`多节点配置 <multi_node>` 指南操作，并在每个节点启动 Ray 前设置 ``RLINF_NODE_RANK``。

.. note::

   Hybrid sharding 从每节点两个 rank 起才有收益。若每个节点只有一个 rank，节点内 shard group 的大小为 1，实际不会发生任何分片，RLinf 会在启动时输出警告；此时请使用 ``full_shard``。
