速查表
======

当你已经熟悉流程，只需要最短可运行命令时，使用本页。

安装
----

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero

启动 Ray
--------

单节点运行可以在本机启动 Ray。

.. code-block:: bash

   ray start --head

多节点运行时，必须在每个节点执行 ``ray start`` 之前设置 ``RLINF_NODE_RANK``。
参见 :doc:`../guides/multi_node`。

运行训练
--------

通过配置名启动具身智能训练。

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

运行评测
--------

使用统一评测入口运行具身智能 benchmark。

.. code-block:: bash

   bash evaluations/run_eval.sh libero/libero_spatial_openpi_pi05_eval

下一步
------

- :doc:`安装 <installation>` — 安装 RLinf 和可选依赖。
- :doc:`快速上手 <vla>` — 运行快速开始训练示例。
- :doc:`启动与扩展 <../guides/launch-scale/index>` — 扩展到多节点运行。
- :doc:`评测 <../evaluations/index>` — 运行独立具身智能评测。
