环境安装
========

评测与训练共用同一套具身环境安装流程。在仓库根目录执行：

.. code-block:: bash

   bash requirements/install.sh embodied --model <model> --env <env>
   source .venv/bin/activate

其中 ``<model>`` 与 ``<env>`` 需与目标 benchmark 匹配。常用组合如下：

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Benchmark
     - 推荐 ``--model``
     - 推荐 ``--env``
   * - LIBERO
     - ``openpi`` / ``openvla-oft`` / ``starvla`` / ``dreamzero`` / ``molmoact2``
     - ``maniskill_libero`` 或 ``libero``
   * - LIBERO / LIBERO-Plus
     - ``fastwam``
     - ``libero`` 或 ``liberoplus``
   * - RoboTwin
     - ``openvla-oft`` / ``openpi`` / ``lingbotvla``
     - ``robotwin``
   * - BEHAVIOR-1K
     - ``openpi``
     - ``behavior``
   * - ManiSkill OOD
     - ``openvla-oft``
     - ``maniskill_libero``
   * - RealWorld
     - ``openpi`` / ``dreamzero``
     - ``franka``
   * - PolaRiS
     - ``openpi``
     - ``polaris``

更多安装选项见 :doc:`../../start/installation`。

Benchmark 专属环境变量
----------------------

按目标 benchmark 需要配置以下变量：

**RoboTwin**

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   export ROBOT_PLATFORM=ALOHA

**BEHAVIOR-1K**

设置 ``OMNIGIBSON_DATA_PATH`` 及相关 OmniGibson 路径，详见 :doc:`../../examples/embodied/behavior`。

**DreamZero**

.. code-block:: bash

   export DREAMZERO_PATH=/path/to/DreamZero

**PolaRiS**

.. code-block:: bash

   export POLARIS_DATA_PATH=/path/to/dataset/PolaRiS-Hub

``run_eval.sh`` 会自动读取上述变量并配置 ``PYTHONPATH``。

下一步
------

完成安装后，跟随 :doc:`quick_tour` 跑通第一个评测，或查看对应 benchmark 指南：:doc:`../guides/index`。
