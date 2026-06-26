Benchmark 指南
==============

本节按 benchmark 组织完整评测流程。每篇指南包含环境准备、示例配置、逐步命令与进阶用法。

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Benchmark
     - 说明
     - 指南
   * - RealWorld
     - Franka 真机评测与部署
     - :doc:`realworld`
   * - SO101
     - 在 SO101 六自由度机械臂上评测后 SFT 的 π₀
     - :doc:`so101_eval`
   * - BEHAVIOR-1K
     - 大规模家居场景仿真
     - :doc:`behavior`
   * - LIBERO
     - 机器人操作基准，含 Spatial / Object / Goal / Long / 90 等套件
     - :doc:`libero`
   * - ManiSkill OOD
     - ManiSkill 分布外泛化评测
     - :doc:`maniskill_ood`
   * - PolaRiS
     - 桌面操作仿真平台
     - :doc:`polaris`
   * - RoboTwin
     - 双臂操作仿真，多任务场景
     - :doc:`robotwin`

.. note::

   IsaacLab、MetaWorld 等 benchmark 尚无 ``evaluations/`` 示例配置，评测可参考 :doc:`../../examples/simulators_index` 中对应训练文档，通过配置回退机制使用 ``examples/embodiment/config/`` 下的训练 YAML 进行评测。

.. toctree::
   :hidden:
   :maxdepth: 1

   realworld
   so101_eval
   behavior
   libero
   maniskill_ood
   polaris
   robotwin
