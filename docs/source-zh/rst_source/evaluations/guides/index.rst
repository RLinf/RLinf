评测基准
========

按 benchmark 选择对应指南，查看完整的依赖安装、权重下载、启动命令与常见问题处理。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: BEHAVIOR-1K
      :link: behavior
      :link-type: doc

      基于 OmniGibson 与 Isaac Sim 4.5 的 50 个家居任务评测（OpenPI π₀.₅）。

   .. grid-item-card:: LIBERO
      :link: libero
      :link-type: doc

      Spatial、Object、Goal、10 与 90 套件，支持 LIBERO-PRO 与 LIBERO-PLUS。

   .. grid-item-card:: ManiSkill OOD
      :link: maniskill_ood
      :link-type: doc

      13 个分布外场景与分布内划分的批量评测。

   .. grid-item-card:: PolaRiS
      :link: polaris
      :link-type: doc

      基于 Isaac Sim 5.1.0 的 DROID 桌面操作评测（OpenPI π₀ / π₀.₅）。

   .. grid-item-card:: RealWorld
      :link: realworld
      :link-type: doc

      Franka PnP、自定义 ``FrankaEnv-v1`` 任务与 Dual Franka RTC 真机评测。

   .. grid-item-card:: RoboLab
      :link: robolab
      :link-type: doc

      使用 π₀.₅ 运行单任务或全部 120 个 DROID joint-position 操作任务。

   .. grid-item-card:: RoboTwin
      :link: robotwin
      :link-type: doc

      基于 OpenVLA-OFT、OpenPI 与 LingbotVLA 的双臂操作评测。

   .. grid-item-card:: SIMPLE
      :link: simple
      :link-type: doc

      使用 Psi0 在 3 档 DR 上评测 8 个 G1 人形机器人全身 Teleop 任务。

   .. grid-item-card:: SuperMimic
      :link: supermimic
      :link-type: doc

      Unitree G1 动作跟踪与站立推力恢复评测。

SGLang 服务评测
---------------

面向通过 RLinf SGLang 具身后端在 LIBERO 上部署的世界动作模型：

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Cosmos3（SGLang）
      :link: cosmos3_sglang
      :link-type: doc

      使用 SGLang diffusers 后端在 LIBERO-Spatial 上评测 Cosmos3。

   .. grid-item-card:: DreamZero（SGLang）
      :link: dreamzero_sglang
      :link-type: doc

      使用 SGLang 后端在 LIBERO-Spatial 上评测 DreamZero WAN2.2-5B。

.. note::

   GenieSim、RoboCasa365 与 RoboDojo 的配置也位于 ``evaluations/`` 下（见 :doc:`../index`）。对于 IsaacLab、MetaWorld 等尚无独立 ``evaluations/`` 目录的模拟器，:doc:`配置回退 <../reference/cli>` 仅支持 ``examples/embodiment/config/`` 中已定义 ``rollout.model``\ （含 ``model_type`` 与 ``num_action_chunks``）的评测就绪配置；若要评测 :doc:`../../examples/simulators_index` 中的纯训练配置，请按 :doc:`../reference/configuration` 改写为评测 YAML。

.. toctree::
   :hidden:
   :maxdepth: 2

   BEHAVIOR-1K <behavior>
   LIBERO <libero>
   ManiSkill OOD <maniskill_ood>
   PolaRiS <polaris>
   RealWorld <realworld>
   RoboLab <robolab>
   RoboTwin <robotwin>
   SIMPLE <simple>
   SuperMimic <supermimic>
   Cosmos3（SGLang） <cosmos3_sglang>
   DreamZero（SGLang） <dreamzero_sglang>
