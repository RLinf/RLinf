评测
====

RLinf 提供统一的具身智能评测入口，支持在仿真或真机环境中并行 rollout，并输出任务级成功率等指标。

快速入门
--------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 快速上手（5 分钟）
      :link: get_started/quick_tour
      :link-type: doc

      端到端跑通首个 LIBERO-Spatial 评测。

   .. grid-item-card:: 安装
      :link: get_started/installation
      :link-type: doc

      配置依赖环境与各 benchmark 专属环境变量。

   .. grid-item-card:: 概览
      :link: get_started/overview
      :link-type: doc

      了解评测架构与 ``evaluations/`` 目录结构。

支持的评测基准
--------------

下表汇总了 ``evaluations/`` 中可通过 ``evaluations/run_eval.sh`` 直接启动的 benchmark 与对应的 SGLang 评测指南。

.. list-table::
   :header-rows: 1
   :widths: 18 28 34 20

   * - 评测基准
     - 任务 / 预设
     - 示例配置
     - 训练指南
   * - :doc:`LIBERO <guides/libero>`
     - ``libero_spatial``、``libero_object``、``libero_goal``、``libero_10``\ （含 Pro / Plus）
     - ``libero/libero_spatial_openpi_pi05_eval.yaml`` 等
     - :doc:`LIBERO <../examples/embodied/libero>`
   * - :doc:`ManiSkill OOD <guides/maniskill_ood>`
     - ``maniskill_ood_template``\ （13 个 OOD 场景 + 训练分布内）
     - ``maniskill/maniskill_ood_openvlaoft_eval.yaml``
     - :doc:`ManiSkill <../examples/embodied/maniskill>`
   * - :doc:`BEHAVIOR-1K <guides/behavior>`
     - ``behavior_r1pro``\ （50 个家居任务）
     - ``behavior/behavior_openpi_pi05_eval.yaml``
     - :doc:`BEHAVIOR-1K <../examples/embodied/behavior>`
   * - :doc:`RoboTwin <guides/robotwin>`
     - ``robotwin_place_empty_cup``、``robotwin_adjust_bottle``、``robotwin_place_shoe``、``robotwin_click_bell``
     - ``robotwin/robotwin_place_empty_cup_openvlaoft_eval.yaml`` 等
     - :doc:`RoboTwin <../examples/embodied/robotwin>`
   * - :doc:`PolaRiS <guides/polaris>`
     - ``polaris_droid_tapeintocontainer``、``polaris_droid_movelattecup`` 等
     - ``polaris/polaris_tapeintocontainer_openpi_pi05_eval.yaml``、``polaris/polaris_movelattecup_openpi_eval.yaml``
     - :doc:`PolaRiS <../examples/embodied/polaris>`
   * - :doc:`RoboLab <guides/robolab>`
     - 120 个 DROID joint-position 操作任务
     - ``robolab/robolab_pi05_droid_jointpos.yaml``
     - :doc:`RoboLab <../examples/embodied/robolab>`
   * - RoboCasa365
     - ``robocasa365`` pretrain 任务切片
     - ``robocasa365/robocasa365_eval_openpi.yaml``
     - :doc:`RoboCasa365 <../examples/embodied/robocasa365>`
   * - :doc:`SIMPLE <guides/simple>`
     - 8 个 G1 人形全身 Teleop 任务（``G1WholebodyOpenOvenTeleop-v0`` 等）
     - ``simple/simple_psi0_eval.yaml``
     - :doc:`Psi0 <../examples/embodied/psi0>`
   * - :doc:`RealWorld <guides/realworld>`
     - ``realworld_franka_sft_env``、``realworld_bin_relocation``、Dual Franka
     - ``realworld/realworld_pnp_eval.yaml``、``realworld/realworld_eval.yaml``、``realworld/realworld_pnp_eval_dreamzero.yaml``
     - :doc:`Franka <../examples/embodied/franka>`
   * - :doc:`Cosmos3 (SGLang) <guides/cosmos3_sglang>`
     - LIBERO-Spatial（SGLang embodied server）
     - ``libero/libero_spatial_cosmos3_eval_sglang.yaml``
     - :doc:`Cosmos3 SFT <../examples/embodied/sft_cosmos3>`
   * - :doc:`DreamZero (SGLang) <guides/dreamzero_sglang>`
     - LIBERO-Spatial（SGLang embodied server）
     - ``libero/libero_spatial_dreamzero_eval_sglang.yaml``
     - :doc:`DreamZero SFT <../examples/embodied/sft_dreamzero>`

.. note::

   **配置回退：** 若 ``evaluations/<benchmark>/<config>.yaml`` 不存在，``run_eval.sh`` 会自动回退到 ``examples/embodiment/config/`` 下同名配置，便于复用训练配置做评测。标准 LIBERO、LIBERO-PRO 与 LIBERO-PLUS 均支持，通过环境变量切换（见 :doc:`guides/libero`）。

参考
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 配置
      :link: reference/configuration
      :link-type: doc

      Hydra YAML 结构与必填字段。

   .. grid-item-card:: 命令行
      :link: reference/cli
      :link-type: doc

      ``run_eval.sh`` 用法与 Hydra 覆盖。

   .. grid-item-card:: 模型
      :link: reference/models
      :link-type: doc

      支持的模型与示例配置。

   .. grid-item-card:: 结果
      :link: reference/results
      :link-type: doc

      日志、指标与视频输出。

非具身的数学推理 LLM 评测请使用 `LLMEvalKit <https://github.com/RLinf/LLMEvalKit>`_。完整安装说明见 :doc:`../start/installation`\ （模型专属 standalone 评测脚本位于 ``toolkits/standalone_eval_scripts/``）。
