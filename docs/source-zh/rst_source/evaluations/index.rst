评测
======

RLinf 提供统一的具身智能评测入口，支持在仿真或真机环境中并行 rollout，并输出任务级成功率等指标。本模块介绍如何安装环境、快速跑通第一个评测，以及在各 benchmark 上完成完整评测流程。

**支持的 Benchmark**

下表列出 ``evaluations/`` 目录中已提供示例配置、且可通过 ``run_eval.sh`` 直接启动的 benchmark。

.. list-table::
   :header-rows: 1
   :widths: 18 28 34

   * - Benchmark
     - 任务 / 环境配置
     - 示例配置文件
   * - RealWorld
     - ``realworld_franka_sft_env``、``realworld_bin_relocation``
     - ``realworld/realworld_eval.yaml``、``realworld/realworld_pnp_eval.yaml``、``realworld/realworld_pnp_eval_dreamzero.yaml``
   * - BEHAVIOR-1K
     - ``behavior_r1pro``
     - ``behavior/behavior_openpi_pi05_eval.yaml``
   * - LIBERO
     - ``libero_spatial``、``libero_object``、``libero_goal``、``libero_10``
     - ``libero/libero_spatial_openpi_pi05_eval.yaml`` 等
   * - ManiSkill OOD
     - ``maniskill_ood_template`` （分布外泛化评测）
     - ``maniskill/maniskill_ood_openvlaoft_eval.yaml``
   * - PolaRiS
     - ``polaris_droid_tapeintocontainer``、``polaris_droid_movelattecup`` 等
     - ``polaris/polaris_tapeintocontainer_openpi_pi05_eval.yaml``、``polaris/polaris_movelattecup_openpi_eval.yaml``
   * - RoboTwin
     - ``robotwin_place_empty_cup``、``robotwin_adjust_bottle``、``robotwin_place_shoe``、``robotwin_click_bell``
     - ``robotwin/robotwin_place_empty_cup_openvlaoft_eval.yaml`` 等

**LIBERO 变体：** 标准 LIBERO、LIBERO-PRO、LIBERO-PLUS 均支持，通过环境变量切换（见 :doc:`guides/libero`）。

**配置回退：** 若 ``evaluations/<benchmark>/<config>.yaml`` 不存在，``run_eval.sh`` 会自动回退到 ``examples/embodiment/config/`` 下同名配置，便于复用训练配置做评测。

快速入门
--------

- :doc:`get_started/overview` — 评测架构与 ``evaluations/`` 目录结构
- :doc:`get_started/installation` — 环境安装与 benchmark 专属环境变量
- :doc:`get_started/quick_tour` — 5 分钟跑通 LIBERO Spatial 评测

Benchmark 指南
--------------

按 benchmark 组织的完整评测流程（环境准备 → 配置 → 启动 → 查看结果）：

- :doc:`guides/realworld` — Franka 真机评测与部署
- :doc:`guides/behavior` — BEHAVIOR-1K
- :doc:`guides/libero` — LIBERO / LIBERO-PRO / LIBERO-PLUS
- :doc:`guides/maniskill_ood` — ManiSkill 分布外泛化评测
- :doc:`guides/polaris` — PolaRiS 桌面操作
- :doc:`guides/robotwin` — RoboTwin 双臂操作

参考
----

- :doc:`reference/configuration` — Hydra YAML 结构与必填字段
- :doc:`reference/cli` — ``run_eval.sh`` 用法与 Hydra 覆盖
- :doc:`reference/models` — 支持的模型与示例配置
- :doc:`reference/results` — 日志、指标与视频输出

相关文档
--------------

- 各 benchmark 的环境搭建与训练示例：:doc:`../examples/simulators_index`
- 环境安装详情：:doc:`../start/installation`
- 数学推理 LLM 评测（非具身）：请参考 `LLMEvalKit <https://github.com/RLinf/LLMEvalKit>`_
- 模型专属 standalone 评测脚本（非统一入口）：``toolkits/standalone_eval_scripts/``

.. toctree::
   :hidden:
   :maxdepth: 2

   get_started/index
   guides/index
   reference/index
