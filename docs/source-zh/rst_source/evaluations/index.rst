模型评测
========

RLinf 提供统一的具身智能评测入口，支持在仿真或真机环境中并行 rollout，并输出任务级成功率等指标。

快速入门
--------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 快速体验（5 分钟）
      :link: get_started/quick_tour
      :link-type: doc

      端到端跑通首个 LIBERO-Spatial 评测。

   .. grid-item-card:: 安装
      :link: ../start/installation
      :link-type: doc

      安装具身仿真环境与模型依赖。

   .. grid-item-card:: 评测基准指南
      :link: guides/index
      :link-type: doc

      按 benchmark 查看完整评测步骤与常见问题。

支持的评测基准
--------------

下表汇总了 ``evaluations/`` 中可通过 ``evaluations/run_eval.sh`` 直接启动的具身评测基准。

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
     - ``maniskill_ood_template``\ （13 个分布外场景 + 分布内场景）
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
     - 120 个 DROID 关节位置控制操作任务
     - ``robolab/robolab_pi05_droid_jointpos.yaml``
     - :doc:`RoboLab <../examples/embodied/robolab>`
   * - RoboCasa365
     - ``robocasa365`` 预训练任务切片
     - ``robocasa365/robocasa365_eval_openpi.yaml``
     - :doc:`RoboCasa365 <../examples/embodied/robocasa365>`
   * - DexJoCo
     - ``dexjoco_water_plant``
     - ``dexjoco/dexjoco_openpi_pi05_eval.yaml``、``dexjoco/dexjoco_cnn_eval.yaml``
     - :doc:`DexJoCo <../examples/embodied/dexjoco>`
   * - RoboDojo
     - Isaac Sim 机械臂操作任务
     - ``robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml``
     - :doc:`RoboDojo <../examples/embodied/robodojo>`
   * - GenieSim
     - 基于仿真服务端的人形操作任务
     - ``geniesim/geniesim_pick_block_color_eval.yaml``
     - :doc:`GenieSim <../examples/embodied/geniesim>`
   * - :doc:`SIMPLE <guides/simple>`
     - 8 个 G1 人形全身遥操作任务（``G1WholebodyOpenOvenTeleop-v0`` 等）
     - ``simple/simple_psi0_eval.yaml``
     - :doc:`Psi0 <../examples/embodied/psi0>`
   * - :doc:`真机评测 <guides/realworld>`
     - ``realworld_franka_sft_env``、``realworld_bin_relocation``、双臂 Franka
     - ``realworld/realworld_pnp_eval.yaml``、``realworld/realworld_eval.yaml``、``realworld/realworld_pnp_eval_dreamzero.yaml``
     - :doc:`Franka <../examples/embodied/franka>`

SGLang 服务评测指南
-------------------

面向通过 RLinf SGLang 具身推理服务在 LIBERO 上评测的世界动作模型：

.. list-table::
   :header-rows: 1
   :widths: 22 26 32 20

   * - 模型指南
     - 评测基准
     - 示例配置
     - 训练指南
   * - :doc:`Cosmos3（SGLang） <guides/cosmos3_sglang>`
     - LIBERO-Spatial（SGLang 具身推理服务）
     - ``libero/libero_spatial_cosmos3_eval_sglang.yaml``
     - :doc:`Cosmos3 SFT <../examples/embodied/sft_cosmos3>`
   * - :doc:`DreamZero（SGLang） <guides/dreamzero_sglang>`
     - LIBERO-Spatial（SGLang 具身推理服务）
     - ``libero/libero_spatial_dreamzero_eval_sglang.yaml``
     - :doc:`DreamZero SFT <../examples/embodied/sft_dreamzero>`

.. note::

   **配置回退：** 若 ``evaluations/<benchmark>/<config>.yaml`` 不存在，``run_eval.sh`` 仅对 ``examples/embodiment/config/`` 中已定义 ``rollout.model``\ （含 ``model_type`` 与 ``num_action_chunks``）的评测就绪配置生效。若要评测只定义了 ``actor.model`` 的训练配置，请按 :doc:`reference/configuration` 将模型预设重绑定到 ``@rollout.model``。标准 LIBERO、LIBERO-PRO 与 LIBERO-PLUS 通过环境变量切换（见 :doc:`guides/libero`）。

评测架构
--------

评测 runner 通过 channel 协调 env worker 与 rollout worker，按 ``env.eval`` 配置执行并行评测，并将 ``eval/success_once``、``eval/return`` 等指标打印到终端并写入日志。

1. **加载配置：** Hydra 读取 ``evaluations/<benchmark>/`` 下的 YAML，并按 ``defaults`` 组合 ``examples/embodiment/config/`` 中的环境与模型预设。
2. **启动 worker：** 按 ``cluster.component_placement`` 在硬件上启动 env worker 与 rollout worker。
3. **并行 rollout：** env worker 重置环境并发送观测，rollout worker 返回动作块，循环执行直到回合结束。
4. **汇总指标：** 计算并记录 ``eval/success_once``、``eval/return`` 等任务级指标。

.. dropdown:: 目录结构

   所有评测入口脚本与 benchmark 配置都位于仓库根目录的 ``evaluations/`` 下：

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - 路径
        - 说明
      * - ``evaluations/eval_embodied_agent.py``
        - 具身评测主程序。
      * - ``evaluations/run_eval.sh``
        - 统一启动脚本，负责推断 benchmark、准备日志目录与初始化仿真环境。
      * - ``evaluations/behavior/``
        - BEHAVIOR-1K 家居任务评测配置。
      * - ``evaluations/geniesim/``
        - GenieSim 服务化仿真评测配置。
      * - ``evaluations/libero/``
        - LIBERO 评测配置（Spatial、Object、Goal、10、Mem、RTC、ApxInf、SGLang）与自定义初始状态。
      * - ``evaluations/maniskill/``
        - ManiSkill 分布外（OOD）泛化评测配置。
      * - ``evaluations/polaris/``
        - PolaRiS 桌面操作评测配置。
      * - ``evaluations/realworld/``
        - Franka 单臂、Dual Franka RTC 与 Dual YAM 真机评测配置。
      * - ``evaluations/robocasa365/``
        - RoboCasa365 pretrain 任务切片评测配置。
      * - ``evaluations/robodojo/``
        - RoboDojo Isaac Sim 评测配置。
      * - ``evaluations/robolab/``
        - RoboLab 120 任务 DROID joint-position 评测配置。
      * - ``evaluations/robotwin/``
        - RoboTwin 双臂操作评测配置。
      * - ``evaluations/simple/``
        - 面向 Psi0 的 SIMPLE 人形机器人全身评测配置。

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

完整的命令行参数、配置字段、支持模型与输出格式见 :doc:`评测参考 <reference/index>`。非具身的数学推理 LLM 评测请使用 `LLMEvalKit <https://github.com/RLinf/LLMEvalKit>`_。完整安装说明见 :doc:`../start/installation`\ （模型专属独立评测脚本位于 ``toolkits/standalone_eval_scripts/``）。

.. toctree::
   :hidden:
   :maxdepth: 2

   快速体验 <get_started/quick_tour>
   评测基准 <guides/index>
   评测参考 <reference/index>
