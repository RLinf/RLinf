支持的模型
==========

评测配置通过 ``defaults`` 组合 ``examples/embodiment/config/model/`` 中的模型预设，并在 ``rollout.model`` 下覆盖 ``model_path`` 等字段。``evaluations/`` 中已有开箱配置的模型如下：

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - 模型
     - ``model_type``
     - 示例配置
   * - π₀ / π₀.₅（OpenPI）
     - ``openpi``
     - ``libero_spatial_openpi_pi05_eval``、``libero_10_apxinf_pi05_eval``\ （ApxInf）、``behavior_openpi_pi05_eval``、``robolab_pi05_droid_jointpos``、``robotwin_adjust_bottle_openpi_eval``、``realworld_pnp_eval`` 等
   * - OpenVLA-OFT
     - ``openvla_oft``
     - ``libero_10_openvlaoft_eval``、``robotwin_place_empty_cup_openvlaoft_eval``、``maniskill_ood_openvlaoft_eval`` 等
   * - StarVLA
     - ``starvla``
     - ``libero_spatial_starvla_eval``
   * - DreamZero
     - ``dreamzero``
     - ``libero_spatial_dreamzero_eval``、``libero_spatial_dreamzero_eval_sglang``、``realworld_pnp_eval_dreamzero``
   * - Cosmos3
     - ``cosmos3``
     - ``libero_spatial_cosmos3_eval_sglang``
   * - Psi0
     - ``psi0``
     - ``simple_psi0_eval``
   * - Evo-1
     - ``evo1``
     - ``libero_spatial_evo1_eval``
   * - FastWAM
     - ``fastwam``
     - ``libero_spatial_fastwam_eval``、``libero_object_fastwam_eval``、``libero_goal_fastwam_eval``、``libero_10_fastwam_eval``
   * - MolmoAct2
     - ``molmoact2``
     - ``libero_spatial_molmoact2_eval``、``libero_object_molmoact2_eval``、``libero_goal_molmoact2_eval``、``libero_10_molmoact2_eval``
   * - LingbotVLA 1.0
     - ``lingbotvla``
     - ``robotwin_click_bell_lingbotvla_eval``、``robotwin_place_shoe_lingbotvla_eval``
   * - LingbotVLA 2.0
     - ``lingbotvlav2``
     - ``robotwin_click_bell_lingbotvlav2_eval``
   * - π₀-FAST（LeRobot）
     - ``pi0_fast``
     - ``libero_10_pi0_fast_eval``

推理后端
--------

评测支持通过 ``rollout.rollout_backend`` 切换四种推理后端：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 后端
     - 说明
   * - ``huggingface``
     - 默认进程内 PyTorch 推理，适用于大多数具身评测配置。
   * - ``sglang``
     - 独立 SGLang 具身服务组，适用于 Cosmos3（:doc:`../guides/cosmos3_sglang`）与 DreamZero（:doc:`../guides/dreamzero_sglang`）等世界动作模型。
   * - ``apxinf``
     - 进程内 Rust/CUDA 低延迟推理后端，支持 OpenPI π₀.₅ 在 LIBERO 上评测（``libero_10_apxinf_pi05_eval``，见 :doc:`../../guides/apxinf`）。
   * - ``phyai``
     - 面向 OpenPI π₀.₅ 的 PhyAI 加速 rollout 后端（安装时传入 ``--engine phyai``，见 :doc:`../../guides/rollout-engines/phyai`）。

模型路径设置
------------

在 YAML 或命令行中将 ``rollout.model.model_path`` 设为本地模型目录或 Hugging Face ``owner/repo``\ （首次运行自动下载到 Hugging Face 缓存）。加载 RL 或 SFT 训练产出的 ``.pt`` 权重时，额外设置 ``runner.ckpt_path``。

Standalone 评测脚本
-------------------

部分模型在 ``toolkits/standalone_eval_scripts/`` 下提供遵循特定协议的独立脚本。这些脚本不走 RLinf 并行环境 worker，优先使用 ``evaluations/run_eval.sh``。
