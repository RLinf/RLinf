支持的模型
==========

评测配置通过 ``defaults`` 引用 ``examples/embodiment/config/model/`` 下的模型 preset，并在 ``rollout.model`` 中覆盖 ``model_path`` 等字段。当前 ``evaluations/`` 中已有示例的模型如下：

.. list-table::
   :header-rows: 1
   :widths: 20 18 42

   * - 模型
     - ``model_type``
     - 示例配置
   * - π₀ / π₀.₅（OpenPI）
     - ``openpi``
     - ``libero_spatial_openpi_pi05_eval``、``libero_goal_openpi_eval``、``robotwin_adjust_bottle_openpi_eval`` 等
   * - OpenVLA-OFT
     - ``openvla_oft``
     - ``libero_10_openvlaoft_eval``、``robotwin_place_empty_cup_openvlaoft_eval``、``maniskill_ood_openvlaoft_eval`` 等
   * - StarVLA
     - ``starvla``
     - ``libero_spatial_starvla_eval``
   * - DreamZero
     - ``dreamzero``
     - ``libero_spatial_dreamzero_eval``、``realworld_pnp_eval_dreamzero``
   * - LingBotVLA
     - ``lingbotvla``
     - ``robotwin_click_bell_lingbotvla_eval``、``robotwin_place_shoe_lingbotvla_eval``

模型路径设置
------------

在 YAML 或命令行中设置 ``rollout.model.model_path`` 指向本地模型目录。若使用 RL 训练产出的 ``.pt`` 权重，可额外设置 ``runner.ckpt_path``。

Standalone 评测脚本
-------------------

部分模型还提供独立于统一入口的评测脚本，位于 ``toolkits/standalone_eval_scripts/``。这些脚本适用于特定评测协议，但不支持并行环境加速。一般推荐优先使用 ``evaluations/run_eval.sh``。
