训练 Pipeline 模式
==================

在具身 FSDP 训练中，通过流水线并行让环境交互、rollout 与 actor 更新重叠，减少 GPU 空等。本指南介绍两个互补开关：按阶段切分 rollout（``rollout.pipeline_stage_num > 1``）以及在 rollout 期间流式发送 micro-batch 给 actor（``runner.use_training_pipeline: True``）。

何时启用
--------

模拟器步进（``env``）与策略推理（``rollout``）运行在不同设备或同机并行、且耗时接近时，将 ``rollout.pipeline_stage_num`` 设为 ``2`` 或更大值：rollout 处理第 ``k`` 批环境的同时，模拟器即可步进第 ``k + 1`` 批环境。

出现以下情况时，进一步开启 ``runner.use_training_pipeline: True``：

- rollout 观测包含大尺寸图像张量，整轮结束后一次性汇总传输会造成明显的通信尖峰。
- env worker 与 actor worker 分布在多卡或多节点（甚至广域网链路）上，需要把轨迹打包为 micro-batch 流式发送，让通信与 actor 训练和 rollout 重叠。

前提条件
--------

- 已有可运行的具身 FSDP 训练配置，且采用 PPO 或 GRPO 类 actor 更新（见 :doc:`具身训练配置 <embodiment_config>`）。
- ``env.train.total_num_envs``\ （开启评测时还包括 ``env.eval.total_num_envs``）必须能被 ``env_world_size * rollout.pipeline_stage_num`` 整除。

最小配置
--------

在具身 FSDP 训练 YAML 中加入以下配置项：

.. code-block:: yaml

   runner:
     use_training_pipeline: True

   rollout:
     pipeline_stage_num: 2

   algorithm:
     adv_type: gae
     normalize_advantages: True
     recompute_logprobs: False

- ``rollout.pipeline_stage_num``：每个 env worker 将并行环境切分的流水阶段数（建议从 ``2`` 起步；不支持多阶段流水的模型保持 ``1``）。
- ``runner.use_training_pipeline``：在 env worker 端将完成的轨迹打包成 micro-batch，并在 rollout 尚未结束时流式发往 actor worker。
- ``algorithm.normalize_advantages``：设为 ``True`` 时，在发往同一 actor rank 的 env worker 间先聚合归一化 advantage，再流式发送。

兼容性与限制
------------

.. warning::

   配置校验会在启动阶段拒绝不兼容组合。开启 ``runner.use_training_pipeline: True`` 前请先核对以下限制。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 功能 / 配置项
     - 与训练流水线模式的兼容性
   * - 算法与 loss
     - ``algorithm.adv_type`` 必须设为 ``gae``，且 ``algorithm.recompute_logprobs`` 必须为 ``False``。仅支持 PPO 与 GRPO 类具身 FSDP actor loss，不支持 ``embodied_sac``、``rlt_ac``、``rlt_td3``、``embodied_dagger``、``embodied_nft``、``embodied_refl`` 和 ``embodied_art``。
   * - 模型约束
     - 部分模型（如 ``molmoact2`` 和 Psi0 SIMPLE）要求 ``rollout.pipeline_stage_num: 1``。
   * - :doc:`异构环境 <hetero_envs>`
     - ``runner.use_training_pipeline: True`` 要求单一训练环境配置，暂不支持与异构训练环境同时开启。
   * - :doc:`观测压缩 <obs_compression>`
     - 完全兼容，带宽受限或跨节点部署时建议同时开启 ``env.obs_compression.enable: True``，在压缩 env 到 rollout 图像传输的同时，用流水线重叠 rollout 到 actor 的训练传输。
   * - :doc:`环境解耦模式 <env_decoupled_mode>`
     - ``rollout.pipeline_stage_num > 1`` 属于固定同步阶段切分，适合各环境步时较均匀的场景；如果不同环境的模拟耗时差异很大，优先选用 :doc:`env_decoupled_mode` 而不是继续增大 ``pipeline_stage_num``。

验证吞吐提升
------------

先用基线配置（``pipeline_stage_num: 1``、``use_training_pipeline: False``）试跑 10–20 步，再开启流水线对比 TensorBoard 指标（指标含义见 :doc:`训练指标 </rst_source/reference/metrics>`）：

1. 对比 ``time/step``：rollout 与 actor 有效重叠后，单步总耗时应明显下降。
2. 查看 ``time/env``、``time/rollout`` 和 ``time/actor``：若某一阶段占比超过 80%，应先调整 GPU 放置（见 :doc:`Placement 策略 <../concepts/placement>`），再微调 ``pipeline_stage_num``。
3. 核对 ``env/success_once`` 与 reward 曲线是否与基线一致，确认流式发送未影响训练稳定性。
