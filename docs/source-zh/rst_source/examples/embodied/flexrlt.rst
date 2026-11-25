FlexRLT
========

基于冻结 VLA 的 prefix 特征在线训练小型 Actor 与 Critic，并结合 STEAM 路由控制策略接管时机。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      TD3 · AC + STEAM

   .. grid-item-card:: 模型
      :text-align: center

      冻结 π₀.₅ + MLP

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      ManiSkill 插孔任务

   .. grid-item-card:: 训练
      :text-align: center

      在线 Off-Policy RL（4 卡）

| **你将完成：** 安装 OpenPI + ManiSkill → 下载权重与 Backbone → 启动 ``maniskill_prefix_stage2_td3_mlp_steam`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`环境安装 </rst_source/start/installation>` · 4 × NVIDIA GPU（建议 80 GB）。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_flow_zh.png
   :alt: FlexRLT 的采集、可选 STEAM 路由、replay 和 actor-critic 更新流程
   :width: 100%

   VLA 保持冻结，Actor 和 Critic 从 replay 学习。本页配置用 STEAM 判断 Actor 接管的时机，其他配置也可以使用环境 gate。

与 RL Token 对比
~~~~~~~~~~~~~~~~

预设 ``maniskill_prefix_stage2_td3_mlp_steam`` 运行 ManiSkill 的 ``PegInsertionSideWideClearance-v1`` 任务：输入主相机与腕部 RGB 图像、语言指令和关节状态，输出 10 步 × 8 维关节增量动作块，使用稀疏插入成功奖励。

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 维度
     - :doc:`RL Token <rlt>`
     - FlexRLT
   * - 表示来源
     - Stage 1 学到的 RL token
     - VLA 特征池化，也兼容学到的 RL token
   * - 在线 RL 前的准备
     - 联合训练 VLA 和 token transformer
     - 特征池化可直接从 base 或 SFT VLA checkpoint 启动
   * - 组件搭配
     - RLT 示例配套的 actor-critic 组合
     - 特征模型、Actor、Critic 和算法可分别选择

本例使用图像 prefix 池化和 TD3。如需替换特征模型、网络头或更新规则，见 :doc:`../../extending/flexrlt`。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: ManiSkill（TD3 + STEAM）
      :sync: flexrlt-maniskill

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. 下载权重与 Backbone**

      将特征 VLA、Expert VLA、STEAM value model、校准 phase head 以及 SigLIP / Gemma backbone 下载到 ``./data/steam_peg``，并将 STEAM value model 的 ``config.json`` 指向本地 backbone 目录：

      .. code-block:: bash

         CKPT_ROOT="$(pwd)/data/steam_peg"
         hf download ChikaYokoyama/maniskill_rlt_1000_new \
           --local-dir "$CKPT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
           --exclude '*/dcp_checkpoint/*'
         hf download ChikaYokoyama/maniskill_rlt_3000_new \
           --local-dir "$CKPT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
           --exclude '*/dcp_checkpoint/*'
         hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
           --local-dir "$CKPT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
           --exclude 'dcp_checkpoint/*'
         hf download ChikaYokoyama/rlt_steam_phase_head --local-dir "$CKPT_ROOT"
         hf download google/siglip-so400m-patch14-384 --local-dir "$CKPT_ROOT/siglip-so400m-patch14-384"
         hf download google/gemma-3-270m --local-dir "$CKPT_ROOT/gemma-3-270m"

         python -c "import json, pathlib; r = pathlib.Path('$CKPT_ROOT'); p = r / 'rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json'; d = json.loads(p.read_text()); d['vision_repo_id'], d['language_repo_id'] = str(r / 'siglip-so400m-patch14-384'), str(r / 'gemma-3-270m'); p.write_text(json.dumps(d, indent=2))"

      **3. 启动训练**

      .. code-block:: bash

         CKPT_ROOT="$(pwd)/data/steam_peg"
         bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam \
           rollout.prefix_feature_model.model_path="$CKPT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_1000/actor" \
           rollout.expert_model.model_path="$CKPT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_3000/actor" \
           rollout.routing_gate.model.model_path="$CKPT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
           rollout.routing_gate.actor_switch.phase_head_path="$CKPT_ROOT/steam_phase_head.pt"

      日志与 TensorBoard 文件保存在 ``logs/<timestamp>-maniskill_prefix_stage2_td3_mlp_steam/``。预热更新完成后，STEAM 在关键阶段启用 Actor 控制，Expert gate 默认在 ``shadow`` 模式下记录接管建议。

      .. dropdown:: 为其他任务校准 STEAM（可选）

         任务、相机或控制策略变化后，采集 20 步 phase 特征并离线拟合 phase head：

         .. code-block:: bash

            bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration
            TRACE_DIR=$(ls -d "$(pwd)"/logs/*-maniskill_prefix_stage2_td3_mlp_steam_calibration/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration | tail -n 1)
            python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
              --phase-head-output "$(pwd)/data/steam_peg/steam_phase_head.pt" \
              --yaml-output "$(pwd)/data/steam_peg/steam_gate_recommendation.yaml" --device cpu

         将新生成的 ``steam_phase_head.pt`` 传给 ``phase_head_path``，并保持 gate 阈值为 ``null`` 以加载建议值。

可视化与结果
------------

运行 ``tensorboard --logdir ./logs`` 查看 ``env/success_once`` 与 ``eval/success_once``，指标说明与日志后端配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

下图比较 RL Token（TD3，8 卡）与 FlexRLT（4 卡，冻结 π₀.₅ + STEAM）在 ManiSkill 插孔任务上的结果，每组使用 64 个训练环境与 256 个评估环境。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_maniskill_sr_zh.png
   :alt: RL Token 与 FlexRLT 在 ManiSkill 插入任务上的训练和评估成功率
   :width: 100%

   左图为训练 SR 原始值和最近 20 步均值，右图为各检查点的评估 SR。

.. list-table:: ManiSkill 插入任务成功率
   :header-rows: 1

   * - 方法
     - GPU 数
     - Train，第 235–254 步均值
     - Eval，第 250 步
     - 截至第 250 步的最高 Eval
   * - RL Token（TD3）
     - 8
     - 57.19%
     - 57.03%
     - 59.38%
   * - FlexRLT，min-Q（运行 1）
     - 4
     - 56.09%
     - 61.72%
     - 61.72%
   * - FlexRLT，min-Q（运行 2）
     - 4
     - 57.34%
     - 62.89%
     - 62.89%
   * - FlexRLT，TD3（Q1）
     - 4
     - 59.61%
     - 56.25%
     - 60.16%
