FlexRLT
========

基于冻结 VLA 的 prefix 特征在线训练小型 actor 与 critic，并结合 STEAM 路由控制策略接管时机。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      冻结 π₀.₅ + MLP

   .. grid-item-card:: 算法
      :text-align: center

      TD3 · AC + STEAM

   .. grid-item-card:: 任务
      :text-align: center

      ManiSkill 插孔任务

   .. grid-item-card:: 硬件
      :text-align: center

      4 × NVIDIA GPU

| **前置条件：** :doc:`环境安装 </rst_source/start/installation>` · 4 卡 GPU（建议 80 GB）· OpenPI + ManiSkill 环境。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_flow_zh.png
   :alt: FlexRLT 的采集、可选 STEAM 路由、replay 和 actor-critic 更新流程
   :width: 100%

   VLA 保持冻结，actor 和 critic 从 replay 学习。本页配置用 STEAM 判断 actor 接管的时机，其他配置也可以使用环境 gate。

任务、观测与 RL Token 对比
~~~~~~~~~~~~~~~~~~~~~~~~~~

预设 ``maniskill_prefix_stage2_td3_mlp_steam`` 运行 ManiSkill 的 ``PegInsertionSideWideClearance-v1`` 任务：输入主相机与腕部图像、语言指令和关节状态，输出 10 步 × 8 维关节增量动作块，使用稀疏插入成功奖励。

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
     - 特征模型、actor、critic 和算法可分别选择

本例使用图像 prefix 池化和 TD3。如需替换特征模型、网络头或更新规则，见 :doc:`../../extending/flexrlt`。

安装与运行
----------

.. include:: _setup_common.rst

可直接拉取 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ 上的具身预装镜像（``source switch_env openpi``），或在本地按下方步骤安装。

.. tab-set::

   .. tab-item:: ManiSkill 插孔任务（TD3 + STEAM）
      :sync: flexrlt-maniskill

      **1. 安装环境**

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env maniskill_libero
         source .venv/bin/activate

      **2. 下载权重与 Backbone**

      将特征 VLA（`maniskill_rlt_1000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_1000_new>`__）、Expert VLA（`maniskill_rlt_3000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_3000_new>`__）、STEAM value model（`rlt_steam_value_model_sft_step_5000 <https://huggingface.co/ChikaYokoyama/rlt_steam_value_model_sft_step_5000>`__）和校准后的 phase head（`rlt_steam_phase_head <https://huggingface.co/ChikaYokoyama/rlt_steam_phase_head>`__）下载到 ``./data/steam_peg``：

      .. code-block:: bash

         CHECKPOINT_ROOT="$(pwd)/data/steam_peg"
         hf download ChikaYokoyama/maniskill_rlt_1000_new \
           --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
           --exclude '*/dcp_checkpoint/*'
         hf download ChikaYokoyama/maniskill_rlt_3000_new \
           --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
           --exclude '*/dcp_checkpoint/*'
         hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
           --local-dir "$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
           --exclude 'dcp_checkpoint/*'
         hf download ChikaYokoyama/rlt_steam_phase_head --local-dir "$CHECKPOINT_ROOT"

      同时下载 STEAM 所用的 SigLIP 与 Gemma backbone：

      .. code-block:: bash

         hf download google/siglip-so400m-patch14-384 \
           --local-dir "$CHECKPOINT_ROOT/siglip-so400m-patch14-384"
         hf download google/gemma-3-270m \
           --local-dir "$CHECKPOINT_ROOT/gemma-3-270m"

      **3. 填写路径并启动**

      在 ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml`` 中填写以下 4 项绝对路径（保留 ``openpi_data.repo_id: maniskill_peginsertionside_joint`` 与 ``norm_stats_path: null``）：

      .. list-table::
         :header-rows: 1
         :widths: 48 52

         * - 字段
           - 本地路径
         * - ``rollout.prefix_feature_model.model_path``
           - ``<CHECKPOINT_ROOT>/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_1000/actor``
         * - ``rollout.expert_model.model_path``
           - ``<CHECKPOINT_ROOT>/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_3000/actor``
         * - ``rollout.routing_gate.model.model_path``
           - ``<CHECKPOINT_ROOT>/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor``
         * - ``rollout.routing_gate.actor_switch.phase_head_path``
           - ``<CHECKPOINT_ROOT>/steam_phase_head.pt``

      在 ``<CHECKPOINT_ROOT>`` 下的 ``rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json`` 中，将 ``vision_repo_id`` 和 ``language_repo_id`` 分别指向本地下载的 SigLIP 与 Gemma 目录（tokenizer 和 image processor 文件留在原 actor 目录）。随后在仓库根目录启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

      日志与 TensorBoard 文件保存在 ``logs/<timestamp>-maniskill_prefix_stage2_td3_mlp_steam/``。数据采集和预热更新完成后，STEAM 允许 actor 在关键阶段接管，expert gate 默认在 ``shadow`` 模式下记录接管建议。

      .. dropdown:: 为自己的任务校准 STEAM（可选）

         任务、相机或控制策略变化后，先采集 20 步 phase 特征：

         .. code-block:: bash

            bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration

         输出位于 ``<run_log_dir>/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration/rank_<rank>/trace_*.pt``。离线拟合 phase head：

         .. code-block:: bash

            TRACE_DIR=/path/to/run_log_dir/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration
            python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
              --phase-head-output "$CHECKPOINT_ROOT/steam_phase_head.pt" \
              --yaml-output "$CHECKPOINT_ROOT/steam_gate_recommendation.yaml" --device cpu

         将训练配置的 ``phase_head_path`` 指向新生成的 ``.pt`` 文件，gate 阈值保持 ``null`` 以加载建议值，并在日常训练中关闭两个采集开关。

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
