FlexRLT：用冻结 VLA 做在线 RL
==========================================

用冻结 VLA 的特征，在线训练小型 actor 和 critic。

概览
----

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_flow_zh.png
   :alt: FlexRLT 的采集、可选 STEAM 路由、replay 和 actor-critic 更新流程
   :width: 100%

   VLA 保持冻结，actor 和 critic 从 replay 学习。本页配置用 STEAM 判断 actor 接管的时机，其他配置也可以使用环境 gate。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 环境
     - 任务
     - 配置
   * - ManiSkill
     - PegInsertionSideWideClearance-v1
     - ``maniskill_prefix_stage2_td3_mlp_steam``

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 观测
     - 动作
     - 奖励
   * - 主相机、腕部相机、语言指令、关节状态
     - 关节增量控制，每个 chunk 为 10 步 × 8 维
     - 稀疏插入成功奖励

preset 使用 4 张 GPU。

和 RL Token 的区别
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 选择
     - :doc:`RL Token <rlt>`
     - FlexRLT
   * - 表示来源
     - Stage 1 学到的 RL token
     - VLA 特征池化，也兼容学到的 RL token
   * - 在线 RL 前的准备
     - 联合训练 VLA 和 token transformer
     - pooling 可直接使用 base 或 SFT VLA checkpoint
   * - 组件搭配
     - RLT 示例配套的 actor-critic 组合
     - 特征模型、actor、critic 和算法可分别选择

本例使用图像 prefix 池化和 TD3。ManiSkill RLT preset 仍可使用 RL token。

安装
----

.. include:: _setup_common.rst

可以使用 ``rlinf/rlinf:agentic-rlinf0.4-maniskill_libero`` 镜像中的 OpenPI 环境（``source switch_env openpi``），也可以本地安装：

.. code:: bash

   # 国内下载可加 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

下载模型
--------

下载特征 VLA、Expert VLA、STEAM value model 和校准后的 phase head：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 资源
     - Hugging Face 仓库
   * - 特征 VLA，step 1000
     - `ChikaYokoyama/maniskill_rlt_1000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_1000_new>`_
   * - Expert VLA，step 3000
     - `ChikaYokoyama/maniskill_rlt_3000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_3000_new>`_
   * - STEAM value model，step 5000
     - `ChikaYokoyama/rlt_steam_value_model_sft_step_5000 <https://huggingface.co/ChikaYokoyama/rlt_steam_value_model_sft_step_5000>`_
   * - Phase head 与其中保存的建议值
     - `ChikaYokoyama/rlt_steam_phase_head <https://huggingface.co/ChikaYokoyama/rlt_steam_phase_head>`_

把四份权重下载到 ``data/steam_peg``，同时下载 STEAM 处理图像和语言所用的 SigLIP 与 Gemma backbone。

.. code:: bash

   CHECKPOINT_ROOT="$(pwd)/data/steam_peg"
   # 为提高国内下载速度，可以使用：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download ChikaYokoyama/maniskill_rlt_1000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/maniskill_rlt_3000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
     --local-dir "$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
     --exclude 'dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_phase_head \
     --local-dir "$CHECKPOINT_ROOT"

   hf download google/siglip-so400m-patch14-384 \
     --local-dir "$CHECKPOINT_ROOT/siglip-so400m-patch14-384"
   hf download google/gemma-3-270m \
     --local-dir "$CHECKPOINT_ROOT/gemma-3-270m"

填写模型路径
------------

编辑 ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml``，把以下字段填为绝对路径。将 ``<CHECKPOINT_ROOT>`` 展开为下载目录：

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

保留 ``openpi_data.repo_id: maniskill_peginsertionside_joint`` 和 ``norm_stats_path: null``。这样每个 OpenPI checkpoint 会读取自己目录里的 ``maniskill_peginsertionside_joint/norm_stats.json``。

然后把 STEAM checkpoint 里的 backbone 路径指到本机下载目录。编辑 ``<CHECKPOINT_ROOT>/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json``：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``config.json`` 字段
     - 本地路径
   * - ``vision_repo_id``
     - ``<CHECKPOINT_ROOT>/siglip-so400m-patch14-384``
   * - ``language_repo_id``
     - ``<CHECKPOINT_ROOT>/gemma-3-270m``

tokenizer 和 image processor 文件留在同一个 ``actor`` 目录。gate 从这里加载它们。

运行
----

在仓库根目录激活环境后启动：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

日志和 TensorBoard 文件保存在 ``logs/<timestamp>-maniskill_prefix_stage2_td3_mlp_steam/``。数据采集和预热更新完成后，STEAM 允许 actor 在关键阶段接管。expert gate 默认在 ``shadow`` 模式下记录接管建议。

预期实验结果
------------

下图比较 RL Token（TD3）与 FlexRLT 在 ManiSkill 插入任务上的训练结果，每组均有 64 个训练环境和 256 个评估环境。RL Token 使用 8 卡，FlexRLT 使用 4 卡，采用冻结的 π0.5 和 STEAM。SR 为至少成功一次的 episode 占比，对应 ``success_once``。

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

为自己的任务校准 STEAM
----------------------

任务、相机或控制行为变化后，先采集 phase 特征：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration

preset 短跑 20 步，使用环境几何 gate，expert gate 保持 shadow。它打开 ``rollout.routing_gate.actor_switch.collect_phase_features: true`` 和 ``algorithm.rlt_gate_calibration.enable: true``。输出位于 ``<run_log_dir>/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration/rank_<rank>/trace_*.pt``。

离线拟合 phase head：

.. code:: bash

   TRACE_DIR=/path/to/run_log_dir/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration
   python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
     --phase-head-output "$CHECKPOINT_ROOT/steam_phase_head.pt" \
     --yaml-output "$CHECKPOINT_ROOT/steam_gate_recommendation.yaml" --device cpu

训练 preset 的 ``phase_head_path`` 指向新文件，gate 阈值留为 ``null``，就会读取保存的建议值；YAML 仅供查看。日常训练关闭两个采集开关。

修改特征、网络或算法，见 :doc:`../../extending/flexrlt`。
