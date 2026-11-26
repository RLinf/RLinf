LingbotVLA 2.0 强化学习
=======================

.. figure:: https://raw.githubusercontent.com/robbyant/lingbot-vla-v2/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/assets/lingbot_vla2_framework.png
   :align: center
   :width: 90%

   LingbotVLA 2.0 框架（图片来源：`Robbyant <https://github.com/robbyant/lingbot-vla-v2>`__）。

用 PPO 在 RoboTwin 2.0 上微调 `LingbotVLA 2.0 <https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin>`__ 的动作专家。RLinf **原生**\ 加载官方发布的 RoboTwin checkpoint，并以它作为训练起点。LingbotVLA 2.0 是基于 Qwen3-VL 底座的视觉-语言-动作模型：它的混合专家（MoE）动作专家通过 flow matching 在 55 维的 latent 动作空间中生成 50 步动作块，再由机器人配置映射为 RoboTwin 的 14 维关节指令。PPO 只更新动作专家，Qwen3-VL 底座和动作专家的 MoE 路由器保持冻结。

概览
----------------------------------------

用 PPO 训练 LingbotVLA 2.0，完成 RoboTwin 2.0 的 ``click_bell`` 任务。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      RoboTwin 2.0

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      Click Bell

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4 GPU（48 GB）

| **你将完成：** 原生安装 → 克隆 RoboTwin 与资产 → 下载检查点 → 填写模型路径 → PPO → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboTwin 仓库与资产 · LingbotVLA 2.0 与 Qwen3-VL 检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - RoboTwin
     - Click Bell
     - ``robotwin_click_bell_ppo_lingbotvlav2``
     - 从官方 RoboTwin checkpoint 出发，用 PPO 训练动作专家。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 头部相机、左右腕部相机，以及 14 维关节状态。
   * - Action
     - 每步 14 维关节指令，由 50 步、每步 55 维的 latent 动作块解码得到。
   * - Reward
     - RoboTwin 任务成功信号。
   * - Prompt
     - RoboTwin episode 的自然语言任务指令。

安装与运行
----------

.. include:: _setup_common.rst

LingbotVLA 1.0 与 2.0 安装的 Python 包都叫 ``lingbotvla``，需要各自使用独立的虚拟环境。安装脚本默认将 `RLinf 维护的 LingbotVLA 2.0 fork <https://github.com/RLinf/lingbot-vla-v2/tree/rlinf>`__ 的 ``rlinf`` 分支克隆到 ``.venv/lingbot-vla-v2``；如需其他位置，在安装前设置 ``LINGBOT_VLA_V2_PATH``。

.. tab-set::

   .. tab-item:: RoboTwin

      **1. 安装环境**

      安装 LingbotVLA 2.0 环境、克隆 RoboTwin 仓库并下载资产：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
         (cd RoboTwin && bash script/_download_assets.sh)

      **2. 下载资源与模型**

      下载 LingbotVLA 2.0 的 RoboTwin checkpoint 与 Qwen3-VL 底座模型：

      .. code-block:: bash

         hf download robbyant/lingbot-vla-v2-6b-robotwin --local-dir ./lingbot-vla-v2-6b-robotwin
         hf download Qwen/Qwen3-VL-4B-Instruct --local-dir ./Qwen3-VL-4B-Instruct

      **3. 启动训练**

      LingbotVLA 2.0 配置文件一览：

      .. list-table::
         :header-rows: 1
         :widths: 32 68

         * - 用途
           - 配置文件
         * - PPO（Click Bell）
           - ``examples/embodiment/config/robotwin_click_bell_ppo_lingbotvlav2.yaml``
         * - 模型默认配置（与评测共用）
           - ``examples/embodiment/config/model/lingbotvlav2.yaml``

      在 ``examples/embodiment/config/model/lingbotvlav2.yaml`` 中填写空白的路径字段（训练与独立评测共用，任何路径留空时启动会报错并提示字段名）：

      .. code-block:: yaml

         model_path: ./lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
         tokenizer_path: ./Qwen3-VL-4B-Instruct
         lingbotvlav2:
           training_config_path: ./lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml
           robot_config_path: ./.venv/lingbot-vla-v2/configs/robot_configs/robotwin.yaml
           stats_path: ./.venv/lingbot-vla-v2/assets/norm_stats/robotwin.json

      先导出 RoboTwin 路径，再启动 PPO 训练：

      .. code-block:: bash

         export ROBOTWIN_PATH=$PWD/RoboTwin
         export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH
         bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2

      .. dropdown:: 进阶配置（可选）

         - **断点续训**：在训练配置中将 ``runner.resume_dir`` 设为已保存的 checkpoint 目录。
         - **Flow-SDE PPO 与价值头预热**：rollout 在每个动作块中随机选一个去噪步按 flow-SDE 采样，其余步沿 ODE 推进。PPO 对该步解码出的 14 个 latent 坐标以 ``logprob_type: token_level`` 和 ``clip_ratio_c: 3.0`` 计算更新，配合 chunk 级奖励与 ``actor.optim.critic_warmup_steps: 20``。
         - **混合精度**：actor 以 ``fp32`` 保存可训练的动作专家权重以避免小学习率更新被舍入，FSDP2 仍以 ``bf16`` 计算，rollout 保留 ``bf16`` 权重副本。
         - **log-prob 漂移保护**：每次更新前 actor 重新计算每个采样 SDE 步的概率。若样本在 rollout 与 actor 间的总 log-probability 差值超过 ``algorithm.max_recompute_logprob_gap: 1.0e-3``，训练停止；最大差值记录在 ``actor/rollout_train_joint_logprob_gap_max``。
         - **独立评测**：按 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>` 运行独立评测，配置为 ``robotwin_click_bell_lingbotvlav2_eval``\ （在新终端中评测前先导出上方两个环境变量）。如需评测 PPO checkpoint，传入其完整权重文件：

           .. code-block:: bash

              bash evaluations/run_eval.sh robotwin robotwin_click_bell_lingbotvlav2_eval \
                runner.ckpt_path=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt

可视化与结果
----------------------------------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，关注任务成功率指标 ``env/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可保存评估视频，完整指标与日志后端说明见 :doc:`训练指标 <../../reference/metrics>`。
