Co-Training
===========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-co/overview.png
   :align: center
   :width: 90%

   仿真-真机协同训练总览。

仿真-真机协同训练通过在仿真中用 PPO 与在真机数据上用 SFT 相结合来训练 π₀.₅ 策略：在提升仿真任务成功率的同时保留真机先验，避免仅靠仿真过拟合而损害 sim-to-real 迁移。技术细节详见 :doc:`Beyond Imitation: RL-Based Sim-Real Co-Training for VLA Models <../../resources/publications/rlinf_co>`。

概览
----------------------------------------

在 ManiSkill 数字孪生上协同训练 π₀.₅——仿真中 PPO + 50 条真机轨迹 SFT（仿真成功率约 35%→50%）。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      PPO + SFT (RL-Co)

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      ManiSkill digital twin

   .. grid-item-card:: 训练
      :text-align: center

      两阶段 sim-real

| **你将完成：** 安装 → 下载资产与 SFT 模型 → SFT（阶段一）→ 协同 RL 训练（阶段二）→ 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · ManiSkill 资产 · SFT 检查点与真机数据（见下文步骤）。

设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本示例仅提供单一演示环境；用于自己的机器人时，请采集数据并构建匹配的仿真场景。

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - 部分
     - 说明
   * - 任务
     - 抓取放置——将桌上的物体放入碗中。
   * - 真机
     - Franka Emika Panda + RealSense；第三人称 RGB（640×480）；7 自由度动作（x, y, z, roll, pitch, yaw, 夹爪）。
   * - 仿真
     - 基于 ManiSkill3 的数字孪生，在布局、相机视角、任务逻辑、语言与动作空间上与真机对齐；动力学经过调校以逼近真实物理。


安装与运行
----------

.. include:: _setup_common.rst

基础 ManiSkill 资产设置见 :doc:`ManiSkill 示例 <maniskill>`。你可以直接在 **阶段二** 使用已发布的 Stage I SFT 权重（推荐），或先在 **阶段一** 自行训练 SFT。

.. tab-set::

   .. tab-item:: 阶段一：SFT 预训练

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. 下载资产与数据**

      下载数字孪生所需的 ManiSkill 资产、Stage I 混合数据集（`RLinf/RLCo-Example-Mix-Data <https://huggingface.co/datasets/RLinf/RLCo-Example-Mix-Data>`__，含 50 条真机轨迹 + 1499 条仿真轨迹），以及 Stage II 所需的 50 条真机轨迹：

      .. code-block:: bash

         hf download RLinf/RLCo-maniskill-assets --repo-type dataset --include "custom_assets/*" --local-dir ./rlinf/envs/sim/maniskill/assets
         hf download RLinf/RLCo-Example-Mix-Data --repo-type dataset --local-dir ./RLCo-Example-Mix-Data
         hf download RLinf/RLCo-Example-Real-Data --repo-type dataset --local-dir ./RLCo-Example-Real-Data

      **3. 启动 SFT 训练**

      参考 `OpenPi 官方代码 <https://github.com/Physical-Intelligence/openpi>`__ 或 :doc:`OpenPI 监督微调 <sft_openpi>` 在 ``./RLCo-Example-Mix-Data`` 上完成 Stage I SFT 训练，然后在阶段二将 ``actor.model.model_path`` 指向产出的权重目录。

   .. tab-item:: 阶段二：仿真-真机协同训练

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. 下载资产、模型与数据**

      下载数字孪生所需的 ManiSkill 资产、现成的 Stage I SFT checkpoint，以及协同训练用的 50 条真机轨迹：

      .. code-block:: bash

         hf download RLinf/RLCo-maniskill-assets --repo-type dataset --include "custom_assets/*" --local-dir ./rlinf/envs/sim/maniskill/assets
         hf download RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT --local-dir ./RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT
         hf download RLinf/RLCo-Example-Real-Data --repo-type dataset --local-dir ./RLCo-Example-Real-Data

      **3. 启动协同训练**

      .. list-table::
         :header-rows: 1
         :widths: 24 50 26

         * - 配方
           - 配置
           - 命令后缀
         * - OpenPI π₀.₅ + RL-Co
           - ``examples/embodiment/config/maniskill_ppo_co_training_openpi_pi05.yaml``
           - ``maniskill_ppo_co_training_openpi_pi05``

      将 ``actor.model.model_path``\ （rollout 会自动沿用同一路径）指向 SFT 权重目录，将 ``actor.sft_data_path`` 指向真机数据目录：

      .. code-block:: yaml

         actor:
           sft_data_path: ./RLCo-Example-Real-Data
           model:
             model_path: ./RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

      在仓库根目录启动协同训练（通用路径、集群和 runner 字段见 :doc:`训练配置 <../../reference/configuration>`；PPO 参数见 :doc:`π0 和 π0.5 模型强化学习训练 <pi0>`）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_co_training_openpi_pi05

.. dropdown:: 进阶配置（可选）

   仓库自带配置已设置协同训练开关与 ``pi05_maniskill_sim_real_co_training`` 数据配置（保持模型架构和归一化统计与 Stage I 一致）：

   .. code-block:: yaml

      actor:
        model:
          openpi:
            config_name: "pi05_maniskill_sim_real_co_training"
        enable_sft_co_train: True
        sft_loss_weight: 0.2

   * ``enable_sft_co_train``：设为 ``True`` 开启协同训练；设为 ``False`` 时退化为纯 PPO 训练。
   * ``sft_loss_weight``：控制 SFT 损失在总损失中的权重。

   配置文件中的 ``batch_size`` 指梯度累积前的微批次大小，实际单次更新数据量为 ``(global_batch_size * input_batch) / (micro_batch_size * num_gpus)``。``global_batch_size`` 和 ``micro_batch_size`` 的设置见 :doc:`./pi0`。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，标准指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

协同训练新增以下损失指标：

- ``train/ppo_loss``：PPO（RL）损失。
- ``train/sft_loss``：真机数据上的 SFT 损失。
- ``actor/total_loss``：联合目标 ``RL_loss + sft_loss_weight * SFT_loss``。
- ``train/loss_ratio``：两项损失幅值比 ``(sft_loss_weight * |SFT_loss|) / |RL_loss|``。若该值持续过大（如超过 ``1e5``），日志会触发警告，此时应降低 ``sft_loss_weight``。

**示例结果**

- 加载 Stage I 权重后：仿真中零样本成功率约 35%。
- 经过 100 步协同训练后：仿真成功率约 50%。

更多关于真机部署效果及消融实验，请参考论文：*Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models*。
