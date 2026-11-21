基于RL的仿真-真机协同训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-co/overview.png
   :align: center
   :width: 90%

   仿真-真机协同训练总览。

仿真-真机协同训练通过在仿真中用 PPO 与在真机数据上用 SFT 相结合来训练 π₀.₅ 策略：
在提升仿真任务成功率的同时保留真机先验，避免仅靠仿真过拟合而损害 sim-to-real 迁移。技术细节详见
:doc:`Beyond Imitation: RL-Based Sim-Real Co-Training for VLA Models <../../resources/publications/rlinf_co>`。

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


安装
----------------------------------------

使用 ``requirements/install.sh`` 安装 OpenPI + ManiSkill 环境（中国大陆用户可在命令末尾追加 ``--use-mirror``，并在执行 ``hf download`` 前设置 ``export HF_ENDPOINT=https://hf-mirror.com``），或从 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ 拉取预构建镜像。基础 ManiSkill 资产设置见 :doc:`ManiSkill 示例 <maniskill>`。

.. tab-set::

   .. tab-item:: OpenPI + ManiSkill

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate


下载资产、模型与数据
----------------------------------------

一次性下载数字孪生所需的 ManiSkill 资产、Stage II 协同训练用的 50 条真机轨迹，以及现成的 Stage I SFT checkpoint（推荐）或 Stage I 混合数据集（`RLinf/RLCo-Example-Mix-Data <https://huggingface.co/datasets/RLinf/RLCo-Example-Mix-Data>`_，含 50 条真机轨迹 + 1499 条仿真轨迹）：

.. tab-set::

   .. tab-item:: 使用 SFT 预训练权重（推荐）

      .. code-block:: bash

         hf download RLinf/RLCo-maniskill-assets --repo-type dataset --include "custom_assets/*" --local-dir ./rlinf/envs/sim/maniskill/assets
         hf download RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT --local-dir ./RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT
         hf download RLinf/RLCo-Example-Real-Data --repo-type dataset --local-dir ./RLCo-Example-Real-Data

   .. tab-item:: 自行训练 Stage I SFT

      下载资产和数据集后，参考 `OpenPi 官方代码 <https://github.com/Physical-Intelligence/openpi>`_ 或 :doc:`OpenPI 监督微调 <sft_openpi>` 运行 Stage I SFT：

      .. code-block:: bash

         hf download RLinf/RLCo-maniskill-assets --repo-type dataset --include "custom_assets/*" --local-dir ./rlinf/envs/sim/maniskill/assets
         hf download RLinf/RLCo-Example-Mix-Data --repo-type dataset --local-dir ./RLCo-Example-Mix-Data
         hf download RLinf/RLCo-Example-Real-Data --repo-type dataset --local-dir ./RLCo-Example-Real-Data


运行
----------------------------------------

本阶段在 PPO 训练循环中加入 50 条真机轨迹的 SFT 损失。默认配置文件为 ``examples/embodiment/config/maniskill_ppo_co_training_openpi_pi05.yaml``。通用路径、集群和 runner 字段见 :doc:`训练配置 <../../reference/configuration>`；PPO 参数见 :doc:`π0 和 π0.5 模型强化学习训练 <pi0>`。

将 ``actor.model.model_path``\ （rollout 会自动沿用同一路径）指向 SFT 权重目录，将 ``actor.sft_data_path`` 指向真机数据目录：

.. code-block:: yaml

   actor:
     sft_data_path: ./RLCo-Example-Real-Data
     model:
       model_path: ./RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

在仓库根目录启动协同训练：

.. code:: bash

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

**TensorBoard**

.. code:: bash

   tensorboard --logdir ./logs --port 6006

**指标**

除标准指标外（见 :doc:`训练指标 <../../reference/metrics>`），协同训练还新增以下指标：

- ``train/ppo_loss``: PPO（RL）损失。
- ``train/sft_loss``: 真机数据上的 SFT 损失。
- ``actor/total_loss``: :math:`\mathcal{L}_{Total} = \mathcal{L}_{RL} + \beta \mathcal{L}_{SFT}`。
- ``train/loss_ratio``: :math:`\frac{\beta \lvert \mathcal{L}_{SFT} \rvert}{\lvert \mathcal{L}_{RL} \rvert}`。若该值持续过大（如 :math:`> 10^5`），日志会触发警告，此时应降低 ``sft_loss_weight``。

**示例结果**

- 加载 Stage I 权重后：仿真中零样本成功率约 35%。
- 经过 100 步协同训练后：仿真成功率约 50%。

更多关于真机部署效果及消融实验，请参考论文：*Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models*。
