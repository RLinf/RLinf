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

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
   # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

请通过镜像内置的 ``switch_env`` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env openpi

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Maniskill 资源下载
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

请先参考 :doc:`ManiSkill 示例 <maniskill>` 下载基础资源。随后下载本示例所需的特定资源：

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill/assets
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/RLCo-maniskill-assets --include "custom_assets/*" --local-dir .

运行
----------------------------------------

Stage I：SFT 预训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

第一阶段旨在通过监督学习快速注入真机与仿真知识，为后续 RL 训练奠定基础。您可以选择 **自行训练** 或 **下载权重**。

**方法A: 使用真机-仿真数据进行 SFT 训练**

我们提供了 LeRobot 格式数据集（50 条真机轨迹 + 1499 条仿真轨迹），托管于 `RLinf/RLCo-Example-Mix-Data <https://huggingface.co/datasets/RLinf/RLCo-Example-Mix-Data>`_。

1. **下载数据集**：

.. code:: bash

   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/RLCo-Example-Mix-Data --local-dir RLCo-Example-Mix-Data

2. **执行训练**：

训练方法请参考 `OpenPi 官方代码 <https://github.com/Physical-Intelligence/openpi>`_ 或 RLinf 文档中的 :doc:`监督训练微调 <sft_openpi>` 章节。

**方法 B：使用 SFT 预训练权重**

跳过训练步骤，直接使用我们提供的 SFT Checkpoint：

.. code:: bash

   # 下载 Spatial-Object-Goal 模型（选择以下任一方式）
   # 方式1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

   # 方式2：使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT --local-dir RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

Stage II：仿真-真机协同 RL 训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本阶段在 PPO 训练循环中加入 SFT 损失，实现协同优化。

**数据准备**

下载用于 Co-Training 的 50 条真机轨迹数据（LeRobot 格式）：

.. code:: bash

   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/RLCo-Example-Real-Data --local-dir RLCo-Example-Real-Data

**关键参数配置**

我们提供 ``maniskill_ppo_co_training_openpi_pi05.yaml`` 配置文件。通用路径、集群和 runner 字段见 :doc:`训练配置 <../../reference/configuration>`；PPO 训练相关参数可参照 :doc:`π0 和 π0.5 模型强化学习训练 <pi0>`。另外需关注以下参数：

**模型加载路径**

将 ``model_path`` 指向 SFT 权重目录，``sft_data_path`` 指向真机数据路径：

.. code-block:: yaml

   rollout:
      model:
         model_path: /path/to/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT
   actor:
      sft_data_path: /path/to/RLCo-Example-Real-Data
      model:
         model_path: /path/to/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT

**Co-Training 策略配置**

.. code-block:: yaml

   actor:
       model:
           openpi:
               config_name: "pi05_maniskill_sim_real_co_training"

       # 开启真机数据协同训练
       enable_sft_co_train: True

       # SFT Loss 权重系数 (beta)
       sft_loss_weight: 0.2

* ``enable_sft_co_train``: 设为 ``True`` 开启协同训练。若为 ``False``，则退化为纯 PPO 训练。
* ``sft_loss_weight``: 控制 SFT Loss (:math:`\mathcal{L}_{SFT}`) 在总 Loss 中的占比权重 :math:`\beta`。

**Python 配置类参考**

在代码层面，``pi05_maniskill_sim_real_co_training`` 对应的配置位于 ``rlinf/models/embodiment/openpi/dataconfig/__init__.py``。需确保 ``model`` 架构与 ``normalization`` 状态与 SFT 阶段保持一致。

**关于 Batch Size 的说明:**

配置文件中的 batch_size 指的是梯度累积前的微批次大小。
实际更新是单批次数据量计算公式为：

.. math::

   \text{True\_Batch\_Size} = \frac{\text{Global\_Batch\_Size} \times \text{Input\_Batch}}{\text{Micro\_Batch\_Size} \times \text{Num\_GPUs}}

对于 ``global_batch_size`` 和 ``micro_batch_size`` 的具体数值设定请参考 :doc:`./pi0`。

**运行脚本**

我们提供了预设脚本，直接运行即可启动训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_co_training_openpi_pi05

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
