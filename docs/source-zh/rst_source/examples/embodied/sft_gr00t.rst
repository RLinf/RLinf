GR00T-N1.7 监督微调
===================

使用 LIBERO 示范数据和 NVIDIA 的 3B 权重，在 RLinf 中微调 GR00T-N1.7。本示例涵盖数据准备、全参数训练和 checkpoint 恢复。

概览
----

模型读取主相机、腕部相机、机器人状态和任务指令，学习 LIBERO 的动作片段。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GR00T-N1.7-3B

   .. grid-item-card:: 方法
      :text-align: center

      全参数 SFT

   .. grid-item-card:: 数据
      :text-align: center

      LIBERO Spatial · LeRobot

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · FSDP

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 数据集
     - 任务套件
     - 配置
   * - LIBERO LeRobot
     - Spatial
     - ``libero_sft_gr00t_n1d7``

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - 观测
     - 主相机与腕部相机 RGB 图像；末端位置、姿态和夹爪状态
   * - 动作
     - 16 步 LIBERO 笛卡尔位移、旋转增量和夹爪控制
   * - 指令
     - 示范轨迹中的任务描述

安装
----

.. include:: _setup_common.rst

安装 GR00T-N1.7 及其支持的仿真依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
   source .venv/bin/activate

安装脚本把上游 GR00T 仓库放在 ``.venv/gr00t``。如果用 ``GR00T_PATH`` 复用已有仓库，请从该仓库复制 ``examples/LIBERO/modality.json``。

下载模型与数据
--------------

先在 Hugging Face 接受 `Cosmos-Reason2-2B <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`_ 的访问条款，并登录有权限的账号。GR00T checkpoint 已包含 backbone 权重，Cosmos 目录提供配置与 tokenizer。

下载模型资产：

.. code-block:: bash

   hf auth login
   hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
   hf download nvidia/Cosmos-Reason2-2B \
     --include '*.json' '*.txt' --local-dir checkpoints/Cosmos-Reason2-2B

下载 Spatial 示范数据，并把上游的模态描述放入数据集元数据目录：

.. code-block:: bash

   hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
     --repo-type dataset --local-dir datasets/libero_spatial
   cp .venv/gr00t/examples/LIBERO/modality.json datasets/libero_spatial/meta/modality.json

保留数据集完整的 ``meta/``、``data/`` 和 ``videos/`` 目录。训练使用该数据集的归一化统计，每次保存 checkpoint 时一并保存。状态与动作转换沿用上游 GR00T 的 LIBERO 配置。

运行
----

在 ``examples/sft/config/libero_sft_gr00t_n1d7.yaml`` 中填写模型与数据的绝对路径：

.. code-block:: yaml

   data:
     train_data_paths:
       - /absolute/path/to/datasets/libero_spatial
   actor:
     model:
       model_path: /absolute/path/to/checkpoints/GR00T-N1.7-3B
       backbone_model_path: /absolute/path/to/checkpoints/Cosmos-Reason2-2B

示例默认使用单机八卡。先用 ``cluster.component_placement.actor`` 指定可用 GPU，再设置 ``actor.global_batch_size`` 与 ``actor.micro_batch_size``，使梯度累积次数为整数。模型以 FP32 加载；FSDP 对 backbone 和 attention 模块使用 BF16，对 projection 模块使用 FP32。视觉编码器、语言模型和 action head 均参与训练，gradient checkpointing 关闭。

启动训练：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_gr00t_n1d7

用 ``runner.max_steps`` 设置更新次数，并让 ``actor.optim.total_training_steps`` 与学习率计划一致。示例设为 20,000 次更新，每 1,000 次保存一次；这是配置示例，尚未测得其 LIBERO 收敛表现。

上游 processor 把动作补齐到 checkpoint 的动作空间，flow-matching loss 使用 mask 排除无效动作维度与时间步。数据流持续采样，因此使用更新次数控制预算，并保留 ``runner.max_epochs: -1``。

可视化与结果
------------

在 ``runner.logger.log_path`` 下用 TensorBoard 查看训练 loss。每份 actor checkpoint 保存模型、优化器、训练 processor、归一化统计与 dataloader 恢复状态。actor 目录包含 ``processor_config.json``、``statistics.json``、``embodiment_id.json`` 和 ``config.json``。恢复时将 ``runner.resume_dir`` 设为已保存的 ``checkpoints/global_step_N`` 目录，步骤见 :doc:`../../guides/resume`。

保留原训练配置，指定已保存的更新目录后恢复：

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/sft"
   python examples/sft/train_vla_sft.py --config-name libero_sft_gr00t_n1d7 \
     runner.resume_dir=/absolute/path/to/checkpoints/global_step_1000

流式 dataloader 通过跳过已读取批次恢复位置，随机图像增强不保证逐位重放。恢复时保持数据集、worker 数量和 GPU 拓扑一致。

训练 loss 反映动作拟合程度；闭环任务成功率需要另行仿真评测。LIBERO 环境准备见 :doc:`gr00t`；导出这份 SFT checkpoint 用于闭环评测前，还需单独验证加载兼容性。
