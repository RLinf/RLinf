JAX 精度对齐的 PyTorch OpenPI 监督微调
========================================

本文档介绍 RLinf 中自包含、与 OpenPI JAX 参考实现精度对齐的 PyTorch OpenPI
监督微调（SFT）流程。该实现支持 ``Pi0`` 和 ``Pi0.5`` 两个流匹配 VLA 变体，注册名为
``model_type: openpi_pytorch``。模型遵循 OpenPI JAX 参考实现的架构和精度行为，并通过
PyTorch 与 FSDP 完成训练。

当前维护的 SFT 配方包括：

- **Pi0 + RoboTwin**
- **Pi0.5 + BEHAVIOR-1K**

当前尚未提供 Pi0 + BEHAVIOR 的维护版 SFT 配置。请使用下面明确列出的模型和数据集配对，
不要混用未列为同一配方的模型模板和数据集配置。


可用配方
--------

每个配方都是 ``examples/sft/config/`` 下的实验配置。实验配置通过 Hydra 引入对应的无路径
模型模板，并提供本地数据集、checkpoint 与归一化统计信息路径。

.. list-table::
   :header-rows: 1
   :widths: 18 18 32 32

   * - 模型
     - 数据集
     - 实验配置
     - 模型模板 / OpenPI 配置
   * - Pi0
     - RoboTwin
     - ``robotwin_sft_openpi_pytorch.yaml``
     - ``model/pi0_pytorch.yaml`` / ``pi0_aloha_robotwin``
   * - Pi0.5
     - BEHAVIOR-1K
     - ``behavior_pi05_vla.yaml``
     - ``model/pi0_5_pytorch.yaml`` / ``pi05_behavior``

两个配方均以 fp32 加载 master weights，并采用 FSDP 混合精度：参数以 bf16 计算，
梯度规约与 buffer 保持 fp32。这样既保持参考实现对齐的优化器行为，又能降低计算和激活显存。
随附配置还启用了梯度检查点。


准备配方
--------

从表中选择实验配置，并替换所有 ``/path/to/...`` 占位符。模型 checkpoint 必须是 RLinf PyTorch
布局（包含 ``model.safetensors`` 与 ``config.json``），对应的
``norm_stats.json`` 也必须位于配置指定的 asset 路径。若起点为 OpenPI JAX checkpoint，
请使用 checkpoint 转换器的 ``jax2rlinf_pytorch`` 模式；完整流程见
``rlinf/utils/ckpt_convertor/openpi/README.md``。

RoboTwin
~~~~~~~~

RoboTwin 配方使用 LeRobot 格式的 RoboTwin 数据集，并采用以下数据设置：

.. code:: yaml

   data:
     train_data_paths: /path/to/robotwin-data
     num_workers: 4
     tolerance_s: 1.0e-4

RoboTwin 使用 14 维 ALOHA 动作和 3 路输入图像。模型配置会将动作补齐到 OpenPI 的 32 维
模型动作空间，并设置 ``num_action_chunks: 50``。在所选配方中设置模型和 assets 路径，例如：

.. code:: yaml

   actor:
     model:
       model_path: /path/to/pi0_base_rlinf_pytorch
       openpi:
         assets_dir: ${actor.model.model_path}
         asset_id: "physical-intelligence/robotwin"
         num_images_in_input: 3

Pi0 checkpoint 对应 ``robotwin_sft_openpi_pytorch.yaml``。

BEHAVIOR-1K
~~~~~~~~~~~

``behavior_pi05_vla.yaml`` 使用 Pi0.5 的流式 BEHAVIOR 数据加载器。它以流匹配去噪目标训练
双臂 R1 Pro 机器人 32 步、23 维的动作块。请在 ``data`` 和 ``actor.model.openpi`` 中配置
数据集根目录、任务选择和与 Pi0.5 匹配的 assets：

.. code:: yaml

   data:
     train_data_paths: /path/to/2025-challenge-demos
     behavior_dataset_root: /path/to/2025-challenge-demos
     repo_id: "behavior-1k/2025-challenge-demos"
     modalities: ["rgb"]
     num_workers: 8
     fine_grained_level: 0
     tolerance_s: 1.0e-4
     tasks: ["turning_on_radio"]
     use_skill: false
     task_subtasks:
       turning_on_radio:
         - "move to radio"
         - "pick up radio from coffee table"
         - "press radio"
         - "place radio on coffee table"

   actor:
     model:
       model_path: /path/to/pi05_base_rlinf_pytorch
       openpi:
         assets_dir: /path/to/assets
         asset_id: "behavior-1k/2025-challenge-demos"

``train_data_paths`` 与 ``behavior_dataset_root`` 指向本地 BEHAVIOR 数据集。``tasks`` 选择
训练任务。``use_skill: false`` 时以主任务文本训练；设为 ``true`` 时，以 ``task_subtasks`` 中
指定的逐帧 REFERENCE 技能文本训练。启用技能训练时，请为所选任务提供明确且有序的技能标签。


启动训练
--------

在仓库根目录启动所需模型和数据集对应的配方：

.. code:: bash

   # Pi0 + RoboTwin
   bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pytorch

   # Pi0.5 + BEHAVIOR-1K
   bash examples/sft/run_vla_sft.sh behavior_pi05_vla

该辅助脚本会设置 SFT 配置路径、记录实际运行命令，并将日志和 checkpoint 写入
``logs/<timestamp>-<config-name>``。checkpoint 按 ``runner.save_interval`` 保存到
``checkpoints/global_step_<N>/``。


转换 SFT checkpoint
-------------------

所有 SFT checkpoint 都使用 ``sft2rlinf_pytorch``；``--config-name`` 选择匹配的
Pi0/Pi0.5 架构，``--dtype fp32`` 用于保留 SFT 的主权重精度。

RoboTwin Pi0：

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2rlinf_pytorch \
       --config-name pi0_aloha_robotwin \
       --dtype fp32 \
       --ckpt /path/to/checkpoints/global_step_30000 \
       --input-norm-stats /path/to/pi0_base_rlinf_pytorch/physical-intelligence/robotwin/norm_stats.json \
       --output-model /path/to/pi0_robotwin_sft_rlinf_pytorch \
       --output-norm-stats /path/to/pi0_robotwin_sft_rlinf_pytorch/physical-intelligence/robotwin/norm_stats.json \
       --reference-model /path/to/pi0_base_rlinf_pytorch

Pi0.5 + BEHAVIOR-1K：

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2rlinf_pytorch \
       --config-name pi05_behavior \
       --dtype fp32 \
       --ckpt /path/to/checkpoints/global_step_30000 \
       --input-norm-stats /path/to/norm_stats.json \
       --output-model /path/to/pi05_behavior_sft_rlinf_pytorch \
       --output-norm-stats /path/to/pi05_behavior_sft_rlinf_pytorch/physical-intelligence/behavior/norm_stats.json

所选的 ``--config-name`` 会在输出配置中保留 RoboTwin 或 BEHAVIOR 的架构。
所有选项以及对应评估配置请参见转换器 README。
