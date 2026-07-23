基于 PyTorch OpenPI (Pi0.5) 的 BEHAVIOR 监督微调
================================================

本文档介绍如何在 RLinf 框架中，对自包含的 **PyTorch OpenPI Pi0.5** 流匹配
（flow-matching）VLA 模型，在 **BEHAVIOR-1K** 任务上进行 **监督微调（SFT）**。
该模型是 Pi0.5 架构的纯 PyTorch 重实现（双专家 Gemma + SigLIP，配合流匹配动作头），
在 RLinf 中以 ``model_type: openpi_pytorch`` 注册。SFT 通常作为进入强化学习前的
第一阶段：模型先模仿高质量示范，后续强化学习才能在良好先验上继续优化。

内容包括
--------

- PyTorch OpenPI SFT 流程是什么，以及如何配置
- FSDP 优化器与混合精度计算所使用的精度约定
- BEHAVIOR 流式数据加载器的相关字段，以及归一化统计 / tokenizer 配置
- 如何启动训练，以及如何转换得到的 checkpoint 用于评估


功能介绍
--------

``openpi_pytorch`` 模型是 Pi0.5 流匹配 VLA 的自包含 PyTorch 移植版本。**需要特别强调的是**：官方 openpi 仓库提供的
PyTorch 实现并未与其 JAX 参考实现对齐，而此处的移植版本在数值上与 JAX 实现严格
对齐。与基于 JAX/LeRobot 的 OpenPI 路径（参见 :doc:`sft_openpi`）不同，它直接从一小组配置字段
构建模型结构（构建阶段不读取 ``config.json``），并且开箱即用地适配 BEHAVIOR-1K。
在 SFT 阶段，策略通过流匹配去噪目标，从 BEHAVIOR 示范中预测双臂 R1 Pro 机器人
32 步、23 维的动作块（action chunk）。


配置说明
--------

该示例拆分为一个可复用、不含路径的 **模型模板**，以及一个提供文件系统路径的
**实验配置**：

- 实验配置：``examples/sft/config/behavior_pi05_vla.yaml``
- 模型模板：``examples/sft/config/model/pi0_5_pytorch.yaml``

实验配置通过 Hydra ``defaults`` 引入该模型模板：

.. code:: yaml

   defaults:
     - model/pi0_5_pytorch@actor.model
     - training_backend/fsdp@actor.fsdp_config
     - override hydra/job_logging: stdout

精度约定
~~~~~~~~

PyTorch OpenPI 的 SFT 配置刻意将 **加载 dtype** 与 **计算 dtype** 分开：

- 模型模板将 ``actor.model.precision`` 设为 ``fp32``（位于 ``pi0_5_pytorch.yaml``）。
  fp32 权重作为 **FSDP 优化器 master** 加载，从而保证 warmup 阶段较小的 LR 更新
  不会因 bf16 舍入而丢失。
- FSDP ``MixedPrecision`` 在 bf16 下计算，同时让梯度 all-reduce 与 buffer 保持
  fp32：

  .. code:: yaml

     actor:
       fsdp_config:
         gradient_checkpointing: True
         mixed_precision:
           param_dtype: bf16     # FSDP 计算 dtype
           reduce_dtype: fp32    # 梯度 all-reduce 保持 fp32

  ``param_dtype`` 是 FSDP 的 **计算** dtype，这里显式设为 bf16，而非从
  ``actor.model.precision`` 插值得到：加载 dtype 选择器与计算 dtype 是两个相互
  独立的开关，因此 fp32-master 加载仍然会以 bf16 进行计算。
- 在双专家 Gemma + SigLIP 骨干上启用了梯度检查点
  （``actor.fsdp_config.gradient_checkpointing: True``），以降低激活值显存占用。
- 学习率调度采用与参考实现完全一致的 warmup + 余弦衰减，通过
  ``actor.optim.lr_scheduler: openpi_cosine`` 选择（warmup 从
  ``peak / (warmup + 1)`` 开始，并在 ``total_training_steps`` 内余弦衰减到
  ``min_lr``）。

流式数据加载器
~~~~~~~~~~~~~~

BEHAVIOR 流式加载器直接从 ``data:`` 段读取其全部参数（没有隐藏默认值）：

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

关键数据字段：

- ``train_data_paths`` / ``behavior_dataset_root``：BEHAVIOR 数据集根目录
  （后者默认等于前者）。
- ``repo_id``：BEHAVIOR 示范数据 repo id（``behavior-1k/2025-challenge-demos``）。
- ``modalities``：加载器消费的输入模态（例如 ``["rgb"]``）。
- ``num_workers``：数据加载器的 worker 进程数。
- ``fine_grained_level`` 与 ``tolerance_s``：流式读取的时间对齐控制参数。
- ``tasks``：要训练的 BEHAVIOR 任务。
- ``use_skill``：为 ``false`` 时在主任务文本上训练；为 ``true`` 时在从
  ``task_subtasks`` 选取的逐帧 REFERENCE 技能文本上训练。
- ``task_subtasks``：每个任务的有序技能标签，当 ``use_skill: true`` 时用于构建
  下标到标签的映射。

归一化统计与 tokenizer
~~~~~~~~~~~~~~~~~~~~~~~

归一化统计与 PaliGemma tokenizer 位于 ``actor.model.openpi`` 下：

.. code:: yaml

   actor:
     model:
       model_path: /path/to/pi05_base_pytorch_new
       openpi:
         assets_dir: /path/to/assets
         asset_id: "behavior-1k/2025-challenge-demos"
         paligemma_tokenizer: /path/to/paligemma_tokenizer/paligemma_tokenizer.model

- ``assets_dir``：存放分位数归一化统计的目录。
- ``asset_id``：在 ``assets_dir`` 下对应本任务统计信息的子路径。
- ``paligemma_tokenizer``：PaliGemma SentencePiece tokenizer 模型
  （从 YAML 解析，而非在代码中硬编码）。

归一化统计会在 ``{assets_dir}/{asset_id}/norm_stats.json`` 处解析。

文件系统路径
~~~~~~~~~~~~

所有文件系统路径都以 ``/path/to/...`` 占位符的形式直接写在配置中。在
``examples/sft/config/behavior_pi05_vla.yaml`` 中将它们改为你自己暂存的资源路径：

- ``data.train_data_paths`` / ``data.behavior_dataset_root``：BEHAVIOR 流式数据集
  根目录。
- ``actor.model.model_path``：训练器加载的新格式 **fp32 基础 checkpoint**。
- ``actor.model.openpi.assets_dir``：归一化统计目录。
- ``actor.model.openpi.paligemma_tokenizer``：PaliGemma SentencePiece tokenizer
  模型。


启动脚本
--------

使用 BEHAVIOR Pi0.5 配置名运行 SFT 辅助脚本：

.. code:: bash

   # 回到仓库根目录
   bash examples/sft/run_vla_sft.sh behavior_pi05_vla

该脚本会将配置名转发给 SFT 入口，并在配置的 ``runner.logger.log_path`` 下写入
日志与 checkpoint。checkpoint 每 ``runner.save_interval`` 步保存一次，位于
``.../checkpoints/global_step_<N>/`` 下。


转换 checkpoint 用于评估
------------------------

可以使用 OpenPI checkpoint 转换器，将 SFT 训练得到的 checkpoint 转换为新格式的
裸 ``Pi0`` 布局（即评估加载器所期望的布局）：

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new \
       --ckpt              /path/to/logs/.../checkpoints/global_step_30000 \
       --input-norm-stats  /path/to/norm_stats.json \
       --output-model      /path/to/pi05_sft_pytorch_new \
       --output-norm-stats /path/to/pi05_sft_pytorch_new/physical-intelligence/behavior/norm_stats.json

``sft2new`` 模式会剥离 wrapper/FSDP key 前缀，将浮点张量转换为 bf16（新格式评估
加载器会校验每个 checkpoint 张量均为 bf16），并原样复制归一化统计文件。其他转换
模式与完整参数说明，请参见转换器包的 README
（``rlinf/utils/ckpt_convertor/openpi/README.md``）。转换后的 checkpoint 即可用于在
BEHAVIOR 上评估；评估配置与启动命令参见 :doc:`behavior`。
