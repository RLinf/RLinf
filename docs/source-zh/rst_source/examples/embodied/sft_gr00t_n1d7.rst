GR00T N1.7 监督微调
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t.png
   :align: center
   :width: 90%

   GR00T 视觉语言动作模型系列。

本页介绍如何在 RoboCasa365 BowlAndCup 示范数据上微调 GR00T N1.7，使用独立随机观测动作窗口，涵盖数据准备、原生 H40 Flow Matching loss，以及共用 VLA FSDP 训练流程。

概览
----

每条训练样本包含当前观测及其后的 40 步动作目标，batch size 按独立窗口计数。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GR00T-N1.7-3B，使用 Cosmos-Reason2-2B 主干

   .. grid-item-card:: 方法
      :text-align: center

      原生 Flow Matching SFT，独立随机窗口采样

   .. grid-item-card:: 数据
      :text-align: center

      GR00T 兼容 LeRobot 格式的 RoboCasa365 BowlAndCup

   .. grid-item-card:: 硬件
      :text-align: center

      八张 NVIDIA GPU 的 FSDP 示例；H40 配方的早期版本曾在八张 A800 上完成两步检查

任务
~~~~

示例配置针对单个操作任务及其示范数据集。

.. list-table::
   :header-rows: 1

   * - 任务
     - 示范内容
     - 配置
   * - BowlAndCup
     - 将杯子放入碗中，再将碗移至目标台面，最后撤离机械臂
     - ``robocasa_bowlandcup_sft_gr00t``

观测与动作
~~~~~~~~~~

每个窗口使用当前观测和随后 40 步动作作为训练样本，未来观测不进入 policy。

.. list-table::
   :header-rows: 1

   * - 字段
     - 内容
   * - Observation
     - 三路 RGB 图像，processor 缩放至 256 × 256 并在训练时裁剪为 230 × 230；另含底盘位姿、末端相对位姿和夹爪状态
   * - Action
     - 每步 12 个真实 RoboCasa 动作值；原生模型张量补齐至 40 × 132
   * - Reward
     - 监督训练不使用 reward
   * - Prompt
     - ``annotation.human.task_description``

准备数据集
----------

将 ``data.train_data_paths`` 指向一个 LeRobot 数据集根目录，其中应包含 parquet 数据、视频和 ``meta/`` 目录。加载器要求 ``meta/`` 下有 ``info.json``、``episodes.jsonl``、``tasks.jsonl``、``stats.json``、``modality.json`` 和 ``embodiment.json``。processor 使用该数据集的归一化统计量。

三路相机 key 为 ``robot0_agentview_left``、``robot0_agentview_right`` 和 ``robot0_eye_in_hand``。状态和动作的模态顺序采用 ``rlinf/data/datasets/gr00t_n1d7/window_dataset.py`` 中的 RoboCasa365 schema。其他 schema 需要先转换，修改数据路径不会改变模态映射。

每个采样窗口必须包含完整的 40 步真实动作监督。轨迹长度为 :math:`L` 时，最后一个合法起点是 :math:`L-40`；长度不足 40 步的轨迹会被拒绝。这与模型动作维度的 padding 是两件事。

若需要检查数据规模，可将 ``data.expected_episodes`` 设为预期轨迹数，否则保持 ``null``。

要在续训时重放相同样本，须保持 parquet、视频、metadata 和采样设置与原训练一致。Sampler 恢复时检查 size、seed、rank 和 world size，但这些检查不验证数据内容完整性。

采样与 Loss
-----------

Sampler 混合所有轨迹的合法窗口起点并打乱顺序，每个窗口独立送入模型。

.. list-table::
   :header-rows: 1

   * - 设置
     - 值
   * - 动作监督长度
     - 40
   * - 窗口选择
     - 混合所有轨迹的合法单窗口起点后打乱
   * - Global / 单卡 micro batch
     - 128 / 16 个窗口
   * - 八卡每次更新的监督窗口数
     - 128

每个采样 epoch 中，全部合法窗口在所有 rank 上合计遍历一次，长轨迹贡献的窗口更多。分布式 sampler 跨 epoch 边界持续采样，不丢弃尾部窗口。采样顺序和数据增强随机种子均由配置 seed 与窗口身份确定。

原生动作头输出带 mask 的逐元素 Flow Matching 平方误差。每个窗口的误差和除以有效动作值数量 :math:`40\times12=480`。全部 40 个时间步均参与监督，补齐动作维度的监督权重为零。原生 collator 将动作目标组成 :math:`[B,40,132]` 张量。

设带 mask 的误差为 :math:`E_{b,h,a}`，监督 mask 为 :math:`M_{b,h,a}`，loss 先对每个窗口的有效动作值求均值，再对窗口求均值：

.. math::

   L = \frac{1}{B}\sum_{b=1}^{B}
       \frac{\sum_{h,a} E_{b,h,a}}{\sum_{h,a} M_{b,h,a}}.

共用 VLA worker 执行 backward 和梯度累积，每个 global batch 只执行一次 AdamW 更新。示例使用八张 GPU、每卡 16 个窗口，每次更新共使用 128 个窗口。

下载模型
--------

准备 GR00T 预训练权重及 Cosmos 主干的本地副本，同时保留配置和 processor 文件。模型构建会严格加载 GR00T state dictionary，并检查原生动作头为 H40、补齐动作维度为 132。

.. code-block:: bash

   hf download nvidia/GR00T-N1.7-3B --local-dir /path/to/GR00T-N1.7-3B
   hf download nvidia/Cosmos-Reason2-2B --local-dir /path/to/Cosmos-Reason2-2B

安装
----

在仓库根目录安装 GR00T N1.7 依赖并激活环境。现有安装器为该模型接受 ``maniskill_libero`` 环境；离线 SFT 从准备好的数据集读取样本，不启动该仿真器。此安装组合在该 SFT 流程下仍待端到端验证。

.. code-block:: bash

   bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
   source .venv/bin/activate

将安装后的 Isaac-GR00T checkout 固定到 adapter 对应的版本。下面使用安装器的默认路径；若用 ``GR00T_PATH`` 指定了已有 checkout，应改为该路径。

.. code-block:: bash

   git -C .venv/gr00t checkout 23ace64f17aa5015259b8609d371eb61a357c776
   uv pip install -e .venv/gr00t --no-deps

运行训练
--------

替换以下三项资产路径后，启动随机窗口示例。``native_sft=true`` 会在共用 VLA SFT 入口中选择原生 GR00T 训练 adapter。

.. code-block:: bash

   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-path config \
     --config-name robocasa_bowlandcup_sft_gr00t \
     data.train_data_paths=/path/to/BowlAndCup/lerobot \
     actor.model.model_path=/path/to/GR00T-N1.7-3B \
     actor.model.backbone_model_path=/path/to/Cosmos-Reason2-2B \
     actor.model.dependency_commit=23ace64f17aa5015259b8609d371eb61a357c776

示例采用 FP32 主参数、FSDP BF16 参数计算、FP32 梯度归约和 AdamW。学习率预热 200 步后，按 30,000 步余弦计划从 :math:`2\times10^{-5}` 退火至 :math:`5\times10^{-6}`，额外 AMP autocast 关闭。

设置 ``runner.max_steps > 0`` 和 ``runner.max_epochs <= 0``；不满足这两个条件时，dataloader 会拒绝启动。Dataloader 的标称长度仅用于共用 runner 的 epoch 计数，sampler 会跨采样 epoch 持续迭代。``StatefulDataLoader`` 保留 sampler 游标用于 checkpoint 恢复，标称长度不会截断采样。

每 5,000 步通过共用 VLA worker 的现有 FSDP save/load 流程保存 model、optimizer、scheduler 和 RNG 状态。Worker 还将 ``StatefulDataLoader`` 状态保存到 ``data.pt``，将各 rank 的 RNG 状态保存到 ``rng.pt``。续训时，``SFTRunner`` 从 ``global_step_<N>`` 目录名恢复 global step。请保留完整 checkpoint 目录用于续训；保存流程不保证整个目录原子写入完成。

构建 dataloader 时，rank 0 将模态定义、归一化统计量和 processor 关键字参数一次性导出到 ``<log_path>/<experiment_name>/processor_bundle.json``，路径由 ``runner.logger.log_path`` 和 ``runner.logger.experiment_name`` 指定。若 bundle 已存在，其内容必须一致，否则停止初始化。该文件位于运行目录，不随每个 checkpoint 重复导出。训练恢复根据相同的模型与数据配置重建 processor，不从该 bundle 恢复。早期实验实现的 checkpoint 是否兼容仍待验证。

可视化与结果
------------

示例将 TensorBoard 日志写入 ``runner.logger.log_path`` 指定的目录。Loss、梯度范数和学习率的说明见 :doc:`训练指标 <../../reference/metrics>`。

当前 H40 随机窗口实现通过了 34 项单元测试。真实数据检查在八张 A800 80GB GPU 上完成了四个 optimizer step，并在第 2、4 步保存 checkpoint；另一轮检查加载第 2 步的 checkpoint 后续训至第 4 步，两轮均正常退出。第 4 步的 dataloader 状态、RNG 状态、scheduler 状态和 Adam 步数一致，但模型张量与 Adam 动量存在差异，因此尚未验证逐位一致的续训结果。

各 checkpoint 使用相同的八个场景种子（7–14）评测，每个场景运行三个 policy seed，共 24 个 episode。评测采用 OSMesa 渲染，每个 episode 最多运行 2,400 个环境步，每次预测 H40 动作并执行其中的八步。成功判定采用环境原生的完整任务成功条件。

.. list-table:: BowlAndCup Checkpoint 评测
   :header-rows: 1

   * - 训练步数
     - 成功 Episode 数
     - 成功率
   * - 5,000
     - 0 / 24
     - 0.00%
   * - 10,000
     - 2 / 24
     - 8.33%
   * - 15,000
     - 5 / 24
     - 20.83%

这些结果由本次提交之外的外部评测脚本生成，该脚本未包含在提交中。这些 checkpoint 与独立评测入口的兼容性仍待验证。通用 rollout 配置见 :doc:`评测 <../../evaluations/index>`。
