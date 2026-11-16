GR00T-N1.7 监督微调
===================

使用 LIBERO Spatial 示范数据全参数微调 GR00T-N1.7，并保存可恢复训练的 checkpoint。本示例从 NVIDIA 的 3B 权重开始，用 FSDP 训练视觉编码器、语言模型和 action head。

概览
----

模型读取双相机图像、机器人状态和任务指令，学习预测 16 步动作片段。

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

      8 张 NVIDIA GPU · FSDP

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 任务套件
     - 示范数据
     - 配置
   * - LIBERO Spatial：10 个任务
     - 432 条轨迹，52,970 个时间步，采样频率 20 Hz
     - ``libero_sft_gr00t_n1d7``

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 训练数据
   * - 观测
     - 主相机与腕部相机 RGB 图像；8 维末端与夹爪状态
   * - 动作
     - 16 步、每步 7 维的笛卡尔位移、旋转增量和夹爪控制
   * - 指令
     - 示范轨迹中的任务描述

安装
----

.. include:: _setup_common.rst

安装并激活 GR00T-N1.7 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
   source .venv/bin/activate

也可以激活此前按上述模型和环境选项安装的已有环境。后续命令均在所选环境、RLinf 仓库根目录执行。示例配置使用单机八卡；调整 GPU 分配时，同时调整 batch size。

下载模型与数据
--------------

先在 Hugging Face 接受 `Cosmos-Reason2-2B <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`_ 的访问条款，再下载 GR00T 权重以及 backbone 配置和 tokenizer：

.. code-block:: bash

   hf auth login
   hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
   hf download nvidia/Cosmos-Reason2-2B \
     --include '*.json' '*.txt' --local-dir checkpoints/Cosmos-Reason2-2B

GR00T checkpoint 包含 backbone 权重，Cosmos 目录用于提供加载所需的配置和 tokenizer。

下载 `Spatial LeRobot 数据集 <https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot>`_，并复制 GR00T 的模态描述文件：

.. code-block:: bash

   hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
     --repo-type dataset --local-dir datasets/libero_spatial
   cp "${GR00T_PATH:-.venv/gr00t}/examples/LIBERO/modality.json" \
     datasets/libero_spatial/meta/modality.json

保留完整的 ``meta/``、``data/`` 和 ``videos/`` 目录。训练会计算这份数据的归一化统计，并随 checkpoint 保存。

运行
----

编辑 ``examples/sft/config/libero_sft_gr00t_n1d7.yaml``，设置数据集、模型权重和 backbone 配置的路径，再设置 GPU 分配、batch size 和训练预算：

.. code-block:: yaml

   cluster:
     component_placement:
       actor: 0-7
   data:
     train_data_paths:
       - /absolute/path/to/datasets/libero_spatial
   runner:
     max_steps: 20000
     save_interval: 1000
   actor:
     micro_batch_size: 1
     global_batch_size: 32
     model:
       model_path: /absolute/path/to/checkpoints/GR00T-N1.7-3B
       backbone_model_path: /absolute/path/to/checkpoints/Cosmos-Reason2-2B
     optim:
       total_training_steps: 20000

全局 batch size 必须能被 GPU 数与 microbatch size 的乘积整除。保留 ``runner.max_epochs: -1``，训练持续采样，到 ``runner.max_steps`` 指定的更新次数后停止。``actor.optim.total_training_steps`` 决定学习率计划长度。20,000 次更新是示例预算，不代表已经测得的收敛目标。

使用修改后的配置启动单机训练。指定本地 Ray 实例，避免连入机器上的其他集群：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   export RAY_ADDRESS=local
   bash examples/sft/run_vla_sft.sh libero_sft_gr00t_n1d7

启动脚本将控制台日志和 TensorBoard 事件写入 ``logs/<timestamp>-libero_sft_gr00t_n1d7/``。训练过程中查看 ``train/loss`` 和全局更新次数；checkpoint 位于该运行目录下的 ``libero_sft_gr00t_n1d7/checkpoints/global_step_N/``。

恢复训练
~~~~~~~~

在原 YAML 中，将 ``runner.resume_dir`` 设为已保存的 ``global_step_N`` 目录的绝对路径，再执行上面的启动命令。保持数据集、GPU 拓扑、batch size 和 dataloader worker 数不变。``runner.max_steps`` 表示包含已完成更新在内的总更新次数。

短程检查保存与恢复流程时，首次启动设置 ``runner.max_steps: 10``、``runner.save_interval: 10``、``actor.optim.lr_warmup_steps: 0`` 和 ``actor.optim.total_training_steps: 20``。运行结束后，把 ``runner.max_steps`` 改为 20，并将 ``runner.resume_dir`` 指向刚保存的 ``global_step_10``，再启动一次。第二次运行应从第 10 次更新继续，保存 ``global_step_20``。checkpoint 恢复详见 :doc:`../../guides/resume`。

.. note::

   dataloader 通过重放数据流恢复读取位置，随机图像增强的结果不保证与恢复前完全一致。

可视化与结果
------------

启动 TensorBoard 查看训练记录：

.. code-block:: bash

   tensorboard --logdir logs

用 ``train/loss`` 观察动作拟合情况，并确认保存的更新目录与预算一致。checkpoint 包含模型、优化器、训练归一化统计、processor 设置和 dataloader 恢复状态。指标定义见 :doc:`../../reference/metrics`。

闭环评测还需要准备包含匹配 processor 设置的模型目录。评测流程见 :doc:`../../evaluations/index`，LIBERO 环境准备见 :doc:`gr00t`。

.. warning::

   评测必须配套使用训练权重和随 checkpoint 保存的 processor 设置。``runner.ckpt_path`` 只加载权重，processor 则从 ``rollout.model.model_path`` 读取。SFT 的 ``actor/`` 目录是训练 checkpoint，不是完整的 Hugging Face 模型目录；评测前需准备带有匹配 processor 的模型目录。
