OpenWAM 监督微调
================

用演示数据微调已发布的 OpenWAM LIBERO checkpoint，再导出模型，用 LIBERO 评估任务成功率。

概览
----

本示例用 FSDP2 联合微调视频预测和动作预测模块。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenWAM Alpha

   .. grid-item-card:: 方法
      :text-align: center

      SFT · FSDP2

   .. grid-item-card:: 数据集
      :text-align: center

      OpenWAM LIBERO

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 张 GPU

**前置条件：**\ 参照 :doc:`安装指南 </rst_source/start/installation>` 配置环境，并准备 8 张至少 80 GB 显存的 GPU。本配置已在 8 张 A800 80 GB 上运行：保留 fp32 主权重，用 bf16 计算，每张 GPU 处理 1 个样本，全局 batch 为 8。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO

      **1. 安装**

      克隆 RLinf，安装 OpenWAM 和 LIBERO，并激活环境：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openwam --env libero
         source .venv/bin/activate

      **2. 下载**

      下载模型和配套的 LeRobot v3 数据集：

      .. code-block:: bash

         hf download OpenWAM/OpenWAM-Alpha-Sim-LIBERO --local-dir ./OpenWAM-Alpha-Sim-LIBERO
         hf download OpenWAM/LIBERO --repo-type dataset --local-dir ./LIBERO

      保留完整的数据集目录，包括视频和元数据。

      **3. 启动**

      按默认配置训练 30,000 步，将日志和 checkpoint 保存到 ``./logs/openwam-sft``\：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_openwam \
           actor.model.model_path=./OpenWAM-Alpha-Sim-LIBERO \
           data.train_data_paths=./LIBERO \
           runner.logger.log_path=./logs/openwam-sft

      每 1,000 步保存一次 checkpoint，目录为 ``./logs/openwam-sft/libero_sft_openwam/checkpoints/global_step_<N>/``\。

      .. dropdown:: 调整训练配置

         8 张 A800 80 GB 可以运行本配置。4 卡运行需要单卡显存更大的 GPU，例如 H200。将 ``cluster.component_placement.actor`` 设为所用 GPU，并确保 ``actor.global_batch_size`` 能被 GPU 数量整除。

         缩短训练时，将 ``runner.max_steps`` 和 ``actor.optim.total_training_steps`` 设为相同的目标步数，并将 ``actor.optim.lr_warmup_steps`` 设为小于总步数的值。将这些覆盖参数追加到上面的启动命令即可。

导出与评估
----------

例如，将第 1,000 步的 checkpoint 导出为 OpenWAM 可直接加载的模型目录：

.. code-block:: bash

   python toolkits/openwam/export_checkpoint.py \
     --rlinf-checkpoint ./logs/openwam-sft/libero_sft_openwam/checkpoints/global_step_1000 \
     --source-checkpoint ./OpenWAM-Alpha-Sim-LIBERO \
     --output ./openwam-libero-step1000

``--source-checkpoint`` 应与训练使用的源模型相同，以保留匹配的配置和分词器。导出还会携带训练保存的归一化统计量。可追加 ``--verify cuda``\，在空闲 GPU 上检查模型是否能加载。

用 OpenWAM 的评估配置运行导出模型：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openwam_eval \
     rollout.model.model_path=./openwam-libero-step1000

默认评估占用 4 张 GPU，运行 3 个 episode。评估各套件的全部任务和初始状态，参见 :ref:`OpenWAM 完整覆盖配置 <openwam-libero-full-coverage>`\。

可视化与结果
------------

打开 TensorBoard 查看总 loss、视频 loss 和动作 loss：

.. code-block:: bash

   tensorboard --logdir ./logs/openwam-sft/tensorboard

对应指标为 ``train/loss``\、\ ``train/loss_video`` 和 ``train/loss_action``\。通用指标参考见 :doc:`训练指标 <../../reference/metrics>`\，独立评估流程见 :doc:`LIBERO 评测 <../../evaluations/guides/libero>`\。

本次参考实验用 8 张 A800 80 GB 微调已发布的 LIBERO checkpoint，约 50 分钟训练 1,000 步，全局 batch 为 8，学习率为 1e-6，warmup 为 100 步，随后按余弦调度衰减。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3a760f24bdc20480a8feba57ecb659ae4111648f/pic/openwam/libero-sft-1000-loss.png
   :align: center
   :width: 100%

   总 loss、视频 loss 和动作 loss 的训练曲线。浅色线为原始数值，橙色线为 50 步滑动平均。日志步数 0–999 对应 1,000 次优化器更新。

训练 1,000 步后，两组 checkpoint 分别评估每套件全部 500 个任务与初始状态组合，即 10 个任务、每任务 50 个初始状态。两组使用相同组合，沿用 OpenWAM 评估预设。

.. list-table:: LIBERO 成功率
   :header-rows: 1

   * - 套件
     - 已发布模型
     - 导出的第 1,000 步模型
   * - Spatial
     - 496/500（99.2%）
     - 489/500（97.8%）
   * - Object
     - 497/500（99.4%）
     - 497/500（99.4%）
   * - Goal
     - 496/500（99.2%）
     - 477/500（95.4%）
   * - Long
     - 495/500（99.0%）
     - 482/500（96.4%）
   * - 四套件合计
     - 1984/2000（99.20%）
     - 1945/2000（97.25%）

训练 loss 下降，总成功率下降了 1.95 个百分点。该结果用于参考 1,000 步训练流程；训练没有单独的验证集。

.. dropdown:: 复现本次结果

   在训练命令后追加 ``runner.max_steps=1000``\、\ ``actor.optim.total_training_steps=1000`` 和 ``actor.optim.lr_warmup_steps=100``\，再导出第 1,000 步模型。使用 :ref:`OpenWAM 完整覆盖配置 <openwam-libero-full-coverage>` 时，Spatial、Object、Goal 设置 ``env.eval.max_steps_per_rollout_epoch=75000``\，Long 设置 ``env.eval.max_steps_per_rollout_epoch=87500``\。

.. dropdown:: 验证与断点恢复

   将单独留出的演示数据作为验证集，设置 ``data.val_data_paths`` 和 ``runner.val_check_interval``\，并沿用训练的归一化统计量。验证集的划分名为 ``train`` 时，设置 ``data.openwam_val_split=train``\。

   验证指标为 ``eval/loss``\、\ ``eval/loss_video`` 和 ``eval/loss_action``\。

   恢复训练时，将 ``runner.resume_dir`` 设为已保存的 ``global_step_<N>`` 目录的绝对路径，该目录应包含 ``actor/``\。默认配置要求 checkpoint 保存数据流和随机数状态，以准确恢复训练。详见 :doc:`../../guides/resume`\。

.. dropdown:: 其他数据源

   数据集须匹配源模型 ``config.yaml`` 中的 ``dataloader.type``\，包括图像字段、机器人状态、动作表示和归一化方式。单数据源使用 ``libero_sft_openwam``\；源模型配置为 ``mixture`` 时使用 ``mixture_sft_openwam``\。

   混合训练的 ``data.train_data_paths`` 指向根目录，每个启用的数据源默认位于 ``<name>`` 子目录。需要单独指定位置时，设置 ``data.openwam.datasets.<name>.dataset_dir``\，并保留该数据源的归一化统计量。
