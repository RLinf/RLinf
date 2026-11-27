RMBench π₀.₅ Short Memory 微调
==============================

使用公开的 Aloha 演示数据微调 RoboTwin π₀.₅ checkpoint，并在 `RMBench <https://rmbench.github.io/>`__ 模拟器中评测。RMBench 考察 policy 能否利用历史观测完成任务，例如在物体被遮挡后仍记住它的位置。

概览
----

policy 根据头部与双腕相机的 8 帧历史 RGB 图像、本体关节状态以及语言指令，预测 32 步双臂关节和夹爪目标位置。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      带 Short Memory 的 π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      SFT

   .. grid-item-card:: 任务
      :text-align: center

      Aloha ``demo_clean``

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA GPU

| **你将完成：** 安装依赖 → 转换演示数据 → 下载基座权重 → 启动 SFT → 查看 ``train/loss``。
| **前置条件：** :doc:`环境安装 </rst_source/start/installation>` · NVIDIA GPU（默认 SFT batch 配置使用 8 × 80 GB GPU）。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 数据
     - 覆盖范围
   * - 公开的 ``demo_clean`` 数据
     - 10 个任务目录，共 500 条演示
   * - 训练集
     - 450 条演示，298,032 帧
   * - 验证集
     - 50 条演示，32,509 帧

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测
     - 头部和双腕相机的 RGB 图像、当前关节和夹爪位置，以及语言指令。
   * - 历史
     - 原始数据为 15 Hz；每个相机每隔 53 步取一帧，共取 8 帧（含当前帧），覆盖约 24.7 秒。
   * - 动作
     - 每次预测 32 步 14 维 Aloha 目标（双臂各 6 个关节和 1 个夹爪）；训练使用关节增量和绝对夹爪位置。

Short Memory 如何使用历史帧见 :doc:`Short Memory <pi0/short_memory>`。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. 安装**

      在 RLinf 根目录安装 OpenPI 训练环境到 ``.venv``\ （用于数据转换、SFT 与推理；评测时会另建 ``.rmbench-venv`` 隔离 SAPIEN/CuRobo 模拟器并下载仿真资产），然后检出固定版本的 RMBench 仓库：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env rmbench && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RMBench.git
         git -C RMBench checkout 7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7

      **2. 准备演示数据**

      下载指定版本的 `RMBench 演示数据 <https://huggingface.co/datasets/TianxingChen/RMBench/tree/5d8ff21038fba32ea549e9eee8497aaba9aab5cf>`__\ ，并将 HDF5 演示转换为 LeRobot v2.1 格式（按任务划分训练集与验证集，仅用训练集计算 ``rmbench_lerobot/norm_stats.json``\ ）：

      .. code-block:: bash

         bash RMBench/scripts/_download_data.sh --revision 5d8ff21038fba32ea549e9eee8497aaba9aab5cf
         python -m toolkits.rmbench.prepare_data --dataset-root RMBench/data/demo_clean \
           --output-dir rmbench_lerobot

      转换后会生成 ``rmbench_lerobot/train`` 和 ``rmbench_lerobot/val``。图像直接写入 Parquet 文件，请为原始数据和转换后的数据集预留充足磁盘空间。

      **3. 下载模型**

      下载起始权重 `RoboTwin π₀.₅ <https://huggingface.co/motus-robotics/pi0.5_robotwin2/tree/effc6e1a99bd4d0edf9447e8ba313def67649cfe>`__\ ：

      .. code-block:: bash

         hf download motus-robotics/pi0.5_robotwin2 model.safetensors \
           --revision effc6e1a99bd4d0edf9447e8ba313def67649cfe --local-dir ./checkpoints/rmbench_base

      **4. 启动训练**

      默认配置使用 8 张 80 GB GPU（每卡 micro-batch size 为 8，global batch size 为 128），全参数微调 11,642 步（约 5 个 epoch）。调整 GPU 数量时请同步修改 ``actor.micro_batch_size``\ ，确保 GPU 数量乘以每卡 batch size 能整除 128：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh rmbench_sft_openpi_pi05 \
           data.train_data_paths="$PWD/rmbench_lerobot/train" \
           actor.model.model_path="$PWD/checkpoints/rmbench_base" \
           actor.model.openpi_data.norm_stats_path="$PWD/rmbench_lerobot/norm_stats.json"

      若只使用当前帧做 SFT，在启动命令末尾追加 ``actor.model.openpi.short_memory=false``。

可视化与结果
------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看训练 loss，指标定义与 Weights & Biases / SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

训练每 2,329 步保存一次 checkpoint（共 11,642 步），保存在 ``logs/<run>/rmbench_pi05_short_memory_sft/checkpoints/global_step_<N>/``。评测时将该 checkpoint 目录与 ``rmbench_lerobot/norm_stats.json`` 传入 :doc:`RMBench 评测指南 <../../evaluations/guides/rmbench>`\ ，即可在模拟器中统计闭环成功率 ``eval/success_once``。
