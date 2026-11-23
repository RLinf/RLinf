在 RMBench 上微调带 Short Memory 的 π₀.₅
================================================

用公开的 Aloha 演示对 RoboTwin π₀.₅ checkpoint 做 SFT，再到 `RMBench <https://rmbench.github.io/>`__ 模拟器里评测。RMBench 的任务需要用到先前的观测。例如，物体被挡住后，policy 仍要记得它的位置。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型

      带 Short Memory 的 π₀.₅

   .. grid-item-card:: 算法

      SFT

   .. grid-item-card:: 任务

      Aloha ``demo_clean``

   .. grid-item-card:: 硬件

      NVIDIA GPU

任务
~~~~

.. list-table::
   :header-rows: 1

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

   * - 字段
     - 说明
   * - 观测
     - 头部和双腕相机的 RGB 图像、当前关节和夹爪位置，以及语言指令。
   * - 历史
     - 原始数据为 15 Hz。每个相机每隔 53 步取一帧，共取 8 帧（含当前帧），覆盖约 24.7 秒。
   * - 动作
     - 每次预测 32 步，每步包含双臂各 6 个关节和 1 个夹爪的位置目标。训练使用关节增量和绝对夹爪位置。

Short Memory 如何使用这些历史帧，见 :doc:`Short Memory <pi0/short_memory>`。

安装
----

在 RLinf 根目录装好 OpenPI 训练环境，再检出指定版本的 RMBench。训练和评测都用这个 RMBench 目录。

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env rmbench
   source .venv/bin/activate
   git clone https://github.com/RoboTwin-Platform/RMBench.git
   git -C RMBench checkout 7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7

准备演示数据
------------

下载指定版本的 `RMBench 演示数据 <https://huggingface.co/datasets/TianxingChen/RMBench/tree/5d8ff21038fba32ea549e9eee8497aaba9aab5cf>`__，再把 HDF5 转成 LeRobot v2.1。脚本按任务划分训练集和验证集，只用训练集生成 ``norm_stats.json``。

.. code-block:: bash

   bash RMBench/scripts/_download_data.sh --revision 5d8ff21038fba32ea549e9eee8497aaba9aab5cf
   python -m toolkits.rmbench.prepare_data --dataset-root RMBench/data/demo_clean \
     --output-dir rmbench_lerobot

转换结束后会得到 ``rmbench_lerobot/train`` 和 ``rmbench_lerobot/val``。图像写在 LeRobot 的 Parquet 文件里，磁盘需要同时容纳下载的数据和转换后的数据。保留 ``rmbench_lerobot/norm_stats.json``，训练和评测都会用到。

下载模型
--------

下载 SFT 用的起始 checkpoint `RoboTwin π₀.₅ <https://huggingface.co/motus-robotics/pi0.5_robotwin2/tree/effc6e1a99bd4d0edf9447e8ba313def67649cfe>`__。

.. code-block:: bash

   hf download motus-robotics/pi0.5_robotwin2 model.safetensors \
     --revision effc6e1a99bd4d0edf9447e8ba313def67649cfe --local-dir checkpoints/rmbench_base

运行
----

在 ``examples/sft/config/rmbench_sft_openpi_pi05.yaml`` 中填写路径。

.. list-table::
   :header-rows: 1

   * - 配置项
     - 取值
   * - ``data.train_data_paths``
     - ``rmbench_lerobot/train`` 的绝对路径
   * - ``actor.model.model_path``
     - ``checkpoints/rmbench_base``
   * - ``actor.model.openpi_data.norm_stats_path``
     - ``rmbench_lerobot/norm_stats.json``

这个配置训练全部权重，跑 11,642 步，global batch size 为 128，约等于 5 个 epoch。原来的训练用了 8 张 80 GB GPU，每张卡的 micro-batch size 为 8。换用其他 GPU 数量时，检查 ``actor.micro_batch_size``。GPU 数量乘以每卡 batch size 必须能整除 128。

在 RLinf 根目录启动训练。

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh rmbench_sft_openpi_pi05

如果只用当前帧做 SFT，把同一份配置里的 ``actor.model.openpi.short_memory`` 设为 ``false``。

可视化与结果
------------

用 TensorBoard 打开 ``logs/<run>/`` 查看训练 loss。评测要填 checkpoint 目录，路径是 ``logs/<run>/rmbench_pi05_short_memory_sft/checkpoints/global_step_<N>/``。准备好这个目录和训练用的 ``norm_stats.json``，接着按 :doc:`RMBench 评测指南 <../../evaluations/guides/rmbench>` 跑模拟器。
