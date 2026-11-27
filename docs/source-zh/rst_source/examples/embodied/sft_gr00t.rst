GR00T-N1.7 监督微调
===================

在 LIBERO Spatial 示范数据上用 FSDP 全参数微调 GR00T-N1.7，保存可恢复训练的 checkpoint，并直接在 LIBERO Spatial 上评测微调后的模型。

概览
----

模型接收主相机和腕部相机 RGB 图像、机器人状态以及任务指令，预测 16 步动作块。

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

      Physical Intelligence LIBERO

   .. grid-item-card:: 硬件
      :text-align: center

      8 张 NVIDIA GPU · FSDP

| **你将完成：** 安装环境 → 下载模型与数据 → 启动 ``run_vla_sft.sh`` → 观察 ``train/loss`` 并评测。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 已获取 ``nvidia/Cosmos-Reason2-2B`` 的 Hugging Face 访问权限 · 8 张 NVIDIA GPU。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 任务套件
     - 示范数据
     - 配置
   * - LIBERO Spatial（10 个任务）
     - 432 条轨迹，52,970 个时间步，10 Hz
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
     - 16 步、每步 7 维的笛卡尔位移、旋转增量与夹爪控制
   * - 指令
     - 示范轨迹附带的任务描述

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO Spatial

      **1. 安装**

      克隆仓库并安装、激活 ``gr00t_n1d7`` 环境：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
         source .venv/bin/activate

      **2. 下载模型与数据**

      先在 Hugging Face 接受 `Cosmos-Reason2-2B <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`_ 的访问条款并登录 ``hf auth login``，再下载基座权重、backbone 配置与 tokenizer 以及 `Physical Intelligence LIBERO <https://huggingface.co/datasets/physical-intelligence/libero>`_ 数据集：

      .. code-block:: bash

         hf auth login
         hf download nvidia/GR00T-N1.7-3B --local-dir ./checkpoints/GR00T-N1.7-3B
         hf download nvidia/Cosmos-Reason2-2B --include '*.json' '*.txt' --local-dir ./checkpoints/Cosmos-Reason2-2B
         hf download physical-intelligence/libero --repo-type dataset --local-dir ./datasets/physical-intelligence-libero

      ``GR00T-N1.7-3B`` 提供预训练权重，``Cosmos-Reason2-2B`` 提供 backbone 配置和 tokenizer。数据集包含 40 个任务、1,693 条轨迹，图像内嵌在 Parquet 文件中；``examples/sft/config/libero_sft_gr00t_n1d7.yaml`` 会筛选 10 个 Spatial 任务，将夹爪动作转为 GR00T 格式，并缓存归一化统计量。

      **3. 启动训练**

      在仓库根目录启动单机 FSDP 训练：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_gr00t_n1d7 \
           data.train_data_paths="$PWD/datasets/physical-intelligence-libero" \
           actor.model.model_path="$PWD/checkpoints/GR00T-N1.7-3B" \
           actor.model.backbone_model_path="$PWD/checkpoints/Cosmos-Reason2-2B"

      日志与 TensorBoard 事件保存在 ``logs/<timestamp>-libero_sft_gr00t_n1d7/``，checkpoint 按 ``runner.save_interval`` 步数保存在 ``libero_sft_gr00t_n1d7/checkpoints/global_step_<N>/``。需要断点续训时，传入 ``runner.resume_dir="$PWD/logs/.../checkpoints/global_step_<N>"`` 并保持数据集、GPU 数量、batch size 与 dataloader worker 数不变（详见 :doc:`../../guides/resume`\ ；数据流重放不保证随机图像增强逐位一致）。

      .. dropdown:: 高级配置（可选）

         调整硬件分配或训练步数时，可在 ``examples/sft/config/libero_sft_gr00t_n1d7.yaml`` 或命令行覆盖以下配置项：

         .. list-table::
            :header-rows: 1
            :widths: 34 66

            * - 配置项
              - 说明
            * - ``cluster.component_placement.actor``
              - FSDP 训练使用的 GPU 编号（默认 ``0-7``）。
            * - ``actor.micro_batch_size`` / ``actor.global_batch_size``
              - 单卡 micro-batch size 与全局 batch size（默认 ``1`` 与 ``32``）；全局 batch size 需能被 GPU 数与 ``micro_batch_size`` 的乘积整除。
            * - ``runner.max_steps`` / ``runner.save_interval``
              - 总训练步数与保存间隔（保持 ``runner.max_epochs: -1``）。
            * - ``actor.optim.total_training_steps``
              - 余弦学习率调度的总步数。
            * - ``data.statistics_path``
              - 按数据集和任务子集哈希缓存统计量的文件前缀；每个 checkpoint 也会同步保存统计量。

.. dropdown:: 评测 SFT checkpoint（可选）

   参考 :doc:`../../evaluations/guides/libero` 直接在 LIBERO Spatial 上评测 ``global_step_<N>``\ （如需继续强化学习后训练，见 :doc:`gr00t`\ ）：

   .. code-block:: bash

      bash evaluations/run_eval.sh libero libero_spatial_gr00t_n1d7_eval \
        rollout.model.model_path="$PWD/logs/<run_dir>/libero_sft_gr00t_n1d7/checkpoints/global_step_3000" \
        rollout.model.backbone_model_path="$PWD/checkpoints/Cosmos-Reason2-2B"

   ``evaluations/libero/libero_spatial_gr00t_n1d7_eval.yaml`` 默认关闭 value head 并设置 ``env.eval.ignore_terminations=False``，直接从 checkpoint 目录读取 processor、归一化统计量与权重，在 40 个并行环境中完成全部 500 条轨迹评测。

可视化与结果
------------

启动 TensorBoard 查看 ``train/loss`` 与训练进度（指标定义见 :doc:`../../reference/metrics`\ ）：

.. code-block:: bash

   tensorboard --logdir ./logs

在 8 张 NVIDIA A800、全局 batch size 为 64 的配置下训练 3,000 步后，LIBERO Spatial 实测结果如下：

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - 模型
     - 配置
     - LIBERO Spatial（``success_once``）
   * - 基座模型（``nvidia/GR00T-N1.7-3B``）
     - 500 条轨迹
     - ``0/500``\ （0.0%）
   * - SFT（3,000 步）
     - 8× A800 · GBS 64 · 500 条轨迹
     - ``367/500``\ （73.4%）
