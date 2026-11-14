OpenWAM 监督微调
================

RLinf 支持使用 OpenWAM 的 LIBERO 数据对 OpenWAM 进行 FSDP 监督微调。本指南以发布的 LIBERO checkpoint 为起点，介绍权重与数据准备、训练配置，以及训练后导出和评测的流程。

安装
----

在 RLinf 仓库根目录安装 OpenWAM 和 LIBERO 环境，并激活环境。已有环境可直接激活：

.. code-block:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

监督微调
--------

准备模型权重
~~~~~~~~~~~~

下载 OpenWAM 发布的 LIBERO checkpoint：

.. code-block:: bash

   hf download OpenWAM/OpenWAM-Alpha-Sim-LIBERO --local-dir /your_path_to/OpenWAM-Alpha-Sim-LIBERO

下载目录应包含权重文件、模型配置、归一化统计文件和 tokenizer。该 checkpoint 已包含所需的模型组件，无需另外下载 Wan 基座模型。后续的 ``actor.model.model_path`` 应指向整个目录，而不是单个权重文件：

.. code-block:: text

   /your_path_to/OpenWAM-Alpha-Sim-LIBERO/
   ├── checkpoint_step_10690.safetensors
   ├── config.yaml
   ├── normalization_stats.npy
   └── tokenizer/
       └── google/umt5-xxl/

准备数据集
~~~~~~~~~~

下载与该 checkpoint 配套的 OpenWAM LIBERO 数据集：

.. code-block:: bash

   hf download OpenWAM/LIBERO --repo-type dataset --local-dir /your_path_to/libero

该数据集已经是 LeRobot v3 格式，无需解压或额外转换。请保留完整的 ``data/``、``meta/`` 和 ``videos/`` 目录，其中 ``meta/libero_normalization_stats.npy`` 是数据集的归一化统计文件：

.. code-block:: text

   /your_path_to/libero/
   ├── data/
   ├── meta/
   │   ├── info.json
   │   ├── tasks.parquet
   │   ├── episodes/
   │   └── libero_normalization_stats.npy
   └── videos/
       ├── observation.images.image/
       └── observation.images.image2/

将 ``data.train_data_paths`` 设置为包含这三个目录的数据集根目录。训练会使用 checkpoint 中的 LIBERO 数据配置；使用自定义数据时，请保持图像字段、机器人状态、动作表示和归一化约定一致。

填写训练配置
~~~~~~~~~~~~

在 ``examples/sft/config/model/openwam.yaml`` 中设置下载后的 checkpoint 路径，并保留训练使用的精度与加载方式：

.. code-block:: yaml

   model_path: /your_path_to/OpenWAM-Alpha-Sim-LIBERO
   precision: fp32
   load_to_device: false

在 ``examples/sft/config/libero_sft_openwam.yaml`` 中设置数据路径。关键字段如下；其余字段可保留配方默认值：

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-7

   data:
     train_data_paths: /your_path_to/libero

   runner:
     max_steps: 30000
     save_interval: 1000

   actor:
     micro_batch_size: 1
     global_batch_size: 8
     optim:
       lr: 1.0e-6
       total_training_steps: 30000

配方默认使用 8 张 GPU、FSDP2 和 bf16 混合精度计算，同时保留 fp32 主权重。修改 GPU 数量时，需要同步修改 ``cluster.component_placement.actor``，并确保 ``actor.global_batch_size`` 能被 actor 的 GPU 数量整除；修改训练步数时，也应同步调整 ``actor.optim.total_training_steps``。

.. note::

   在 H200 上，使用 LIBERO 数据、``micro_batch_size: 1`` 和 ``global_batch_size: 8`` 的已有测试中，8 卡训练每卡峰值已分配显存为 55.5 GiB，保留显存为 83.7 GiB；4 卡时分别为 72.3 GiB 和 113.3 GiB。

启动训练与查看结果
~~~~~~~~~~~~~~~~~~

在 RLinf 仓库根目录运行：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openwam

启动脚本会把本次运行的日志目录设置为 ``logs/<timestamp>-libero_sft_openwam/``。通过 ``run_embodiment.log`` 查看训练日志，或在 TensorBoard 中查看 ``train/loss``、``train/loss_video`` 和 ``train/loss_action``。将下面的 RUN_DIR 替换为实际运行目录：

.. code-block:: bash

   RUN_DIR="/absolute/path/to/RLinf/logs/<timestamp>-libero_sft_openwam"
   tensorboard --logdir "$RUN_DIR/tensorboard"

checkpoint 默认每 1000 步保存一次，位于 ``$RUN_DIR/libero_sft_openwam/checkpoints/global_step_<N>/``。

导出与评测
----------

训练保存的是 RLinf checkpoint。先将其导出为 OpenWAM checkpoint 目录，再用于推理或评测。使用上一步的 RUN_DIR，导出第 1000 步的 checkpoint：

.. code-block:: bash

   python toolkits/openwam/export_checkpoint.py \
     --rlinf-checkpoint "$RUN_DIR/libero_sft_openwam/checkpoints/global_step_1000" \
     --source-checkpoint /your_path_to/OpenWAM-Alpha-Sim-LIBERO \
     --output /your_path_to/openwam-libero-step1000

``--source-checkpoint`` 应使用训练起点的 checkpoint 目录，以保留配套的配置、tokenizer 和归一化文件。导出脚本会处理 RLinf 权重分片；如果需要检查导出结果能否在 GPU 上重新加载，可以额外传入 ``--verify cuda``。

在 LIBERO 评测配置中将 ``rollout.model.model_path`` 设置为导出目录 ``/your_path_to/openwam-libero-step1000``，再按照 :doc:`../../evaluations/guides/libero` 中的 OpenWAM 说明运行评测。RoboTwin checkpoint 的评测流程见 :doc:`../../evaluations/guides/robotwin`。

可选配置
--------

验证与断点续训
~~~~~~~~~~~~~~

启用训练期间的验证时，将 ``data.val_data_paths`` 指向独立的验证数据集，并设置 ``runner.val_check_interval``。上文下载的 OpenWAM LIBERO 数据集只有 ``train`` split；不要直接把同一份训练数据作为验证集。如果独立验证数据集也只提供 ``train`` split，需要设置 ``data.openwam_val_split: train``，并确保其使用与训练一致的归一化统计量。验证指标记录为 ``eval/loss``、``eval/loss_video`` 和 ``eval/loss_action``。

断点续训时，在训练配置中将 ``runner.resume_dir`` 设置为 ``$RUN_DIR/libero_sft_openwam/checkpoints/global_step_<N>`` 对应的实际绝对路径，而不是其下的 ``actor/`` 目录。配方默认启用 ``runner.strict_resume: true``，要求 checkpoint 包含数据流和随机数状态；使用缺少这些状态的旧 checkpoint 时会报错。更多说明见 :doc:`../../guides/resume`。

其他数据源与混合训练
~~~~~~~~~~~~~~~~~~~~

使用其他 OpenWAM checkpoint 时，先确认其 ``config.yaml`` 中的 ``dataloader.type`` 与数据集格式一致。单数据源使用 ``libero_sft_openwam.yaml`` 配方并替换权重与数据路径；仅当 checkpoint 配置为 ``mixture`` 时使用 ``mixture_sft_openwam.yaml``。

混合训练的 ``data.train_data_paths`` 只填一个根目录，各启用的数据源默认从 ``<root>/<name>`` 读取。需要单独指定某个数据源的位置时，修改 ``data.openwam.datasets.<name>.dataset_dir``，并保留该数据源配套的归一化统计文件。
