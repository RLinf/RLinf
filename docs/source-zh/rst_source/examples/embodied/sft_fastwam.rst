FastWAM 监督微调与评测
========================

.. figure:: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
   :align: center
   :width: 90%

   Fast-WAM 使用视频扩散世界-动作模型预测机器人动作。

RLinf 支持使用四个 LIBERO suite 的 LeRobot 数据对 FastWAM 进行 FSDP 监督微调，
并在对应 suite 上完成评测。策略输入为主视角、腕部 RGB 图像、8 维机器人状态和
语言指令，每次预测 32 步动作，执行 10 步后重新规划。

安装
----

.. include:: _setup_common.rst

华为昇腾请参阅 :ref:`sft-fastwam-hardware`；下面的命令用于安装 NVIDIA 环境。

进行 SFT 时安装 FastWAM：

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam
   source .venv/bin/activate

进行评测时安装 FastWAM 和 LIBERO：

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam --env libero
   source .venv/bin/activate

监督微调
--------

下载并解压四套 LIBERO LeRobot 数据：

.. code-block:: bash

   huggingface-cli download yuanty/LIBERO-fastwam \
     --repo-type dataset \
     --local-dir /your_path_to/LIBERO-fastwam

   for archive in /your_path_to/LIBERO-fastwam/*.tar.gz; do
     tar -xzf "$archive" -C /your_path_to/LIBERO-fastwam
   done

下载归一化统计文件：

.. code-block:: bash

   huggingface-cli download yuanty/fastwam \
     libero_uncond_2cam224_dataset_stats.json \
     --local-dir /your_path_to/fastwam

下载 Wan 组件并生成 ActionDiT backbone：

.. code-block:: bash

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
     --local-dir /your_path_to/checkpoints/Wan-AI/Wan2.2-TI2V-5B
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
     --include "google/umt5-xxl/*" \
     --local-dir /your_path_to/checkpoints/Wan-AI/Wan2.1-T2V-1.3B

   export DIFFSYNTH_MODEL_BASE_PATH=/your_path_to/checkpoints
   python .venv/FastWAM/scripts/preprocess_action_dit_backbone.py \
     --model-config .venv/FastWAM/configs/model/fastwam.yaml \
     --output /your_path_to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
     --device cuda \
     --dtype bfloat16

预计算 T5 embedding：

.. code-block:: bash

   python .venv/FastWAM/scripts/precompute_text_embeds.py \
     task=libero_uncond_2cam224_1e-4 \
     "data.train.dataset_dirs=[/your_path_to/LIBERO-fastwam/libero_spatial_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_object_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_goal_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_10_no_noops_lerobot]" \
     "data.train.text_embedding_cache_dir=/your_path_to/text_embeds_cache/libero" \
     model.redirect_common_files=true

在 ``examples/sft/config/libero_sft_fastwam.yaml`` 和
``examples/sft/config/model/fastwam.yaml`` 中填写路径。关键字段如下：

.. code-block:: yaml

   data:
     # 四个 LIBERO suite 的 LeRobot 数据。
     train_data_paths:
       - /your_path_to/LIBERO-fastwam/libero_spatial_no_noops_lerobot
       - /your_path_to/LIBERO-fastwam/libero_object_no_noops_lerobot
       - /your_path_to/LIBERO-fastwam/libero_goal_no_noops_lerobot
       - /your_path_to/LIBERO-fastwam/libero_10_no_noops_lerobot
     text_embedding_cache_dir: /your_path_to/text_embeds_cache/libero  # 预计算的 T5 embedding

   actor:
     model:
       model_path: null  # 从 base model 初始化。
       dataset_stats_path: /your_path_to/fastwam/libero_uncond_2cam224_dataset_stats.json
       freeze_non_dit: true  # 仅训练 MoT expert 和 proprio encoder。
       fastwam:
         overrides:
           # 预处理后的 ActionDiT backbone。
           - model.action_dit_pretrained_path=/your_path_to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

启动 SFT：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_fastwam

.. _sft-fastwam-hardware:

在不同硬件后端上运行
----------------------------------------

NVIDIA 使用上面的安装与启动步骤。华为昇腾 CANN 支持 FastWAM SFT 和 LIBERO 评测，沿用相同的模型权重、归一化统计、数据集和文本 embedding 缓存。

华为昇腾 CANN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用昇腾 LIBERO 容器，或已安装 CANN 与 NPU 驱动的宿主机：

.. include:: _ascend_libero.rst

在容器内使用 RLinf 代码创建 FastWAM 环境，也可以直接在昇腾宿主机上运行相同命令：

.. code-block:: bash

   bash requirements/install.sh --platform ascend embodied --model fastwam --env libero
   source .venv/bin/activate

仅运行 SFT 时可省略 ``--env libero``。国内下载可添加 ``--use-mirror``。安装器会选择昇腾 PyTorch 环境，并跳过 CUDA flash-attention。

按上文准备 Wan 组件、ActionDiT backbone、数据集和 T5 embedding 缓存。在昇腾主机上预处理 ActionDiT 时，将 ``--device cuda`` 改为 ``--device cpu``。CUDA 不可用时，文本 embedding 脚本也会使用 CPU。保持 ``DIFFSYNTH_MODEL_BASE_PATH`` 指向包含 ``Wan-AI`` 的目录。使用原始 Wan ``.pth`` 文件时，在上游模型 overrides 和文本 embedding 命令中设置 ``model.redirect_common_files=false``。

按上文在 ``examples/sft/config/libero_sft_fastwam.yaml`` 和 ``examples/sft/config/model/fastwam.yaml`` 中填写数据与模型路径。使用八张 NPU 时，可先设置 ``actor.micro_batch_size: 1`` 和 ``actor.global_batch_size: 8``，并启用 ``actor.fsdp_config.gradient_checkpointing`` 以降低显存占用。根据可用设备调整 ``cluster.component_placement.actor``，同时保证 global batch size 能被 micro batch size 与 NPU 数量之积整除。然后在仓库根目录使用相同配置启动 SFT：

.. code-block:: bash

   export DIFFSYNTH_MODEL_BASE_PATH=/your_path_to/checkpoints
   bash examples/sft/run_vla_sft.sh libero_sft_fastwam

评测时，按下文设置 checkpoint 和归一化统计路径，将 MuJoCo 与 PyOpenGL 设为通过 OSMesa 使用 CPU 渲染，再运行相同的 LIBERO 评测入口：

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa
   bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

在运行评测的容器或宿主机中安装 OSMesa 运行库。将 ``libero_spatial_fastwam_eval`` 替换为下表中的其他 suite 配置即可评测对应任务。这些配置默认将 rollout worker 放在四张设备上，可通过 ``cluster.component_placement`` 调整使用的 NPU。

RLinf 构建 FastWAM 时会选择 worker 所在的设备。在 NPU worker 上，昇腾适配自动生效，并为视频与动作 DiT expert 启用融合 RoPE 和 RMSNorm。

评测
----

下载发布的 checkpoint：

.. code-block:: bash

   huggingface-cli download yuanty/fastwam \
     libero_uncond_2cam224.pt \
     --local-dir /your_path_to/fastwam

在 ``examples/embodiment/config/model/fastwam.yaml`` 中设置路径：

.. code-block:: yaml

   model_path: /your_path_to/fastwam/libero_uncond_2cam224.pt  # FastWAM checkpoint
   dataset_stats_path: /your_path_to/fastwam/libero_uncond_2cam224_dataset_stats.json  # 归一化统计文件

分别运行四个 suite：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval
   bash evaluations/run_eval.sh libero libero_object_fastwam_eval
   bash evaluations/run_eval.sh libero libero_goal_fastwam_eval
   bash evaluations/run_eval.sh libero libero_10_fastwam_eval

配置使用 4 张 GPU，因为 20 个环境槽位可以均匀分配给 4 个 worker。每个环境
执行 25 个 episode，共覆盖 ``20 * 25 = 500`` 个初始状态；同样的 20 个环境
槽位无法均匀分配给 8 个 worker。

.. list-table::
   :header-rows: 1
   :widths: 18 42 20 20

   * - Suite
     - 配置
     - ``max_episode_steps``
     - ``max_steps_per_rollout_epoch``
   * - Spatial
     - ``libero_spatial_fastwam_eval``
     - ``400``
     - ``10000``
   * - Object
     - ``libero_object_fastwam_eval``
     - ``400``
     - ``10000``
   * - Goal
     - ``libero_goal_fastwam_eval``
     - ``400``
     - ``10000``
   * - Long
     - ``libero_10_fastwam_eval``
     - ``700``
     - ``17500``
