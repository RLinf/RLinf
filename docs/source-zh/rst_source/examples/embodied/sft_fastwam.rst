FastWAM 监督微调与评测
========================

.. figure:: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
   :align: center
   :width: 90%

   Fast-WAM 使用视频扩散世界-动作模型预测机器人动作。

RLinf 支持使用四个 LIBERO suite 的 LeRobot 数据对 FastWAM 做 FSDP 监督微调，并在对应 suite 上完成评测。策略输入为主视角、腕部 RGB 图像、8 维机器人状态和语言指令，每次预测 32 步动作，执行 10 步后重新规划。

概览
--------

在四个 LIBERO 套件上使用 FSDP 对 FastWAM 做监督微调，并运行闭环评测。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      FastWAM

   .. grid-item-card:: 方法
      :text-align: center

      FSDP SFT · 评测

   .. grid-item-card:: 数据
      :text-align: center

      LIBERO（LeRobot）

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · :ref:`Huawei Ascend CANN <sft-fastwam-hardware>`

| **你将完成：** 安装环境 → 下载 LIBERO 数据与 Wan 组件 → 预处理 embedding → 启动 ``run_vla_sft.sh`` 或 ``run_eval.sh``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · LIBERO LeRobot 数据集与 Wan2.2 / Wan2.1 权重（见下文）。

.. _sft-fastwam-hardware:

安装与运行
----------

.. include:: _setup_common.rst

NVIDIA CUDA 用户可按 SFT 和评测标签页操作；NPU 用户请使用“华为昇腾 CANN”标签页。

.. tab-set::

   .. tab-item:: SFT 训练（LIBERO）
      :sync: fastwam-sft

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model fastwam && source .venv/bin/activate

      **2. 下载数据与 Wan 组件**

      下载 LIBERO LeRobot 数据集、归一化统计以及 Wan2.2 / Wan2.1 组件：

      .. code-block:: bash

         hf download yuanty/LIBERO-fastwam --repo-type dataset --local-dir ./LIBERO-fastwam
         for archive in ./LIBERO-fastwam/*.tar.gz; do
           tar -xzf "$archive" -C ./LIBERO-fastwam
         done
         hf download yuanty/fastwam libero_uncond_2cam224_dataset_stats.json --local-dir ./fastwam
         hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan-AI/Wan2.2-TI2V-5B
         hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/umt5-xxl/*" --local-dir ./checkpoints/Wan-AI/Wan2.1-T2V-1.3B

      **3. 预处理 ActionDiT backbone 与 T5 embedding**

      .. code-block:: bash

         export DIFFSYNTH_MODEL_BASE_PATH=./checkpoints
         python .venv/FastWAM/scripts/preprocess_action_dit_backbone.py \
           --model-config .venv/FastWAM/configs/model/fastwam.yaml \
           --output ./fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
           --device cuda \
           --dtype bfloat16

         python .venv/FastWAM/scripts/precompute_text_embeds.py \
           task=libero_uncond_2cam224_1e-4 \
           "data.train.dataset_dirs=[./LIBERO-fastwam/libero_spatial_no_noops_lerobot,./LIBERO-fastwam/libero_object_no_noops_lerobot,./LIBERO-fastwam/libero_goal_no_noops_lerobot,./LIBERO-fastwam/libero_10_no_noops_lerobot]" \
           "data.train.text_embedding_cache_dir=./text_embeds_cache/libero" \
           model.redirect_common_files=true

      **4. 在** ``examples/sft/config/libero_sft_fastwam.yaml`` **和** ``examples/sft/config/model/fastwam.yaml`` **中填写路径：**

      .. code-block:: yaml

         # examples/sft/config/libero_sft_fastwam.yaml
         data:
           train_data_paths:
             - ./LIBERO-fastwam/libero_spatial_no_noops_lerobot
             - ./LIBERO-fastwam/libero_object_no_noops_lerobot
             - ./LIBERO-fastwam/libero_goal_no_noops_lerobot
             - ./LIBERO-fastwam/libero_10_no_noops_lerobot
           text_embedding_cache_dir: ./text_embeds_cache/libero

      .. code-block:: yaml

         # examples/sft/config/model/fastwam.yaml
         dataset_stats_path: ./fastwam/libero_uncond_2cam224_dataset_stats.json
         fastwam:
           overrides:
             - model.action_dit_pretrained_path=./fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

      **5. 启动 SFT：**

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_fastwam

   .. tab-item:: 评测（LIBERO）
      :sync: fastwam-evaluation

      **1. 安装含 LIBERO 的环境并下载发布权重与统计文件：**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model fastwam --env libero && source .venv/bin/activate
         hf download yuanty/fastwam libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json --local-dir ./fastwam

      **2. 在** ``examples/embodiment/config/model/fastwam.yaml`` **中设置路径：**

      .. code-block:: yaml

         model_path: ./fastwam/libero_uncond_2cam224.pt
         dataset_stats_path: ./fastwam/libero_uncond_2cam224_dataset_stats.json

      **3. 分别运行四个 suite**\ （使用 4 张 GPU，将 20 个环境槽位 × 25 个 episode = 500 个初始状态均匀分配给 4 个 worker）：

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval
         bash evaluations/run_eval.sh libero libero_object_fastwam_eval
         bash evaluations/run_eval.sh libero libero_goal_fastwam_eval
         bash evaluations/run_eval.sh libero libero_10_fastwam_eval

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

   .. tab-item:: 华为昇腾 CANN
      :sync: ascend

      昇腾支持 FastWAM SFT 和 LIBERO 评测，使用相同的模型权重、数据集和文本 embedding 缓存。

      **1. 安装环境**

      使用昇腾 LIBERO 容器，或已安装 CANN 和 NPU 驱动的主机。

      .. include:: _ascend_libero.rst

      在容器或昇腾主机的 RLinf 仓库目录中创建 FastWAM 环境：

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model fastwam --env libero
         source .venv/bin/activate

      只运行 SFT 时可省略 ``--env libero``\ 。

      **2. 准备模型和数据**

      运行 SFT 前，完成“SFT 训练（LIBERO）”标签页的第 2–4 步，下载资源、完成预处理并设置路径。预处理 ActionDiT 时使用 ``--device cpu``\ ；CUDA 不可用时，文本 embedding 脚本也会使用 CPU。

      将 ``DIFFSYNTH_MODEL_BASE_PATH`` 指向包含 ``Wan-AI`` 的目录。加载原始 Wan ``.pth`` 权重时，在文本 embedding 命令和模型的 ``fastwam.overrides`` 列表中设置 ``model.redirect_common_files=false``\ 。

      **3. 启动 SFT**

      设置好数据和模型路径后，在 ``examples/sft/config/libero_sft_fastwam.yaml`` 中配置 8 张 NPU：

      .. code-block:: yaml

         cluster:
           component_placement:
             actor: 0-7
         actor:
           micro_batch_size: 1
           global_batch_size: 8
           fsdp_config:
             gradient_checkpointing: true

      根据可用 NPU 调整 placement。global batch size 需能被 micro batch size 与 NPU 数量的乘积整除。

      .. code-block:: bash

         export DIFFSYNTH_MODEL_BASE_PATH=./checkpoints
         bash examples/sft/run_vla_sft.sh libero_sft_fastwam

      **4. 运行 LIBERO 评测**

      按“评测（LIBERO）”标签页下载官方 checkpoint 和归一化统计文件，并设置路径。评测配置默认使用 4 个设备，启动前请根据可用 NPU 调整 ``cluster.component_placement``\ 。

      在评测环境中安装 OSMesa 运行库，然后启用 CPU 渲染并启动评测：

      .. code-block:: bash

         export DIFFSYNTH_MODEL_BASE_PATH=./checkpoints
         export MUJOCO_GL=osmesa
         export PYOPENGL_PLATFORM=osmesa
         bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

      评测 Object、Goal 或 Long 时，使用“评测（LIBERO）”标签页列出的对应配置。

      NPU 运行时会自动启用融合 RoPE 和 RMSNorm。

.. dropdown:: 进阶配置（可选）

   ``examples/sft/config/model/fastwam.yaml`` 默认从基础模型初始化（``model_path: null``），仅训练 MoT expert 和 proprio encoder（``freeze_non_dit: true``）：

   .. code-block:: yaml

      actor:
        model:
          model_path: null
          freeze_non_dit: true

可视化与结果
------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，跟踪 SFT 阶段的训练损失及评测阶段的 ``env/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可录制评测视频；完整指标见 :doc:`训练指标 <../../reference/metrics>`，套件评测选项见 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>`。
