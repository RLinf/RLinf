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
