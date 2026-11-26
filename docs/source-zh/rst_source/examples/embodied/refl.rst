ReFL
====

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward 概览，图片来自作者的官方仓库。

使用奖励反馈学习（ReFL）微调 Stable Diffusion 3.5 Medium，让模型生成的图像获得更高的 ImageReward 分数。训练使用文本提示词，也可以加入图文对做监督训练。下方实验的 Lpre 权重为 ``0.01``，训练 50 步后，ImageReward 评估均值为 ``1.1980``。

概览
----

本示例使用 ``sd3_refl_imagereward`` 配置，在四张 NVIDIA GPU 上训练 SD3 的 LoRA 参数。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      ReFL

   .. grid-item-card:: 模型
      :text-align: center

      SD3.5 Medium

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      文本提示词、图文对

   .. grid-item-card:: 训练
      :text-align: center

      LoRA · FSDP2 · BF16

任务
~~~~

训练通过 ImageReward 分数优化文生图效果。

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 模型
     - 配置
     - 奖励
   * - SD3.5 Medium
     - ``sd3_refl_imagereward``
     - ImageReward 人类偏好分数

观测与动作
~~~~~~~~~~

每条提示词生成一张图像，由 ImageReward 打分。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 数据
     - 含义
   * - 观测
     - 文本提示词。
   * - 动作
     - 生成的 RGB 图像。
   * - 奖励
     - 每张图像对应一个 ImageReward 分数，越高越好。

ReFL 如何训练
-------------

ReFL 先采样到去噪后期，再用一步预测得到干净图像。冻结的 ImageReward 模型为图像打分，奖励梯度回传到生成模型的 LoRA 参数。加入图文监督损失 Lpre 后，每次奖励更新前先执行一次监督更新。

本示例将 `ImageReward 论文 <https://arxiv.org/abs/2304.05977>`_ 中的方法用于 SD3 的流匹配模型。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: SD3.5 Medium

      **1. 安装环境**

      按 :ref:`安装说明 <diffusion-model-setup>` 准备 diffusion 环境，再按 :ref:`模型下载说明 <diffusion-model-checkpoints>` 下载 SD3.5 Medium。

      **2. 下载模型**

      下载 ImageReward 权重、tokenizer 和官方提示词文件：

      .. code-block:: bash

         export IR_MODEL="$PWD/models/ImageReward"
         export IR_DATA="$PWD/data/ImageReward"
         mkdir -p "$IR_MODEL" "$IR_DATA"

         hf download zai-org/ImageReward ImageReward.pt med_config.json --local-dir "$IR_MODEL"
         python -c 'from transformers import BertTokenizer; BertTokenizer.from_pretrained("bert-base-uncased")'

         for file in refl_data.json test.json; do
           curl --fail --location \
             "https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/data/$file" \
             --output "$IR_DATA/$file"
         done

      ``refl_data.json`` 用于训练，``test.json`` 用于评估。配置中的 ``actor.refl_data.exclude_path`` 指向评估文件，从训练数据中排除这些提示词。

      **3. 启动训练**

      编辑 ``examples/diffusion/config/sd3_refl_imagereward.yaml``。将 ``actor.model.model_path`` 设为 SD3.5 Medium 目录。将 ``actor.refl_reward_model.model_path`` 和 ``actor.refl_reward_model.med_config`` 分别设为下载的 ``ImageReward.pt`` 和 ``med_config.json`` 文件路径。

      将 ``actor.refl_data.path`` 设为 ``refl_data.json`` 路径，``actor.refl_data.exclude_path`` 和 ``env.eval.dataset.path`` 均设为 ``test.json`` 路径。在同一 YAML 中配置 GPU 0–3：

      .. code-block:: yaml

         cluster:
           component_placement:
             actor,env,rollout: 0-3

      默认配置训练 50 步，每步 512 个样本，全局 batch size 为 64，对应 8 次奖励更新。在仓库根目录启动训练：

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh sd3_refl_imagereward

默认 ``algorithm.refl_pre_loss_weight=0.0``，只使用奖励损失。每 10 步评估并保存一次 checkpoint，评估分数记录为 ``eval/image_reward``。日志和 checkpoint 写入 ``logs/<timestamp>-sd3_refl_imagereward/``。actor checkpoint 位于其中的 ``<experiment_name>/checkpoints/global_step_<N>/actor/``，每份约占 9.4 GiB。将 ``runner.resume_dir`` 设为上一级 ``global_step_<N>`` 目录，即可恢复训练状态，包括数据位置和采样随机状态。

配置奖励更新
~~~~~~~~~~~~

``actor.refl_samples_per_step`` 设置每步样本数，取全局 batch size 的整数倍。奖励更新的主要参数如下：

.. list-table::
   :header-rows: 1
   :widths: 32 15 53

   * - 参数
     - 默认值
     - 作用
   * - ``algorithm.refl_margin``
     - ``2.0``
     - 图像得分达到或超过此阈值时，奖励梯度为零。
   * - ``algorithm.refl_grad_scale``
     - ``1e-3``
     - 奖励损失的缩放系数。
   * - ``algorithm.refl_xt_source``
     - ``rollout``
     - 选择去噪轨迹中的后期状态；``resample`` 对最终潜变量重新加噪。

加入图文监督训练
~~~~~~~~~~~~~~~~

将图像与同名的 UTF-8 ``.txt`` 描述文件放在同一目录，也可以分子目录存放：

.. code-block:: text

   image_text_data/
     00000.jpg
     00000.txt
     00001.png
     00001.txt

下方实验使用 `PD3M <https://huggingface.co/datasets/Spawning/PD3M>`_ 的 20,000 对图文数据。可展开下方步骤在独立环境中准备数据，也可直接使用自己的图文目录。

.. dropdown:: 可选：准备 20,000 对 PD3M 图文数据

   脚本读取前两个元数据分片，保留有授权信息、宽高均不小于 512 像素、长边不超过短边两倍的图像。排除 iNaturalist 来源及描述单色图像或文档的样本，再以随机种子 ``42`` 打乱，取前 20,000 条。

   .. code-block:: bash

      export PD3M_DIR="$PWD/data/PD3M"
      uv venv --python 3.10 "$PD3M_DIR/.venv"
      uv pip install --python "$PD3M_DIR/.venv/bin/python" \
        huggingface_hub pyarrow img2dataset==1.47.0

      "$PD3M_DIR/.venv/bin/hf" download Spawning/PD3M \
        metadata/pd3m.00.parquet metadata/pd3m.01.parquet \
        --repo-type dataset --local-dir "$PD3M_DIR"

      "$PD3M_DIR/.venv/bin/python" - "$PD3M_DIR" <<'PYTHON'
      import random
      import re
      import sys

      import pyarrow as pa
      import pyarrow.parquet as pq

      root = sys.argv[1]
      columns = ["id", "url", "caption", "width", "height", "license", "source"]
      rows = []
      for shard in ("00", "01"):
          rows += pq.read_table(
              f"{root}/metadata/pd3m.{shard}.parquet", columns=columns
          ).to_pylist()

      mono = re.compile(r"black[- ]and[- ]white|grayscale|sepia|monochrome", re.I)
      document = re.compile(
          r"\b(book|page|newspaper|document|letter|map|poster|stamp|coin|text (on|at|in))\b",
          re.I,
      )
      selected = [
          row for row in rows
          if row["license"]
          and row["source"] != "iNaturalist"
          and min(row["width"], row["height"]) >= 512
          and max(row["width"], row["height"]) <= 2 * min(row["width"], row["height"])
          and not mono.search(row["caption"])
          and not document.search(row["caption"])
      ]
      random.Random(42).shuffle(selected)
      pairs = [{key: row[key] for key in ("id", "url", "caption")}
               for row in selected[:20000]]
      pq.write_table(pa.Table.from_pylist(pairs), f"{root}/pd3m_refl.parquet")
      PYTHON

      export PRE_DATA="$PD3M_DIR/images_512"
      "$PD3M_DIR/.venv/bin/img2dataset" \
        --url_list="$PD3M_DIR/pd3m_refl.parquet" \
        --input_format=parquet --url_col=url --caption_col=caption \
        --output_format=files --output_folder="$PRE_DATA" \
        --image_size=512 --resize_mode=keep_ratio --resize_only_if_bigger=True \
        --number_sample_per_shard=1000 --processes_count=16 --thread_count=32 \
        --timeout=60 --retries=2 --enable_wandb=False

下载后的图像和描述文件位于 ``$PRE_DATA``。训练加载时将图像缩放并中心裁剪到 512 × 512。配置中的 ``actor.refl_pre_data.caption_prefix`` 为 ``"The image shows "``，用于去掉 PD3M 描述开头的这段文字。要保留自己的描述原文，将该配置项设为 ``""``。

在同一 YAML 中，将 ``actor.refl_pre_data.path`` 设为下载后的 ``images_512`` 目录，``algorithm.refl_pre_loss_weight`` 设为 ``0.01``；对比实验使用 ``0.1``。每组实验都用以下命令启动：

.. code-block:: bash

   bash examples/diffusion/run_diffusion.sh sd3_refl_imagereward

每步包含 8 次监督更新和 8 次奖励更新。建议使用 Lpre 权重 ``0.01``。下方实验中，这组的评估分数比 ``0.1`` 组高。

可视化与结果
------------

实验使用 4 张 NVIDIA A800 80 GB GPU、SD3.5 Medium、LoRA（rank 为 ``32``，alpha 为 ``64``）、FSDP2 和 BF16。学习率为 ``1e-4``，seed 为 ``42``，每步训练 512 个样本。三组实验各训练 50 步。

图像分辨率为 512 × 512，训练去噪 10 步，评估去噪 40 步，评估 guidance scale 为 ``4.5``。每次评估从 463 条提示词生成 512 张图像，报告平均 ImageReward 分数。

.. list-table:: 平均评估 ImageReward
   :header-rows: 1
   :widths: 20 16 16 16 16 16

   * - Lpre 权重
     - Step 10
     - Step 20
     - Step 30
     - Step 40
     - Step 50
   * - ``0`` （reward-only）
     - 0.9331
     - 1.0391
     - 1.1059
     - 1.1123
     - 1.1692
   * - ``0.1``
     - 0.1973
     - 0.6263
     - 0.6783
     - 0.7831
     - 0.9248
   * - ``0.01``
     - 0.5075
     - 0.7576
     - 0.9299
     - 1.1453
     - 1.1980

更换打分器或生成模型见 :ref:`ReFL 接口说明 <refl-model-interface>`，日志查看方式见 :doc:`/rst_source/guides/logger`。
