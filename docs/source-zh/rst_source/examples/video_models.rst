图像与视频生成
==============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 与 Wan2.2 图像/视频生成模型。

使用 Diffusion-NFT 或 DiffusionART，以 OCR 奖励微调 SD3 和 Wan2.2。两种方法共用模型、prompt 数据集和 diffusion 环境。先按本页准备依赖、模型与数据，再进入对应的训练指南。

选择训练方法
------------

根据需要选择训练方法；各指南包含对应配置、更新参数和预期结果。

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Diffusion-NFT
      :link: diffusion_nft
      :link-type: doc

      根据相对奖励构造正负速度候选，包含 SD3 图像 OCR 和 Wan2.2 图像/视频 OCR 配置。

   .. grid-item-card:: DiffusionART
      :link: diffusion_art
      :link-type: doc

      在 prompt 组内构造共享奖励引导，并向冻结的速度目标训练，包含 SD3 图像 OCR 和 Wan2.2 视频 OCR 配置。

Prompt、生成结果与奖励
----------------------

每条 prompt 描述图像或视频，并用引号标出目标文字，例如写有 "OPEN" 的标牌。diffusion 环境对生成结果打分后结束 episode。``algorithm.group_size`` 决定同一 prompt 生成多少个候选，用于比较奖励。

.. list-table:: 共用训练数据
   :header-rows: 1
   :widths: 24 76

   * - 数据
     - 含义
   * - 观测
     - 来自 ``env.*.dataset`` 的文本 prompt。
   * - 动作
     - 生成的图像或视频。
   * - 奖励
     - OCR 文字相似度，范围为 ``[0, 1]``，越高越好。

图像 OCR 用归一化 Levenshtein 距离比较识别文字与引号内的目标文字；精确匹配子串时也计为 ``1``。视频 OCR 对各帧使用同一个 scorer：NFT 视频配置保留逐帧奖励，ART 视频配置则取帧均值，得到每个候选的单一奖励。除 reward 外，还应检查生成样本，因为 OCR 不衡量视觉质量或时序一致性。

.. _diffusion-model-setup:

安装
----

先满足 :doc:`RLinf 安装的前置条件 </rst_source/start/installation>`，再在 GPU 训练机器的仓库根目录执行：

.. code:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

安装器创建 Python 3.10 环境，安装 Diffusers、PEFT、Transformers、PaddleOCR 和 CPU PaddlePaddle。两个 ART 配置及 SD3 NFT 配置使用 CPU OCR。需要其他环境目录时，传入 ``--venv /path/to/venv``，并激活对应环境。

Wan NFT 配置设置了 ``env.*.reward.use_gpu: True``。使用 GPU OCR 时，通过 ``--paddle-gpu`` 安装 GPU PaddlePaddle（CUDA 11.x 或 12.x），然后激活环境：

.. code:: bash

   bash requirements/install.sh embodied --model diffusion --paddle-gpu
   source .venv/bin/activate

.. note::

   Wan2.2 请使用 diffusion 安装器。其依赖支持 ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``；为其他模型准备的环境可能固定了较旧的 Diffusers 版本。

.. _diffusion-model-checkpoints:

下载模型
--------

下载所选配置对应的 checkpoint。SD3.5 Medium 需要先在 Hugging Face 接受模型访问条款。

.. code:: bash

   huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
      --local-dir /path/to/stable-diffusion-3.5-medium

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
      --local-dir /path/to/Wan2.2-TI2V-5B-Diffusers

命令将 Diffusers checkpoint 保存到本地。在所选训练 YAML 中，将 ``actor.model.model_path`` 设为对应目录。

.. _diffusion-ocr-dataset:

下载数据集
----------

两种方法都使用 `NVlabs/DiffusionNFT <https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`__ 的 OCR prompt。将训练集和测试集下载到同一目录：

.. code:: bash

   mkdir -p /path/to/dataset/ocr
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt \
      -o /path/to/dataset/ocr/train.txt
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt \
      -o /path/to/dataset/ocr/test.txt

每行是一条 prompt。``env.*.dataset.split`` 选择 ``train.txt`` 或 ``test.txt``。在所选训练 YAML 中，将 ``env.train.dataset.path`` 和 ``env.eval.dataset.path`` 设为该目录。模型与数据准备完成后，继续阅读 :doc:`Diffusion-NFT <diffusion_nft>` 或 :doc:`DiffusionART <diffusion_art>`。

.. warning::

   启动训练前，将下载命令和所选训练 YAML 中的 ``/path/to/...`` 占位路径替换为实际路径。

.. toctree::
   :hidden:
   :maxdepth: 1

   Diffusion-NFT <diffusion_nft>
   DiffusionART <diffusion_art>
