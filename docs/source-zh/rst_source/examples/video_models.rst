图像与视频生成
==============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 与 Wan2.2 图像/视频生成模型。

使用 **Diffusion-NFT** 或 **DiffusionART** 以 OCR 奖励微调 SD3 和 Wan2.2，或使用 **ReFL** 以 ImageReward 微调 SD3.5 Medium。Diffusion-NFT 与 DiffusionART 共用下方的依赖、模型与 OCR prompt 数据集准备流程；ReFL 在独立页面内给出完整安装与运行步骤。

选择训练方法
------------

根据需要选择训练方法；各指南包含对应配置、更新参数和预期结果。

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: DiffusionART
      :link: diffusion_art
      :link-type: doc

      在 prompt 组内构造共享奖励引导，并向冻结的速度目标训练，包含 SD3 图像 OCR 和 Wan2.2 视频 OCR 配置。

   .. grid-item-card:: Diffusion-NFT
      :link: diffusion_nft
      :link-type: doc

      根据相对奖励构造正负速度候选，包含 SD3 图像 OCR 和 Wan2.2 图像/视频 OCR 配置。

   .. grid-item-card:: ReFL
      :link: embodied/refl
      :link-type: doc

      使用可微 ImageReward 反馈及可选图文监督更新微调 SD3.5 Medium。

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

先满足 :doc:`RLinf 安装的前置条件 </rst_source/start/installation>`\ （中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``）：

.. tab-set::

   .. tab-item:: CPU OCR（SD3 / ART）

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion && source .venv/bin/activate

   .. tab-item:: GPU OCR（Wan NFT）

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion --paddle-gpu && source .venv/bin/activate

安装器会创建 Python 3.10 环境，并安装 Diffusers、PEFT、Transformers 与 PaddleOCR（``--paddle-gpu`` 支持 CUDA 11.x / 12.x，供 Wan NFT 配置使用）。需要其他目录时可追加 ``--venv /path/to/venv``。

.. note::

   Wan2.2 请使用 diffusion 安装器。其依赖支持 ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``；为其他模型准备的环境可能固定了较旧的 Diffusers 版本。

.. _diffusion-model-checkpoints:

下载模型
--------

下载所选配置对应的 checkpoint（使用 SD3.5 Medium 前需先在 Hugging Face 接受访问条款）：

.. tab-set::

   .. tab-item:: SD3.5 Medium

      .. code-block:: bash

         hf download stabilityai/stable-diffusion-3.5-medium --local-dir ./stable-diffusion-3.5-medium

   .. tab-item:: Wan2.2 TI2V 5B

      .. code-block:: bash

         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./Wan2.2-TI2V-5B-Diffusers

启动 ``run_diffusion.sh`` 时可直接通过 Hydra 命令行覆盖模型与数据路径（也可以修改所选 YAML 中的 ``actor.model.model_path``、``env.train.dataset.path`` 和 ``env.eval.dataset.path``）：

.. code-block:: bash

   bash examples/diffusion/run_diffusion.sh sd3_nft_ocr \
       actor.model.model_path=./stable-diffusion-3.5-medium \
       env.train.dataset.path=./ocr_data \
       env.eval.dataset.path=./ocr_data

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

   DiffusionART <diffusion_art>
   Diffusion-NFT <diffusion_nft>
   ReFL <embodied/refl>
