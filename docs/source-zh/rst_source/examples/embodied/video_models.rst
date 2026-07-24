视频生成模型的强化学习
======================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   在 RLinf 中使用 Diffusion-NFT 训练视频生成模型。

使用 RLinf 对 SD3 和 Wan2.2 生成模型进行 Diffusion-NFT 强化学习微调。该流程复用
具身 runner，将图像/视频生成模型接入 ``actor.model``，并通过 ``gen_reward``
对生成媒体进行一步式 reward 打分。

概览
----------------------------------------

在 OCR prompt 数据集上训练 SD3 图像生成模型，以及 Wan2.2 单帧/视频生成模型。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      ``gen_reward``

   .. grid-item-card:: 算法
      :text-align: center

      Diffusion-NFT

   .. grid-item-card:: 任务
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · GPU

| **你将完成：** 安装依赖 → 下载模型与数据集 → 选择配置 → 启动 ``run_generation.sh`` → 观察 ``avg`` reward。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · SD3 或 Wan2.2 checkpoint · OCR prompt 数据集。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

根据生成模型、输出类型和训练目标选择对应配置。

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - 模型
     - 配置
     - 输出
     - 说明
   * - SD3
     - ``sd3_nft_ocr``
     - Image
     - 对 SD3 图像生成模型进行 OCR reward 微调。
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_nft_ocr``
     - Image
     - 对 Wan2.2 单帧生成结果进行 OCR reward 微调。
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_nft_video_ocr``
     - Video
     - 对 Wan2.2 多帧视频生成结果进行 OCR reward 微调。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 从 OCR prompt 数据集中采样的文本 prompt。
   * - Action
     - SD3 或 Wan2.2 生成的图像或视频输出。
   * - Reward
     - OCR reward 对生成媒体的文本匹配得分。
   * - Prompt
     - 用于条件控制生成模型的自然语言文本。

安装
----------------------------------------

.. include:: _setup_common.rst

**自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 --use-mirror 到下面的 install.sh 命令
   bash requirements/install.sh embodied --model generation
   source .venv/bin/activate

如果要使用自定义虚拟环境目录，可以传入 ``--venv <dir>``：

.. code:: bash

   bash requirements/install.sh embodied --model generation --venv /path/to/venv
   source /path/to/venv/bin/activate

该命令会创建 Python 3.10 环境，并安装 SD3、Wan2.2 和 OCR reward 所需的
Diffusers、PEFT、Transformers、PaddleOCR、PaddlePaddle 等依赖。

.. warning::

   当前 generation 配置使用 ``/path/to/...`` 占位路径。在启动前，请按配置文件旁边的注释替换模型与数据集路径，或通过命令行覆盖 ``actor.model.model_path`` 和 ``env.*.dataset.path``。

.. note::

   Wan2.2 需要支持 ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` 的 Diffusers 版本。
   请优先使用上面的安装命令，不要直接复用已经为其他模型固定旧版 Diffusers 的具身环境。

下载模型
----------------------------------------

训练前需要下载对应的 Diffusers checkpoint，并把 ``actor.model.model_path``
设置为本地模型目录。

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - 模型
     - Hugging Face Repo
     - 本地路径示例
   * - Stable Diffusion 3.5 Medium
     - `stabilityai/stable-diffusion-3.5-medium <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium>`__
     - ``/path/to/stable-diffusion-3.5-medium``
   * - Wan2.2 TI2V 5B
     - `Wan-AI/Wan2.2-TI2V-5B-Diffusers <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>`__
     - ``/path/to/Wan2.2-TI2V-5B-Diffusers``

.. code:: bash

   huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
      --local-dir /path/to/stable-diffusion-3.5-medium

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
      --local-dir /path/to/Wan2.2-TI2V-5B-Diffusers

下载完成后，在启动命令中将同一个本地路径传给 ``actor.model.model_path``。
Stable Diffusion 3.5 Medium 需要先在 Hugging Face 上接受模型访问条款。

下载数据集
----------------------------------------

SD3 和 Wan2.2 示例使用
`NVlabs/DiffusionNFT <https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`__
中的 OCR prompt dataset。数据集目录需要包含 ``train.txt`` 和 ``test.txt``；
每一行是一个 prompt，``env.*.dataset.split`` 决定读取哪个文件。

.. code:: bash

   mkdir -p /path/to/dataset/ocr
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt \
      -o /path/to/dataset/ocr/train.txt
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt \
      -o /path/to/dataset/ocr/test.txt

启动时，将 train/eval 的 dataset path 都指向包含这些文件的目录。

运行
----------------------------------------

向 ``run_generation.sh`` 传入 generation 配置名，并通过命令行覆盖模型和数据集路径。

.. code:: bash

   bash examples/generation/run_generation.sh sd3_nft_ocr \
      actor.model.model_path=/path/to/stable-diffusion-3.5-medium \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

.. code:: bash

   bash examples/generation/run_generation.sh wan22_ti2v_5b_nft_video_ocr \
      actor.model.model_path=/path/to/Wan2.2-TI2V-5B-Diffusers \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

训练流程包括：

1. 从 OCR prompt 数据集中采样 prompt group。
2. 使用 SD3 或 Wan2.2 生成图像或视频候选。
3. 通过 OCR reward 为生成媒体打分。
4. 对生成模型应用 Diffusion-NFT 更新。

进一步配置请修改 ``examples/generation/config/*.yaml``。优先看
``algorithm.group_size``、``algorithm.nft_target_space``、``algorithm.nft_tau``
和 ``actor.model.*.use_lora``。

可视化与结果
----------------------------------------

在生成的日志目录上打开 TensorBoard。主要观察 ``avg`` reward 以及
``actor/*`` 训练指标。如果在 ``env.*.video_cfg`` 中启用媒体保存，生成样本会写入
``video_base_dir``。

.. code:: bash

   tensorboard --host 0.0.0.0 --logdir logs/

训练指标定义和 logger 目录结构请参考 :doc:`Training Metrics </rst_source/reference/metrics>`。
