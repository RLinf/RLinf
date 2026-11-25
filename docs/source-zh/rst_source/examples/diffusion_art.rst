DiffusionART
============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   使用 DiffusionART 训练的 SD3 与 Wan2.2 生成模型。

使用 DiffusionART 微调 SD3 或 Wan2.2，通过 OCR 奖励改善图像和视频中的文字渲染。完成 :doc:`共用准备步骤 <video_models>` 后，选择对应配置启动训练。

概览
----

两个配置均使用 LoRA、FSDP2 和确定性 ODE 采样，奖励来自生成文字与提示词中目标文字的 OCR 相似度。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      DiffusionART

   .. grid-item-card:: 模型
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      OCR 提示词

   .. grid-item-card:: 训练
      :text-align: center

      LoRA · FSDP2 · BF16

任务
~~~~

选择图像生成或视频生成配置。

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - 模型
     - 配置
     - 任务
   * - SD3.5 Medium
     - ``sd3_art_ocr``
     - 图像 OCR
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_art_video_ocr``
     - 文生视频 OCR

观测与动作
~~~~~~~~~~

每个生成结果对应一个 OCR 分数，视频奖励取采样帧的分数均值。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 数据
     - 含义
   * - 观测
     - 文本提示词，目标文字用引号标出。
   * - 动作
     - 生成的图像或视频。
   * - 奖励
     - OCR 相似度分数，范围为 ``[0, 1]``。

ART 如何训练
------------

ART 根据同一提示词下各候选的奖励和最终潜变量计算共享更新方向。训练使用生成轨迹中的去噪状态，两个更新窗口共用第一次更新前冻结的速度目标。

算法推导与参考实现见 `DiffusionART 论文 <https://arxiv.org/abs/2610.12572>`_、`官方代码 <https://github.com/vanzll/DiffusionART>`_ 和 `项目页 <https://vanzll.github.io/diffusionart-project/>`_。

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。两个 ART 配置均使用 CPU OCR，完整共用说明见 :doc:`video_models`。

.. tab-set::

   .. tab-item:: SD3.5 Medium（图像 OCR）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion && source .venv/bin/activate

      **2. 下载模型与数据**

      先在 Hugging Face 同意 SD3.5 Medium 的使用协议，再下载权重与 OCR 提示词数据集：

      .. code-block:: bash

         hf download stabilityai/stable-diffusion-3.5-medium --local-dir ./stable-diffusion-3.5-medium
         mkdir -p ./ocr_data
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt -o ./ocr_data/train.txt
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt -o ./ocr_data/test.txt

      **3. 启动训练**

      在 ``examples/diffusion/config/sd3_art_ocr.yaml`` 中设置 ``actor.model.model_path``、``env.train.dataset.path`` 和 ``env.eval.dataset.path``\ （GPU 分配由 ``cluster.component_placement`` 控制），然后启动训练：

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh sd3_art_ocr

   .. tab-item:: Wan2.2 TI2V 5B（视频 OCR）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion && source .venv/bin/activate

      **2. 下载模型与数据**

      .. code-block:: bash

         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./Wan2.2-TI2V-5B-Diffusers
         mkdir -p ./ocr_data
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt -o ./ocr_data/train.txt
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt -o ./ocr_data/test.txt

      **3. 启动训练**

      在 ``examples/diffusion/config/wan22_ti2v_5b_art_video_ocr.yaml`` 中设置 ``actor.model.model_path``、``env.train.dataset.path`` 和 ``env.eval.dataset.path``\ （GPU 分配由 ``cluster.component_placement`` 控制），然后启动训练：

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr

训练日志写入 ``logs/<timestamp>-<config>/``。图像 OCR 分数记录为 ``env/avg``，视频 OCR 分数记录为 ``env/video_ocr``，指标定义见 :doc:`训练指标 </rst_source/reference/metrics>`。

配置 ART 更新
~~~~~~~~~~~~~

每轮样本数为 ``env.train.total_num_envs * env.train.rollout_epoch``。默认 ``algorithm.art_updates_per_rollout=2``，每轮分两次更新，``actor.global_batch_size`` 取每轮样本数的一半。例如，96 个环境、12 个 rollout epoch 对应每轮 1152 个样本，全局 batch size 为 576。

主要更新参数如下：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 参数
     - 作用
   * - ``algorithm.art_timestep_fraction``
     - 用于训练的去噪状态比例；SD3 为 ``0.5``，Wan 为 ``0.25``。
   * - ``algorithm.art_guidance_gain``
     - 奖励引导幅度，默认为 ``1.0``。
   * - ``algorithm.art_corrective_mask_enabled``
     - 启用偏移修正 mask，默认为 ``True``。
   * - ``algorithm.art_mask_threshold``
     - 偏移修正阈值，默认为 ``1e-5``，使用 ``sde_kl`` 度量。

.. _diffusion-art-results:

预期实验结果
------------

实验使用 4 张 NVIDIA A800 80 GB GPU、Wan2.2 TI2V 5B、LoRA（rank 为 ``32``，alpha 为 ``64``）、FSDP2、BF16 和 CPU OCR。生成设置为 512 × 512 分辨率、5 帧、10 个去噪步骤，guidance scale 为 ``1.0``。

ART 使用 ``wan22_ti2v_5b_art_video_ocr`` 配置，NFT 使用 ``wan22_ti2v_5b_nft_video_ocr`` 配置。训练奖励取最后 5760 条轨迹的 ``env/video_ocr`` 均值。多次运行的结果用均值 ± 样本标准差表示。

第一组实验中，ART 使用 ``video_level`` 奖励，NFT 使用 ``frame_level`` 奖励。

.. list-table:: 生成 40,320 条轨迹时的 Wan 视频 OCR 结果
   :header-rows: 1
   :widths: 24 8 16 12 20 20

   * - 方法
     - 运行次数
     - 每轮轨迹数
     - 学习率
     - 训练奖励
     - 训练耗时（h）
   * - ART
     - 1
     - 1152
     - 1e-4
     - 0.8584
     - 10.01
   * - NFT
     - 2
     - 576
     - 4e-4
     - 0.7565 ± 0.0544
     - 11.29 ± 0.52

在这组配置下，生成 40,320 条轨迹后，ART 的训练奖励比 NFT 均值高 **13.5%**，训练耗时少 **11.3%**。耗时取训练循环的计时结果。

第二组实验中，两种方法的每轮轨迹数和学习率相同，均使用 ``video_level`` 奖励，各运行一次。表中比较两次运行的前 29 轮。

.. list-table:: 生成 33,408 条轨迹时的 Wan 视频 OCR 结果
   :header-rows: 1
   :widths: 18 22 18 22 20

   * - 方法
     - 每轮轨迹数
     - 学习率
     - 训练奖励
     - 训练耗时（h）
   * - ART
     - 1152
     - 1e-4
     - 0.8446
     - 8.31
   * - NFT
     - 1152
     - 1e-4
     - 0.1076
     - 8.23

复现实验时，在对应配置中设置模型与数据目录，并使用以下共用设置：

.. dropdown:: 进阶配置（可选）

   .. code-block:: yaml

      cluster:
        component_placement:
          actor,env,rollout: "0-3"
      algorithm:
        group_size: 8
      actor:
        seed: 42
        model:
          wan22_ti2v_5b:
            max_generation_batch_size: 2
            compile_transformer_forward: True
            offload_auxiliary_modules: True
      env:
        train:
          total_num_envs: 96
          seed: 42
          reward:
            use_gpu: False
        eval:
          seed: 42
          reward:
            use_gpu: False
      runner:
        val_check_interval: -1

各组实验的 batch 和训练步数设置如下。两组 NFT 均使用 ``algorithm.nft_tau=[1.0,0.01,0,70]``。

.. list-table:: 分组训练配置
   :header-rows: 1
   :widths: 46 18 18 18

   * - 参数
     - ART，1152 条/轮
     - NFT，576 条/轮
     - NFT，1152 条/轮
   * - ``env.train.rollout_epoch``
     - 12
     - 6
     - 12
   * - ``actor.global_batch_size``
     - 576
     - 288
     - 576
   * - ``actor.micro_batch_size``
     - 2
     - 1
     - 2
   * - ``actor.optim.lr``
     - 1e-4
     - 4e-4
     - 1e-4
   * - ``env.train.reward.reward_type``
     - ``video_level``
     - ``frame_level``
     - ``video_level``
   * - ``runner.max_epochs`` / ``runner.max_steps``
     - 70
     - 70
     - 200
   * - ``runner.save_interval``
     - 10
     - -1
     - 10
