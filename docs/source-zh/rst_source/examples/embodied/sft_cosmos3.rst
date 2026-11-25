Cosmos3 监督微调
================

将 NVIDIA Cosmos3-Nano（OmniMoT 视频世界模型）按 LIBERO 动作空间做 SFT 微调，适配 LIBERO 仿真器。

概览
----------------------------------------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Cosmos3-Nano · OmniMoT

   .. grid-item-card:: 方法
      :text-align: center

      FSDP2 SFT

   .. grid-item-card:: 数据
      :text-align: center

      LIBERO（LeRobot）

   .. grid-item-card:: 硬件
      :text-align: center

      8 张 A800 80GB GPU

| **你将完成：** 安装环境 → 下载并将基座权重转为 DCP → 配置数据路径 → 启动 ``run_vla_sft.sh`` → 转换 checkpoint 用于评测。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Cosmos3-Nano 基座权重（DCP）· Wan2.2 VAE · LIBERO LeRobot v3 数据集。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO

      **1. 安装环境**

      安装 Cosmos3 训练环境并克隆 ``cosmos-framework``\ （用于数据变换、归一化统计与 DCP 转换）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero && source .venv/bin/activate
         git clone https://github.com/NVIDIA/cosmos-framework.git ./cosmos-framework
         export COSMOS_FRAMEWORK_PATH="$PWD/cosmos-framework"

      **2. 下载模型**

      下载 ``nvidia/Cosmos3-Nano`` diffusers 基座权重，并将 SFT 启动与 DCP 转换所需的配套 Hugging Face 资产缓存至 ``HF_HOME``\ （离线机器开启 ``HF_HUB_OFFLINE=1`` 时直接读取缓存）：

      .. code-block:: bash

         # 基座模型（diffusers 格式，下文转换为 DCP）与 Wan2.2 VAE
         hf download nvidia/Cosmos3-Nano --local-dir ./Cosmos3-Nano
         hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e Wan2.2_VAE.pth --local-dir ./Wan2.2-TI2V-5B

         # 配套资产直接下载至 HF 缓存（供离线模式使用）
         export HF_HOME=${HF_HOME:-~/.cache/huggingface}
         hf download Qwen/Qwen3-VL-8B-Instruct
         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "vae/*"

      **3. 准备基座模型与数据**

      Cosmos3 SFT 从 **DCP 格式** 的 Cosmos3-Nano 基座热启动，只训练动作头。diffusers 目录把分片放在 ``transformer/`` 子目录而非顶层，SFT 无法直接加载，必须先在 **同一训练 venv** 内将 ``./Cosmos3-Nano`` 转为 DCP（DCP 的 ``.metadata`` pickle 与保存时的 Python 版本绑定）：

      .. code-block:: bash

         export HF_HOME=${HF_HOME:-~/.cache/huggingface}
         export HF_HUB_OFFLINE=1
         export TRANSFORMERS_OFFLINE=1
         python -m cosmos_framework.scripts.convert_model_to_dcp \
           --checkpoint-path ./Cosmos3-Nano \
           --no-use-ema-weights \
           -o ./Cosmos3-Nano-DCP

      训练数据直接使用原始 LIBERO LeRobot v3 布局（含 ``meta/`` 与 ``data/``），无需离线预处理：Cosmos3 数据加载器在加载时通过 cosmos-framework 的 ``action_policy_libero_all_nano`` 配方在线应用 ``frame_wise_relative`` + ``rot6d`` + ``quantile_rot`` 变换。在 ``examples/sft/config/libero_sft_cosmos3.yaml`` 中配置模型、VAE 和数据路径：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Cosmos3-Nano-DCP
             wan_vae_path: ./Wan2.2-TI2V-5B/Wan2.2_VAE.pth
             load_to_device: false

         data:
           train_data_paths: ./LIBERO_LeRobot
           data_type: "libero_all"          # LIBERO 全部套件，也可单独训练 "libero_10"

      .. warning::

         务必保持 ``actor.model.load_to_device: false``。初始化时 ``net``\ （bf16 约 27 GB）+ ``net_ema``\ （fp32 约 54 GB）≈ 81 GB，单张 80 GB 卡上直接上卡会在 FSDP2 分片前就 OOM。

      **4. 启动训练**

      在仓库根目录启动 SFT（每 ``save_interval`` 步保存到 ``checkpoints/global_step_<N>/``，日志写入 ``logs/YYYYMMDD-HHMMSS-libero_sft_cosmos3/run_embodiment.log``；断点续训时将 ``runner.resume_dir`` 指向已保存的 ``global_step_<N>`` 目录）：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_cosmos3

.. dropdown:: 关键配置

   大多数字段沿用 ``examples/sft/config/libero_sft_cosmos3.yaml`` 即可，重点配置项如下：

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - 字段
        - 说明
      * - ``actor.model.model_path``
        - Cosmos3-Nano 基座权重目录（DCP）。
      * - ``actor.model.wan_vae_path``
        - Wan2.2 VAE 权重（``Wan2.2_VAE.pth``）。
      * - ``actor.model.load_to_device``
        - 必须设为 ``false``，让 FSDP2 从 CPU 直接分片到 GPU。
      * - ``actor.model.ema_enabled``
        - Cosmos3 power-EMA（rate 0.1），在显存中保存一份 fp32 副本；若仍 OOM 可临时设为 ``false``。
      * - ``actor.micro_batch_size`` / ``global_batch_size``
        - 每卡微批与全局批（默认 32 / 2048）。
      * - ``actor.optim.lr`` / ``lr_warmup_steps``
        - 学习率与 warmup（默认 5e-5 / 500）；``model/cosmos3.yaml`` 中为动作头配置了额外 5× 学习率倍率。
      * - ``runner.max_steps`` / ``save_interval``
        - 训练步数与存档间隔（默认 5000 / 500）。

.. dropdown:: 动作表示（10 维 rot6d 与 7 维 axis-angle）

   Cosmos3 内部使用 **10 维 rot6d** 表示动作，LIBERO 环境使用 **7 维 axis-angle**\ （rot6d 连续，可避免 axis-angle 在 ±π 处的不连续问题）：

   .. list-table::
      :header-rows: 1
      :widths: 24 20 56

      * - 表示
        - 维度
        - 组成
      * - ``raw_action_dim``\ （模型侧）
        - 10
        - 3 平移 + 6 rot6d + 1 夹爪
      * - ``action_dim``\ （环境侧）
        - 7
        - 3 平移 + 3 axis-angle + 1 夹爪

   训练时 Cosmos3 数据加载器在线将 7 维 LIBERO 动作转为 10 维 rot6d，推理时再转回 7 维。训练与评测必须使用同一配方和同一归一化统计文件（通过 ``action_stats_path`` 引用 ``libero_native_frame_wise_relative_rot6d.json``），详见 :doc:`SGLang 评测 <../../evaluations/guides/cosmos3_sglang>`。

.. dropdown:: 转换 Checkpoint

   SFT 产出的 ``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`` 为 FSDP2 格式，不能直接用于评测。使用 ``toolkits/cosmos3/convert_checkpoint.py`` 将其转换为 diffusers 目录 ``model_diffusers``\ （依次去掉 ``omni.`` 前缀、存为 DCP、导出 HF 并拆分 diffusers 组件；传入 ``--use-ema-weights`` 可保留 EMA 权重）：

   .. code-block:: bash

      export SRC="./checkpoints/global_step_500/actor/model_state_dict/full_weights.pt"
      export OUT="./converted"
      export LIBERO_ROOT="./LIBERO_LeRobot"
      mkdir -p "$OUT"

      python toolkits/cosmos3/convert_checkpoint.py --src "$SRC" --out "$OUT"

   .. warning::

      转换须在 **cosmos-framework 环境** 下运行，要求 ``transformers 4.57.x`` 与 ``diffusers 0.39.0``\ （包含 ``Cosmos3OmniTransformer``）、Wan2.2 diffusers 格式 VAE 以及离线 HF 缓存（``HF_HUB_OFFLINE=1``）。

   将转换得到的 ``model_diffusers`` 填作 :doc:`Cosmos3 SGLang 评测 <../../evaluations/guides/cosmos3_sglang>` 的 ``rollout.model.model_path``。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 查看训练日志与曲线，指标定义见 :doc:`训练指标 <../../reference/metrics>`。

.. dropdown:: 常见问题

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - 现象
        - 处理
      * - 初始化 OOM（单卡 80 GB 也爆）
        - 设 ``actor.model.load_to_device: false``。
      * - 加载报动作头 shape 不匹配
        - 确认 ``keys_to_skip_loading`` 跳过了基座 DROID 8 维动作头（``model/cosmos3.yaml`` 默认已配）。
      * - Hugging Face 拉取卡住
        - 启用离线模式与本地缓存：``HF_HUB_OFFLINE=1``、``TRANSFORMERS_OFFLINE=1``，并将 ``HF_HOME`` 指向本地缓存。
      * - 动作异常或 loss 异常
        - 核对训练与评测两侧的 ``action_normalization``\ （``quantile_rot``）与统计文件是否一致。
