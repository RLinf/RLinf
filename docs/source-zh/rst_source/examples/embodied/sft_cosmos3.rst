Cosmos3 监督微调与 NFT 训练（LIBERO）
========================================

在 LIBERO 演示数据上使用 FSDP2 微调 NVIDIA Cosmos3-Nano 或 Cosmos3-Edge，将权重导出为 Diffusers 目录，并基于 SGLang rollout 继续开展 Diffusion-NFT 强化学习训练。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Cosmos3-Nano · Cosmos3-Edge · OmniMoT

   .. grid-item-card:: 方法
      :text-align: center

      SFT · Diffusion-NFT（GRPO）

   .. grid-item-card:: 数据
      :text-align: center

      LIBERO（LeRobot v3）

   .. grid-item-card:: 硬件
      :text-align: center

      4–8 张 A800 / A100 80 GB GPU

| **你将完成：** 安装依赖 → 准备基座权重与 LIBERO 数据 → 运行 SFT → 导出 ``model_diffusers`` → 评测或继续开展 Edge NFT 训练。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Cosmos3 基座权重（DCP）· Wan2.2 VAE · LIBERO LeRobot v3 数据集。

安装
----

.. include:: _setup_common.rst

安装 Cosmos3 训练环境（``--env libero`` 会一并安装 LIBERO 仿真环境）：

.. tab-set::

   .. tab-item:: 海外源

      .. code-block:: bash

         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero
         source .venv/bin/activate

   .. tab-item:: 国内镜像

      .. code-block:: bash

         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero --use-mirror
         source .venv/bin/activate

克隆 ``cosmos-framework`` 并导出路径，供 RLinf 加载 Cosmos3 数据变换与归一化统计文件：

.. code-block:: bash

   git clone https://github.com/NVIDIA/cosmos-framework.git ./cosmos-framework
   export COSMOS_FRAMEWORK_PATH="$PWD/cosmos-framework"

Nano SFT 离线缓存
~~~~~~~~~~~~~~~~~

Nano SFT 和基座 DCP 转换会读取 Hugging Face 缓存中的 ``Qwen3-VL-8B-Instruct`` 与 ``Wan2.2-TI2V-5B``。离线机器需先将它们下载到 ``HF_HOME``，启动前再设置 ``HF_HUB_OFFLINE=1`` 和 ``TRANSFORMERS_OFFLINE=1``：

.. tab-set::

   .. tab-item:: 海外源

      .. code-block:: bash

         export HF_HOME=${HF_HOME:-~/.cache/huggingface}
         hf download Qwen/Qwen3-VL-8B-Instruct
         hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e Wan2.2_VAE.pth
         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "vae/*"

   .. tab-item:: 国内镜像

      .. code-block:: bash

         export HF_HOME=${HF_HOME:-~/.cache/huggingface}
         export HF_ENDPOINT=https://hf-mirror.com
         hf download Qwen/Qwen3-VL-8B-Instruct
         hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e Wan2.2_VAE.pth
         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "vae/*"

准备数据
--------

将 ``data.train_data_paths`` 指向原始 LIBERO LeRobot v3 目录（``data_type: libero_all`` 要求目录下包含 ``libero_spatial``、``libero_object``、``libero_goal`` 和 ``libero_10``）。数据加载器会在训练时在线把 LIBERO 的 7 维 axis-angle 动作转为 10 维 rot6d，并用 ``libero_native_frame_wise_relative_rot6d.json`` 完成 quantile 归一化。

.. code-block:: yaml

   data:
     train_data_paths: ./data/LIBERO_LeRobot_v3
     data_type: "libero_all"   # 也可设为 "libero_10" 仅训练单个套件

训练与评测必须使用相同的 ``action_normalization``\ （``quantile_rot``）和同一份 ``action_stats_path`` 统计文件。

动作表示说明
~~~~~~~~~~~~

Cosmos3 模型内部预测 10 维 rot6d 动作，LIBERO 环境接收 7 维 axis-angle 动作：

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - 表示
     - 维度
     - 组成
   * - ``raw_action_dim``\ （模型侧）
     - 10
     - 3 平移 + 6 rot6d + 1 夹爪
   * - ``action_dim``\ （环境侧）
     - 7
     - 3 平移 + 3 axis-angle + 1 夹爪

训练时数据加载器在线将 7 维 LIBERO 动作转为 10 维 rot6d；评测和 RL rollout 阶段再把 10 维输出反归一化并转回 7 维送入 LIBERO。

Cosmos3-Nano SFT
----------------

Nano SFT 从 DCP 格式的 ``nvidia/Cosmos3-Nano`` 基座权重热启动。先下载 Hugging Face Diffusers 快照：

.. tab-set::

   .. tab-item:: 海外源

      .. code-block:: bash

         hf download nvidia/Cosmos3-Nano --local-dir ./assets/Cosmos3-Nano

   .. tab-item:: 国内镜像

      .. code-block:: bash

         export HF_ENDPOINT=https://hf-mirror.com
         hf download nvidia/Cosmos3-Nano --local-dir ./assets/Cosmos3-Nano

在训练用虚拟环境中将 Diffusers 快照转为 DCP，确保 DCP 元数据与训练 Python 版本一致：

.. code-block:: bash

   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   python -m cosmos_framework.scripts.convert_model_to_dcp \
     --checkpoint-path ./assets/Cosmos3-Nano \
     --no-use-ema-weights \
     -o ./assets/Cosmos3-Nano-DCP-py311

编辑 ``examples/sft/config/libero_sft_cosmos3.yaml``，填写本地模型、tokenizer、VAE、统计文件与数据路径：

.. code-block:: yaml

   data:
     train_data_paths: ./data/LIBERO_LeRobot_v3

   actor:
     model:
       model_path: ./assets/Cosmos3-Nano-DCP-py311
       tokenizer_path: ./assets/Qwen3-VL-8B-Instruct-tokenizer
       wan_vae_path: ./assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       action_stats_path: ./cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json

在仓库根目录启动 Nano SFT：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_cosmos3

脚本使用 FSDP2 在各 GPU 上分片训练，日志写入 ``logs/<时间戳>-libero_sft_cosmos3/run_embodiment.log``，每隔 ``runner.save_interval`` 步将 checkpoint 存至 ``checkpoints/global_step_<N>/``。断点续训时设置 ``runner.resume_dir`` 即可。

.. warning::

   SFT 配置必须保持 ``actor.model.load_to_device: false``。Cosmos3 初始化时会同时构建在线网络与 FP32 EMA 副本，若在 FSDP2 分片前直接搬到单张 80 GB GPU 上会触发 OOM。

Cosmos3-Edge SFT
----------------

Edge SFT 使用 ``examples/sft/config/libero_sft_cosmos3_edge.yaml``，采用 Nemotron 主干和 Edge processor。先下载完整 Edge 快照与 Wan2.2 VAE：

.. tab-set::

   .. tab-item:: 海外源

      .. code-block:: bash

         hf download nvidia/Cosmos3-Edge \
           --revision 344d602b128d1bbdacb43b08d0a3626f46343e29 \
           --local-dir ./assets/Cosmos3-Edge
         hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
           --local-dir ./assets/Wan2.2-TI2V-5B

   .. tab-item:: 国内镜像

      .. code-block:: bash

         export HF_ENDPOINT=https://hf-mirror.com
         hf download nvidia/Cosmos3-Edge \
           --revision 344d602b128d1bbdacb43b08d0a3626f46343e29 \
           --local-dir ./assets/Cosmos3-Edge
         hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
           --local-dir ./assets/Wan2.2-TI2V-5B

在同一训练虚拟环境中将 Edge 快照转为 DCP（清空 VAE bucket 和凭据字段以直接读取本地 VAE 文件）：

.. code-block:: bash

   COSMOS_DEVICE=cpu python -m cosmos_framework.scripts.convert_model_to_dcp \
     --checkpoint-path "$PWD/assets/Cosmos3-Edge" \
     --config-file "$COSMOS_FRAMEWORK_PATH/cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml" \
     --output-path "$PWD/assets/Cosmos3-Edge-DCP-py311" \
     --experiment-overrides \
       'model.config.tokenizer.bucket_name=""' \
       'model.config.tokenizer.object_store_credential_path_pretrained=""' \
       "model.config.tokenizer.vae_path=$PWD/assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
       model.config.vlm_config.tokenizer.repository=null \
       "model.config.vlm_config.tokenizer.tokenizer_type=$PWD/assets/Cosmos3-Edge"

保留原始 ``./assets/Cosmos3-Edge`` 目录供 ``actor.model.tokenizer_path`` 读取 Edge processor 元数据与 tokenizer。

在 ``examples/sft/config/libero_sft_cosmos3_edge.yaml`` 中配置本地路径（``data_type: libero_all`` 至少需要 4 张 GPU）：

.. code-block:: yaml

   data:
     train_data_paths: ./data/LIBERO_LeRobot_v3

   actor:
     model:
       model_path: ./assets/Cosmos3-Edge-DCP-py311
       tokenizer_path: ./assets/Cosmos3-Edge
       wan_vae_path: ./assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       action_stats_path: ./cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json

在仓库根目录启动 Edge SFT：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_cosmos3_edge

SFT 关键配置
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 字段
     - 说明
   * - ``actor.model.model_path``
     - Cosmos3-Nano 或 Cosmos3-Edge 基座 DCP 目录。
   * - ``actor.model.tokenizer_path``
     - Nano 指向 Qwen3-VL tokenizer 目录；Edge 指向完整 Edge 快照目录。
   * - ``actor.model.wan_vae_path``
     - ``Wan2.2_VAE.pth`` 路径。
   * - ``actor.model.load_to_device``
     - 必须设为 ``false``，让 FSDP2 直接从 CPU 分片上卡。
   * - ``actor.model.ema_enabled``
     - 开启 Cosmos3 power-EMA（rate 为 0.1）；显存紧张时可设为 ``false``。
   * - ``actor.micro_batch_size`` / ``global_batch_size``
     - 每卡微批与全局批大小（Nano：32 / 2048；Edge：64 / 256）。
   * - ``runner.max_steps`` / ``save_interval``
     - 总训练步数与保存间隔。

转换 Checkpoint
---------------

SFT 产出的权重位于 ``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``。SGLang 评测和 NFT 训练都需要 Diffusers 组件目录 ``model_diffusers``。使用架构匹配的 Diffusers 目录作为模板即可同时支持 Nano 与 Edge 导出：

.. code-block:: bash

   python -m toolkits.cosmos3.convert_checkpoint \
     --src ./results/checkpoints/global_step_10000/actor/model_state_dict/full_weights.pt \
     --base-model-path ./assets/Cosmos3-Edge \
     --out ./assets/cosmos3-edge-policy-liberoall-10k

转换脚本会校验全部 transformer 权重键名与形状，将 SFT 权重写入 ``./assets/cosmos3-edge-policy-liberoall-10k/model_diffusers``，并保留模板的 tokenizer、VAE、scheduler 和配置文件。输出目录必须尚不存在且位于模板目录之外。默认导出在线权重；传入 ``--use-ema-weights`` 可导出 EMA 权重。

.. note::

   若没有匹配的 Diffusers 模板且已配置 ``cosmos-framework`` 导出环境（``transformers 4.57.x`` + ``diffusers 0.39.0``），可省略 ``--base-model-path`` 并设置 ``LIBERO_ROOT`` 走四步转换。参数说明见 ``python -m toolkits.cosmos3.convert_checkpoint --help``。

.. _cosmos3-sglang-install:

安装 SGLang 推理与 RL 环境
--------------------------

若要在单个环境中同时运行 Cosmos3-Edge NFT 训练与 SGLang 评测，使用 ``--engine sglang`` 一次性安装 Cosmos3、LIBERO 以及 SGLang 0.5.19 Diffusion 运行时（自动包含固定版本 ``FxxxxU/sglang@cf565fa`` 的 Cosmos3 扩展）：

.. tab-set::

   .. tab-item:: 海外源

      .. code-block:: bash

         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero --engine sglang --venv .venv-sglang
         source .venv-sglang/bin/activate

   .. tab-item:: 国内镜像

      .. code-block:: bash

         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero --engine sglang --venv .venv-sglang --use-mirror
         source .venv-sglang/bin/activate

.. note::

   若仅运行 SGLang 评测而不需要训练侧依赖，可省略 ``--model cosmos3``，直接运行 ``bash requirements/install.sh embodied --env libero --engine sglang --venv .venv-sglang``。如需复用本地已有的 ``FxxxxU/sglang`` 目录而非自动克隆到 ``.venv-sglang/sglang-cosmos3``，在运行 ``install.sh`` 前导出 ``SGLANG_COSMOS3_PATH`` 即可。

Cosmos3-Edge NFT 训练
---------------------

导出 ``model_diffusers`` 后，在 LIBERO-Spatial 上使用 Diffusion-NFT 与 SGLang rollout 启动 Cosmos3-Edge 训练。先编辑 ``examples/embodiment/config/libero_spatial_nft_cosmos3_edge_sglang.yaml`` 中的本地路径：

.. code-block:: yaml

   actor:
     model:
       model_path: ./assets/cosmos3-edge-policy-liberoall-10k/model_diffusers
       wan_vae_path: ./assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       action_stats_path: ./cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json

在仓库根目录启动训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_nft_cosmos3_edge_sglang

每个 rollout worker 会在初始化时于分配的 GPU 上启动专属 SGLang Diffusion server，训练结束时自动关闭。权重同步采用 patch 模式：第 0 步建立完整基线权重，后续步骤仅同步发生变化的权重元素。若 actor 与 rollout 同卡部署显存紧张，可同时开启 ``rollout.enable_offload: true`` 与 ``actor.enable_offload: true``。

训练完成后，按 :doc:`Cosmos3 SGLang 评测 <../../evaluations/guides/cosmos3_sglang>` 导出并评测最终 checkpoint。

可视化与结果
------------

使用 TensorBoard 查看训练曲线与日志：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

各项指标定义见 :doc:`训练指标 </rst_source/reference/metrics>`。

常见问题
--------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 现象
     - 处理
   * - 80 GB GPU 初始化即 OOM
     - 保持 ``actor.model.load_to_device: false``；同卡运行 SGLang NFT 时同时开启 ``rollout.enable_offload: true`` 和 ``actor.enable_offload: true``。
   * - 报 ``sglang version 0.0.0.dev... is not supported``
     - 带 ``SETUPTOOLS_SCM_PRETEND_VERSION=0.5.19`` 重新安装 ``../sglang-cosmos3/python``。
   * - SFT 热启动报动作头形状不匹配
     - 确认 ``keys_to_skip_loading`` 已跳过基座 DROID 8 维动作头（``examples/embodiment/config/model/cosmos3.yaml`` 默认已配置）。
   * - 离线环境卡在 Hugging Face 下载
     - 预先下载到 ``HF_HOME`` 并导出 ``HF_HUB_OFFLINE=1`` 与 ``TRANSFORMERS_OFFLINE=1``。
   * - 动作异常或成功率为 0
     - 核对训练与推理两侧的 ``action_normalization``\ （``quantile_rot``）、``action_stats_path``、``num_inference_steps``、``num_frames`` 与 ``size`` 是否一致。
