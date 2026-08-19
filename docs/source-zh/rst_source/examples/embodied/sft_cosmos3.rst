Cosmos3 监督微调（LIBERO）
================================

把 NVIDIA Cosmos3-Nano（OmniMoT 视频世界模型）按照 libero 的动作空间进行 sft 微调，适配 libero 仿真器。

概览
----------------------------------------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Cosmos3-Nano · OmniMoT

   .. grid-item-card:: 方法
      :text-align: center

      在 Libero 仿真器上的 SFT

   .. grid-item-card:: 数据
      :text-align: center

      LIBERO（LeRobot）

   .. grid-item-card:: 硬件
      :text-align: center

      8个 A800 80GB GPU

| **你将完成：** 安装 → 准备基座模型与数据 → ``run_vla_sft.sh`` 训练 → 获得一个适配 libero 仿真器的 Cosmos3 模型。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Cosmos3-Nano 基座权重（DCP）· Wan2.2 VAE · 下载好的 LIBERO LeRobot 数据集。

安装
----------------------------------------

.. include:: _setup_common.rst

安装 Cosmos3（如需仿真评测，加 ``--env libero``）：

.. code-block:: bash

   unset PYTHONPATH
   # 国内用户可加 --use-mirror 加速下载。
   bash requirements/install.sh embodied --model cosmos3 --env libero
   source .venv/bin/activate

Cosmos3 的数据变换与归一化统计依赖 cosmos framework，训练前设置：

.. code-block:: bash

   export COSMOS_FRAMEWORK_PATH=/path/to/cosmos-framework

离线缓存 Qwen3-VL-8B-Instruct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cosmos3-Nano 的理解塔基于 Qwen3-VL-8B-Instruct，SFT 启动时会经 ``AutoConfig`` / ``from_pretrained`` 从 Hugging Face 拉取 ``Qwen/Qwen3-VL-8B-Instruct``（用于加载 tokenizer ``Qwen2TokenizerFast`` 与 VLM 配置，给语言指令分词）。

这是sft训练时**唯一**会从 HF 下载的模型；无外网或无缓存的机器上，worker 会在网络重试处卡死。

因此训练前要把 Qwen3-VL-8B-Instruct 提前下好放进 HF 本地缓存，并开离线模式让 ``from_pretrained`` 直接读缓存。

**下载 Qwen3-VL-8B-Instruct（任选其一）**

方式 A——直接下载进 HF 缓存（推荐，缓存布局自动正确）：

.. code-block:: bash

   export HF_HOME=${HF_HOME:-~/.cache/huggingface}
   # 国内可用镜像加速：export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download Qwen/Qwen3-VL-8B-Instruct

下载后会落在 ``$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/``（含 ``snapshots/<rev>/``、``refs/main``、``blobs/``）。

方式 B——已有下载好的副本（普通目录或另一份缓存），软链进 HF 缓存：

.. code-block:: bash

   mkdir -p "$HF_HOME/hub"
   ln -sfn /path/to/models--Qwen--Qwen3-VL-8B-Instruct \
     "$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct"

.. note::

   软链的目标必须是 HF 缓存布局（``snapshots/`` / ``refs/`` / ``blobs/``），不是 ``--local-dir`` 下载的平铺目录。平铺目录 ``from_pretrained`` 离线时识别不了，要么用方式 A，要么 ``ln -sfn`` 指向一份缓存布局的目录。

**步骤 2：训练时开离线模式**

在 ``run_vla_sft.sh``（或你的 shell）里设：

.. code-block:: bash

   export HF_HOME=${HF_HOME:-~/.cache/huggingface}
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   # 缺缓存资源时的兜底镜像（离线 + 缓存命中时不会走它）：
   # export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

``HF_HUB_OFFLINE=1`` 让 worker 只从 ``$HF_HOME/hub`` 读取、不发网络请求；``TRANSFORMERS_OFFLINE=1`` 对 transformers 库做同样约束。两者都设上，避免离线机器上卡在网络重试。

准备基座模型
----------------------------------------

Cosmos3 SFT **从基座模型 Cosmos3-Nano 热启动，只训练动作头**。``actor.model.model_path`` 必须指向一个 **DCP 格式**的基座权重目录，``wan_vae_path`` 指向 Wan2.2 VAE：

.. code-block:: yaml

   defaults:
     - model/cosmos3@actor.model      # 引入 examples/sft/config/model/cosmos3.yaml

   actor:
     model:
       model_path: /path/to/Cosmos3-Nano-DCP
       wan_vae_path: /path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       load_to_device: false          # 见下方 warning

准备基座 DCP 权重
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**第 1 步：下载 Cosmos3-Nano（diffusers 格式）**

从 Hugging Face 下载 ``nvidia/Cosmos3-Nano``。它是 **diffusers/safetensors 格式**（``model_index.json`` + ``transformer/*.safetensors`` 分片 + ``vae/`` + ``text_tokenizer/`` 等），**不是 DCP**——SFT 还不能直接用，需第 2 步转成 DCP。

.. code-block:: bash

   # 国内可用镜像加速：export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download nvidia/Cosmos3-Nano \
     --local-dir /path/to/Cosmos3-Nano

.. note::

   基座用 ``--local-dir`` 下载到**普通目录**即可。

**第 2 步：转成 DCP**

SFT 不能直接从 diffusers 目录加载基座：cosmos3 的基座加载器 ``_load_base_weights`` 用 ``_is_safetensors_checkpoint`` 判定格式——它只检查路径**顶层**是否有 ``*.safetensors`` 文件（非递归）。diffusers 目录把分片放在 ``transformer/`` 等子目录里、顶层没有，判定为 ``False``，于是落到 DCP 加载分支（``_load_model`` + ``CustomLoadPlanner``），而 diffusers 目录又不是 DCP，会加载失败。

因此必须先把 diffusers 转成 DCP，再给 ``model_path`` 用。用 cosmos_framework 的 ``convert_model_to_dcp.py``：它读取 ``nvidia/Cosmos3-Nano``（diffusers）、重建模型、用 ``torch.distributed.checkpoint`` 存成分片 DCP。

.. code-block:: bash

   # 在训练用的 venv 里跑（保证 DCP metadata 与训练 Python 版本一致）
   python -m cosmos_framework.scripts.convert_model_to_dcp \
     --checkpoint-path /path/to/Cosmos3-Nano \
     --no-use-ema-weights \
     -o /path/to/Cosmos3-Nano-DCP

**这条命令做什么：** 读取 ``nvidia/Cosmos3-Nano``（diffusers/safetensors）→ 经 ``Cosmos3OmniModel.from_pretrained_dcp`` 重建 → 用 ``torch.distributed.checkpoint`` 存成分片 DCP（``model/__0_*.distcp`` + ``.metadata`` + ``checkpoint.json``）。产物即上面 ``actor.model.model_path`` 指向的目录。

.. note::

   - ``--checkpoint-path`` 既接受模型名（``Cosmos3-Nano``，触发 HF 下载）也接受本地路径（已下载的 diffusers 目录）。离线机器先下好 diffusers 再指本地路径，避免联网。
   - DCP 的 ``.metadata`` 是 pickle 序列化、**与保存时的 Python 版本绑定**——所以 ``convert_model_to_dcp`` 必须在与训练**相同的 Python / venv** 里跑，产出的 DCP 才能在训练时加载（本地目录名带 ``-py311`` 后缀就是这个意思：在 py3.11 下重存的 DCP）。
   - 这与 Qwen3-VL 的 HF 缓存是两回事：本步转换的是基座模型权重，Qwen3-VL 是 tokenizer（见上方「离线缓存 Qwen3-VL-8B-Instruct」）。

准备数据
----------------------------------------

训练数据用**原始** LeRobot v3 布局（含 ``meta/``、``data/``）的 LIBERO 即可，通过 ``data.train_data_paths`` 指定本地目录。

``frame_wise_relative`` + ``rot6d`` + ``quantile_rot`` 这套动作转换**不需要预先处理数据集**——它是在加载时由 cosmos 的 ``PackingDataLoader`` 在线完成的：``rlinf/data/datasets/cosmos3/dataloader.py`` 调 cosmos framework 的 ``make_config()`` + experiment ``action_policy_libero_all_nano``，其 dataset 配置里带 ``action_space="frame_wise_relative"`` / ``rotation_space="6d"`` / ``action_normalization="quantile_rot"``，cosmos3的 dataset 会在 load 时把原始 7 维动作转成 10 维 rot6d 并做 quantile 归一化。

quantile 归一化**统计文件** ``libero_native_frame_wise_relative_rot6d.json``（由 cosmos framework 的数据生成器一次性算好），评测侧用 ``action_stats_path`` 引用。训练与评测必须用同一套 recipe（同样的 ``rotation_space`` / ``action_normalization`` + 同一个 stats 文件），否则去归一化会错位。

.. code-block:: yaml

   data:
     train_data_paths: /path/to/LIBERO_LeRobot
     data_type: "libero_all"          # LIBERO 全部 task suite， 可以单独训练 "libero_10" 这个任务
     num_workers: 4
     prefetch_factor: 4
     val_ratio: 0.01

拉起训练
----------------------------------------

在 RLinf 仓库根目录执行：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_cosmos3

**这条命令做什么：** 读取 ``examples/sft/config/libero_sft_cosmos3.yaml``，在每张 GPU 上用 FSDP2 分片 Cosmos3 模型并训练，checkpoint 每 ``save_interval`` 步存入 ``.../checkpoints/global_step_<N>/``，日志写到 ``logs/<时间戳>-libero_sft_cosmos3/run_embodiment.log``。

.. warning::

   **初始化必设** ``actor.model.load_to_device: false``。Cosmos3 构建时 ``net``（bf16 ~27GB）+ ``net_ema``（fp32 ~54GB）≈ 81GB，单张 80GB 卡上默认的 eager ``model.to(device)`` 会在 FSDP2 ``fully_shard`` 分片**之前**就 OOM。设为 ``false`` 让模型留在 CPU，由 ``fully_shard`` 直接分片上卡。

断点续训：设 ``runner.resume_dir`` 指向某个 ``global_step_<N>`` 目录后重跑。

关键配置
----------------------------------------

大多数字段照抄 ``libero_sft_cosmos3.yaml`` 即可。真正需要关注的：

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
     - **必须 ``false``**（见上方 warning）。
   * - ``actor.model.ema_enabled``
     - Cosmos3 power-EMA（rate=0.1），每个优化步更新 ``net_ema``；占一份 fp32 网络显存，若仍 OOM 可临时设 ``false``。
   * - ``actor.micro_batch_size`` / ``global_batch_size``
     - 每卡微批与全局批（示例 32 / 2048）。
   * - ``actor.optim.lr`` / ``lr_warmup_steps``
     - 学习率与 warmup（示例 5e-5 / 500）。动作头额外有 5× LR 倍率（``model/cosmos3.yaml`` 的 ``lr_multipliers``）。
   * - ``runner.max_steps`` / ``save_interval``
     - 训练步数与存档间隔（示例 5000 / 500）。


特殊说明：action 转换 rot6d 10 维 vs axis-angle 7 维
---------------------------------------------------------

Cosmos3 内部用 **10 维 rot6d** 表示动作，LIBERO 环境用 **7 维 axis-angle**。这是理解整条链路的关键：

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - 表示
     - 维度
     - 组成
   * - ``raw_action_dim``（模型侧）
     - 10
     - 3 平移 + 6 rot6d + 1 夹爪
   * - ``action_dim``（环境侧）
     - 7
     - 3 平移 + 3 axis-angle + 1 夹爪

用 rot6d 是因为 axis-angle 在 ±π 处不连续，扩散模型难以回归；rot6d（旋转矩阵前两列）连续、更易学。

训练时 cosmos 的数据加载器会在线把 LIBERO 的 7 维动作转成 10 维 rot6d（见下方「准备数据」）；推理时再把模型输出的 10 维 rot6d 转回 7 维喂环境，详见 :doc:`SGLang 评测 <../../evaluations/guides/cosmos3_sglang>`。


转换 Checkpoint
----------------------------------------

SFT 产出的 ``.../checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`` 是 FSDP2 分片格式，**不能直接评测**。Cosmos3 评测默认使用 diffusers 组件目录 ``model_diffusers``（含 ``model_index.json`` / ``transformer/`` / ``vae/`` / ``text_tokenizer/`` / ``scheduler/``）。从 ``full_weights.pt`` 到 ``model_diffusers`` 分四步（依赖 cosmos framework）：

.. code-block:: text

   1) full_weights.pt  --去掉 omni. 前缀-->  model.safetensors
   2) model.safetensors  --转 DCP 目录-->     model_dcp
   3) model_dcp  --cosmos_framework.scripts.export_model + cosmos config-->  model_hf
   4) model_hf   --cosmos_framework.scripts.convert_model_to_diffusers-->    model_diffusers
      并将 model_index.json 的 _class_name 改为 Cosmos3OmniDiffusersPipeline

先设好路径变量（下面四步都引用）：

.. code-block:: bash

   # SFT 产出的 RLinf checkpoint（omni.net.* 前缀）
   SRC="/path/to/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt"
   # 转换工作目录（四步产物都放这里）
   OUT="/path/to/converted"
   mkdir -p "$OUT"

**第 1 步：** ``full_weights.pt`` → ``model.safetensors``（去掉 ``omni.`` 前缀，让 cosmos 能识别 ``net.*`` / ``net_ema.*``）。

.. code-block:: bash

   python - <<'PY'
   import torch, os
   from safetensors.torch import save_file

   SRC  = os.environ["SRC"]
   OUTD = os.path.join(os.environ["OUT"], "model_safetensors")
   os.makedirs(OUTD, exist_ok=True)

   print("Loading full_weights.pt (~91GB)...")
   sd = torch.load(SRC, map_location="cpu", weights_only=False)
   print(f"Loaded {len(sd)} keys")

   # Strip "omni." prefix: omni.net.* -> net.*, omni.net_ema.* -> net_ema.*
   stripped = {(k[5:] if k.startswith("omni.") else k): v.contiguous() for k, v in sd.items()}
   print(f"Stripped to {len(stripped)} keys (e.g. {list(stripped.keys())[:3]})")

   dst = os.path.join(OUTD, "model.safetensors")
   save_file(stripped, dst)
   print(f"Saved {os.path.getsize(dst)/1e9:.1f} GB -> {dst}")
   PY

.. note::

   ``export`` 命令前先 ``export SRC=... OUT=...``（上面代码块读这两个环境变量）。``weights_only=False`` 因为 RLinf 存的是带元信息的 state dict。

**第 2 步：** ``model.safetensors`` → ``model_dcp``（转成 DCP 目录，``export_model`` 只认 DCP 格式）。

.. code-block:: bash

   python - <<'PY'
   import os, math, torch
   from safetensors.torch import load_file
   import torch.distributed.checkpoint as dcp
   from torch.distributed.checkpoint.filesystem import FileSystemWriter
   from cosmos_framework.checkpoint.dcp import CustomSavePlanner

   SRC = os.path.join(os.environ["OUT"], "model_safetensors", "model.safetensors")
   OUT = os.path.join(os.environ["OUT"], "model_dcp", "model")
   os.makedirs(OUT, exist_ok=True)

   state_dict = load_file(SRC)
   print(f"Loaded {len(state_dict)} keys")

   # ~5GB per shard（与 transformers 默认 max_shard_size 对齐）
   nbytes  = sum(v.numel() * v.element_size() for v in state_dict.values() if isinstance(v, torch.Tensor))
   nshards = max(1, math.ceil(nbytes / (5 * 1024**3)))
   writer  = FileSystemWriter(OUT, thread_count=nshards)
   dcp.save(state_dict=state_dict, storage_writer=writer, planner=CustomSavePlanner())
   print(f"Saved DCP -> {OUT}")
   PY

.. note::

   这一步依赖 ``cosmos_framework.checkpoint.dcp.CustomSavePlanner``，所以须在 cosmos framework 环境里跑（与下面 3、4 步同环境）。产物 ``$OUT/model_dcp/`` 即下一步 ``export_model --checkpoint-path`` 的输入。

**第 3、4 步：** DCP → HF → diffusers（cosmos framework 自带脚本）。

.. code-block:: bash

   python -m cosmos_framework.scripts.export_model \
     --checkpoint-path "$OUT/model_dcp" \
     --config-file /path/to/cosmos3_action_libero/config.yaml \
     --no-use-ema-weights \
     -o "$OUT/model_hf"

   python -m cosmos_framework.scripts.convert_model_to_diffusers \
     --checkpoint-path "$OUT/model_hf" \
     -o "$OUT/model_diffusers"

第 3 步把 DCP 权重按 cosmos 训练配置导出成 HF 格式 ``model_hf``；

第 4 步把 HF 权重拆成 diffusers 组件目录 ``model_diffusers``（SGLang 评测的输入）。第 4 步后须把 ``model_diffusers/model_index.json`` 的 ``_class_name`` 改为 ``Cosmos3OmniDiffusersPipeline``。

.. warning::

   转换须在 **cosmos framework 环境** 下进行：``transformers 4.57.x`` + ``diffusers 0.39.0``（含 ``Cosmos3OmniTransformer``）。更高版本 transformers 会在 ``save_pretrained`` 处崩溃、且其 diffusers 缺 cosmos3 类。转换还需 Wan2.2 VAE（diffusers 格式）与离线 HF 缓存（``HF_HUB_OFFLINE=1``）。具体路径与版本以你的 cosmos framework 部署为准。

转换得到的 ``model_diffusers`` 即 :doc:`Cosmos3 SGLang 评测 <../../evaluations/guides/cosmos3_sglang>` 的 ``rollout.model.model_path``。

可视化与结果
----------------------------------------

查看训练日志与曲线：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

训练指标含义见 :doc:`训练指标 </rst_source/reference/metrics>`。

常见问题
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 现象
     - 处理
   * - 初始化 OOM（单卡 80GB 也爆）
     - 设 ``actor.model.load_to_device: false``。
   * - 加载报动作头 shape 不匹配
     - 确认 ``keys_to_skip_loading`` 跳过了基座 DROID 8 维动作头（``model/cosmos3.yaml`` 默认已配）。
   * - HuggingFace 拉取卡住
     - 离线 + 本地缓存：``HF_HUB_OFFLINE=1``、``TRANSFORMERS_OFFLINE=1``，``HF_HOME`` 指向本地缓存。
   * - 动作乱 / loss 异常
     - 核对训练与评测两侧 ``action_normalization``（``quantile_rot``）与统计文件同源。
