使用 ImageReward 微调 SD3
=========================

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward 概览，图片来自 ImageReward 作者的官方仓库。

使用 Reward Feedback Learning（ReFL）和可微的图像偏好奖励微调 Stable Diffusion 3.5。先运行纯奖励训练，再按需加入图文监督更新，比较它们对生成质量的影响。本示例复用 RLinf 的 diffusion 环境、rollout worker、FSDP actor、权重同步和训练 runner。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      Diffusion 图像生成

   .. grid-item-card:: 算法
      :text-align: center

      ReFL

   .. grid-item-card:: 任务
      :text-align: center

      文生图偏好微调

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA；下文采用四卡配置

配置文件为 ``examples/diffusion/config/sd3_refl_imagereward.yaml``。共享的 diffusion 环境和 NFT 示例见 :doc:`../video_models`。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 模型
     - 奖励
     - 训练数据
   * - 使用 LoRA 的 SD3.5 Medium
     - ImageReward 人类偏好分数
     - 文本 prompt；可选的图文对用于监督更新

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 观测
     - ``task_descriptions``：一批文本 prompt。
   * - 动作
     - 生成的 RGB 图像；每张图像对应一个完整的环境 episode。
   * - 奖励
     - ``avg`` 和 ``image_reward``：相同的标量分数，越高越好。
   * - 更新输入
     - rollout 保存的晚期去噪状态和 prompt embedding，均已 detach。

安装
----

在仓库根目录安装 diffusion 环境；通用安装要求见 :doc:`/rst_source/start/installation`：

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

diffusion 依赖选用 ``image-reward==1.5``，该版本依赖 ``timm==0.6.13``。复用已有环境时，先检查这两个版本；较新的 ``timm`` 可能与 ImageReward 的依赖要求冲突。

准备模型与 Prompt
-----------------

先在 Hugging Face 接受 SD3.5 模型许可，并通过 ``hf auth login`` 登录，以下载需要授权的权重。随后下载生成模型、奖励权重、ImageReward 使用的 BERT tokenizer 和官方 prompt：

.. code-block:: bash

   export SD3_MODEL="$PWD/models/stable-diffusion-3.5-medium"
   export IR_MODEL="$PWD/models/ImageReward"
   export IR_DATA="$PWD/data/ImageReward"
   mkdir -p "$SD3_MODEL" "$IR_MODEL" "$IR_DATA"

   hf download stabilityai/stable-diffusion-3.5-medium --local-dir "$SD3_MODEL"
   hf download zai-org/ImageReward ImageReward.pt med_config.json --local-dir "$IR_MODEL"
   python -c 'from transformers import BertTokenizer; BertTokenizer.from_pretrained("bert-base-uncased")'

   for file in refl_data.json test.json; do
     curl --fail --location \
       "https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/data/$file" \
       --output "$IR_DATA/$file" || break
   done

在启动训练的 shell 中保留这些环境变量。每个节点都需要能够从 Hugging Face 缓存加载 ``bert-base-uncased``；离线运行前，应在每个节点准备好缓存。

prompt JSON 的顶层是列表，每项包含 ``text`` 或 ``prompt``。loader 去除首尾空白，按原顺序去重，并从训练集中排除 ``env.train.dataset.exclude_path`` 中的 prompt。上述固定版本的官方文件会得到 9,756 条训练 prompt 和 463 条评测 prompt。

运行纯奖励训练
--------------

先只优化奖励目标，作为后续比较监督更新影响的对照。以下四卡配置每个 runner step 生成 512 个训练样本，按 global batch size 64 更新：

.. code-block:: bash

   REFL_ARGS=(
     actor.model.model_path="$SD3_MODEL"
     rollout.model.model_path="$SD3_MODEL"
     actor.model.sd3.lora_path=null
     actor.refl_reward_model.model_path="$IR_MODEL/ImageReward.pt"
     actor.refl_reward_model.med_config="$IR_MODEL/med_config.json"
     env.train.dataset.path="$IR_DATA/refl_data.json"
     env.train.dataset.exclude_path="$IR_DATA/test.json"
     env.eval.dataset.path="$IR_DATA/test.json"
     env.train.total_num_envs=4
     env.train.rollout_epoch=128
     env.eval.total_num_envs=4
     env.eval.rollout_epoch=128
     actor.global_batch_size=64
     actor.micro_batch_size=1
     actor.refl_pre_data.num_workers=0
     rollout.pipeline_stage_num=1
     env.train.reward.batch_size=1
     env.eval.reward.batch_size=1
     actor.seed=42
     env.train.seed=42
     env.eval.seed=0
     algorithm.update_epoch=1
     algorithm.refl_grad_scale=1e-3
     runner.max_epochs=200
     runner.max_steps=200
     runner.val_check_interval=10
     runner.save_interval=10
     runner.resume_dir=null
     runner.ckpt_path=null
   )

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     algorithm.refl_pre_loss_weight=0.0 \
     runner.logger.experiment_name=sd3_refl_reward_only

此命令从原始模型开始训练 200 步，每 10 步评测并保存。请按 :doc:`/rst_source/start/installation` 配置机器上的 GPU；micro-batch size 1 只是起点，不保证一定满足显存限制。actor 会加载可微的 ImageReward 模型，环境中的奖励打分也会占用显存。

加入图文监督更新
----------------

将 ``algorithm.refl_pre_loss_weight`` 设为正值，会在奖励更新之外继续训练生成模型的去噪能力。这个可选的预训练目标损失称为 Lpre。请将本地图文对整理成 img2dataset 的 ``files`` 布局：

.. code-block:: text

   image_text_data/
     00000.jpg
     00000.txt
     shard_01/
       00001.png
       00001.txt

loader 会递归查找 ``.jpg``、``.jpeg``、``.png`` 和 ``.webp`` 图像。每张图像旁需要有同名的 UTF-8 ``.txt`` caption。图像会经过缩放、512 像素中心裁剪、随机水平翻转，并归一化到 [-1, 1]。默认的 ``caption_prefix`` 仅在 caption 以 ``The image shows`` 加一个空格开头时移除该前缀。

下载 PD3M 子集
~~~~~~~~~~~~~~

如果还没有图文数据，可使用公开的 `PD3M 数据集 <https://huggingface.co/datasets/Spawning/PD3M>`_。其 metadata 保存 caption 和图像 URL，图像文件则位于独立的 PD12M S3 bucket。以下示例从两个 metadata 分片筛选 20,000 条候选，不下载全量数据。此阶段只需要 CPU，建议在申请 GPU 前完成。

先将 ``DATA_ROOT`` 指向数据盘。例如，可使用 ``export DATA_ROOT=/mnt/data/refl`` 将资源放在挂载的数据盘，或指定自己的可写目录。下方命令会为 img2dataset 创建独立环境，避免改动 diffusion 训练环境。运行时保持 diffusion 环境激活，由其中的 ``hf`` 命令下载 metadata。下方固定 PyArrow 版本以兼容 NumPy 1.26，并安装图像检查需要的 Pillow。

示例固定了 metadata revision 和 shuffle seed。筛选要求 license 字段存在、URL 和 caption 非空、短边至少 512 像素、长宽比不超过 2，并排除 iNaturalist 及命中下列黑白图像或文档关键词的 caption。这些规则只是示例数据筛选方法，不保证图像质量，也不复现论文中的训练集。下载后仍需结合目标任务抽查图像与 caption。

.. code-block:: bash

   export DATA_ROOT="${DATA_ROOT:-$PWD/data/refl}"
   export PD3M_DIR="${PD3M_DIR:-$DATA_ROOT/PD3M}"
   export PRE_DATA="$PD3M_DIR/images_512"
   export PD3M_VENV="$DATA_ROOT/.venv-img2dataset"

   (
     set -e
     unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
     export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_ROOT/.cache/uv}"
     export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$DATA_ROOT/.cache/uv-python}"
     export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
     export NO_ALBUMENTATIONS_UPDATE=1
     mkdir -p "$PD3M_DIR" "$TMPDIR"

     if [ ! -x "$PD3M_VENV/bin/python" ]; then
       uv venv --python 3.10 "$PD3M_VENV"
     fi
     uv pip install --python "$PD3M_VENV/bin/python" \
       img2dataset==1.47.0 numpy==1.26.4 \
       pyarrow==19.0.1 pillow==11.1.0 \
       opencv-python-headless==4.11.0.86 albumentations==1.4.24

     hf download Spawning/PD3M \
       metadata/pd3m.00.parquet metadata/pd3m.01.parquet \
       --repo-type dataset \
       --revision 2a5eb24a8dccf245acd8e56341761aee06da0bdf \
       --local-dir "$PD3M_DIR"

     "$PD3M_VENV/bin/python" - <<'PY'
   import os
   import random
   import re
   from pathlib import Path

   import pyarrow as pa
   import pyarrow.parquet as pq

   root = Path(os.environ["PD3M_DIR"])
   columns = ["id", "url", "caption", "width", "height", "license", "source"]
   rows = []
   for shard in ("00", "01"):
       rows.extend(pq.read_table(
           root / "metadata" / f"pd3m.{shard}.parquet", columns=columns,
       ).to_pylist())

   monochrome = re.compile(r"black[- ]and[- ]white|grayscale|sepia|monochrome", re.I)
   document = re.compile(
       r"\b(book|page|newspaper|document|letter|map|poster|stamp|coin|text (on|at|in))\b",
       re.I,
   )
   candidates = []
   for row in rows:
       caption = row["caption"] or ""
       width, height = row["width"] or 0, row["height"] or 0
       if (
           row["url"] and caption.strip() and row["license"]
           and row["source"] != "iNaturalist"
           and min(width, height) >= 512
           and max(width, height) <= 2 * min(width, height)
           and not monochrome.search(caption)
           and not document.search(caption)
       ):
           candidates.append(row)

   assert len(candidates) >= 20000, f"Need 20000 candidates, found {len(candidates)}"
   random.Random(42).shuffle(candidates)
   table = pa.Table.from_pylist(candidates[:20000]).select(columns)
   manifest = root / "pd3m_refl.parquet"
   if manifest.exists():
       assert pq.read_table(manifest).equals(table), (
           f"Existing manifest differs: {manifest}; use a new PD3M_DIR"
       )
   else:
       pq.write_table(table, manifest)
   print(f"Metadata rows: {len(rows)}; eligible: {len(candidates)}; selected: {len(table)}")
   print("Manifest:", manifest)
   PY

     "$PD3M_VENV/bin/img2dataset" \
       --url_list="$PD3M_DIR/pd3m_refl.parquet" \
       --input_format=parquet --url_col=url --caption_col=caption \
       --save_additional_columns='["id", "license", "source"]' \
       --output_format=files --output_folder="$PRE_DATA" \
       --image_size=512 --resize_mode=keep_ratio \
       --resize_only_if_bigger=True \
       --number_sample_per_shard=1000 \
       --processes_count=16 --thread_count=32 \
       --timeout=60 --retries=2 \
       --incremental_mode=incremental --enable_wandb=False

     "$PD3M_VENV/bin/python" - <<'PY'
   import os
   from pathlib import Path

   import pyarrow.parquet as pq
   from PIL import Image

   root = Path(os.environ["PRE_DATA"])
   count = pq.read_table(Path(os.environ["PD3M_DIR"]) / "pd3m_refl.parquet").num_rows
   stats = list(root.glob("*_stats.json"))
   expected_shards = (count + 999) // 1000
   assert len(stats) == expected_shards, (
       f"Incomplete download: {len(stats)}/{expected_shards} shard stats files"
   )
   images = sorted(root.rglob("*.jpg"))
   invalid = []
   for path in images:
       try:
           assert path.with_suffix(".txt").read_text(encoding="utf-8").strip(), "empty caption"
           with Image.open(path) as image:
               image.convert("RGB").load()
       except Exception as error:
           invalid.append(f"{path}: {error}")
   assert not invalid, f"Invalid image-caption pairs: {invalid[:5]}"
   assert len(images) >= 64, f"Need at least 64 valid pairs, found {len(images)}"
   print(f"Finished shards: {len(stats)}; valid image-caption pairs: {len(images)}")
   print("Lpre path:", root)
   PY
   )

命令会将图像保存为短边 512 的 JPEG，并写入同名 ``.txt`` caption；每张图像的 metadata 保留原始 ID、license 和 source。训练时 Lpre loader 再做中心裁剪。请随实验保留原始 metadata 和 ``pd3m_refl.parquet``，以便复现数据选择。数据集卡标注的许可为 CDLA-Permissive-2.0，同时应检查 manifest 中保留的逐图像许可信息。

``hf download`` 沿用当前 Hugging Face endpoint 和缓存设置。若使用 ``HF_ENDPOINT=https://hf-mirror.com``，应在这一步之前导出；该设置只影响 metadata 下载，图像 URL 仍指向 S3。请为原图传输预留时间和磁盘空间：缩放减少的是落盘图像大小，不会减少从服务器获取的原始字节数。训练时保持与 BERT tokenizer 下载阶段相同的 ``HF_HOME``。

示例使用 16 个进程，每个进程 32 个线程；共享机器或带宽受限时应降低这两个值。20,000 条输入按每片 1,000 条划分，全部完成后应有 20 个 ``*_stats.json``。统计文件在对应分片结束时才生成，因此 JPG 数量增加而统计文件数暂时为零是可能的。可在另一个 shell 中检查进度：

.. code-block:: bash

   find "$PRE_DATA" -type f -name '*.jpg' | wc -l
   find "$PRE_DATA" -maxdepth 1 -type f -name '*_stats.json' | wc -l

20,000 是候选数量，不是成功图文对数量；URL 失败或图像无法解码都会减少最终数量。下载末尾的检查会验证所有已写入 JPEG 和 caption，并要求全部分片完成；请记录其打印的最终图文对数量。最低 64 对的检查只是为了支持下文的 smoke，不代表数据多样性足以用于正式训练；若下载成功率很低，应先检查原因。显式设置的 ``--incremental_mode=incremental`` 在重跑时跳过已有完成统计的分片，即使其中存在失败 URL，也不会自动重试这些分片内的每张失败图片。修改候选列表或分片大小时，应使用新的输出目录，避免误用旧分片。若 manifest 一致性检查失败，请另设 ``PD3M_DIR``，不要原地覆盖。

启动监督训练
~~~~~~~~~~~~

图文检查通过后，``PRE_DATA`` 就是 image-text loader 接受的目录。若已有相同格式的数据，也可直接将 ``PRE_DATA`` 指向现有目录，再启动独立实验：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     runner.logger.experiment_name=sd3_refl_pre01

此实验保持纯奖励实验的奖励系数和 rollout 设置。将 ``algorithm.refl_pre_loss_weight`` 改为 ``0.01``，同时使用新的实验名，可比较更低权重的效果。保留 ``actor.refl_pre_data.num_workers=0``，让 actor 进程直接加载图像，避免在 accelerator 初始化后创建 DataLoader 子进程。

两种更新如何交互
~~~~~~~~~~~~~~~~

actor 从晚期 rollout 状态预测干净 latent，解码后计算奖励。VAE 和 ImageReward 的参数保持冻结，但梯度可以经过它们回传到生成模型。奖励损失为：

.. math::

   L_{\mathrm{reward}} = \lambda\,\mathbb{E}
   [\max(0, m - r(\hat{x}_0, c))],

其中 ``refl_grad_scale`` 控制 :math:`\lambda`，``refl_margin`` 控制 :math:`m`。``refl_xt_source=rollout`` 使用保存的晚期状态；``resample`` 则在采样到的晚期时间对最终 rollout latent 重新加噪。

SD3 的监督目标采用 flow matching：

.. math::

   z_t = (1-t)z_0 + t\epsilon, \qquad
   L_{\mathrm{pre}} = \mathbb{E}\left[
   \lVert v_\theta(z_t,t,c) - (\epsilon-z_0)\rVert^2\right].

对于每个 global batch，启用 Lpre 时，actor 先用 ``refl_pre_loss_weight * Lpre`` 做一次 optimizer step，再用 Lreward 做另一次 step。两次更新共享同一个 AdamW 及其动量统计，因此降低 Lpre 系数不等于按相同比例减小参数更新。系数为零时，会跳过图文数据加载和监督更新。

Lpre 学习的是提供的图文数据，而不是约束模型保持原始 SD3 的输出。数据分布不匹配或两个目标之间的干扰都可能降低 ImageReward 分数。应与纯奖励实验对比，并检查生成图片；奖励提高本身不能证明视觉质量改善。原始 SD1.4 方法见 `ImageReward 论文 <https://arxiv.org/abs/2304.05977>`_；本示例将其中的去噪监督目标适配为 SD3 的 flow matching。

调整监督权重
------------

调整监督权重，是为了判断图文去噪学习能否保留有用的生成能力，还是干扰了偏好优化。Lpre 学习的是所选图文数据的去噪任务，不是 KL 惩罚、对初始 SD3 输出的匹配约束，也不是第二个奖励。它是否有益取决于 caption 与图像是否匹配、图像质量，以及数据分布是否接近目标 prompt。

结合优化器理解权重
~~~~~~~~~~~~~~~~~~

用 :math:`\alpha` 表示 ``refl_pre_loss_weight``。提高它会放大监督 step 提交的梯度，但 actor 执行的是连续两次 AdamW step，不是将两个损失相加后更新一次：

.. code-block:: text

   alpha > 0: step(alpha * Lpre), then step(Lreward)
   alpha = 0: step(Lreward)

AdamW 的参数更新同时依赖当前梯度及其一阶、二阶历史统计。在忽略 weight decay 时，可近似写成 :math:`\Delta\theta=-\eta\hat{m}/(\sqrt{\hat{v}}+\epsilon)`。缩放梯度序列也会改变这些统计量；两个损失还共享优化器历史，因此 ``alpha=0.01`` 并不意味着监督参数更新只有 ``alpha=1`` 时的 1%。尤其要区分零和很小的正数：零会完全跳过监督 step，正数则仍保留这次 step 及其与奖励 step 的交互。weight decay 和梯度裁剪也会影响更新。具体规则见 `AdamW 更新公式 <https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_。

按上文配置，一个 runner step 包含八个 global batch。启用 Lpre 后对应八次监督更新和八次奖励更新；权重为零时只有八次奖励更新。比较时应对齐 runner step 和生成样本数量，并记录额外计算开销，而不是认为两者的 optimizer step 数相同。

固定其余变量再比较
~~~~~~~~~~~~~~~~~~

先完成下文的短程 GPU smoke，再固定 ``algorithm.refl_grad_scale=1e-3``、学习率、batch size、prompt 划分、图像子集、seed 和评测频率，只改变监督权重。每组都从相同初始模型和全新优化器开始；共享的 ``REFL_ARGS`` 已显式清空 LoRA/checkpoint 路径并设置 seed。相同 seed 能控制一部分波动，但开启 Lpre 仍可能改变训练中随机数的消耗顺序。

将 ``0.0`` 作为相同步数的纯奖励对照。``1.0 / 0.75 / 0.5 / 0.1 / 0.01`` 是由强到弱的诊断扫描，不是论文指定值，也不是已经验证适合 SD3 的最优值。时间紧张时可先比较 ``0.0 / 0.01 / 0.1``；否则可用下面的命令让六组各跑 50 个 runner step：

.. code-block:: bash

   (
     set -e
     stamp="$(date +%Y%m%d_%H%M%S)"
     for weight in 0.0 1.0 0.75 0.5 0.1 0.01; do
       CUDA_VISIBLE_DEVICES=0,1,2,3 \
       bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
         "${REFL_ARGS[@]}" \
         actor.refl_pre_data.path="$PRE_DATA" \
         algorithm.refl_pre_loss_weight="$weight" \
         runner.max_epochs=50 runner.max_steps=50 \
         runner.logger.experiment_name="sd3_refl_pre${weight//./p}_50steps_${stamp}"
     done
   )

各组使用独立实验名，每 10 步评测并保存，当前组成功结束后才启动下一组。即使以前的纯奖励实验效果不错，也应保留这组同条件零权重对照，避免把训练时长或评测设置的差异误当成 Lpre 的影响。

结合曲线和图片判断
~~~~~~~~~~~~~~~~~~

用留出集的 ``eval/image_reward`` 和记录的图片筛选候选，不要只看训练 loss。actor 奖励来自采样的晚期状态预测，与最终生成图像的评测奖励并不是同一种测量。

.. list-table:: Lpre 对照实验的判断依据
   :header-rows: 1
   :widths: 27 34 39

   * - 现象
     - 支持的判断
     - 下一步检查
   * - 权重越低，留出集奖励越接近纯奖励对照
     - 监督更新可能与偏好优化存在竞争。
     - 检查 caption 和图像是否贴合任务，并与零权重组比较。
   * - 所有正权重都相近地退化，但零权重正常
     - 降低系数尚未消除额外 AdamW step 的影响。
     - 检查数据质量和梯度，不要假设再缩小十倍就一定能解决。
   * - 中间权重奖励接近、图片质量更好
     - 可能存在有用的质量折中。
     - 检查 prompt 覆盖、多样性，并用其他 seed 复验。
   * - 训练奖励上升，但留出集奖励或图片质量下降
     - 可能出现过拟合或奖励利用。
     - 用固定留出 prompt 检查伪影，再决定是否增强奖励更新。

``train/actor/refl_pre_loss`` 记录未乘权重的去噪 MSE，而 ``train/actor/refl_reward_loss`` 已包含 ``refl_grad_scale``。两种梯度范数也已包含各自损失系数，但其比值不是参数更新比，更不应强行调成 1。先确认损失和梯度范数保持有限，再按各指标自己的定义比较随训练的变化，而不是直接比较两个不同 loss 的数值大小。样本奖励超过 ``refl_margin`` 时，对应奖励梯度也会消失；奖励梯度范数下降时，还需结合奖励分布判断。

用长跑确认候选权重
~~~~~~~~~~~~~~~~~~

短程筛选后，再将候选权重延长到 200 步。例如可独立比较 ``0.1`` 和 ``0.01``；这两个值只是候选，不代表它们一定优于纯奖励训练：

.. code-block:: bash

   (
     set -e
     stamp="$(date +%Y%m%d_%H%M%S)"
     for weight in 0.1 0.01; do
       CUDA_VISIBLE_DEVICES=0,1,2,3 \
       bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
         "${REFL_ARGS[@]}" \
         actor.refl_pre_data.path="$PRE_DATA" \
         algorithm.refl_pre_loss_weight="$weight" \
         runner.max_epochs=200 runner.max_steps=200 \
         runner.logger.experiment_name="sd3_refl_pre${weight//./p}_200steps_${stamp}"
     done
   )

若尚无同条件的 200 步纯奖励对照，还应使用相同命令将 ``0.0`` 组延长到 200 步。比较相同步数的 checkpoint，并用多个 seed 复验后，再将小幅分数差异归因于权重。完成这一轮后，再单独扫描 ``refl_grad_scale`` 或学习率；同时改变 Lpre 权重、奖励系数和学习率，会让实验无法区分 Lpre 本身的影响。

验证训练路径
------------

带 Lpre 长跑前，先检查训练、评测、checkpoint 保存和数据迭代器换轮。完整数据集通过验证后，用软链接建立独立的 64 对图文子集，保留正式训练使用的 ``PRE_DATA``，不复制图像内容：

.. code-block:: bash

   export DATA_ROOT="${DATA_ROOT:-$PWD/data/refl}"
   mkdir -p "$DATA_ROOT"
   export SMOKE_PRE_DATA="$(mktemp -d "$DATA_ROOT/pd3m-smoke.XXXXXX")"
   python - <<'PY'
   import os
   import random
   from pathlib import Path

   source = Path(os.environ["PRE_DATA"])
   target = Path(os.environ["SMOKE_PRE_DATA"])
   images = sorted(path for path in source.rglob("*.jpg") if path.is_file())
   assert len(images) >= 64, f"Need 64 image-caption pairs, found {len(images)}"
   for index, image in enumerate(random.Random(42).sample(images, 64)):
       caption = image.with_suffix(".txt")
       assert caption.read_text(encoding="utf-8").strip(), f"Empty caption: {caption}"
       (target / f"{index:05d}.jpg").symlink_to(image.resolve())
       (target / f"{index:05d}.txt").symlink_to(caption.resolve())
   print("Smoke image-caption directory:", target)
   PY

按下面的四卡设置，每个 actor rank 在一个 runner step 中消耗其 16 对图文，第 2、3 步就会重新创建迭代器。使用这个子集运行三步检查：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$SMOKE_PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     env.train.rollout_epoch=16 \
     env.eval.rollout_epoch=1 \
     runner.max_epochs=3 runner.max_steps=3 \
     runner.val_check_interval=3 runner.save_interval=3 \
     runner.logger.experiment_name=sd3_refl_smoke

确认进程正常退出，并且实验目录下的 ``checkpoints/global_step_3/actor`` 包含 checkpoint 文件。smoke 通过表示这些边界路径已执行，不保证长跑显存稳定或奖励提升；正式训练仍使用完整的 ``PRE_DATA`` 目录。

在 diffusion 环境运行 CPU 单元测试：

.. code-block:: bash

   pytest tests/unit_tests/test_diffusion.py

测试使用临时生成的数据和随机初始化的小模型；外部 ImageReward 权重由小型本地模块替代，RLinf scorer 则正常运行。测试覆盖 prompt 排除、图像预处理、可微奖励、SD3 rollout 与前向的一致性、监督梯度及配置组合。它们不验证真实权重下的奖励提升，也不验证多卡 FSDP 执行；缺少可选依赖时，相应测试会跳过。

将 ``env.eval.video_cfg.save_video=true`` 可把评测样本保存到 ``env.eval.video_cfg.video_base_dir``。diffusion 环境将图像重复为视频帧，以复用录像 wrapper。

结合生成图片观察 ``eval/image_reward`` 和 ``train/actor/refl_reward``。启用 Lpre 时，再检查 ``train/actor/refl_pre_loss``、``train/actor/refl_pre_grad_norm`` 和 ``train/actor/refl_reward_grad_norm``；梯度范数之比不是 AdamW 参数更新之比。每次评测抽取 512 个样本，而测试集有 463 条 prompt，因此其中 49 条会重复；不同实验应保持相同的评测配置。共享日志流程见 :doc:`/rst_source/guides/logger`。
