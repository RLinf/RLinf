# FastWAM 官方实现对齐说明

本文说明 `codex/fastwam-official-alignment` 分支中 FastWAM LIBERO SFT 的修改内容、验证方式和当前仍存在的实现边界。

## 1. 目标与基准

本分支以官方 FastWAM 仓库为行为基准，并将其接入 RLinf 的 embodied FSDP SFT pipeline。官方依赖固定在：

```text
45d8e1458921d83f8ad6cf9ce993d371208dabd0
```

目标是对齐影响训练结果的语义：模型初始化、数据内容与顺序、可训练模块、精度、loss、optimizer、scheduler、梯度裁剪、随机数和训练步数计算。

这不是把官方 `Accelerate + DeepSpeed ZeRO-1` launcher 原样搬入 RLinf；RLinf 仍使用自己的 Ray/runner/FSDP2/checkpoint 基础设施。因此后端通信、checkpoint 文件格式和评测入口不保证字节级相同，见第 5 节。

## 2. 具体修改

### 2.1 训练配置

`examples/sft/config/libero_sft_fastwam.yaml`

- 使用官方四个 LIBERO LeRobot 数据集：`spatial`、`object`、`goal`、`libero_10`。
- 对齐官方默认训练参数：
  - `batch_size=16`/rank
  - `num_workers=8`
  - `lr=1e-4`
  - `num_epochs=10`
  - `weight_decay=1e-2`
  - `betas=(0.9, 0.95)`
  - `eps=1e-8`
  - `max_grad_norm=1.0`
  - `seed=42`
  - `log_every=10`
  - `save_every=2000`
- 增加 `fastwam_cosine` scheduler：5% `LinearLR` warmup，随后 cosine decay，`eta_min=0.01 * lr`。
- 根据实际数据集长度、world size、global batch 和 epoch/step 限制动态计算总训练步数，避免使用错误的固定 horizon。
- 默认关闭 `mot_checkpoint_mixed_attn` 对应的 gradient checkpointing，与官方 LIBERO task 配置一致。
- 官方 `eval_every=200` 暂未打开，因为 RLinf 通用 SFT evaluator 尚未提供官方 Wan2.2 rollout/video metrics。

### 2.2 模型与训练模块

`rlinf/models/embodiment/fastwam/__init__.py`

- 复用官方 FastWAM 模型、processor 和模型配置。
- 模型初始化后冻结全部参数，只训练官方训练路径中的 video/action DiT（MoT）和 proprio encoder。
- 记录可训练参数根节点，便于启动日志核对。

`rlinf/models/embodiment/fastwam/fastwam_policy.py`

- SFT loss 直接复用官方 `training_loss`。
- 修复 FSDP2 下 context、video latent、action、VAE 和 proprio 输入的 dtype/device 处理。
- 保留官方 batched inference 行为，并同步 gradient-checkpointing 开关。

`examples/sft/config/model/fastwam.yaml`

- 默认按官方 base-model SFT 路径加载 Wan2.2 video DiT 和 ActionDiT backbone。
- 使用官方 T5 cache、VAE 和模型组件路径。
- 明确 `precision: bf16`，不再使用旧的 fp32/legacy 参数默认值。

### 2.3 混合精度与 FSDP2

`rlinf/workers/sft/fsdp_sft_worker.py`

- FastWAM 使用 `bf16` model parameters + `bf16 autocast`，与官方 `Accelerator(mixed_precision="bf16")` 对齐。
- FastWAM 的参数/梯度由 FSDP2 管理，但不再额外启用一套 FSDP mixed-precision cast policy，避免形成第二条精度转换路径。
- 每次训练 step 强制保持冻结的 VAE/text encoder 为 eval，只有 DiT/MoT 和 proprio encoder 处于 train。
- 在模型初始化后调用官方 seed 初始化逻辑，保证 DataLoader worker 的随机性策略一致。
- fresh run 会避免通用 RLinf optimizer warmup 额外消耗官方的第一步 AdamW 状态；resume 则保留 checkpoint 中的 optimizer/state。

`rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`

- 对 FastWAM 的 bf16 路径使用专门日志，避免误报通用 FSDP fp32 warning。

`examples/sft/config/libero_sft_fastwam.yaml`

- FastWAM 使用 whole-root FSDP2 wrap。
- 不对 MoT 内部 expert/block 做 per-block auto-wrap，因为官方 MoT 会直接访问 cross-expert attention 的内部参数，分块 sharding 会导致参数没有在访问前 all-gather。

### 2.4 数据顺序、worker 和 scheduler

`rlinf/data/datasets/fastwam.py`

- 复用官方数据处理、prompt、动作/视频帧设置、dataset stats 和 text embedding cache。
- 实现与官方 `ResumableEpochSampler` + Accelerate `BatchSamplerShard(even_batches=True)` 一致的分布式 batch 顺序。
- 使用官方 worker seed 初始化函数。
- 归一化 OmegaConf `ListConfig`，支持四套数据目录和单套 smoke override。

`rlinf/hybrid_engines/fsdp/utils.py`

- 新增官方等价的 `fastwam_cosine` scheduler。

### 2.5 数据与环境准备

`examples/sft/prepare_fastwam_sft.sh`

- 下载官方 LIBERO checkpoint 和 dataset stats。
- 下载 Wan2.2 VAE/T5、Wan2.1 T5 和官方 video DiT 权重。
- 预处理 ActionDiT backbone。
- 下载并解压官方 LIBERO LeRobot 数据集。
- 调用官方脚本预计算 T5 text embeddings。

`examples/sft/run_vla_sft.sh`、`requirements/install.sh`

- 统一 FastWAM venv、模型根目录、数据根目录和 text cache 的环境变量。
- 保留 RLinf 原有 runner 入口，同时支持 FastWAM 所需的路径和安装依赖。

### 2.6 文档与测试

- 更新中英文 FastWAM SFT 文档，说明官方参数、FSDP2 限制和当前评测边界。
- `tests/unit_tests/models/test_fastwam.py` 覆盖配置组合、批量 inference 和 checkpoint 开关。
- 增加真实 FastWAM FSDP SFT e2e 配置入口。

## 3. 如何验证

以下命令在 RLinf 仓库根目录执行。先激活已经准备好的环境，并根据实际位置设置路径：

```bash
cd /mnt/public2/wph/codes/develop_async/RLinf_fastwam
source .venv/bin/activate

export FASTWAM_PATH="$PWD/.venv/FastWAM"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export FASTWAM_DATASET_ROOT="/mnt/public2/wph/datasets/LIBERO-fastwam"
export FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$FASTWAM_DATASET_ROOT/text_embeds_cache/libero"
```

### 3.1 静态检查和单元测试

```bash
bash -n examples/sft/run_vla_sft.sh examples/sft/prepare_fastwam_sft.sh

python -m py_compile \
  rlinf/data/datasets/fastwam.py \
  rlinf/hybrid_engines/fsdp/utils.py \
  rlinf/hybrid_engines/fsdp/fsdp_model_manager.py \
  rlinf/models/embodiment/fastwam/__init__.py \
  rlinf/models/embodiment/fastwam/fastwam_policy.py \
  rlinf/workers/sft/fsdp_sft_worker.py

pytest -q tests/unit_tests/models/test_fastwam.py
git diff --check
```

预期：单元测试 `3 passed`，且无 syntax/diff-check 错误。

### 3.2 准备官方资产

首次运行或资产不完整时：

```bash
bash examples/sft/prepare_fastwam_sft.sh
```

如果只想先验证单套数据，可以设置 `FASTWAM_DATASET_DIR`；完整对齐验证应使用四套数据的默认配置。下载模型和数据需要可访问 Hugging Face，必要时先配置本机代理或镜像。

### 3.3 两卡 one-step pipeline smoke

该命令验证真实模型加载、两卡 FSDP2、数据读取、forward/backward、optimizer step 和 checkpoint 保存。为了控制显存，使用 `micro_batch_size=1/global_batch_size=2`，它不是官方完整训练 batch：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/sft/run_vla_sft.sh libero_sft_fastwam \
  runner.max_epochs=-1 \
  runner.max_steps=1 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=2 \
  data.num_workers=2
```

启动日志应至少包含：

```text
[FSDP] AMP is enabled with precision: torch.bfloat16
```

并且两个 rank 都能看到 FastWAM 模型、数据集和 trainable DiT/MoT/proprio 信息。完成后应生成：

```text
logs/<timestamp>-libero_sft_fastwam/libero_sft_fastwam/checkpoints/global_step_1/actor/
```

### 3.4 checkpoint resume

将上一步生成的目录设置为 `runner.resume_dir`，把最大步数改为 2：

```bash
export FASTWAM_RESUME_DIR="/absolute/path/to/logs/<timestamp>-libero_sft_fastwam/libero_sft_fastwam/checkpoints/global_step_1"

CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/sft/run_vla_sft.sh libero_sft_fastwam \
  runner.resume_dir="$FASTWAM_RESUME_DIR" \
  runner.max_epochs=-1 \
  runner.max_steps=2 \
  runner.save_interval=-1 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=2 \
  data.num_workers=2
```

预期：两个 rank 成功恢复 model、optimizer、data loader state 和 RNG state，并完成 step 2，不出现 shape、dtype 或 state restore 错误。

### 3.5 按官方默认参数训练

显存允许时，在两卡上保持官方的每卡 batch 16，应将 global batch 改为 32：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/sft/run_vla_sft.sh libero_sft_fastwam \
  actor.micro_batch_size=16 \
  actor.global_batch_size=32 \
  data.num_workers=8
```

官方 8 卡配置使用 `global_batch_size=128`。不要把两卡 smoke 的 `global_batch_size=2` 与官方训练曲线直接比较。

## 4. 已完成的验证结果

本分支已经完成：

- 真实两卡 FastWAM FSDP2 forward/backward/optimizer step。
- 两卡 step-1 checkpoint 保存。
- 从该 checkpoint 恢复并完成 step 2。
- FastWAM unit tests：`3 passed`。
- shell syntax、Python compile 和 `git diff --check`。
- sampler 与官方 Accelerate batch sharding 的组合测试。
- scheduler 与官方 warmup + cosine 组合的数值对照测试。

已观察到但不影响训练的环境警告：torchcodec 无法加载 `libnvrtc.so.13` 时会回退到 torchvision/pyav；如需消除该警告，应补齐对应 CUDA runtime，而不是修改 FastWAM 训练逻辑。

## 5. 当前仍未完全等同的部分

1. **分布式后端**：官方 launcher 使用 Accelerate + DeepSpeed ZeRO-1；本分支使用 RLinf FSDP2。训练语义已对齐，但 collective、参数分片和 optimizer state 的底层实现不同。
2. **官方评测 loop**：官方每 `eval_every=200` 进行 Wan2.2 rollout、视频指标和动作指标；RLinf 当前 generic SFT evaluator 尚未接入这一套，因此配置中 `val_check_interval=-1`。
3. **checkpoint 格式**：官方和 RLinf 的保存目录、metadata、DCP shard 组织不同；RLinf 内部 resume 已验证，但不能直接互换官方 checkpoint 目录。
4. **launcher/logging**：官方使用 `scripts/train_zero1.sh` 和 Accelerate；RLinf 使用 `run_vla_sft.sh`、Ray runner 和 RLinf logger。
5. **双卡完整训练曲线**：当前真实 GPU 验证是 smoke/resume；完整十 epoch、多卡规模和 LIBERO rollout 指标仍需要专门的长任务验证。

因此，本分支可以称为“训练语义对齐 + RLinf FSDP2 集成”，不应宣称已经达到官方后端和评测流程的字节级复刻。

## 6. 重要文件索引

| 文件 | 作用 |
| --- | --- |
| `examples/sft/config/libero_sft_fastwam.yaml` | 官方训练参数、数据集、FSDP2 和 AMP 配置 |
| `examples/sft/config/model/fastwam.yaml` | FastWAM 模型与精度默认配置 |
| `rlinf/data/datasets/fastwam.py` | 数据处理、分布式 sampler、worker seed |
| `rlinf/models/embodiment/fastwam/__init__.py` | 官方模型构造和 trainable module 选择 |
| `rlinf/models/embodiment/fastwam/fastwam_policy.py` | 官方 loss/inference 与 RLinf policy 适配 |
| `rlinf/workers/sft/fsdp_sft_worker.py` | bf16、train/eval mode、训练 horizon、resume 行为 |
| `rlinf/hybrid_engines/fsdp/utils.py` | 官方等价 FastWAM scheduler |
| `examples/sft/prepare_fastwam_sft.sh` | 模型、数据、ActionDiT 和 text cache 准备 |
| `docs/fastwam_official_alignment.md` | 本验证与差异说明 |
