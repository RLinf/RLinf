# Classifier-Free Guidance for Embodied RL (Pi06)

本文档介绍 Pi06 pipeline 中 Classifier-Free Guidance (CFG) 的原理、阶段关系和完整使用流程。

---

## 1. 原理概述

Pi06 将 **Classifier-Free Guidance (CFG)** 从扩散模型领域引入到具身智能的策略学习中。核心思想：

- **训练时**：根据 advantage 标签，为每个样本选择正向（positive）或负向（negative）引导 prompt，以一定概率随机退化为无条件输入，让模型同时学习两种分布
- **推理时**：同时计算有条件和无条件两个去噪方向，通过插值得到最终动作

去噪公式（推理）：

```
v_t = (1 - w) * v_uncond + w * v_cond
```

其中 `w = cfgrl_guidance_scale`。`w > 1` 增强引导效果，`w = 1` 纯有条件生成，`w = 0` 纯无条件生成。

这个过程需要一条 **数据处理和模型训练的 pipeline** 来支撑，共 5 个阶段，严格按顺序执行。

---

## 2. Pipeline 阶段详解

```
┌──────────────────────┐
│ Stage 1              │
│ Compute Returns      │  为原始数据集添加 reward / return 列
│                      │  输入: LeRobot 数据集 (SFT + Rollout)
│                      │  输出: 数据集原地更新 (parquet + stats.json)
└──────────┬───────────┘
           │  Stage 2 需要数据集中有 return 列来构造 value 训练目标
           │
┌──────────▼───────────┐
│ Stage 2              │
│ Train Value Model    │  训练 critic 模型学习 V(s)
│                      │  输入: Stage 1 处理后的数据集 + 基础模型
│                      │  输出: Value model checkpoint (SafeTensors)
└──────────┬───────────┘
           │  Stage 3 需要训练好的 value model 来估计 V(s)
           │
┌──────────▼───────────┐
│ Stage 3              │
│ Compute Advantages   │  用 V(s) 计算 TD(N) advantage，标记正/负样本
│                      │  输入: Stage 1 的数据集 + Stage 2 的 value checkpoint
│                      │  输出: advantages.parquet (每个样本一个 True/False 标签)
└──────────┬───────────┘
           │  Stage 4 需要 advantage 标签来区分正/负引导
           │
┌──────────▼───────────┐
│ Stage 4              │
│ CFG SFT Training     │  用 advantage 标签 + 引导 prompt 训练 CFG 模型
│                      │  输入: Stage 1 的数据集 + Stage 3 的 advantage 标签 + 基础模型
│                      │  输出: CFG action model checkpoint (SafeTensors)
└──────────┬───────────┘
           │  Stage 5 使用 Stage 4 的 CFG 模型做完整 RL 调试
           │  (也可独立使用基础模型运行)
┌──────────▼───────────┐
│ Stage 5 (可选)        │
│ Debug One-Iter       │  完整 pi06 RL 循环调试 (离线模式)
│                      │  输入: Stage 3 的 advantage 标签 + 基础模型
│                      │  输出: 训练后的 checkpoint
└──────────────────────┘
```

### 阶段间的数据依赖关系

| 阶段 | 依赖 | 为什么需要 |
|------|------|-----------|
| **Stage 1 → Stage 2** | Stage 2 读取数据集中的 `return` 列作为 value model 的训练目标。return 由 `G_t = r_t + gamma * G_{t+1}` 反向累积而来，代表从当前时刻到 episode 结束的累积收益 |
| **Stage 2 → Stage 3** | Stage 3 加载 Stage 2 训练出的 value model，对每个状态推理 `V(s_t)` 和 `V(s_{t+N})`，计算 TD(N) advantage: `A = reward_sum + gamma^N * V(s_{t+N}) - V(s_t)`。advantage 大于阈值（默认 top 30%）的样本标记为 `is_success=True`（正向），其余为 `False`（负向） |
| **Stage 3 → Stage 4** | Stage 4 的 CFG 训练核心就是根据 `is_success` 标签选择引导 prompt。`True` 的样本使用 `[POSITIVE][POSITIVE]\nTask: ...` 作为条件，`False` 的使用 `[NEGATIVE][NEGATIVE]\nTask: ...`。模型通过这种条件化学习区分好/差动作的分布 |
| **Stage 3 → Stage 5** | Stage 5 是完整 RL 循环的离线调试，同样需要 advantage 标签来驱动 CFG 训练 |

### 数据集说明

Pipeline 使用两种数据集：

- **SFT 数据集**：人类示范数据，所有 episode 都是成功的。reward = -1/step，最后一步 = 0。return 单调递减
- **Rollout 数据集**：策略采集的数据，包含成功和失败 episode。失败 episode 的最后一步 reward = `failure_reward`

两者在所有阶段中一起处理，Stage 3 会用统一阈值对两个数据集的 advantage 做 True/False 二分。

---

## 3. 关键文件索引

| 文件 | 说明 |
|------|------|
| **Stage 1** | |
| `examples/process/compute_returns.py` | 计算 return 的主脚本（Hydra 入口） |
| `examples/process/run_compute_returns.sh` | Stage 1 启动器 |
| `examples/process/config/compute_returns_test.yaml` | Stage 1 测试配置 |
| **Stage 2** | |
| `examples/vla_lib_sft/train_vla_lib_sft.py` | Value model 训练入口 |
| `examples/vla_lib_sft/run_vla_lib_sft.sh` | Stage 2 启动器 |
| `examples/vla_lib_sft/config/libero_value_model_test.yaml` | Stage 2 测试配置 |
| `rlinf/runners/vla_lib_sft_runner.py` | `VlaLibSFTRunner` — 训练循环 |
| `rlinf/workers/vla_lib_sft/fsdp_value_sft_worker.py` | `FSDPValueSftWorker` — FSDP 训练 |
| `rlinf/models/embodiment/vla_lib_value_model/` | Value model（critic head）定义 |
| `rlinf/datasets/vla_lib/value_mixture_dataset.py` | Value 训练数据集 |
| `rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf_vla_lib.sh` | Value checkpoint 转换脚本 |
| **Stage 3** | |
| `examples/process/compute_advantages.py` | 计算 advantage 的主脚本（支持 torchrun） |
| `examples/process/run_compute_advantages.sh` | Stage 3 启动器 |
| `examples/process/config/compute_advantages_test.yaml` | Stage 3 测试配置 |
| **Stage 4** | |
| `examples/cfg/train_cfg_sft.py` | CFG SFT 训练入口 |
| `examples/cfg/run_cfg_sft.sh` | Stage 4 启动器 |
| `examples/cfg/config/libero_cfg_sft_test.yaml` | Stage 4 测试配置 |
| `rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py` | CFG 模型：训练 forward + 推理 sample_actions |
| `rlinf/datasets/transforms/tokenize_transforms.py` | `TokenizePromptWithGuidance` — 生成正/负 guidance prompt |
| `rlinf/workers/cfg/fsdp_cfg_worker.py` | FSDP CFG 训练 worker |
| `rlinf/workers/cfg/utils.py` | `DatasetWithAdvantage` — 将 advantage 标签注入样本 |
| `rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh` | Action model checkpoint 转换脚本 |
| **Stage 5** | |
| `examples/embodiment/train_debug_one_iter.py` | Debug one-iter 入口 |
| `examples/embodiment/run_debug_one_iter.sh` | Stage 5 启动器 |
| `examples/embodiment/config/one_iter_debug_libero10_test.yaml` | Stage 5 测试配置 |
| `rlinf/runners/debug_pi06_runner.py` | `DebugPi06Runner` — 离线 RL 训练循环 |
| `rlinf/workers/actor/debug_fsdp_actor_worker_cfg.py` | `DebugCFGFSDPActor` — 离线 CFG actor |

---

## 4. 完整使用流程

### 前置条件

```bash
# 1. 激活 openpi 环境
source switch_env openpi

# 2. 确认 GPU 可用
nvidia-smi

# 3. 确认以下文件存在
#    - 基础模型:          /path/to/pi05_base_pytorch/
#    - norm_stats.json:   /path/to/pi05_base_pytorch/physical-intelligence/libero/norm_stats.json
#    - Paligemma tokenizer: ~/.cache/openpi/big_vision/paligemma_tokenizer.model
```

### Stage 1: Compute Returns

**作用**：为数据集添加 `reward`、`return`、`prompt` 列，更新 `meta/stats.json`。

**配置**（`examples/process/config/compute_returns_test.yaml`）：
```yaml
data:
  datasets:
    - dataset_path: /path/to/sft_dataset      # SFT 人类示范数据
      type: sft
      failure_reward: -300.0
    - dataset_path: /path/to/rollout_dataset   # Rollout 采集数据
      type: rollout
      failure_reward: -300.0
  gamma: 1.0       # 折扣因子，1.0 表示不折扣
```

**启动**：
```bash
bash examples/process/run_compute_returns.sh compute_returns_test
```

**验证**：
```bash
python3 -c "
import json
for p in ['/path/to/sft_dataset', '/path/to/rollout_dataset']:
    stats = json.load(open(f'{p}/meta/stats.json'))
    assert 'return' in stats and 'reward' in stats
    print(f'{p}: return=[{stats[\"return\"][\"min\"]}, {stats[\"return\"][\"max\"]}]')
print('Stage 1 PASS')
"
```

**产出**：两个数据集的 parquet 文件原地更新（新增 return/reward/prompt 列）。

---

### Stage 2: Train Value Model

**作用**：训练 critic 模型学习 `V(s)`（状态值函数），用于下一步计算 advantage。

**配置**（`examples/vla_lib_sft/config/libero_value_model_test.yaml`）：
```yaml
data:
  datasets:
    - dataset_path: /path/to/sft_dataset       # 同 Stage 1
    - dataset_path: /path/to/rollout_dataset   # 同 Stage 1
  num_return_bins: 201              # 分类 loss 的 bin 数量
  normalize_to_minus_one_zero: true # 归一化到 [-1, 0]

actor:
  micro_batch_size: 2
  global_batch_size: 16    # = micro_batch_size × num_gpus (8 GPUs)
  model:
    model_path: /path/to/pi05_base_pytorch
    freeze_vlm: True       # 冻结 VLM 骨干，只训练 critic head
    expert_loss_type: categorical  # 分类损失
```

**启动**：
```bash
bash examples/vla_lib_sft/run_vla_lib_sft.sh libero_value_model_test
```

**转换 checkpoint**（Stage 3 需要 SafeTensors 格式）：
```bash
# 找到训练产出的 checkpoint
LATEST_LOG=$(ls -td logs/vla_lib_sft/libero_value_model_test-* | head -1)
CKPT_DIR="${LATEST_LOG}/vla_lib_value_sft/checkpoints/global_step_5/actor/model_state_dict"

# 用 symlink 避免路径中冒号导致 Hydra 解析错误
ln -sfn "${CKPT_DIR}" /tmp/step2_ckpt

# FSDP .pt → HuggingFace SafeTensors
bash rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf_vla_lib.sh \
    convertor.ckpt_path=/tmp/step2_ckpt/full_weights.pt \
    convertor.save_path=/tmp/step2_ckpt
```

**验证**：
```bash
ls /tmp/step2_ckpt/*.safetensors && echo "Stage 2 PASS"
```

**产出**：`/tmp/step2_ckpt/` 目录下的 SafeTensors 文件（即训练好的 value model）。

---

### Stage 3: Compute Advantages

**作用**：用 Stage 2 的 value model 推理 `V(s_t)` 和 `V(s_{t+N})`，计算 advantage 并标记正/负样本。

**公式**：
```
A_t = normalize(sum of r_{t:t+N}) + gamma^N * V(s_{t+N}) - V(s_t)
is_success = (A_t >= threshold)     # threshold = top 30% 分位点
```

**配置**（`examples/process/config/compute_advantages_test.yaml`）：
```yaml
advantage:
  value_checkpoint: /path/to/value_model   # ← Stage 2 产出的 /tmp/step2_ckpt
  positive_quantile: 0.3     # top 30% 标记为 True
  value_mode: expert_categorical
  tag: "test"                # 产出文件名: advantages_test.parquet

data:
  datasets:
    - dataset_path: /path/to/sft_dataset
    - dataset_path: /path/to/rollout_dataset
```

**启动**：
```bash
# 单 GPU 运行（--nproc 1），通过命令行覆盖 value_checkpoint 路径
bash examples/process/run_compute_advantages.sh compute_advantages_test \
    --nproc 1 \
    advantage.value_checkpoint=/tmp/step2_ckpt
```

**验证**：
```bash
python3 -c "
import pandas as pd
for p in ['/path/to/sft_dataset', '/path/to/rollout_dataset']:
    df = pd.read_parquet(f'{p}/meta/advantages_test.parquet')
    n_pos = df['is_success'].sum()
    print(f'{p}: {len(df)} samples, {n_pos} positive ({100*n_pos/len(df):.1f}%)')
print('Stage 3 PASS')
"
```

**产出**：每个数据集下 `meta/advantages_test.parquet`，包含列：`advantage` (float)、`is_success` (bool)、`value_current`、`value_next`。

---

### Stage 4: CFG SFT Training

**作用**：使用 Stage 3 的 advantage 标签，训练 Classifier-Free Guidance 模型。

**训练逻辑**：
1. 加载数据集，读取 `meta/advantages_test.parquet` 中的 `is_success` 标签
2. 每个样本根据标签选择引导 prompt：
   - `is_success=True` → `[POSITIVE][POSITIVE]\nTask: {task}`
   - `is_success=False` → `[NEGATIVE][NEGATIVE]\nTask: {task}`
3. 以 `unconditional_prob` (默认 0.3) 概率随机忽略引导，使用原始 prompt
4. 计算 flow matching loss

**配置**（`examples/cfg/config/libero_cfg_sft_test.yaml`）：
```yaml
data:
  advantage_tag: "test"         # ← 必须与 Stage 3 的 tag 一致
  datasets:
    - path: /path/to/sft_dataset
    - path: /path/to/rollout_dataset

actor:
  micro_batch_size: 2
  global_batch_size: 16         # = micro_batch_size × num_gpus
  model:
    model_path: /path/to/pi05_base_pytorch
    model_type: cfg_model       # ← 必须设为 cfg_model
    openpi:
      unconditional_prob: 0.3   # 30% 概率无条件训练
      cfgrl_guidance_scale: 1.0 # 推理时 guidance 强度
      guidance_type: positive   # 推理时使用 positive guidance
```

**启动**：
```bash
bash examples/cfg/run_cfg_sft.sh libero_cfg_sft_test
```

**转换 checkpoint**：
```bash
LATEST_LOG=$(ls -td logs/cfg_sft/libero_cfg_sft_test-* | head -1)
CKPT_DIR="${LATEST_LOG}/cfg_sft_test/checkpoints/global_step_5/actor/model_state_dict"
ln -sfn "${CKPT_DIR}" /tmp/step4_ckpt

bash rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh fsdp_model_convertor \
    convertor.ckpt_path=/tmp/step4_ckpt/full_weights.pt \
    convertor.save_path=/tmp/step4_ckpt \
    model.model_path=/path/to/pi05_base_pytorch \
    model.openpi.config_name=pi05_libero
```

**验证**：
```bash
ls /tmp/step4_ckpt/*.safetensors && echo "Stage 4 PASS"
```

**产出**：CFG action model checkpoint（SafeTensors），可用于推理。

---

### Stage 5（可选）: Debug One-Iter

**作用**：验证完整 pi06 RL 训练循环——包含 actor 初始化、advantage 数据集加载、离线训练。使用单 GPU 运行。

**配置**（`examples/embodiment/config/one_iter_debug_libero10_test.yaml`）：
```yaml
cluster:
  component_placement:
    env, rollout, actor: 0-0     # 单 GPU

data:
  advantage_tag: "test"          # ← 与 Stage 3/4 一致
  datasets:
    - path: /path/to/rollout_dataset

runner:
  offline_training_step: 5       # 离线训练步数
  val_check_interval: -1         # 不做环境评估

actor:
  micro_batch_size: 2
  global_batch_size: 2           # 单 GPU，所以 = micro_batch_size × 1
  model:
    model_type: cfg_model
    openpi:
      unconditional_prob: 0.3
      cfgrl_guidance_scale: 1.0
      guidance_type: positive
```

**启动**：
```bash
bash examples/embodiment/run_debug_one_iter.sh one_iter_debug_libero10_test
```

**转换 checkpoint**：
```bash
LATEST_LOG=$(ls -td logs/debug_one_iter_debug_libero10_test-* | head -1)
CKPT_DIR="${LATEST_LOG}/debug_pi06_test/checkpoints/global_step_5/actor/model_state_dict"
ln -sfn "${CKPT_DIR}" /tmp/step5_ckpt

bash rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh fsdp_model_convertor \
    convertor.ckpt_path=/tmp/step5_ckpt/full_weights.pt \
    convertor.save_path=/tmp/step5_ckpt \
    model.model_path=/path/to/pi05_base_pytorch \
    model.openpi.config_name=pi05_libero
```

---

### 快速参考：一键顺序执行

```bash
source switch_env openpi

# Stage 1: Compute Returns
bash examples/process/run_compute_returns.sh compute_returns_test

# Stage 2: Train Value Model + Convert Checkpoint
bash examples/vla_lib_sft/run_vla_lib_sft.sh libero_value_model_test
CKPT=$(ls -td logs/vla_lib_sft/libero_value_model_test-* | head -1)/vla_lib_value_sft/checkpoints/global_step_5/actor/model_state_dict
ln -sfn "$CKPT" /tmp/step2_ckpt
bash rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf_vla_lib.sh \
    convertor.ckpt_path=/tmp/step2_ckpt/full_weights.pt convertor.save_path=/tmp/step2_ckpt

# Stage 3: Compute Advantages
bash examples/process/run_compute_advantages.sh compute_advantages_test \
    --nproc 1 advantage.value_checkpoint=/tmp/step2_ckpt

# Stage 4: CFG SFT Training + Convert Checkpoint
bash examples/cfg/run_cfg_sft.sh libero_cfg_sft_test
CKPT=$(ls -td logs/cfg_sft/libero_cfg_sft_test-* | head -1)/cfg_sft_test/checkpoints/global_step_5/actor/model_state_dict
ln -sfn "$CKPT" /tmp/step4_ckpt
bash rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh fsdp_model_convertor \
    convertor.ckpt_path=/tmp/step4_ckpt/full_weights.pt convertor.save_path=/tmp/step4_ckpt \
    model.model_path=/path/to/pi05_base_pytorch model.openpi.config_name=pi05_libero

# Stage 5 (可选): Debug One-Iter
bash examples/embodiment/run_debug_one_iter.sh one_iter_debug_libero10_test
```
