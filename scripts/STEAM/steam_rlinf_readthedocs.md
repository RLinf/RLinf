# STEAM：用于离线策略优化的集成优势建模

来源页面：<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/steam.html#id7>

> 说明：本文档根据页面正文整理为 Markdown 版本，保留主要标题层级、算法流程、关键配置、命令、表格与文件结构，便于离线阅读和复制使用。

## 1. 简介

STEAM 是 RLinf 中用于离线策略优化的一套流程。其核心思想是使用成对分类的进度评论器（progress critic）和深度集成（deep ensemble）对已有数据进行逐帧打分，再将 conservative worst-of-N 集成估计转换为优势标签，最后用这些标签驱动与 RECAP 类似的无分类器引导（Classifier-Free Guidance, CFG）训练。

与 RECAP 相同，STEAM 不需要在线环境交互，因此适用于真实机器人等难以大规模在线采样的场景。二者的差异主要在价值信号来源：STEAM 不直接回归折扣回报，而是从帧对中学习时间进度（temporal progress）评论器，并通过集成方式降低单一预测器在分布外 rollout 状态上高估优势的风险。

## 2. 概览

| 项目 | 内容 |
|---|---|
| 目标 | 离线提升策略，无需新采样 |
| 算法 | STEAM，采用 worst-of-N 集成 |
| 模型 | SigLIP + Gemma3 评论器，策略模型为 π₀.₅ |
| 环境 / 数据 | LeRobot 格式数据集 |
| 训练流程 | 离线三阶段 |

整体流程为：先 SFT 一个集成进度评论器，再计算集成优势，之后进行 CFG 策略训练并评测。

前置条件包括：RLinf 安装环境、SigLIP + Gemma3 + π₀.₅ 检查点，以及 LeRobot 格式数据集。

## 3. 流程

一次 STEAM 运行包含两个 STEAM 特有阶段，并接一个 CFG 训练阶段。

```text
┌────────────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
│  Step 1                │     │  Step 2                │     │  Step 3              │
│  STEAM Value Model SFT │────▶│  Compute Ensemble      │────▶│  CFG Training        │
│                        │     │  Advantages            │     │                      │
│  Train an ensemble of  │     │  Worst-of-N ensemble   │     │  Train the policy    │
│  pair-classification   │     │  signed score -> bool  │     │  with classifier-    │
│  progress critics      │     │  advantage labels      │     │  free guidance       │
└────────────────────────┘     └────────────────────────┘     └──────────────────────┘
```

### 3.1 核心步骤

1. **Value Model SFT**  
   训练一组进度评论器。每个成员由 SigLIP、Gemma3 backbone 和分类头组成，输入帧对 \((o_t, o_{t+k})\)，输出有符号帧步幅所在的 bin。因此模型学习的是时间进度，而不是直接回归回报。

2. **Compute Ensemble Advantages**  
   对每一帧，所有集成成员在帧对 \((o_t, o_{t+k})\) 上推理。随后采用 worst-of-N 聚合规则：

   \[
   A = \min_m A_m
   \]

   得到连续优势分数 `advantage_continuous`，其范围为 \([-1, 1]\)。之后根据阈值或分位数规则将每帧标记为正样本或负样本。

3. **CFG Training**  
   将优势标签交给 CFG 阶段。高优势样本作为条件输入，低优势样本作为无条件输入，从而完成基于 classifier-free guidance 的策略优化。

## 4. STEAM 工作原理

### 4.1 优势建模

STEAM，全称 Self-supervised Temporal Ensemble Advantage Modeling，即自监督时序集成优势建模。它仅利用专家演示中的时序顺序学习优势，不依赖奖励、人工标注或外部价值模型。

对于专家轨迹中的帧对 \((f_i, f_j)\)，时序偏移定义为有符号帧步幅 \(j-i\)。未来帧相对于当前帧表示前向进度；若反向输入帧对，则会产生负偏移，用于描述退回或逆向行为。偏移还会根据轨迹长度进行归一化，使目标更接近时序效率而非原始步数。更短、更高效的执行会获得更高分数，缓慢或次优执行会获得较低分数。

每个预测器由 SigLIP 视觉编码器、Gemma3 语言模型和任务相关预测头组成。模型将帧对与语言指令映射到 `num_bins` 个时序偏移 bin 上，并通过交叉熵损失训练。逐成员优势由预测 bin 的期望值减去固定基线偏移得到：

\[
A_m = \frac{2}{N}\left( E_{b \sim p_{\theta_m}}[b] - b_{\mathrm{ref}} \right) \in [-1, 1]
\]

其中，\(E_b[b]\) 为预测器 \(m\) 对 bin 索引的期望，\(b_{\mathrm{ref}}\) 是确定性参考基线，即在最长 episode 上、固定前瞻 \(H\) 对应的、长度归一化后的真实偏移。若 `num_bins == 2`，该模型退化为二分类进度判别器。

### 4.2 优势估计

单个预测器在分布外 rollout 状态上可能高估优势。STEAM 使用多个预测器组成集成，并采用保守的 worst-of-N 规则：

\[
A_{\text{STEAM}} = \min_{m \in \{1, \dots, M\}} A_m
\]

聚合结果会写入 `advantage_continuous`。同时，逐成员均值、最小值、方差等诊断量也会被记录。

由于不同数据源的优势分布可能不同，`advantage_continuous` 会按数据源转换为布尔 `advantage`。支持两种 `label_mode`：

- `threshold`：rollout 帧满足 `advantage_continuous > positive_threshold` 时标记为 True；sft 帧默认标记为 True。
- `quantile`：rollout 帧中分数最高的 `rollout_quantile` 比例被标记为 True。若设置 `expert_quantile`，则 sft 帧也会按自身分数排序，只保留最高的 `expert_quantile` 比例为 True。

### 4.3 CFG 训练

STEAM 的优势标签用于驱动 OpenPI（π₀.₅）策略上的 CFG 阶段。高优势样本作为条件输入，低优势样本作为无条件输入。与 CFG 相关的机制包括：

- `positive_only_conditional`
- `unconditional_prob`
- `cfgrl_guidance_scale`

## 5. 安装

### 5.1 克隆 RLinf 仓库

```bash
# 中国大陆用户可使用以下镜像以获得更快下载速度：
# git clone https://ghfast.top/github.com/RLinf/RLinf.git
git clone https://github.com/RLinf/RLinf.git
cd RLinf
```

### 5.2 安装依赖

STEAM 与 RECAP 共用 OpenPI 环境。

#### 方式一：Docker 镜像

```bash
docker run -it --rm --gpus all \
   --shm-size 20g \
   --network host \
   --name rlinf \
   -v .:/workspace/RLinf \
   rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

# 为提高国内下载速度，可以使用：
docker run -it --rm --gpus all \
   --shm-size 20g \
   --network host \
   --name rlinf \
   -v .:/workspace/RLinf \
   docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

# 挂载用户目录
docker run -d --gpus all \
   --shm-size 120g \
   --network host \
   --name rlinf \
   -v /home/yz:/home/yz \
   docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \
   tail -f /dev/null

docker run -it -d --gpus '"device=1"' \
   --shm-size 20g \
   --network host \
   --name rlinf \
   -v /home/yz:/home/yz \
   docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
```

```bash
先看容器是否还在运行：

  docker ps -a --filter name=rlinf

  如果它还在运行，直接进入：

  docker exec -it rlinf /bin/bash

  停止容器运行
  docker stop rlinf

  如果它已经退出，删除旧容器：

  docker rm rlinf

  然后用显式 bash 重新启动：

退出镜像
  Ctrl+D
```
进入容器后，切换到 OpenPI 虚拟环境：
```bash
source switch_env openpi
```


#### 方式二：自建环境

```bash
# 为提高国内依赖安装速度，可以添加 --use-mirror 到下面的 install.sh 命令
bash requirements/install.sh embodied --model openpi --env maniskill_libero
source .venv/bin/activate
```

## 6. 下载模型

STEAM 价值模型由两个预训练 backbone 组成：

- SigLIP-so400m：`google/siglip-so400m-patch14-384`，作为视觉编码器。
- Gemma3-270M：`google/gemma-3-270m`，作为语言模型与分词器。

```bash
# 下载模型（选择任一方法）
# 方法 1: 使用 git clone
git lfs install
git clone https://huggingface.co/google/siglip-so400m-patch14-384
git clone https://huggingface.co/google/gemma-3-270m

# 方法 2: 使用 huggingface-hub
# 为提升国内下载速度，可以设置：
# export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface-hub
hf download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
hf download google/gemma-3-270m --local-dir gemma-3-270m
```

在模型配置 `examples/offline_rl/config/model/steam_value_model.yaml` 中设置路径：

```yaml
actor:
  model:
    vision_repo_id: /path/to/siglip-so400m-patch14-384
    language_repo_id: /path/to/gemma-3-270m
    tokenizer_path: /path/to/gemma-3-270m
```

## 7. 数据准备

STEAM 使用 LeRobot 格式数据集，数据分为两类：

- **SFT 数据集**：专家级演示，通常是成功轨迹。
- **Rollout 数据集**：在线交互采集的轨迹，可能包含成功、失败以及人工介入数据。

示例数据配置如下：

```yaml
data:
  train_data_paths:
    - dataset_path: /path/to/sft_dataset
      type: sft
    - dataset_path: /path/to/rollout_dataset
      type: rollout
```

注意：Step 1 与 Step 2 中的 `train_data_paths` 和 `data.k` 必须保持一致。优势计算需要使用与评论器训练时相同的时间步幅对帧对打分。

### 7.1 流程 Tag 系统

STEAM 使用 `advantage tag` 在各步骤之间传递数据。与 RECAP 不同，STEAM 没有单独的 compute returns 步骤，因此没有 `returns_tag`。唯一需要关注的是 `advantage_tag`：

- Step 2 的 `advantage.tag` 写出 `meta/advantages_{tag}.parquet`。
- Step 3 的 `data.advantage_tag` 读取对应的优势文件。

| 步骤 | 配置字段 | 说明 |
|---|---|---|
| 2 | `advantage.tag` | 写入 `meta/advantages_{tag}.parquet` |
| 3 | `data.advantage_tag` | 读取 `meta/advantages_{tag}.parquet` |

## 8. Step 1：价值模型 SFT

该阶段训练集成进度评论器。每个成员由 SigLIP + Gemma3 backbone 和分类头组成。成员从共享 backbone 克隆而来，并对 value head 重新设种子，使集成方差可以作为认知不确定性信号。

配置文件位于：

- `examples/offline_rl/config/steam_value_model_sft.yaml`
- `examples/offline_rl/config/model/steam_value_model.yaml`

关键字段示例：

```yaml
data:
  train_data_paths:
    - dataset_path: /path/to/sft_dataset
      type: sft               # 此阶段仅需专家数据
  k: 32                       # 最大有符号步幅 K（帧对时间尺度）
  # 评论器逐帧加载的图像（视角）名称；必须与检查点训练时的视角一致。
  # 缺失的视角会被补成零占位（mask=False）。
  camera_keys: [face_view, left_wrist_view, right_wrist_view]
actor:
  micro_batch_size: 32
  global_batch_size: 512
  model:
    num_bins: 32              # 2 = 二分类进度；>2 = 多 bin（偶数）
    ensemble_size: 3          # 集成中评论器数量
    fusion_hidden_dim: 512
    freeze_vision_encoder: false
    freeze_language_model: false
    use_gradient_checkpointing: true
  optim:
    lr: 5.0e-5
    value_lr: 5.0e-5
```

### 8.1 关键参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `data.k` | `required` | 最大有符号步幅 \(K\)。多 bin 模式下，`2*K` 必须是 `num_bins` 的整数倍。 |
| `actor.model.num_bins` | `2` | bin 数量。`2` 为二分类进度，`>2` 且为偶数时表示多 bin 有符号步幅分类。 |
| `actor.model.ensemble_size` | `1` | 集成成员数。大于 1 时启用 worst-of-N 聚合和不确定性统计。 |

### 8.2 启动命令

```bash
bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft

# 命令行覆盖配置字段：
bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft data.k=8

# 后台运行
nohup bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft > /home/yz/projects/outputs/logs/RLinf_smovla_V3_0720_valueTrain0720.log 2>&1 &
echo $!

# 断点续训
nohup bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft \
    +runner.resume_dir=/home/yz/projects/RLinf/logs/steam_sft/steam_value_model_sft-20260713-07:00:10/steam_sft/checkpoints/global_step_1000 \
    > /home/yz/projects/outputs/logs/RLinf_V8-1_0713-2.log 2>&1 &

# 核心代码：
ls projects/RLinf/rlinf/workers/sft/fsdp_steam_sft_worker.py
```

数据格式：<(image_t, image_tk, prompt), label>，label为[-1,1]中共num_bins个均匀值，数值代表t-tk的进展长度，正负表示t->tk是否有进展

输入：任务 prompt + 两帧多相机图像
输出：判断第二帧相对第一帧是 progress 还是 regress
损失：num_bins-way cross entropy


### 8.3 输出

- 检查点位于 `logs/steam_sft/{config_name}-{timestamp}/.../checkpoints/global_step_{N}/actor`。
- TensorBoard 日志会同步输出。

### 8.4 关键指标

- `train/actor/loss`：有符号步幅 bin 上的交叉熵。
- `train/actor/accuracy`：最优 bin 分类准确率。
- `train/actor/grad_norm`：梯度范数。

## 9. Step 2：计算集成优势

该阶段使用训练好的集成评论器对每一帧推理，并写出逐帧优势标签。

配置文件为：

```text
examples/offline_rl/config/steam_compute_advantages_ensemble.yaml
```

配置示例：

```yaml
advantage:                    
  value_checkpoint: /path/to/steam_value_ensemble/checkpoints/global_step_N/actor
  batch_size: 256
  label_mode: quantile        # 必填："threshold" 或 "quantile"
  rollout_quantile: 0.3       # rollout 帧最高的 30% 标为 True
  expert_quantile: 0.8        # 可选：sft 帧最高的 80% 标为 True
  tag: steam_k32_ensemble3_q30
data:
  k: 32                       # 必须与 Step 1 的 data.k 一致
  camera_keys: [face_view, left_wrist_view, right_wrist_view]
  train_data_paths:
    - dataset_path: /path/to/sft_dataset
      type: sft
    - dataset_path: /path/to/rollout_dataset
      type: rollout
```

### 9.1 关键参数

`label_mode` 决定哪些参数生效：

- `threshold` 模式下，只有 `advantage.positive_threshold` 起作用。它是 \([-1, 1]\) 内的有符号分数阈值。rollout 帧分数高于该阈值时为正样本，sft 帧恒为正样本。
- `quantile` 模式下，`positive_threshold` 被忽略。该模式使用 `rollout_quantile` 和 `expert_quantile` 分别在 rollout 与 sft 数据池中选择分数最高的样本。若省略 `expert_quantile`，则所有 sft 帧都会被标为正。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `advantage.value_checkpoint` | `required` | Step 1 集成检查点路径，通常指向 `actor` 目录。 |
| `advantage.label_mode` | `required` | `threshold` 或 `quantile`，无默认值，需要显式设置。 |
| `advantage.positive_threshold` | `null` | \([-1, 1]\) 内的有符号分数阈值，仅 `label_mode=threshold` 时有效。 |
| `advantage.rollout_quantile` | `null` | rollout 帧被标为 True 的最高比例，仅 `label_mode=quantile` 时有效且必填。 |
| `advantage.expert_quantile` | `null` | sft 帧被标为 True 的最高比例，仅 `label_mode=quantile` 时有效，可选。 |
| `advantage.tag` | `required` | 输出 tag，写入 `meta/advantages_{tag}.parquet`。 |
| `data.k` | `required` | 帧对步幅，必须与 Step 1 训练时的 `data.k` 一致。 |

### 9.2 启动命令

```bash
# 自动检测 GPU 数；单卡与 torchrun 多卡均支持。
bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh steam_compute_advantages_ensemble

# 后台启动
nohup bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh steam_compute_advantages_ensemble > /home/yz/projects/outputs/logs/RLinf_V8-1_task2_valueInfer0720.log 2>&1 &

# 指定 GPU 数：
bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh steam_compute_advantages_ensemble --nproc 4

# 核心代码：
ls /home/yz/projects/RLinf/examples/offline_rl/advantage_labeling/steam/process/compute_advantages_ensemble.py
```

### 9.3 输出文件

- `meta/advantages_{tag}.parquet`：逐帧优势标签文件，包含 `advantage`、`advantage_continuous`、`ensemble_signed_score`、逐成员值，以及集成熵、方差等诊断量。
- `meta/mixture_config.yaml`：每个 tag 一条记录，包含 `label_mode`、阈值、`ensemble_size`、`num_bins` 和正样本计数。

## 10. Step 3：CFG 训练

策略优化直接读取 Step 2 产生的 STEAM 优势 parquet 文件。需要将 CFG 配置中的 `data.advantage_tag` 指向 Step 2 的 `advantage.tag`。

```bash
bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi \
    data.advantage_tag=steam_k32_ensemble3_q30
```

完整 CFG 配置与参数可参考 RLinf 文档中的 RECAP / CFG 训练阶段。

## 11. STEAM 实验结果

文档中给出了四个真实机器人操作任务上的对比结果。对比方法包括 BC、HG-DAgger、RECAP 和 STEAM。

### 11.1 成功率（%）

| 任务 | BC | HG-DAgger | RECAP | STEAM |
|---|---:|---:|---:|---:|
| Towel Folding | 33.3 | 40 | 55.6 | 92.3（↑59） |
| Chips Checkout | 39.5 | 53.3 | 53.3 | 93.8（↑54.3） |
| Pick-and-Place | 63.8 | — | 53.8 | 80（↑16.2） |
| Cola Restocking | 52 | — | 52.9 | 75（↑23） |

### 11.2 吞吐量（每小时成功 episode 数）

| 任务 | BC | HG-DAgger | RECAP | STEAM |
|---|---:|---:|---:|---:|
| Towel Folding | 42 | 48 | 39 | 58 |
| Chips Checkout | 16.3 | 22.0 | 23.9 | 47.5 |
| Pick-and-Place | 230 | — | 161 | 254 |
| Cola Restocking | 71 | — | 46 | 90 |

总体来看，STEAM 在四个任务上将成功率提升至 75%–93.8%，并取得最高吞吐量。其中 Towel Folding 与 Chips Checkout 的成功率提升最明显。

## 12. 进阶用法

### 12.1 合并集成检查点

独立训练的单模型成员，或从已有集成中抽取的成员，可以合并为一个集成推理检查点。每个 `--member` 可以是一个检查点路径，也可以通过 `PATH:idx` 从已有集成中抽取第 `idx` 个成员。

```bash
python examples/offline_rl/advantage_labeling/steam/process/merge_steam_ensemble.py \
    --member /path/to/seed1/checkpoints/global_step_5000/actor \
    --member /path/to/seed2/checkpoints/global_step_5000/actor \
    --member /path/to/ensemble/checkpoints/global_step_6000/actor:2 \
    --output /path/to/merged/actor
```

合并逻辑位于：

```text
rlinf.models.embodiment.value_model.steam.checkpoint_merge.merge_ensemble_checkpoints
```

### 12.2 阈值 / 分位数重标注

若希望改变标注阈值，但不想重新运行 GPU 推理，可以对已有优势 parquet 重标注。该过程为纯 CPU 流程，会复用 `advantage_continuous`。

```bash
python examples/offline_rl/advantage_labeling/steam/process/relabel_advantages.py \
    --dataset_paths /path/to/sft_ds /path/to/rollout_ds \
    --source_tag steam_k32_ensemble3_q30 \
    --new_tag steam_k32_ensemble3_q20 \
    --mode quantile --rollout_quantile 0.2
```

重标注脚本位于：

```text
examples/offline_rl/advantage_labeling/steam/process/relabel_advantages.py
```

### 12.3 可视化优势

可以从优势 parquet 渲染分布图、逐成员诊断图、不确定性图、逐 episode 图以及 episode 时间线等。

```bash
python examples/offline_rl/advantage_labeling/steam/process/visualize_advantage.py \
    --dataset /path/to/dataset \
    --tag steam_k32_ensemble3_q30 \
    --output outputs/steam_viz
```

## 13. 可视化与结果

训练指标可参考 RLinf 文档中的“训练指标”部分。启动 TensorBoard 的命令为：

```bash
tensorboard --logdir ./logs --port 6006
```

## 14. 文件结构

STEAM 的步骤脚本主要位于 `examples/` 下。其中包括模型推理与标注逻辑；模型和数据集代码位于 `rlinf/models` 与 `rlinf/data/datasets` 下；与 RECAP 共享的后处理逻辑位于 `rlinf/data/process/` 下。

```text
examples/offline_rl/
├── config/                                  # 共享生产配置
│   ├── steam_value_model_sft.yaml           # Step 1
│   ├── steam_compute_advantages_ensemble.yaml   # Step 2
│   ├── cfg_rl_openpi.yaml                   # Step 3（CFG，与 RECAP 共用）
│   └── model/
│       └── steam_value_model.yaml           # 价值模型架构默认配置
├── advantage_labeling/
│   └── steam/
│       ├── train_steam.py                   # Step 1：价值模型 SFT 入口
│       ├── run_steam_sft.sh                 # Step 1 启动脚本
│       └── process/                         # Step 2：自包含入口脚本（与 recap 一致）
│           ├── compute_advantages_ensemble.py     # Step 2：集成推理 + 标注（Hydra）
│           ├── relabel_advantages.py              # CLI：重标注优势（CPU）
│           ├── merge_steam_ensemble.py            # CLI：合并集成检查点
│           ├── visualize_advantage.py             # 优势可视化
│           └── run_compute_advantages_ensemble.sh # Step 2 启动脚本
└── policy_optimization/
    └── cfg_rl/
        ├── train_cfg.py                      # Step 3：CFG 策略训练
        └── run_cfg_rl.sh                     # Step 3 启动脚本

rlinf/
├── models/embodiment/value_model/steam/     # 评论器、集成、配置、合并
│   ├── modeling_steam.py / modeling_critic.py
│   ├── ensemble_modeling_critic.py          # worst-of-N + coerce_to_ensemble
│   └── checkpoint_merge.py
├── data/datasets/steam/                     # pair_dataset.py、mixture.py、binning.py
└── data/process/                            # 共享、模型无关（RECAP + STEAM）
    ├── advantage.py                         # 分位数阈值 + 布尔标签
    ├── distributed.py                       # 分片推理辅助
    └── mixture_config.py                    # meta/mixture_config.yaml tag I/O
```
